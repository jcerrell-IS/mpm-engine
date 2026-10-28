"""Blender-side renderer for one NCLaw trajectory panel. Run inside Blender only.

Invoked by sim/nclaw_blender_render.py as

    Blender --background --python experiments/nclaw/blender_scene.py -- \
        --npz out/nclaw_suite/dumps/water_spot_mild_truth.npz \
        --out-dir out/nclaw_suite/blender/water_spot_mild/truth \
        --material water --spacing 0.015625 --floor-z 0.0567 --stride 2

The script opens the npz once, builds one scene, and renders numbered PNG stills
in a single Blender session: the particle cloud lives in one mesh with a fixed
vertex count whose coordinates are overwritten per frame with foreach_set, and a
geometry-nodes modifier turns those vertices into a surface (jelly, plasticine,
water) or into instanced grains (sand).

This module imports bpy and therefore never runs under the repo venv. It is
outside ident/ and imports nothing from the project.

Blender 5.2 API notes
---------------------
Resolution Mode on Points to Volume and on Volume to Mesh is no longer a node
property with enum identifiers. It is a MENU input socket whose accepted values
are the interface labels "Amount" and "Size"; assigning the old 'VOXEL_AMOUNT'
raises TypeError.

The render engine enum lists only BLENDER_EEVEE under --background, but assigning
scene.render.engine = 'CYCLES' works, and the cycles addon preferences report the
Metal GPU even though compute_device_type has an empty enum item list there.

GeometryNodeRandomValue does not exist; the function node namespace holds the
random value node in 5.2.

Node group sockets go through node_group.interface.new_socket(name, in_out=,
socket_type=). Material.use_nodes and World.use_nodes still work but warn that
they go away in Blender 6.0.

Per frame the vertex coordinates are written with mesh.vertices.foreach_set("co"),
then mesh.update(), obj.update_tag() and scene.frame_set() force the geometry
nodes tree to re-evaluate. Renders of two different frames were confirmed to
differ by md5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import bpy  # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore

# NCLaw's own per material render colours, from
# NCLaw/experiments/configs/env/blob/bsdf_pcd/*.yaml. Those files feed Mitsuba
# rgb reflectances, which are sRGB encoded, while a Blender Base Color socket is
# linear. Feeding the raw numbers in produced pale washed out surfaces (their
# saturated red jelly came out salmon), so the values are decoded here.
NCLAW_SRGB = {
    "jelly": (0.92941176, 0.32941176, 0.23137255),
    "plasticine": (0.59607843, 0.98431373, 0.59607843),
    "sand": (0.96078431, 0.88235294, 0.63529412),
    "water": (0.45490196, 0.80000000, 0.95686275),
}


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


NCLAW_RGB = {k: tuple(srgb_to_linear(c) for c in v) for k, v in NCLAW_SRGB.items()}
# Their plasticine value decodes to a neon green that does not read as clay, so
# that one material keeps their hue at clay saturation instead.
NCLAW_RGB["plasticine"] = (0.36, 0.52, 0.33)

# Light power per unit squared camera distance, so a fitted camera keeps its look.
KEY_ENERGY = 55.0
FILL_ENERGY = 16.0
RIM_ENERGY = 13.0
WORLD_COLOR = (0.26, 0.28, 0.32)
GROUND_COLOR = (0.42, 0.42, 0.44)

# Fixed three quarter view direction: the camera sits at target + dist * CAM_DIR.
# The driver fits target and dist to both panels of a cell so the two renders share
# one camera.
CAM_DIR = (0.5137, -0.7757, 0.3676)
CAM_LENS = 50.0


# ----------------------------------------------------------------------------- scene


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*WORLD_COLOR, 1.0)
    bg.inputs["Strength"].default_value = 1.0


def make_camera(target, dist: float) -> None:
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = CAM_LENS
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    loc = Vector(target) + dist * Vector(CAM_DIR)
    cam.location = loc
    cam.rotation_euler = (Vector(target) - loc).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    print(f"[scene] camera at {tuple(round(v, 3) for v in loc)} "
          f"target {tuple(round(v, 3) for v in target)} dist {dist:.3f}", flush=True)


def add_area_light(name: str, loc, target, size: float, energy: float) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.size = size
    data.energy = energy
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    direction = Vector(target) - Vector(loc)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_lights(target, dist: float) -> None:
    """Key, fill and rim placed relative to the fitted camera, so the lighting
    keeps the same ratios whatever the cell extent."""
    t = Vector(target)
    key = t + dist * Vector((0.62, -0.72, 1.05))
    fill = t + dist * Vector((-0.95, -0.45, 0.42))
    rim = t + dist * Vector((0.05, 1.05, 0.55))
    add_area_light("key", key, t, 1.5 * dist, KEY_ENERGY * dist * dist)
    add_area_light("fill", fill, t, 2.2 * dist, FILL_ENERGY * dist * dist)
    add_area_light("rim", rim, t, 1.5 * dist, RIM_ENERGY * dist * dist)


def make_ground(floor_z: float) -> None:
    mesh = bpy.data.meshes.new("ground")
    s = 30.0
    mesh.from_pydata(
        [(-s, -s, floor_z), (s, -s, floor_z), (s, s, floor_z), (-s, s, floor_z)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new("ground", mesh)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new("ground_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*GROUND_COLOR, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    obj.data.materials.append(mat)


# ------------------------------------------------------------------------ materials


def make_material(material: str) -> bpy.types.Material:
    rgb = NCLAW_RGB[material]
    mat = bpy.data.materials.new(f"{material}_mat")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    if material == "jelly":
        b.inputs["Roughness"].default_value = 0.15
        b.inputs["IOR"].default_value = 1.45
        b.inputs["Transmission Weight"].default_value = 0.35
        b.inputs["Subsurface Weight"].default_value = 0.5
        b.inputs["Subsurface Scale"].default_value = 0.06
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.25)
        mat.use_raytrace_refraction = True
        mat.use_screen_refraction = True
    elif material == "plasticine":
        b.inputs["Roughness"].default_value = 0.6
        b.inputs["Specular IOR Level"].default_value = 0.35
        b.inputs["Subsurface Weight"].default_value = 0.12
        b.inputs["Subsurface Scale"].default_value = 0.005
    elif material == "water":
        b.inputs["Roughness"].default_value = 0.02
        b.inputs["IOR"].default_value = 1.33
        b.inputs["Transmission Weight"].default_value = 1.0
        mat.use_raytrace_refraction = True
        mat.use_screen_refraction = True
        mat.refraction_depth = 0.02
    elif material == "sand":
        b.inputs["Roughness"].default_value = 0.9
        b.inputs["Specular IOR Level"].default_value = 0.25
    else:
        raise SystemExit(f"unknown material {material}")
    return mat


# -------------------------------------------------------------------- geometry nodes


def _io(ng) -> tuple:
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = ng.nodes.new("NodeGroupInput")
    gout = ng.nodes.new("NodeGroupOutput")
    gin.location = (-800, 0)
    gout.location = (600, 0)
    return gin, gout


def build_volume_group(mat, spacing: float, radius_scale: float, voxel_scale: float,
                       threshold: float):
    """Mesh to Points, Points to Volume, Volume to Mesh: a closed surface."""
    ng = bpy.data.node_groups.new("surface", "GeometryNodeTree")
    gin, gout = _io(ng)
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    p2v = ng.nodes.new("GeometryNodePointsToVolume")
    v2m = ng.nodes.new("GeometryNodeVolumeToMesh")
    smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    for i, n in enumerate((m2p, p2v, v2m, smooth, setmat)):
        n.location = (-600 + 220 * i, 0)

    voxel = voxel_scale * spacing
    m2p.inputs["Radius"].default_value = 0.5 * spacing
    p2v.inputs["Density"].default_value = 1.0
    # Blender 5.2: Resolution Mode is a MENU socket whose values are the UI
    # labels "Amount" and "Size", not the old enum identifiers.
    p2v.inputs["Resolution Mode"].default_value = "Size"
    p2v.inputs["Voxel Size"].default_value = voxel
    p2v.inputs["Radius"].default_value = radius_scale * spacing
    v2m.inputs["Resolution Mode"].default_value = "Size"
    v2m.inputs["Voxel Size"].default_value = voxel
    v2m.inputs["Threshold"].default_value = threshold
    v2m.inputs["Adaptivity"].default_value = 0.0
    smooth.inputs["Shade Smooth"].default_value = True
    setmat.inputs["Material"].default_value = mat

    lk = ng.links.new
    lk(gin.outputs[0], m2p.inputs["Mesh"])
    lk(m2p.outputs["Points"], p2v.inputs["Points"])
    lk(p2v.outputs["Volume"], v2m.inputs["Volume"])
    lk(v2m.outputs["Mesh"], smooth.inputs["Mesh"])
    lk(smooth.outputs["Geometry"], setmat.inputs["Geometry"])
    lk(setmat.outputs["Geometry"], gout.inputs[0])
    return ng


def build_grain_group(mat, spacing: float, radius_scale: float, subdivisions: int):
    """Mesh to Points then icosphere instances: sand reads as grains."""
    ng = bpy.data.node_groups.new("grains", "GeometryNodeTree")
    gin, gout = _io(ng)
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    ico = ng.nodes.new("GeometryNodeMeshIcoSphere")
    smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    iop = ng.nodes.new("GeometryNodeInstanceOnPoints")
    m2p.location = (-500, 120)
    ico.location = (-500, -220)
    smooth.location = (-280, -220)
    setmat.location = (-60, -220)
    iop.location = (200, 0)

    m2p.inputs["Radius"].default_value = 0.5 * spacing
    ico.inputs["Radius"].default_value = radius_scale * spacing
    ico.inputs["Subdivisions"].default_value = subdivisions
    smooth.inputs["Shade Smooth"].default_value = True
    setmat.inputs["Material"].default_value = mat

    lk = ng.links.new
    lk(gin.outputs[0], m2p.inputs["Mesh"])
    lk(ico.outputs["Mesh"], smooth.inputs["Mesh"])
    lk(smooth.outputs["Geometry"], setmat.inputs["Geometry"])
    lk(m2p.outputs["Points"], iop.inputs["Points"])
    lk(setmat.outputs["Geometry"], iop.inputs["Instance"])
    lk(iop.outputs["Instances"], gout.inputs[0])
    return ng


# ------------------------------------------------------------------------- rendering


def setup_render(engine: str, width: int, height: int, samples: int,
                 exposure: float) -> None:
    scene = bpy.context.scene
    r = scene.render
    r.resolution_x = width
    r.resolution_y = height
    r.resolution_percentage = 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    # AgX desaturates the NCLaw colours badly (their saturated red jelly rendered
    # as pale pink), so use the plain sRGB transform and control exposure instead.
    for name in ("Standard", "AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = name
            break
        except TypeError:
            continue
    scene.view_settings.exposure = exposure
    print(f"[scene] view transform {scene.view_settings.view_transform} "
          f"exposure {exposure}", flush=True)
    if engine == "cycles":
        # 'CYCLES' is absent from the engine enum listing in --background but
        # assignment still works in Blender 5.2.
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = dev.type == "METAL"
        except Exception as exc:  # pragma: no cover
            print(f"[scene] Metal setup failed, staying on CPU: {exc}")
        scene.cycles.device = "GPU"
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 8
        scene.cycles.transmission_bounces = 8
    else:
        scene.render.engine = "BLENDER_EEVEE"
        ev = scene.eevee
        ev.taa_render_samples = samples
        ev.use_raytracing = True
        ev.ray_tracing_method = "SCREEN"
        ev.ray_tracing_options.use_denoise = True
        ev.ray_tracing_options.resolution_scale = "1"
        ev.use_shadows = True
        ev.shadow_ray_count = 2
        ev.shadow_step_count = 8


_reported: list[str] = []


def report_geometry(obj, n_part: int, tag: str) -> None:
    """Print the evaluated geometry size once, so an unseen render can be checked."""
    if _reported:
        return
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    verts = polys = 0
    data = getattr(ev, "data", None)
    if isinstance(data, bpy.types.Mesh):
        verts, polys = len(data.vertices), len(data.polygons)
    inst = sum(1 for i in dg.object_instances if i.is_instance and i.parent == ev)
    print(f"[scene] {tag}: particles={n_part} evaluated verts={verts} "
          f"polys={polys} instances={inst}", flush=True)
    _reported.append(tag)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--spacing", type=float, required=True)
    ap.add_argument("--floor-z", type=float, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--engine", default="eevee", choices=["eevee", "cycles"])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", default="", help="comma separated frame indices; "
                                                "overrides --stride (stills mode)")
    ap.add_argument("--cam-target", default="0.5,0.5,0.3")
    ap.add_argument("--cam-dist", type=float, default=2.7)
    ap.add_argument("--exposure", type=float, default=-0.6)
    ap.add_argument("--smooth-repeat", type=int, default=6,
                    help="Smooth modifier iterations on the surfaced mesh")
    ap.add_argument("--radius-scale", type=float, default=2.0)
    ap.add_argument("--voxel-scale", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--grain-scale", type=float, default=0.6)
    ap.add_argument("--grain-subdiv", type=int, default=1)
    args = ap.parse_args(argv)

    x = np.load(args.npz)["x"]  # (frames, N, 3), float32, metres, z up
    n_frames, n_part = x.shape[0], x.shape[1]
    if args.frames:
        frames = [int(v) for v in args.frames.split(",")]
    else:
        frames = list(range(0, n_frames, args.stride))
    os.makedirs(args.out_dir, exist_ok=True)

    target = tuple(float(v) for v in args.cam_target.split(","))
    reset_scene()
    setup_render(args.engine, args.width, args.height, args.samples, args.exposure)
    make_camera(target, args.cam_dist)
    make_lights(target, args.cam_dist)
    make_ground(args.floor_z)

    mesh = bpy.data.meshes.new("particles")
    mesh.from_pydata([tuple(p) for p in x[frames[0]].astype(float)], [], [])
    mesh.update()
    obj = bpy.data.objects.new("particles", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mat = make_material(args.material)
    if args.material == "sand":
        ng = build_grain_group(mat, args.spacing, args.grain_scale, args.grain_subdiv)
    else:
        ng = build_volume_group(mat, args.spacing, args.radius_scale,
                                args.voxel_scale, args.threshold)
    modifier = obj.modifiers.new("surface", "NODES")
    modifier.node_group = ng
    if args.material != "sand" and args.smooth_repeat > 0:
        # Volume to Mesh follows the union of the particle spheres, which shows
        # every particle as a lump; a plain Smooth modifier on the surfaced mesh
        # removes that without touching the silhouette.
        smooth_mod = obj.modifiers.new("relax", "SMOOTH")
        smooth_mod.factor = 1.0
        smooth_mod.iterations = args.smooth_repeat

    scene = bpy.context.scene
    flat = np.empty(3 * n_part, dtype=np.float32)
    times = []
    for out_i, fi in enumerate(frames):
        flat[:] = x[fi].reshape(-1)
        mesh.vertices.foreach_set("co", flat)
        mesh.update()
        obj.update_tag()
        scene.frame_set(out_i)
        bpy.context.view_layer.update()
        report_geometry(obj, n_part, f"frame {fi}")
        scene.render.filepath = os.path.join(args.out_dir, f"{out_i:04d}.png")
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        times.append(dt)
        print(f"[scene] frame {fi} -> {out_i:04d}.png  {dt:.2f} s", flush=True)
    print(f"[scene] {len(frames)} stills, {args.engine}, "
          f"mean {np.mean(times):.2f} s/frame, total {np.sum(times):.1f} s",
          flush=True)


if __name__ == "__main__":
    main()
