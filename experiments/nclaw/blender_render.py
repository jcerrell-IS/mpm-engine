"""Blender side-by-side videos of the NCLaw comparison: truth versus recovered law.

Replaces the matplotlib scatter panels of sim/nclaw_suite_video.py with realistic
renders in the spirit of NCLaw's own figures: the same trajectory dumps, the same
fixed camera for both panels, surfaced jelly, plasticine and water, and sand drawn
as instanced grains. Per cell the driver renders two panels of PNG stills through
Blender, encodes each panel, and stacks them into one mp4 with per panel labels.

Run:
    .venv/bin/python -m experiments.nclaw.blender_render                     # all cells
    .venv/bin/python -m experiments.nclaw.blender_render water_spot_mild
    .venv/bin/python -m experiments.nclaw.blender_render --still 60 water_spot_mild
    .venv/bin/python -m experiments.nclaw.blender_render --engine eevee sand_blub_mild

Outputs:
    out/nclaw_suite/blender/<cell>/{truth,rec}/NNNN.png   stills
    out/nclaw_suite/blender/<cell>/{truth,rec}.mp4        per panel encodes
    out/nclaw_suite/videos_blender/<cell>.mp4             final side by side

Measurements and deviations
---------------------------
Engine. Blender 5.2 lists only BLENDER_EEVEE in the render engine enum under
--background, yet assigning 'CYCLES' works and Metal picks up the M3 Max GPU.
Measured at 960x720 on this machine: EEVEE 0.57 to 0.76 s per frame; Cycles at 64
samples with denoising 1.5 s (sand), 1.6 to 1.9 s (water), 1.8 s (plasticine),
2.3 s (jelly cube) per frame. The first Cycles frame on a cold Metal kernel cache
cost 105 s; later sessions reuse the cache and start at full speed. Cycles is
therefore the default and it gives water real refraction instead of EEVEE's screen
space approximation. The full water_spot_mild cell, meaning 63 stills per panel,
two panels and the ffmpeg passes, took 223.6 s wall clock, so about 1.9 minutes
per panel against the 5 minute budget. --engine eevee is kept for iteration.

Surfacing resolution. The suggested radius of 1.6 x particle spacing with a voxel
fine enough to close the surface renders every particle as a visible lump: at
0.35 x spacing the water frame 60 surface came out with 165310 vertices and read
as a heap of balls rather than a liquid. Laplacian smoothing cannot fix it because
the bumps are as wide as the sphere radius, about 9 voxels: 30 iterations of the
Smooth modifier moved the image by a mean of 2.6 grey levels out of 255. The
surface here therefore uses radius 2.0 x spacing with voxel size 1.0 x spacing
(18442 vertices for the same frame) plus 6 smoothing iterations, which reads as a
liquid and renders faster. Voxel 0.6 x spacing was tried in between and was still
bumpy.

Particle spacing. Taken as mean(volume0)^(1/3) from the dump, which is 0.0156 m
for the 2 particles per cell scenes and 0.0104 m for sand at 3 per cell. The
measured nearest neighbour distance is smaller and drifts during flow: sand 0.0086
at frame 0 and 0.0067 at frame 70, water 0.0128 and 0.0109. The surfacing radius
of 2.0 x spacing therefore sits at 2.4 to 2.9 x the realised neighbour distance,
which is what closes the surface.

Grain radius. Kept at 0.6 x spacing. Enlarging it does almost nothing to the
silhouette, so the grains already overlap: rasterising the projected grains at
frame 40 of sand_blub_mild gives 5.39 percent frame coverage at 0.6 x spacing and
5.77 percent at 1.0 x, with fill inside the silhouette bounding box moving only
from 44.0 to 45.7 percent. The 44 percent is the shape of the spread pile, not
gaps between grains. The rendered coverage of 4.64 percent agrees with the 5.39
percent analytic value, which is how the grain geometry was checked without
looking at the image.

Camera. Fitted per cell rather than fixed at the unit box, since the cells occupy
very different parts of the box: the look at point is the time averaged particle
centroid and the distance is bisected until every per frame bounding box corner of
both panels sits inside 95 percent of the frame. Both panels of a cell share the
fit, so truth and recovered stay comparable. Aiming at the centre of the swept
bounding box was tried first and left the material low in frame under a large
empty floor.

Colour. The NCLaw bsdf_pcd values are Mitsuba rgb reflectances and are sRGB
encoded, while a Blender Base Color socket is linear, so they are decoded before
use; feeding the raw numbers in turned their saturated red jelly into pale salmon.
Their plasticine value decodes to a neon green that does not read as clay, so that
one material keeps their hue at clay saturation, (0.36, 0.52, 0.33) linear.

Tone mapping. View transform Standard with exposure -0.6, not the Blender default
AgX: AgX desaturated the NCLaw palette to the point where the jelly read as pink.

Floor. The plane sits at the minimum particle height over both panels of the cell
(0.0567 m for water_spot_mild), not at the nominal 3 cell boundary height
0.09375 m, because particles penetrate the boundary condition by roughly a
particle radius and a plane at 0.09375 m would hide them.

Lighting. World background and key light were dimmed from the first attempt after
measuring the render: mean pixel value 224 of 255 with standard deviation 16 was a
washed out frame, and the values used here give mean 191 to 197 with standard
deviation 21 to 23.

Frame count and rate. Stride 2 and 25 fps as in sim/nclaw_suite_video.py, so 63
frames and 2.52 s per video, matching the matplotlib versions frame for frame.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.nclaw.suite_video import CELLS  # noqa: E402

BLENDER = os.environ.get("BLENDER",
                         "/Applications/Blender.app/Contents/MacOS/Blender")
FFMPEG = os.environ.get("FFMPEG", "/opt/homebrew/bin/ffmpeg")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

OUT = ROOT / "out" / "nclaw_suite"
DUMPS = OUT / "dumps"
STILLS = OUT / "blender"
VIDEOS = OUT / "videos_blender"
SCENE_SCRIPT = Path(__file__).resolve().parent / "blender_scene.py"

PANELS = (("truth", "truth"), ("rec", "recovered"))


def run(cmd: list[str], tag: str) -> None:
    print(f"[{tag}] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# Same three quarter direction as experiments/nclaw/blender_scene.py CAM_DIR.
CAM_DIR = np.array([0.5137, -0.7757, 0.3676])
CAM_LENS = 50.0
SENSOR = 36.0


def camera_basis(target: np.ndarray, dist: float):
    loc = target + dist * CAM_DIR
    fwd = target - loc
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return loc, fwd, right, up


def frame_extent(pts: np.ndarray, target: np.ndarray, dist: float,
                 width: int, height: int) -> float:
    """Largest normalised screen coordinate of pts; 1.0 is the frame edge."""
    loc, fwd, right, up = camera_basis(target, dist)
    p = pts - loc
    depth = p @ fwd
    if depth.min() <= 1e-3:
        return 1e9
    half_x = 0.5 * SENSOR
    half_y = 0.5 * SENSOR * min(1.0, height / width)
    u = (p @ right) / depth * (CAM_LENS / half_x)
    v = (p @ up) / depth * (CAM_LENS / half_y)
    return float(max(np.abs(u).max(), np.abs(v).max()))


def fit_camera(pts: np.ndarray, centroid: np.ndarray, width: int, height: int,
               margin: float = 0.95) -> tuple[np.ndarray, float]:
    """Target and distance that hold every point inside margin of the frame.

    The look at point is the time averaged particle centroid rather than the
    centre of the swept bounding box: aiming at the box centre left the material
    sitting low in frame with empty floor above it. Both panels of a cell are
    fitted together, so truth and recovered share one camera and stay comparable.
    """
    target = centroid
    lo, hi = 0.2, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if frame_extent(pts, target, mid, width, height) > margin:
            lo = mid
        else:
            hi = mid
    return target, hi


def scene_geometry(stem: str, stride: int, width: int, height: int):
    """Spacing from mean particle volume, floor from the lowest particle, and a
    camera fitted to the extent of both panels."""
    truth = np.load(DUMPS / f"{stem}_truth.npz")
    rec = np.load(DUMPS / f"{stem}_rec.npz")
    spacing = float(np.mean(truth["volume0"]) ** (1.0 / 3.0))
    xt, xr = truth["x"], rec["x"]
    floor_z = float(min(xt[..., 2].min(), xr[..., 2].min()))
    corners = []
    centroids = []
    for x in (xt, xr):
        sub = x[::stride]
        lo = sub.min(axis=1)
        hi = sub.max(axis=1)
        centroids.append(sub.reshape(-1, 3).mean(axis=0))
        for mask in range(8):
            corners.append(np.where(
                [[bool(mask >> k & 1) for k in range(3)]] * len(lo), hi, lo))
    pts = np.concatenate(corners, axis=0)
    centroid = np.mean(centroids, axis=0).astype(float)
    target, dist = fit_camera(pts, centroid, width, height)
    return spacing, floor_z, target, dist


def render_panel(cell: str, panel: str, material: str, stem: str, spacing: float,
                 floor_z: float, target, dist: float, args) -> Path:
    out_dir = STILLS / cell / panel
    if out_dir.exists() and not args.reuse_stills:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        BLENDER, "--background", "--python", str(SCENE_SCRIPT), "--",
        "--npz", str(DUMPS / f"{stem}_{panel}.npz"),
        "--out-dir", str(out_dir),
        "--material", material,
        "--spacing", f"{spacing:.6f}",
        "--floor-z", f"{floor_z:.6f}",
        "--stride", str(args.stride),
        "--engine", args.engine,
        "--samples", str(args.samples),
        "--width", str(args.width),
        "--height", str(args.height),
        "--cam-target", ",".join(f"{v:.5f}" for v in target),
        "--cam-dist", f"{dist:.5f}",
        "--exposure", str(args.exposure),
    ]
    if args.still is not None:
        cmd += ["--frames", str(args.still)]
    run(cmd, f"{cell}/{panel}")
    return out_dir


def encode_panel(still_dir: Path, fps: int) -> Path:
    mp4 = still_dir.with_suffix(".mp4")
    run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(still_dir / "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
        "encode")
    return mp4


def stack_panels(left: Path, right: Path, dest: Path, fps: int) -> Path:
    def label(idx: int, text: str, out: str) -> str:
        return (f"[{idx}:v]drawtext=fontfile={FONT}:text='{text}':"
                f"x=26:y=22:fontsize=36:fontcolor=0x1a1a1a:"
                f"box=1:boxcolor=white@0.55:boxborderw=12[{out}]")

    filt = ";".join([label(0, "truth", "l"), label(1, "recovered", "r"),
                     "[l][r]hstack=inputs=2[v]"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", filt, "-map", "[v]", "-r", str(fps),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dest)],
        "stack")
    return dest


def make_cell(cell: str, args) -> Path | None:
    material, stem, _subtitle = CELLS[cell]
    spacing, floor_z, target, dist = scene_geometry(stem, args.stride,
                                                   args.width, args.height)
    print(f"[{cell}] material={material} stem={stem} spacing={spacing:.6f} "
          f"floor_z={floor_z:.6f} cam_target={np.round(target, 3)} "
          f"cam_dist={dist:.3f} engine={args.engine}", flush=True)
    t0 = time.time()
    dirs = {}
    for panel, _text in PANELS:
        dirs[panel] = render_panel(cell, panel, material, stem, spacing, floor_z,
                                   target, dist, args)
    if args.still is not None:
        for panel, _text in PANELS:
            print(f"[{cell}] still {panel}: {dirs[panel] / '0000.png'}", flush=True)
        print(f"[{cell}] stills done in {time.time() - t0:.1f} s", flush=True)
        return None
    mp4s = {p: encode_panel(dirs[p], args.fps) for p, _t in PANELS}
    dest = stack_panels(mp4s["truth"], mp4s["rec"], VIDEOS / f"{cell}.mp4", args.fps)
    if args.clean_stills:
        for panel, _text in PANELS:
            shutil.rmtree(dirs[panel])
    print(f"[{cell}] wrote {dest} in {time.time() - t0:.1f} s "
          f"({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cells", nargs="*", default=None)
    ap.add_argument("--engine", default="cycles", choices=["cycles", "eevee"])
    ap.add_argument("--samples", type=int, default=64,
                    help="Cycles samples or EEVEE render samples")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--exposure", type=float, default=-0.6)
    ap.add_argument("--still", type=int, default=None,
                    help="render only this dump frame per panel and stop")
    ap.add_argument("--reuse-stills", action="store_true",
                    help="keep existing PNGs instead of clearing the still dir")
    ap.add_argument("--clean-stills", action="store_true",
                    help="delete the PNG still dirs after encoding")
    args = ap.parse_args()
    cells = args.cells or list(CELLS)
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; known: {list(CELLS)}")
    for cell in cells:
        make_cell(cell, args)


if __name__ == "__main__":
    main()
