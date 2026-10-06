"""Watertight-mesh seeding and content-based .ply dispatch in warpmpm.vehicle.

solidify_watertight fills only between entry and exit crossings of each grid column, so a
closed mesh's voids stay empty and the particle volume tracks the mesh volume.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from warpmpm.vehicle import is_gaussian_ply, load_vehicle, solidify_columns, solidify_watertight


def _particle_volume(pts, h):
    return len(pts) * h ** 3


def test_box_volume_matches_mesh():
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    h = 0.05
    pts = solidify_watertight(mesh, h)
    assert mesh.is_watertight
    assert _particle_volume(pts, h) == pytest.approx(mesh.volume, rel=0.02)
    lo, hi = mesh.bounds
    assert np.all(pts >= lo - 1e-6) and np.all(pts <= hi + 1e-6)


def test_void_between_stacked_bodies_stays_empty():
    # Two closed boxes stacked with a 0.4 m gap, like a body above ground clearance.
    lower = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    lower.apply_translation((0.0, 0.0, 0.1))           # z in [0.0, 0.2]
    upper = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    upper.apply_translation((0.0, 0.0, 0.7))           # z in [0.6, 0.8]
    mesh = trimesh.util.concatenate([lower, upper])
    h = 0.05

    pts = solidify_watertight(mesh, h)
    in_gap = (pts[:, 2] > 0.2 + 1e-6) & (pts[:, 2] < 0.6 - 1e-6)
    assert not in_gap.any()
    assert _particle_volume(pts, h) == pytest.approx(lower.volume + upper.volume, rel=0.02)

    # The column fill used for open splat shells cannot see the gap and fills it.
    surface = np.asarray(mesh.sample(20_000, seed=0), dtype=np.float64)
    col = solidify_columns(surface, h)
    assert ((col[:, 2] > 0.25) & (col[:, 2] < 0.55)).any()


def test_ply_dispatch_reads_the_header_not_the_suffix(tmp_path):
    mesh_ply = tmp_path / "hull.ply"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_ply)
    assert not is_gaussian_ply(mesh_ply)

    splat_ply = tmp_path / "scene.ply"
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    splat_ply.write_bytes(header.encode("ascii") + np.zeros(len(props), "<f4").tobytes())
    assert is_gaussian_ply(splat_ply)


def test_load_vehicle_seeds_a_watertight_ply_mesh(tmp_path):
    path = tmp_path / "hull.ply"
    mesh = trimesh.creation.box(extents=(1.7, 4.3, 1.5))
    mesh.export(path)
    h = 0.1
    body = load_vehicle(path, spacing=h)
    assert body.mesh is not None and body.mesh.is_watertight
    assert _particle_volume(body.particles, h) == pytest.approx(mesh.volume, rel=0.05)


def test_touching_parts_fill_solid():
    # Two closed boxes sharing a face: the exit from the lower and the entry into the upper sit
    # at the same z and face opposite ways, so both must be kept.
    lower = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    lower.apply_translation((0.0, 0.0, 0.1))
    upper = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    upper.apply_translation((0.0, 0.0, 0.3))
    h = 0.05
    pts = solidify_watertight(trimesh.util.concatenate([lower, upper]), h)
    assert _particle_volume(pts, h) == pytest.approx(0.4, rel=0.02)


def test_column_grazing_a_ridge_keeps_the_body_below():
    # A rhombus prism above a box; one column passes exactly through the rhombus's side ridge.
    h, w = 0.125, 0.4375
    profile = [(0.0, 1.0), (w, 1.5), (0.0, 2.0), (-w, 1.5)]
    ridge = trimesh.convex.convex_hull(np.array([(x, y, z) for x, z in profile for y in (-0.5, 0.5)]))
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    box.apply_translation((0.0, 0.0, 0.25))
    pts = solidify_watertight(trimesh.util.concatenate([ridge, box]), h)
    on_ridge_column = np.isclose(pts[:, 0], -w)
    assert np.count_nonzero(on_ridge_column & (pts[:, 2] < 0.5)) > 0
