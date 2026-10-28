"""Scaffolding shared by the robotics-era studies in this package.

Five pieces, each extracted because two or more scripts carried a verbatim or
near-verbatim copy of it:

1. Path anchors (ENGINE_ROOT, REPO_ROOT, engine_out, add_repo_to_path). Every
   script used to compute these with `Path(__file__).resolve().parents[N]`,
   which silently breaks whenever a file changes directory depth. Import the
   anchors instead so the arithmetic lives in one place.
2. press_scene: the free 3D dough block on a sticky floor, squeezed by an
   attached plate tool. predict_volume_franka.press_force,
   rollout_force_video.force_series and predict_volume_rollout.squeeze_dump
   built this scene with identical literals; only what they recorded per frame
   differed, so the recording loops stay in the scripts.
3. lobedness: the 3-fold boundary-mode metric for the three-prong studies.
   three_prong and dough_franka_threeprong carried numerically identical
   implementations.
4. cotrack: run CoTracker3 over a rendered frame folder on a masked query grid.
   rollout_franka_cotracker and surface_track_test differed only in the
   frame resize and the pixel predicate that picks query points, both now
   arguments.

Not consolidated on purpose:
  - the per-frame recording loops around press_scene. They produce the
    published numbers (tau_y=192, eta=55 grid-impulse; FE 11 percent versus
    Bingham 34 percent held-out force) and cannot be re-run cheaply here, so
    they are left byte-for-byte as they were.
  - dough_fe_viscous.squeeze and the quasi-2D plane-strain slab in
    realdata_pipeline and speckle_particle_videos. They look similar to
    press_scene but differ in the tool padding, the wall colliders and the
    material, so sharing one builder would need flags for every difference.
  - the matplotlib figure code. Each figure is bespoke.

This module holds no experiment of its own and writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# experiments/robotics/common.py -> experiments/robotics -> experiments -> mpm_engine
ENGINE_ROOT = Path(__file__).resolve().parents[2]
# ... -> video2sim, the staging tree that holds perception/ and the shared literature
REPO_ROOT = ENGINE_ROOT.parent


def engine_out(*parts: str) -> Path:
    """Path under the engine's out/ tree. Does not create it."""
    return ENGINE_ROOT.joinpath("out", *parts)


def add_repo_to_path() -> Path:
    """Put the video2sim root on sys.path so `perception.*` imports resolve.

    The perception package is not installed into the environment; it is imported
    from the staging tree next to the engine. Returns REPO_ROOT for callers that
    also want to build paths from it.
    """
    p = str(REPO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return REPO_ROOT


def add_engine_to_path() -> Path:
    """Put the engine root on sys.path so `examples.*` and `experiments.*` resolve
    when a script is executed by file path rather than with `python -m`."""
    p = str(ENGINE_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return ENGINE_ROOT


def press_scene(tau_y, eta, geom, n_grid=52, density=1000.0, dt=1.0e-4,
                substeps=20, v_plate=0.08, press_strain=0.5, device="auto"):
    """Build the sticky-floor dough block with a plate tool poised above it.

    Returns a dict with everything the recording loops need:
      s, be, tool  the solver, the WarpMPMBackend and the attached tool handle
      cx, cy       the plate centre in the horizontal plane
      bh           the plate half-extents
      floor, ch    floor height and initial dough height
      dough_top    floor + ch, the undeformed top surface
      z0           the plate's initial centre height
      fdt, nf      the frame timestep and the number of frames to press

    The caller drives the press itself; this function does not step the solver.
    """
    from warpmpm import GridConfig, Solver, block, newtonian
    from warpmpm.coupling.backend import WarpMPMBackend

    grid = GridConfig(n_grid=n_grid, grid_lim=0.4)
    cw, cd, ch = geom
    pos, vol, floor = block(grid, size=geom, ppc=2)
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(eta=eta, density=density, bulk_modulus=9.0e5).with_yield(tau_y))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    cx = cy = grid.grid_lim * 0.5
    dough_top = floor + ch
    bh = (0.5 * cw + 0.015, 0.5 * cd + 0.015, 0.6 * grid.dx)
    be = WarpMPMBackend(solver=s)
    z0 = dough_top + bh[2]
    tool = be.attach_tool((cx, cy, z0), bh)
    fdt = dt * substeps
    nf = round(press_strain * ch / v_plate / fdt)
    return dict(grid=grid, s=s, be=be, tool=tool, cx=cx, cy=cy, bh=bh,
                floor=floor, ch=ch, dough_top=dough_top, z0=z0, fdt=fdt, nf=nf)


def lobedness(x0, xf, cx, cy):
    """Strength of the period-120-degree (3-fold) boundary mode, before vs after.

    Projects to the x-y plane, takes the maximum radius in each of 48 angular
    bins, and reports the amplitude of the 3-per-revolution Fourier mode
    normalized by the mean radius of the initial shape. A triangular or
    three-lobed cross-section gives a strong component.
    """
    def profile(x):
        dx, dy = x[:, 0] - cx, x[:, 1] - cy
        ang = np.arctan2(dy, dx)
        rad = np.sqrt(dx * dx + dy * dy)
        bins = np.linspace(-np.pi, np.pi, 49)
        idx = np.clip(np.digitize(ang, bins) - 1, 0, len(bins) - 2)
        rmax = np.array([rad[idx == b].max() if np.any(idx == b) else np.nan
                         for b in range(len(bins) - 1)])
        return bins[:-1] + np.diff(bins) / 2, rmax

    rbar = np.nanmean(profile(x0)[1]) + 1e-9

    def mode3(x):
        th, r = profile(x)
        ok = np.isfinite(r)
        r = r[ok] - np.nanmean(r)
        t = th[ok]
        return float(np.abs(np.sum(r * np.exp(-3j * t))) / max(ok.sum(), 1) / rbar)

    return mode3(x0), mode3(xf)


def warm_dough_pixels(f0, pts):
    """Query-point predicate for the arm renders: warm (low-blue) pixels are dough,
    the grey Franka and table are not."""
    yi = pts[:, 1].astype(int)
    xi = pts[:, 0].astype(int)
    r, g, b = f0[yi, xi, 0], f0[yi, xi, 1], f0[yi, xi, 2]
    return (r > b + 12) & (g > b + 4) & (r > 70)


def nonwhite_pixels(f0, pts, thresh=720.0):
    """Query-point predicate for the plain surface renders: anything darker than a
    white background is on the dough."""
    val = f0[pts[:, 1].astype(int), pts[:, 0].astype(int)].sum(1)
    return val < thresh


def cotrack(frames_dir, spacing=7, device="cpu", select=None, resize=None):
    """Run CoTracker3 offline on a folder of f_*.png frames.

    Query points are a `spacing`-pixel grid filtered by `select(frame0, points)`;
    pass warm_dough_pixels or nonwhite_pixels, or None to keep the whole grid.
    `resize` is an optional (width, height) applied to every frame before
    tracking. Returns (tracks (T, N, 2) in pixels, visibility (T, N)).
    """
    import torch
    from PIL import Image

    files = sorted(Path(frames_dir).glob("f_*.png"))
    imgs = np.stack([
        np.asarray(Image.open(f).convert("RGB").resize(resize) if resize
                   else Image.open(f).convert("RGB"))
        for f in files
    ]).astype(np.float32)
    video = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None]        # (1, T, 3, H, W)
    f0 = imgs[0]
    H, W = f0.shape[:2]
    ys = np.arange(spacing, H - spacing, spacing)
    xs = np.arange(spacing, W - spacing, spacing)
    GX, GY = np.meshgrid(xs, ys)
    pts = np.stack([GX.ravel(), GY.ravel()], -1).astype(np.float32)
    if select is not None:
        pts = pts[select(f0, pts)]
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    with torch.no_grad():
        tr, vis = model(video.to(device), queries=torch.from_numpy(q)[None].to(device))
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()
