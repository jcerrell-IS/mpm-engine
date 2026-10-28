"""The one scene every stage of the counterfactual loop shares.

A colored splat blob is squeezed by a scripted plate. Identification needs to
re-simulate this scene at candidate parameters, so the scene lives here once:
observe.py runs it at the hidden truth, identify.py runs it at candidates,
and the counterfactual stage edits it (obstacle, height, tilt) explicitly.

The camera is orthographic and owned by this module for the same reason: the
projection that renders the observed frames is byte-for-byte the projection
that reprojects simulated particles onto the tracks; there is no separate
calibration step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GRID_N, GRID_LIM = 48, 0.4
DT, SUBSTEPS, FRAMES = 2.0e-5, 20, 96
TRUTH = {"tau_y": 900.0, "eta": 80.0}          # hidden from identification
IMAGE = 512


def build(tau_y: float, eta: float, device: str = "auto"):
    """The splat blob and its press, at the given dough parameters."""
    from warpmpm.core.solver import GridConfig
    from warpmpm.materials import newtonian
    from warpmpm.splats import make_synthetic_cloud
    from warpmpm.splats.scene import SplatScene

    grid = GridConfig(n_grid=GRID_N, grid_lim=GRID_LIM)
    cloud = make_synthetic_cloud(shape="box", n=6000, sh_degree=0, seed=0)
    material = newtonian(eta=float(eta), density=1200.0).with_yield(float(tau_y))
    scene = SplatScene(cloud, grid=grid, material=material, device=device,
                       fill=True, filler_appearance="inherit",
                       filler_kwargs={"k": 8}, cov_mode="step", floor="sticky")

    x0 = scene.solver.x()
    lo, hi = x0.min(0), x0.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    half = (0.5 * (hi[0] - lo[0]) + 0.01, 0.5 * (hi[1] - lo[1]) + 0.01,
            0.6 * grid.dx)
    clearance = 0.01
    z_top = hi[2] + half[2] + clearance
    z_stop = scene.floor_z + half[2] + 0.6 * (hi[2] - lo[2])
    tool = scene.solver.add_box((cx, cy, z_top), half, velocity=(0.0, 0.0, 0.0))
    plan = {"cx": cx, "cy": cy, "z_top": z_top, "z_stop": z_stop,
            "settle": 12, "v_desc": 1.0,
            "dt_ctrl": DT * SUBSTEPS}
    return scene, tool, plan


def step_frame(scene, tool, plan, f: int, z_box: float) -> float:
    """Advance one frame with the scripted press; returns the new plate height."""
    if f >= plan["settle"] and z_box > plan["z_stop"]:
        v = -min(plan["v_desc"], (z_box - plan["z_stop"]) / plan["dt_ctrl"])
        scene.solver.set_box(tool, center=(plan["cx"], plan["cy"], z_box),
                             velocity=(0.0, 0.0, v))
        z_box = z_box + v * plan["dt_ctrl"]
    scene.step(dt=DT, substeps=SUBSTEPS)
    return z_box


def camera_from_frame0(x0: np.ndarray, pad: float = 1.35) -> dict:
    """Fixed orthographic camera: rotate, then scale to pixels."""
    az, el = np.deg2rad(40.0), np.deg2rad(18.0)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0],
                   [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el), np.cos(el)]])
    R = Rx @ Rz
    u = x0 @ R.T
    c = 0.5 * (u.max(0) + u.min(0))
    span = pad * float((u.max(0) - u.min(0)).max())
    scale = IMAGE / span
    return {"R": R, "center": c, "scale": scale, "image": IMAGE}


def project(cam: dict, x: np.ndarray) -> np.ndarray:
    """World positions to (u, v) pixels plus depth, (N, 3)."""
    u = (x @ cam["R"].T - cam["center"]) * cam["scale"]
    px = u[:, 0] + cam["image"] / 2.0
    py = cam["image"] / 2.0 - u[:, 2]
    return np.stack([px, py, u[:, 1]], axis=1)


def render_frame(cam: dict, x: np.ndarray, colors: np.ndarray, sizes: np.ndarray,
                 path: Path) -> None:
    """Painter's 2D scatter at pixel coordinates, the observation image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = project(cam, x)
    order = np.argsort(p[:, 2])                 # far to near
    fig = plt.figure(figsize=(IMAGE / 100, IMAGE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.scatter(p[order, 0], p[order, 1], c=np.clip(colors[order], 0, 1),
               s=sizes[order], linewidths=0)
    ax.set_xlim(0, IMAGE)
    ax.set_ylim(IMAGE, 0)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)
