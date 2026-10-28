"""Side-by-side videos of the NCLaw comparison: truth vs recovered-law rollout.

Renders the primary cells of docs/nclaw_comparison_plan.md from the suite dumps
(out/nclaw_suite/dumps/<material>_<shape>[_mild]_{truth,rec}.npz): two rotating
3D panels on identical axes, particles coloured by speed, so the recovered law's
match to the truth is visible frame by frame. Adapted from the original video2sim renderer.

Run:  .venv/bin/python -m experiments.nclaw.suite_video            # all primary cells
      .venv/bin/python -m experiments.nclaw.suite_video jelly_armadillo_mild
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ident.io.schema import load_dump  # noqa: E402

OUT = ROOT / "out" / "nclaw_suite"
VIDEOS = OUT / "videos"

# scene key -> (material, dump stem pair, panel subtitle with recovered theta)
CELLS: dict[str, tuple[str, str, str]] = {
    "jelly_cube": ("jelly", "jelly_cube", "reconstruction (training scene)"),
    "jelly_armadillo_mild": ("jelly", "jelly_armadillo_mild", "their held-out mesh"),
    "plasticine_bunny_mild": ("plasticine", "plasticine_bunny_mild", "their held-out mesh"),
    "sand_blub_mild": ("sand", "sand_blub_mild", "their held-out mesh"),
    "water_spot_mild": ("water", "water_spot_mild", "their held-out mesh"),
    "jelly_cross": ("jelly", "jelly_cross", "their trajectory vs our recovered law"),
    "plasticine_cross": ("plasticine", "plasticine_cross", "their trajectory vs our recovered law"),
    "sand_cross": ("sand", "sand_cross", "their trajectory vs our recovered law"),
    "water_cross": ("water", "water_cross", "their trajectory vs our recovered law"),
}


def _theta_line(material: str) -> str:
    ident = json.loads((OUT / f"identify_{material}.json").read_text())
    th = ident["theta_engine"]
    return ", ".join(f"{k}={v:.4g}" for k, v in th.items())


def make_view(cell: str, stride: int = 2, fps: int = 25) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    material, stem, subtitle = CELLS[cell]
    t = load_dump(str(OUT / "dumps" / f"{stem}_truth.npz"))
    r = load_dump(str(OUT / "dumps" / f"{stem}_rec.npz"))
    nf = min(t.meta.n_frames, r.meta.n_frames)
    n = min(t.meta.n_particles, r.meta.n_particles)
    Xt, Vt = t.x[:nf, :n], t.v[:nf, :n]
    Xr, Vr = r.x[:nf, :n], r.v[:nf, :n]
    times = t.times[:nf]
    rmse_mm = float(np.sqrt(((Xt - Xr) ** 2).sum(-1).mean()) * 1e3)

    lo = min(Xt.min(), Xr.min())
    hi = max(Xt.max(), Xr.max())
    pad = 0.02 * (hi - lo)
    lim = (lo - pad, hi + pad)
    spd_ref = np.concatenate([np.linalg.norm(Vt[nf // 4], axis=1),
                              np.linalg.norm(Vr[nf // 4], axis=1)])
    vmax = float(np.percentile(spd_ref, 99)) or 0.5

    frames = list(range(0, nf, stride))
    fig = plt.figure(figsize=(11.5, 5.8))
    panels = [("truth", Xt, Vt), (f"recovered ({_theta_line(material)})", Xr, Vr)]
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]

    def draw(ax, X, V, fi, title):
        ax.clear()
        spd = np.linalg.norm(V[fi], axis=1)
        ax.scatter(X[fi][:, 0], X[fi][:, 1], X[fi][:, 2], s=3, c=spd, cmap="viridis",
                   vmin=0, vmax=vmax, edgecolors="none", depthshade=True)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_zlim(*lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=18, azim=-60 + 0.2 * fi)
        ax.set_title(title, fontsize=9)

    def upd(fi):
        for ax, (title, X, V) in zip(axes, panels, strict=True):
            draw(ax, X, V, fi, title)
        fig.suptitle(f"NCLaw comparison, {material}: {subtitle}\n"
                     f"law identified from one cube throw; {rmse_mm:.2f} mm RMS "
                     f"deviation   [t={times[fi]:.2f}s]", fontsize=12, y=0.99)
        return tuple(axes)

    anim = animation.FuncAnimation(fig, upd, frames=frames, blit=False)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    p = VIDEOS / f"{cell}.mp4"
    anim.save(str(p), writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    print(f"wrote {p}  ({len(frames)} frames, {n} particles, {rmse_mm:.2f} mm RMS)")
    return p


if __name__ == "__main__":
    cells = sys.argv[1:] or list(CELLS)
    for c in cells:
        make_view(c)
