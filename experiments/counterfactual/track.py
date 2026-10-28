"""Stage 2 of the counterfactual loop: tracks from the rendered frames.

CoTracker3 runs on the observed PNGs with query points seeded at the frame-0
projections of chosen splats, so every track corresponds to one known splat
and no matching problem exists downstream. The frame-0 configuration is
scene knowledge, the same assumption every method in the NCLaw comparison
enjoys; everything after frame 0 comes from the tracker alone.

Query selection is occlusion-aware: the image is tiled, and each tile
contributes its nearest-to-camera splat, so queries sit on the front surface
the camera sees.

This module opens the hidden truth only after tracking, to report tracking
quality (pixel error of tracks against the projected true motion of the same
splats); identification never reads it.

Run from the engine root:

    .venv/bin/python -m experiments.counterfactual.track

Writes out/counterfactual/tracks.npz.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments.counterfactual import scene as S

OUT = Path(__file__).resolve().parents[2] / "out" / "counterfactual"
TILE = 24                     # px; one query per tile, front surface only


def select_queries(x0_vis: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    p = S.project(cam, x0_vis)
    n_tiles = S.IMAGE // TILE
    best = {}
    for i, (px, py, depth) in enumerate(p):
        tx, ty = int(px // TILE), int(py // TILE)
        if not (0 <= tx < n_tiles and 0 <= ty < n_tiles):
            continue
        k = (tx, ty)
        if k not in best or depth < p[best[k], 2]:
            best[k] = i
    idx = np.array(sorted(best.values()))
    return idx, p[idx, :2]


def main() -> None:
    import imageio.v2 as imageio
    import torch

    cam = dict(np.load(OUT / "camera.npz"))
    truth = np.load(OUT / "observe_truth.npz")
    n_vis = int(truth["n_visible"])
    x0_vis = truth["x"][0][:n_vis]
    idx, pix0 = select_queries(x0_vis, cam)
    print(f"[track] {len(idx)} queries on the front surface")

    frames = sorted((OUT / "frames").glob("f_*.png"))
    imgs = np.stack([imageio.imread(f)[..., :3] for f in frames]).astype(np.float32)
    video = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None]
    q = np.concatenate([np.zeros((len(idx), 1), np.float32),
                        pix0.astype(np.float32)], 1)
    device = "cpu"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device).eval()
    t0 = time.time()
    with torch.no_grad():
        tr, vis = model(video.to(device), queries=torch.from_numpy(q)[None].to(device))
    tracks = tr[0].cpu().numpy()
    vismask = vis[0].cpu().numpy()
    print(f"[track] tracked {tracks.shape[1]} points over {tracks.shape[0]} "
          f"frames in {time.time() - t0:.0f}s")

    # validation only: pixel error against the projected true motion
    x_true = truth["x"][:, :n_vis][:, idx]
    proj = np.stack([S.project(cam, xf)[:, :2] for xf in x_true])
    err = np.linalg.norm(tracks - proj, axis=-1)
    print(f"[track] pixel error vs hidden truth: p50 {np.median(err):.2f} px, "
          f"p95 {np.percentile(err, 95):.2f} px (image {S.IMAGE} px)")
    np.savez(OUT / "tracks.npz", tracks=tracks, visibility=vismask,
             splat_indices=idx, query_pixels=pix0,
             pixel_error_p50=float(np.median(err)),
             pixel_error_p95=float(np.percentile(err, 95)))
    print(f"[track] wrote {OUT / 'tracks.npz'}")


if __name__ == "__main__":
    main()
