"""Stage 3 of the counterfactual loop: the law from the observed frames.

The scan re-simulates the shared scene at candidate (tau_y, eta), renders
each candidate through the same camera and renderer as the observation,
masks the frames, and scores the normalized MSE of the silhouette width,
height, and area time series. simulate_tracks is an alternative pixel-track
objective kept for diagnostics; main does not use it. Coarse grid, then one
refinement round around the minimum. Nothing is differentiated, and the
truth file is opened only after the scan, for reporting.

Run from the engine root, after track:

    .venv/bin/python -m experiments.counterfactual.identify

Writes out/counterfactual/identify.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.counterfactual import scene as S

OUT = Path(__file__).resolve().parents[2] / "out" / "counterfactual"


def simulate_tracks(tau_y: float, eta: float, idx: np.ndarray,
                    cam: dict) -> np.ndarray:
    sc, tool, plan = S.build(tau_y, eta)
    n_vis = sc.n_visible
    out = [S.project(cam, sc.state()["pos"].detach().cpu().numpy()[:n_vis][idx])[:, :2]]
    z_box = plan["z_top"]
    for f in range(1, S.FRAMES + 1):
        z_box = S.step_frame(sc, tool, plan, f - 1, z_box)
        x = sc.state()["pos"].detach().cpu().numpy()[:n_vis][idx]
        out.append(S.project(cam, x)[:, :2])
    return np.stack(out)


def silhouette_series(frame_dir: Path) -> np.ndarray:
    """Per-frame (width, height, area) of the non-background pixel mask."""
    import imageio.v2 as imageio
    rows = []
    for f in sorted(frame_dir.glob("f_*.png")):
        img = imageio.imread(f)[..., :3]
        mask = (img < 250).any(axis=-1)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            rows.append((0.0, 0.0, 0.0))
            continue
        rows.append((float(xs.max() - xs.min()), float(ys.max() - ys.min()),
                     float(mask.sum())))
    return np.asarray(rows)


def simulate_silhouette(tau_y: float, eta: float, cam: dict,
                        tmp: Path) -> np.ndarray:
    """Render the candidate through the SAME pipeline and mask it identically."""
    from warpmpm.splats.render import _colors_from_state
    sc, tool, plan = S.build(tau_y, eta)
    n_vis = sc.n_visible
    st = sc.state()
    colors = np.clip(np.asarray(_colors_from_state(st, cam["R"][1], 0)), 0, 1)
    cov6 = st["cov6"].detach().cpu().numpy()
    sigma = np.sqrt(np.clip((cov6[:, 0] + cov6[:, 3] + cov6[:, 5]) / 3.0, 0, None))
    sizes = np.clip(sigma * cam["scale"] * 2.0, 1.0, 40.0)
    tmp.mkdir(parents=True, exist_ok=True)
    x = st["pos"].detach().cpu().numpy()
    if not np.isfinite(x).all():
        return None
    S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis], tmp / "f_0000.png")
    z_box = plan["z_top"]
    for f in range(1, S.FRAMES + 1):
        z_box = S.step_frame(sc, tool, plan, f - 1, z_box)
        x = sc.state()["pos"].detach().cpu().numpy()
        if not np.isfinite(x).all():
            return None
        S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis],
                       tmp / f"f_{f:04d}.png")
    return silhouette_series(tmp)


def main() -> None:
    import shutil

    cam = dict(np.load(OUT / "camera.npz"))
    obs = silhouette_series(OUT / "frames")
    scale = np.abs(obs).mean(0) + 1e-9

    tried: dict[tuple, float] = {}

    def score(tau_y: float, eta: float) -> float:
        key = (round(float(tau_y), 3), round(float(eta), 3))
        if key in tried:
            return tried[key]
        t0 = time.time()
        tmp = OUT / "_candidate_frames"
        sim = simulate_silhouette(tau_y, eta, cam, tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        if sim is None:
            tried[key] = float("inf")
            print(f"[identify] tau_y={tau_y:g} eta={eta:g}: DIVERGED")
            return tried[key]
        mse = float(np.mean(((sim - obs) / scale) ** 2))
        tried[key] = mse
        print(f"[identify] tau_y={tau_y:g} eta={eta:g}: silhouette MSE "
              f"{mse:.5f} ({time.time() - t0:.0f}s)")
        return mse

    for ty in (200, 400, 800, 1600, 3200, 6400):
        for et in (20, 80, 320):
            score(ty, et)
    best = min(tried, key=tried.get)
    for fty in (0.7, 0.85, 1.2, 1.4):
        score(best[0] * fty, best[1])
    for fet in (0.4, 0.6, 1.6, 2.5):
        score(best[0], best[1] * fet)
    best = min(tried, key=tried.get)

    truth = json.loads(str(np.load(OUT / "observe_truth.npz")["truth"]))
    res = {
        "tau_y": best[0], "eta": best[1],
        "silhouette_mse_at_best": tried[best],
        "scan": {f"{k[0]:g},{k[1]:g}": v for k, v in sorted(tried.items())},
        "n_rollouts": len(tried),
        "truth": truth,
        "tau_y_error_pct": 100.0 * (best[0] / truth["tau_y"] - 1.0),
        "eta_error_pct": 100.0 * (best[1] / truth["eta"] - 1.0),
        "objective": ("normalized MSE of the silhouette width, height, and "
                      "area time series, candidate frames rendered and "
                      "masked identically to the observation"),
        "track_objective_falsified": (
            "the objective uses the silhouette because the tracker "
            "under-reports displacement (median tracked-to-true ratio 0.65 "
            "at 0.45 px per frame of press motion); the silhouette needs no "
            "tracker, and under a displacement-controlled press the material "
            "signal is the lateral squeeze-out"),
    }
    (OUT / "identify.json").write_text(json.dumps(res, indent=2))
    print(f"[identify] tau_y {best[0]:g} (truth {truth['tau_y']:g}, "
          f"{res['tau_y_error_pct']:+.1f} pct), eta {best[1]:g} "
          f"(truth {truth['eta']:g}, {res['eta_error_pct']:+.1f} pct), "
          f"{len(tried)} rollouts")


if __name__ == "__main__":
    main()
