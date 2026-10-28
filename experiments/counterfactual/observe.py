"""Stage 1 of the counterfactual loop: the observed event.

Runs the shared scene at the HIDDEN truth parameters and writes only what a
camera would give into frames/: rendered PNGs of the
colored splat blob under the press. Everything else that comes out of the
simulator, the per-frame particle positions above all, goes into
observe_truth.npz and is read only by validation; identification never
opens it.

Run from the engine root:

    .venv/bin/python -m experiments.counterfactual.observe

Writes out/counterfactual/{frames/,observe_truth.npz,camera.npz}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.counterfactual import scene as S

OUT = Path(__file__).resolve().parents[2] / "out" / "counterfactual"


def main() -> None:
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sc, tool, plan = S.build(**S.TRUTH)
    n_vis = sc.n_visible
    st = sc.state()
    x = st["pos"].detach().cpu().numpy()
    cam = S.camera_from_frame0(x)
    np.savez(OUT / "camera.npz", **{k: v for k, v in cam.items()})

    from warpmpm.splats.render import _colors_from_state  # DC colors, camera-lit
    colors = np.clip(np.asarray(_colors_from_state(st, cam["R"][1], 0)), 0, 1)
    cov6 = st["cov6"].detach().cpu().numpy()
    sigma = np.sqrt(np.clip((cov6[:, 0] + cov6[:, 3] + cov6[:, 5]) / 3.0, 0, None))
    sizes = np.clip(sigma * cam["scale"] * 2.0, 1.0, 40.0)

    xs, times = [x.copy()], [0.0]
    S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis],
                   frames_dir / "f_0000.png")
    z_box = plan["z_top"]
    t0 = time.time()
    for f in range(1, S.FRAMES + 1):
        z_box = S.step_frame(sc, tool, plan, f - 1, z_box)
        st = sc.state()
        x = st["pos"].detach().cpu().numpy()
        xs.append(x.copy())
        times.append(f * plan["dt_ctrl"])
        S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis],
                       frames_dir / f"f_{f:04d}.png")
        if f % 8 == 0:
            print(f"[observe] frame {f}/{S.FRAMES} ({time.time()-t0:.0f}s)")
    np.savez_compressed(
        OUT / "observe_truth.npz", x=np.stack(xs), times=np.asarray(times),
        n_visible=n_vis, truth=json.dumps(S.TRUTH))
    print(f"[observe] wrote {S.FRAMES + 1} frames and the hidden truth "
          f"({time.time()-t0:.0f}s, {len(x)} particles, {n_vis} visible)")


if __name__ == "__main__":
    main()
