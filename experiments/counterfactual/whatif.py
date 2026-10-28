"""Stages 4 and 5 of the counterfactual loop: resimulate, then ask what-ifs.

Every clip below runs at the IDENTIFIED parameters from identify.json, never
the truth, rendered by the same camera that produced the observation:

  resim         the observed event replayed at the identified law, written as
                a side-by-side video against the observed frames, the
                validation run before any counterfactual
  deeper_press  the plate driven to 85 percent of the blob height instead of
                60 percent: an event that never happened
  offset_press  the plate shifted a third of the blob width sideways, so the
                material squeezes out asymmetrically
  stiffer_what_if  the same press if the material had ten times the
                viscosity: a material counterfactual on the same event

Run from the engine root, after identify:

    .venv/bin/python -m experiments.counterfactual.whatif

Writes out/counterfactual/videos/<name>.mp4 and whatif.json.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.counterfactual import scene as S

OUT = Path(__file__).resolve().parents[2] / "out" / "counterfactual"


def run_clip(name: str, tau_y: float, eta: float, cam: dict,
             edit=None, log=print) -> Path:
    import imageio.v2 as imageio

    sc, tool, plan = S.build(tau_y, eta)
    if edit is not None:
        edit(sc, tool, plan)
    n_vis = sc.n_visible
    st = sc.state()
    from warpmpm.splats.render import _colors_from_state
    colors = np.clip(np.asarray(_colors_from_state(st, cam["R"][1], 0)), 0, 1)
    cov6 = st["cov6"].detach().cpu().numpy()
    sigma = np.sqrt(np.clip((cov6[:, 0] + cov6[:, 3] + cov6[:, 5]) / 3.0, 0, None))
    sizes = np.clip(sigma * cam["scale"] * 2.0, 1.0, 40.0)

    tmp = OUT / f"_frames_{name}"
    tmp.mkdir(parents=True, exist_ok=True)
    x = st["pos"].detach().cpu().numpy()
    S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis], tmp / "f_0000.png")
    z_box = plan["z_top"]
    t0 = time.time()
    for f in range(1, S.FRAMES + 1):
        z_box = S.step_frame(sc, tool, plan, f - 1, z_box)
        x = sc.state()["pos"].detach().cpu().numpy()
        if not np.isfinite(x).all():
            log(f"[whatif] {name}: went non-finite at frame {f}; clip truncated")
            break
        S.render_frame(cam, x[:n_vis], colors[:n_vis], sizes[:n_vis],
                       tmp / f"f_{f:04d}.png")
    log(f"[whatif] {name}: simulated and rendered in {time.time() - t0:.0f}s")
    return tmp


def main() -> None:
    import imageio.v2 as imageio

    ident = json.loads((OUT / "identify.json").read_text())
    tau_y, eta = float(ident["tau_y"]), float(ident["eta"])
    cam = dict(np.load(OUT / "camera.npz"))
    videos = OUT / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    def deeper(sc, tool, plan):
        x0 = sc.solver.x()
        h = x0[:, 2].max() - x0[:, 2].min()
        plan["z_stop"] -= 0.25 * h            # 60 -> 85 percent of the height

    def offset(sc, tool, plan):
        x0 = sc.solver.x()
        w = x0[:, 0].max() - x0[:, 0].min()
        plan["cx"] += w / 3.0

    clips = {
        "resim": None,
        "deeper_press": deeper,
        "offset_press": offset,
    }
    made = {}
    for name, edit in clips.items():
        tmp = run_clip(name, tau_y, eta, cam, edit)
        frames = sorted(tmp.glob("f_*.png"))
        if name == "resim":
            obs = sorted((OUT / "frames").glob("f_*.png"))
            imgs = [np.concatenate([imageio.imread(a)[..., :3],
                                    imageio.imread(b)[..., :3]], axis=1)
                    for a, b in zip(obs, frames, strict=True)]
            path = videos / "resim_side_by_side.mp4"
        else:
            imgs = [imageio.imread(f)[..., :3] for f in frames]
            path = videos / f"{name}.mp4"
        imageio.mimsave(path, imgs, fps=24)
        made[name] = str(path)
        shutil.rmtree(tmp)
    tmp = run_clip("stiffer_what_if", tau_y, 10.0 * eta, cam, None)
    imgs = [imageio.imread(f)[..., :3] for f in sorted(tmp.glob("f_*.png"))]
    imageio.mimsave(videos / "stiffer_what_if.mp4", imgs, fps=24)
    made["stiffer_what_if"] = str(videos / "stiffer_what_if.mp4")
    shutil.rmtree(tmp)

    (OUT / "whatif.json").write_text(json.dumps(
        {"identified": {"tau_y": tau_y, "eta": eta}, "videos": made}, indent=2))
    print(f"[whatif] wrote {len(made)} clips under {videos}")


if __name__ == "__main__":
    main()
