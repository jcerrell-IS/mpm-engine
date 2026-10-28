"""Validate the identified law: twin prediction vs the real pour, plus the deliverable
side-by-side video with a live transfer-curve strip.

Consumes:
  out/pour_wf/<ep>/observations.npz     perception (real V(t) via the graduation curve,
                                        real onset from the receiver's floor crop)
  out/pour_wf/<ep>/identify.json        the weak-form eta_hat
  out/pour_recorded_twin/<ep>/metrics.csv and side_by_side.mp4
                                        the single twin run at eta_hat (the only
                                        simulation in the pipeline)
  out/pour_recorded_twin/<ep>/archive_eta<EEE>[_v<VVV>]/metrics.csv
                                        comparison twins at other (eta, fill) values:
                                        archive_eta141 is the handbook viscosity,
                                        archive_eta420_v325 the V0-scan alternative

Produces (out/pour_wf/<ep>/):
  validation.png            real vs predicted transfer curves + errors
  validation.json           final-volume / RMS / onset numbers for every run
  side_by_side_curve.mp4    [real | twin] with the V(t) race below (the paper video)

Onsets compare like for like: the real one is the first frame with amber pixels below
the receiver's 30 mL graduation (pour_perception.py), the twin's the first frame with
particles landed below that graduation (n_rcv_pool in metrics.csv). The older
2 mL-in-the-cavity arrival (in-flight liquid included) is kept alongside.

Run AFTER the twin:  python examples/pour_recorded_twin.py --eta <eta_hat>
  python experiments/pour/pour_validate.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PRE_ROLL = 1.0    # twin's episode clock: t_twin = t_pourclock + PRE_ROLL
ARCHIVE_RE = re.compile(r"^archive_eta(\d{3})(?:_v(\d{3}))?$")
POOL_PARTICLES = 3    # landed particles (~0.02 mL) that call the twin's onset


def read_metrics(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    sim = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    sim["t_pour"] = sim["t"] - PRE_ROLL              # onto the pour clock
    return sim


def load_all(episode: str):
    wf = REPO / "out" / "pour_wf" / episode
    tw = REPO / "out" / "pour_recorded_twin" / episode
    obs = dict(np.load(wf / "observations.npz"))
    ident = json.loads((wf / "identify.json").read_text())
    sim = read_metrics(tw / "metrics.csv")
    others = []
    for d in sorted(tw.glob("archive_eta*")):
        m = ARCHIVE_RE.match(d.name)
        if m and (d / "metrics.csv").exists():
            others.append(dict(eta=int(m.group(1)) / 100.0,
                               volume_ml=float(m.group(2) or 300),
                               sim=read_metrics(d / "metrics.csv")))
    return obs, ident, sim, others, wf, tw


def numbers(obs, sim) -> dict:
    t_r = obs["t"]
    v_r = obs["rcv_vol"] * 1e6
    ok = np.isfinite(v_r)
    t_s, v_s = sim["t_pour"], sim["ml_rcv"]
    # real settled value: median of the last second of readable data
    late = ok & (t_r > t_r[ok].max() - 1.0)
    v_final_real = float(np.median(v_r[late]))
    v_final_sim = float(v_s[-1])
    m = ok & (t_r >= t_s.min()) & (t_r <= t_s.max())
    v_sim_i = np.interp(t_r[m], t_s, v_s)
    rms = float(np.sqrt(np.mean((v_r[m] - v_sim_i) ** 2)))
    out = dict(v_final_real_mL=v_final_real, v_final_sim_mL=v_final_sim,
               v_final_err_mL=v_final_sim - v_final_real,
               v_final_err_pct=100 * (v_final_sim - v_final_real)
               / max(v_final_real, 1e-9),
               rms_overlap_mL=rms, n_real_points=int(m.sum()),
               # arrival: first sim frame with > 2 mL anywhere in the cavity
               sim_arrival_2ml_s=(float(t_s[np.argmax(v_s > 2.0)])
                                  if (v_s > 2.0).any() else np.nan))
    if "n_rcv_pool" in sim:
        landed = sim["n_rcv_pool"] >= POOL_PARTICLES
        out["sim_onset_pool_s"] = float(t_s[np.argmax(landed)]) if landed.any() else np.nan
    if "rcv_onset_s" in obs:
        out["real_onset_pool_s"] = float(obs["rcv_onset_s"])
    return out


def plot(obs, ident, sim, others, num, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ok = np.isfinite(obs["rcv_vol"])
    ax.plot(obs["t"][ok], obs["rcv_vol"][ok] * 1e6, ".", ms=3, color="k",
            label="real: receiver level -> graduation curve [mL]")
    ax.plot(sim["t_pour"], sim["ml_rcv"], "-", color="tab:blue", lw=1.8,
            label=f"twin at identified eta = {ident['eta']:.2f} Pa.s (count) [mL]")
    colors = ("tab:orange", "tab:green", "tab:red", "tab:purple")
    for o, col in zip(others, colors, strict=False):
        lab = f"twin at eta = {o['eta']:.2f} Pa.s"
        if o["volume_ml"] != 300:
            lab += f", {o['volume_ml']:.0f} mL fill"
        ax.plot(o["sim"]["t_pour"], o["sim"]["ml_rcv"], "-", color=col, lw=1.2,
                alpha=0.85, label=lab + " [mL]")
    # (the twin's level_vol_rcv readout is omitted: at peak tilt the source cup dips
    # into the receiver's cavity frustum and a handful of its particles wreck the
    # 0.97-quantile level; the count channel is unaffected)
    if np.isfinite(num.get("real_onset_pool_s", np.nan)):
        ax.axvline(num["real_onset_pool_s"], color="k", lw=0.6, ls=":", alpha=0.7)
    ax.axhline(num["v_final_real_mL"], color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("t - t_send [s]")
    ax.set_ylabel("transferred volume [mL]")
    ax.set_title(f"prediction: final {num['v_final_sim_mL']:.1f} vs real "
                 f"{num['v_final_real_mL']:.1f} mL "
                 f"({num['v_final_err_pct']:+.1f}%), RMS {num['rms_overlap_mL']:.1f} mL")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def curve_strip(obs, sim, ident, t_now: float, width_px: int, height_px: int,
                t_lo: float, t_hi: float, v_max: float):
    """One matplotlib frame of the V(t) race with a time cursor, as an RGB array."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ok = np.isfinite(obs["rcv_vol"])
    tr, vr = obs["t"][ok], obs["rcv_vol"][ok] * 1e6
    m = tr <= t_now
    ax.plot(tr[m], vr[m], ".", ms=2.5, color="#222222", label="real (graduations)")
    ms_ = sim["t_pour"] <= t_now
    ax.plot(sim["t_pour"][ms_], sim["ml_rcv"][ms_], "-", lw=2, color="tab:blue",
            label=f"twin @ eta={ident['eta']:.2f} Pa.s")
    ax.axvline(t_now, color="tab:red", lw=1)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, v_max)
    ax.set_ylabel("mL")
    ax.set_xlabel("t - t_send [s]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def compose_video(obs, sim, ident, tw: Path, out_mp4: Path, strip_h: int = 240):
    import imageio.v2 as imageio

    src = tw / "side_by_side.mp4"
    if not src.exists():
        print(f"{src} missing; skipping the composed video")
        return None
    reader = imageio.get_reader(src)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 60)
    t0 = float(sim["t_pour"][0]) - 1.0 / fps        # frame 0 of the twin video
    t_lo, t_hi = t0, float(sim["t_pour"][-1])
    v_max = 1.15 * max(np.nanmax(sim["ml_rcv"]),
                       np.nanmax(np.where(np.isfinite(obs["rcv_vol"]),
                                          obs["rcv_vol"], 0.0)) * 1e6)
    strip_cache = {}
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2,
                            output_params=["-movflags", "+faststart"]) as wtr:
        for k, frame in enumerate(reader):
            t_now = t0 + k / fps
            key = round(t_now * 10)                  # refresh the strip at 10 Hz
            if key not in strip_cache:
                strip_cache.clear()
                strip_cache[key] = curve_strip(obs, sim, ident, t_now,
                                               frame.shape[1], strip_h,
                                               t_lo, t_hi, v_max)
            strip = strip_cache[key]
            if strip.shape[1] != frame.shape[1]:
                pad = frame.shape[1] - strip.shape[1]
                strip = np.pad(strip, ((0, 0), (0, max(pad, 0)), (0, 0)),
                               constant_values=255)[:, :frame.shape[1]]
            wtr.append_data(np.vstack([frame, strip]))
    reader.close()
    print("wrote", out_mp4)
    return out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="ep0001")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    obs, ident, sim, others, wf, tw = load_all(args.episode)
    num = numbers(obs, sim)
    num["eta"] = ident["eta"]
    num["other_runs"] = []
    for o in others:
        n = numbers(obs, o["sim"])
        num["other_runs"].append(dict(
            eta=o["eta"], volume_ml=o["volume_ml"], v_final_sim_mL=n["v_final_sim_mL"],
            v_final_err_pct=n["v_final_err_pct"], rms_overlap_mL=n["rms_overlap_mL"],
            sim_arrival_2ml_s=n["sim_arrival_2ml_s"],
            sim_onset_pool_s=n.get("sim_onset_pool_s", np.nan)))
    (wf / "validation.json").write_text(json.dumps(num, indent=2))
    print(json.dumps(num, indent=2))
    plot(obs, ident, sim, others, num, wf / "validation.png")
    print("wrote", wf / "validation.png", "and", wf / "validation.json")
    if not args.no_video:
        compose_video(obs, sim, ident, tw, wf / "side_by_side_curve.mp4")


if __name__ == "__main__":
    main()
