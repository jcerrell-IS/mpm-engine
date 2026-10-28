"""Hold-time lookup table: how long to hold the cup at the roll's end pose to pour a target
volume, predicted by the twin at the identified viscosity and, for contrast, at the
handbook value.

Motion: the episode's recorded roll to its end pose (58.7 deg on ep0001), a dwell of
`hold` seconds there (the robot's dwell_s), then the recorded return. The hold is the only
control variable; the angle stays the calibrated pose, so every point is in the regime
where the drain rate Q = F/eta, not the cup's geometry, sets the volume. Each point is one
twin run (examples/pour_recorded_twin.py --hold); hold 0 is the validated ep0001 twin
itself, read from the existing run directories.

Run:
  python experiments/pour/pour_hold_sweep.py               # ~2.5 min per new point on the GPU
  python experiments/pour/pour_hold_sweep.py --dry         # table and plot from existing runs
  python experiments/pour/pour_hold_sweep.py --plan 150    # the dwell to command for 150 mL

Outputs (out/pour_wf/<episode>/):
  hold_table.json   V(hold) per eta, and the hold that reaches each target volume
  hold_table.png    the curves with the targets read off
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))

from pour_recorded_twin import OUT_ROOT, run

WF_ROOT = REPO / "out" / "pour_wf"
HOLDS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
TARGETS = (100.0, 115.0, 130.0, 145.0, 160.0, 175.0)
ETA_HANDBOOK = 1.41


def final_ml(metrics_csv: Path) -> float:
    with metrics_csv.open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["ml_rcv"])


def hold0_metrics(episode: str, eta: float, eta_hat: float) -> Path:
    """The existing hold-0 run: the primary directory at eta_hat, an archive otherwise."""
    tw = OUT_ROOT / episode
    if abs(eta - eta_hat) < 5e-3:
        return tw / "metrics.csv"
    return tw / f"archive_eta{round(eta * 100):03d}" / "metrics.csv"


def sweep(episode: Path, etas, dry: bool, device: str) -> dict:
    ident = json.loads((WF_ROOT / episode.name / "identify.json").read_text())
    eta_hat = round(float(ident["eta"]), 2)
    table = {}
    for eta in etas:
        pts = {}
        m0 = hold0_metrics(episode.name, eta, eta_hat)
        if m0.exists():
            pts[0.0] = final_ml(m0)
        for h in HOLDS:
            d = OUT_ROOT / f"{episode.name}_hold{h:g}_eta{eta:.2f}"
            if (d / "metrics.csv").exists():
                pts[h] = final_ml(d / "metrics.csv")
            elif not dry:
                res = run(episode, device=device, video=False, side_by_side=False, eta=eta,
                          hold=h)
                pts[h] = res["final_ml"]
            print(f"eta {eta:.2f} hold {h:g} s -> {pts.get(h, float('nan')):.1f} mL", flush=True)
        table[eta] = dict(sorted(pts.items()))
    return eta_hat, table


def invert(holds, vols, target: float):
    """Hold that reaches `target`, by interpolation on the monotone V(hold); None outside."""
    holds, vols = np.asarray(holds), np.asarray(vols)
    if len(vols) < 2 or not (vols.min() <= target <= vols.max()):
        return None
    order = np.argsort(vols)
    return float(np.interp(target, vols[order], holds[order]))


def plan(episode: str, targets) -> None:
    """Operator read-out from hold_table.json: the dwell_s to command for each target, the
    local sensitivity, and what the handbook viscosity would deliver at that dwell."""
    tab = json.loads((WF_ROOT / episode / "hold_table.json").read_text())
    eta_hat = tab["eta_hat"]
    run_hat = tab["runs"][f"{eta_hat:.2f}"]
    h_hat, v_hat = np.asarray(run_hat["hold_s"]), np.asarray(run_hat["final_ml"])
    hb = tab["runs"].get(f"{ETA_HANDBOOK:.2f}")
    print(f"{episode}: 300 mL fill, the recorded roll to its end pose, dwell there, the "
          f"recorded return; twin at eta_hat = {eta_hat:.2f} Pa.s")
    print(" target   dwell_s   dV/dt at that dwell    handbook-eta twin would deliver")
    for t in targets:
        h = invert(h_hat, v_hat, t)
        if h is None:
            print(f" {t:5.0f} mL   out of the table's range "
                  f"({v_hat.min():.0f}-{v_hat.max():.0f} mL)")
            continue
        slope = float(np.gradient(v_hat, h_hat)[np.argmin(np.abs(h_hat - h))])
        other = (f"{np.interp(h, hb['hold_s'], hb['final_ml']):.0f} mL" if hb else "--")
        print(f" {t:5.0f} mL   {h:6.2f}    {slope:5.1f} mL/s ({slope * 0.1:.1f} mL per 0.1 s)"
              f"    {other}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, default=REPO / "pouring_real_data" / "ep0001")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry", action="store_true", help="no new runs; table from existing")
    ap.add_argument("--etas", type=float, nargs="*", default=None,
                    help="viscosities to sweep (default: eta_hat and the handbook 1.41)")
    ap.add_argument("--plan", type=float, nargs="+", metavar="ML",
                    help="target volumes: print the dwell to command from hold_table.json")
    args = ap.parse_args()
    episode = args.episode.resolve()
    if args.plan:
        plan(episode.name, args.plan)
        return
    ident = json.loads((WF_ROOT / episode.name / "identify.json").read_text())
    etas = args.etas or [round(float(ident["eta"]), 2), ETA_HANDBOOK]
    eta_hat, table = sweep(episode, etas, args.dry, args.device)

    out = WF_ROOT / episode.name
    result = dict(episode=episode.name, eta_hat=eta_hat, targets_ml=list(TARGETS), runs={},
                  hold_for_target_s={})
    for eta, pts in table.items():
        holds, vols = list(pts.keys()), list(pts.values())
        result["runs"][f"{eta:.2f}"] = dict(hold_s=holds, final_ml=vols)
        result["hold_for_target_s"][f"{eta:.2f}"] = {
            f"{t:.0f}": invert(holds, vols, t) for t in TARGETS}
    (out / "hold_table.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (eta, pts), col in zip(table.items(), ("tab:blue", "tab:orange", "tab:green"),
                               strict=False):
        holds, vols = np.array(list(pts.keys())), np.array(list(pts.values()))
        lab = f"twin at eta = {eta:.2f} Pa.s" + (" (identified)" if abs(eta - eta_hat) < 5e-3
                                                 else " (handbook)")
        ax.plot(holds, vols, "o-", color=col, lw=1.6, ms=5, label=lab)
    for t in TARGETS:
        ax.axhline(t, color="0.75", lw=0.7, ls=":")
        h = result["hold_for_target_s"][f"{eta_hat:.2f}"].get(f"{t:.0f}")
        if h is not None:
            ax.plot([h, h], [0, t], color="tab:blue", lw=0.7, ls="--")
            ax.annotate(f"{t:.0f} mL: hold {h:.2f} s", xy=(h, t), xytext=(4, -12),
                        textcoords="offset points", fontsize=8, color="tab:blue")
    ax.axhline(97.2, color="k", lw=0.8, ls="-.", label="real ep0001, no hold (97.2 mL)")
    ax.set_xlabel("hold at the roll's end pose [s]  (the robot's dwell_s)")
    ax.set_ylabel("transferred volume [mL]")
    ax.set_ylim(0, None)
    ax.set_title(f"{episode.name}: hold-time lookup table from the twin (300 mL fill, 58.7 deg)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hold_table.png", dpi=130)
    plt.close(fig)
    print(json.dumps(result["hold_for_target_s"], indent=2))
    print("wrote", out / "hold_table.json", "and", out / "hold_table.png")
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
