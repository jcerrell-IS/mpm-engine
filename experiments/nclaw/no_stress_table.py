"""The no-stress tier table: our rollouts, the full-channel run, their published cells.

Reads the ``compare_<material>*_nostress.json`` files ``compare.py --no-stress``
writes and prints one markdown table per material, plus the recovered
parameters, the refusals and the identification wall times. The full-channel
column is the addendum of docs/four_method_comparison.md, entered here as data
so a tier row is always read next to the row it degrades from; the published
column is NCLaw's own, from ``suite.NCLAW_PUBLISHED``.

Run:  .venv/bin/python -m experiments.nclaw.no_stress_table [--tier nostress]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "out" / "nclaw_cross_compare"

# per-material flag tag of the comparison, matching the full-channel run
TAGS = {"jelly": "_nclawbc_sub1", "plasticine": "_nclawbc",
        "sand": "_nclawbc_nclawlaw_sub1", "water": "_nclawbc_nclawlaw"}

# the full-channel addendum of docs/four_method_comparison.md, per scene group
FULL_CHANNEL = {
    "jelly": {"dataset": 3.0e-6, "time": 8.8e-6, "vel": 1.8e-6},
    "plasticine": {"dataset": 7.2e-7, "time": 9.0e-7, "vel": 4.0e-7},
    "sand": {"dataset": 6.2e-9, "time": 9.1e-9, "vel": 1.2e-9},
    "water": {"dataset": 3.8e-7, "time": 6.3e-6, "vel": 1.7e-7},
}
SCENE_ROLE = {"dataset": "reconstruction", "time": "time", "vel": "velocity"}


def _role(scene: str) -> str:
    return "vel" if scene.startswith("vel") else scene


def material_rows(material: str, tier: str = "nostress") -> dict:
    from experiments.nclaw.suite import NCLAW_PUBLISHED
    path = OUT / f"compare_{material}{TAGS[material]}_{tier}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    res = json.loads(path.read_text())
    published = NCLAW_PUBLISHED[material]
    legs = [k for k in next(iter(res["scenes"].values())) if k == "truth_theta"
            or k.startswith("recovered")]
    rows = []
    for scene, cells in res["scenes"].items():
        role = _role(scene)
        pub = published[SCENE_ROLE[role]]
        row = {"scene": scene, "role": role, "published": pub,
               "full_channel": FULL_CHANNEL[material][role]}
        for leg in legs:
            row[leg] = cells[leg]["mse"]
        row["margin_vs_published"] = pub / cells["recovered"]["mse"]
        rows.append(row)
    return {"result": res, "rows": rows, "legs": legs}


def render(material: str, tier: str = "nostress") -> str:
    got = material_rows(material, tier)
    res, rows, legs = got["result"], got["rows"], got["legs"]
    recovered_legs = [leg for leg in legs if leg.startswith("recovered")]
    head = (["scene", "our sim, correct properties"]
            + [f"our sim, {leg.replace('recovered', 'recovered')}" for leg in recovered_legs]
            + ["full-channel recovered", "their published", "margin"])
    lines = [f"### {material}", "",
             "| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in rows:
        cells = [r["scene"], f"{r['truth_theta']:.1e}"]
        cells += [f"{r[leg]:.1e}" for leg in recovered_legs]
        cells += [f"{r['full_channel']:.1e}", f"{r['published']:.1e}",
                  f"{r['margin_vs_published']:.0f}x"]
        lines.append("| " + " | ".join(cells) + " |")
    vel = [r for r in rows if r["role"] == "vel"]
    if vel:
        # the margin of the mean cell, the addendum's convention: their published
        # number divided by the mean MSE over the velocity scenes, not the mean of
        # the per-scene ratios, which a single easy scene would inflate
        mean_rec = float(np.mean([r["recovered"] for r in vel]))
        lines.append("| vel mean | "
                     + f"{np.mean([r['truth_theta'] for r in vel]):.1e} | "
                     + " | ".join(f"{np.mean([r[leg] for r in vel]):.1e}"
                                  for leg in recovered_legs)
                     + f" | {vel[0]['full_channel']:.1e} | {vel[0]['published']:.1e} | "
                     + f"{vel[0]['published'] / mean_rec:.0f}x |")
    lines += ["", f"theta: {json.dumps(res['theta_recovered'], default=float)}",
              f"refused: {res['identify_diagnostics']['refused_parameters']}",
              f"identify wall time: {res['wall_identify_s']:.1f} s "
              f"({json.dumps(res.get('wall_times_s'), default=float)})",
              f"provenance: {json.dumps(res.get('channel_provenance'), default=float)}"]
    for name, spec in (res.get("theta_variants") or {}).items():
        lines += ["", f"variant {name}: theta "
                  f"{json.dumps(spec['theta'], default=float)}",
                  f"  provenance: {spec.get('provenance')}",
                  f"  refused: {spec.get('refused')} {spec.get('note', '')}",
                  f"  diagnostics: {json.dumps(spec.get('diagnostics'), default=float)}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="nostress")
    ap.add_argument("--materials", default="jelly,plasticine,sand,water")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = "\n\n".join(render(m, a.tier)
                       for m in a.materials.split(",") if m.strip())
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
