"""Stages CLI for the elastic identification campaign.

Run:  .venv/bin/python -m experiments.elastic <stage> [flags]

Stages, in the order they build on each other:

  recover              sphere drop, radial-window recovery, re-simulate and compare
  shape                learn on a rectangle, predict the held-out star geometry
  errors               reconstruction versus generalization position error
  sample-complexity    estimator spread versus row count against Gauss-Markov
  grid-gate            Step 0 acceptance gate, grid-consistent versus radial window
  sequential           the Bayesian posterior of E and nu versus frame
  sequential-rollout   rollout error versus frames observed, with the online surrogate
  plastic              von-Mises drop plus the stress-basis check
  plastic-gate         same material at two drop heights, yield identified or refused
  plastic-gate-figure  the deviatoric-strain histogram of that gate
  plastic-sequential   recursive G and yield versus frame
  hyperelastic         Mooney and Yeoh coefficient recovery, single and multi-probe
  hyperelastic-fe      the same rows through the trained function-encoder bases
  fe-basis             build the four-family hyperelastic basis and its error bound
  all                  recover, grid-gate, plastic-gate, sample-complexity, fe-basis

Stages reuse dumps that already exist, in out/elastic first and then in the
pre-consolidation directories (out/elastic_drop, out/plastic_drop, out/hyperelastic
under either this repository or the video2sim staging tree), so a re-run does not
re-simulate a 676 MB trajectory. Pass --force to simulate anyway.
"""
from __future__ import annotations

import argparse

STAGES = [
    "recover", "shape", "errors", "sample-complexity", "grid-gate",
    "sequential", "sequential-rollout", "plastic", "plastic-gate",
    "plastic-gate-figure", "plastic-sequential", "hyperelastic",
    "hyperelastic-fe", "fe-basis", "all",
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.elastic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--shapes", default="sphere,box",
                   help="grid-gate: which dumps to gate")
    ap.add_argument("--margins", default="2,3,4",
                   help="grid-gate: collider clearance sweep, in cells")
    ap.add_argument("--n-grid", type=int, default=48,
                   help="grid-gate: the grid resolution the dump was produced at")
    ap.add_argument("--dump", default="truth.npz",
                   help="sequential stages: which dump to stream")
    ap.add_argument("--which", default="mooney", choices=["mooney", "yeoh", "all"],
                   help="hyperelastic: which family to recover")
    ap.add_argument("--yield-stress", type=float, default=2.0e4,
                   help="plastic-gate: the yield stress of the two drops")
    a = ap.parse_args(argv)

    from . import drop, fe_basis, grid_gate, hyperelastic, plastic, sequential

    if a.stage in ("recover", "all"):
        drop.stage_recover()
    if a.stage == "shape":
        drop.stage_shape()
    if a.stage == "errors":
        drop.stage_errors()
    if a.stage in ("sample-complexity", "all"):
        drop.stage_sample_complexity()
    if a.stage in ("grid-gate", "all"):
        grid_gate.run(shapes=tuple(a.shapes.split(",")),
                      margins=tuple(float(m) for m in a.margins.split(",")),
                      n_grid=a.n_grid)
    if a.stage == "sequential":
        sequential.stage_elastic_figure(a.dump)
    if a.stage == "sequential-rollout":
        sequential.stage_rollout_vs_frames(a.dump)
    if a.stage == "plastic":
        plastic.stage_truth()
    if a.stage in ("plastic-gate", "all"):
        plastic.stage_gate(a.yield_stress)
    if a.stage == "plastic-gate-figure":
        plastic.stage_gate_figure()
    if a.stage == "plastic-sequential":
        sequential.stage_plastic_figure()
    if a.stage == "hyperelastic":
        hyperelastic.stage_recover(a.which)
    if a.stage == "hyperelastic-fe":
        hyperelastic.stage_recover_fe()
    if a.stage in ("fe-basis", "all"):
        fe_basis.run()


if __name__ == "__main__":
    main()
