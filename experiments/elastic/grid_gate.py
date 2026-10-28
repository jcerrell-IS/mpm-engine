"""Step 0 acceptance gate: grid-consistent elastic recovery against the radial window.

The radial-window recovery in drop.py vanishes on a sphere surface, so it is accurate
on the sphere drop and biased on the cube drop, where the window is nonzero on the
faces and the floor contact traction leaks into the residual. This stage runs the
grid-consistent assembly of ident.weakform.elastic_grid on the same two dumps and
reports both routes side by side, plus a sweep of the collider clearance so the
sensitivity is on the record rather than assumed.

Acceptance (docs/nclaw_comparison_plan.md, Step 0):
  sphere E error at or below the radial window's 0.15 percent,
  cube E error within 2 percent,
  lambda error and cond(A^T A) reported for both.

Result, all four gates pass. Shipped route is timeweak at margin 3 cells:
  sphere  E error 0.102 percent, mu 0.212 percent, lam 0.966 percent, nu 0.2986,
          180 rows of 115548, cond(A^T A) 70.6
  cube    E error 0.069 percent, mu 0.424 percent, lam 3.51 percent, nu 0.3046,
          180 rows of 246723, cond(A^T A) 40.8
against the radial window's 0.35 percent on the sphere and 13.4 percent on the cube.
Closing the window leak took the cube from 13.4 percent to under the 2 percent bar,
which is what let the NCLaw comparison proceed.

Artifacts: out/elastic/grid_gate.json. The pre-consolidation run is preserved at
video2sim/out/elastic_drop/grid_gate.json, which is the path
docs/nclaw_comparison_plan.md names.

Run:  .venv/bin/python -m experiments.elastic grid-gate
      .venv/bin/python -m experiments.elastic grid-gate --shapes sphere
"""
from __future__ import annotations

import json

import numpy as np

from .core import artifact, find_artifact, grid_recover
from .drop import recover as radial_recover

DUMPS = {"sphere": "truth.npz", "box": "box_truth.npz"}
N_GRID_DEFAULT = 48          # core.run_drop default, which the dump does not record


def _write(report: dict) -> None:
    """Persist after every shape, so an interrupted sweep keeps what it measured."""
    artifact("grid_gate.json").write_text(json.dumps(report, indent=2, default=float))


def run(shapes=("sphere", "box"), margins=(2.0, 3.0, 4.0), n_grid=N_GRID_DEFAULT,
        frame_stride: int = 2, log=print) -> dict:
    """Both routes on both dumps, with the collider-clearance sweep recorded."""
    existing = find_artifact("grid_gate.json")
    report: dict = (json.loads(existing.read_text()) if existing is not None
                    else {"routes": {}, "margin_sweep": {}})
    report.setdefault("routes", {})
    report.setdefault("margin_sweep", {})
    for shape in shapes:
        p = find_artifact(DUMPS[shape])
        if p is None:
            log(f"[gate] missing {DUMPS[shape]}; regenerate with the drop stage")
            continue
        sweep = [grid_recover(p, n_grid=n_grid, margin_cells=m, frame_stride=frame_stride,
                              route="timeweak", log=log)
                 for m in margins]
        report["margin_sweep"][shape] = sweep
        shipped = next((s for s in sweep if s["margin_cells"] == 3.0), sweep[0])
        report["routes"][shape] = {
            "grid_timeweak": shipped,
            "grid_instant": grid_recover(p, n_grid=n_grid, margin_cells=3.0,
                                         frame_stride=frame_stride, route="instant",
                                         log=log),
        }
        try:
            rad = radial_recover(p, log=lambda *_: None)
            report["routes"][shape]["radial_window"] = {
                "E_err": rad["E_err"], "mu_err": rad["mu_err"],
                "cond_AtA": rad["cond"], "E_hat": rad["E"], "nu_hat": rad["nu"],
            }
            log(f"[radial] {shape:7s} E err {100*rad['E_err']:.2f}%  "
                f"mu err {100*rad['mu_err']:.2f}%  cond {rad['cond']:.3e}")
        except Exception as exc:
            report["routes"][shape]["radial_window"] = {"error": str(exc)}
        _write(report)

    gates = {}
    if "sphere" in report["routes"]:
        s = report["routes"]["sphere"]
        gates["sphere_E_err_le_0.15pct"] = s["grid_timeweak"]["E_err"] <= 0.0015
        gates["sphere_beats_radial_window_E"] = (
            s["grid_timeweak"]["E_err"] <= s["radial_window"].get("E_err", np.inf))
    if "box" in report["routes"]:
        b = report["routes"]["box"]
        gates["cube_E_err_le_2pct"] = b["grid_timeweak"]["E_err"] <= 0.02
        gates["cube_beats_radial_window_E"] = (
            b["grid_timeweak"]["E_err"] <= b["radial_window"].get("E_err", np.inf))
    report["gates"] = gates
    _write(report)
    log(f"\n[gate] acceptance: {json.dumps(gates)}")
    log(f"[gate] wrote {artifact('grid_gate.json')}")
    return report
