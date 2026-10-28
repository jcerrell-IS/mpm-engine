"""Differentiable-simulation baseline: gradient descent through an MPM rollout.

Purpose. This campaign is the comparison method for the convex weak-form
identification, the row that NCLaw and DPSI occupy: fit material parameters by
backpropagating (here forward-mode differentiating) a particle-position loss
through a whole MPM rollout. The TrackEUCLID invariant stands. warp-mpm is never
made differentiable; the baseline differentiates a separate minimal JAX MPM
transcribed from the warp kernels (forward.py), and every truth trajectory and
every scored rollout still comes from warp.

Layout:
  forward.py   the differentiable JAX MLS-MPM, transcribed substep for substep
               from src/warpmpm/kernels, with the two AD guards it needs
  identify.py  the stage driver: validate, landscape, fit, refine, ls, report

Key results (docs/four_method_comparison.md, campaign of 2026-08-21, one thrown
cube per material at grid 20, NCLaw's position-MSE metric):

  cross-engine forward gap at truth theta   9.1e-13 to 3.3e-10 MSE, five to
                                            eight orders below every published
                                            NCLaw cell
  equal-budget best of 5 inits vs the
  convex weak-form solve on the same dump   jelly mu +0.4 / lam -1.3 percent in
                                            52.4 min, against +0.8 / -4.8
                                            percent in 5.0 s
                                            plasticine mu +63 / lam -59 /
                                            tau_y -0.6 percent in 30.8 min,
                                            against +0.4 / +1.0 / -0.4 percent
                                            in 5.4 s
                                            sand phi +0.004 percent in 72.1
                                            min, against +6.5 percent in 2.3 s
                                            water K +0.0002 percent in 45.9
                                            min, against -7.4 percent in 2.1 s
  continued past the equal-budget stop      the plasticine degeneracy is a
                                            valley, not a wall: mu to 0.02
                                            percent, lam to 0.01 percent, and
                                            the descent then beats the convex
                                            solve on all seven parameters at
                                            774x to 3068x its total wall time
  sand loss landscape                       unimodal and smooth at coarse and
                                            fine scale, all 5 inits converge
                                            (spread 1.00), so the granular
                                            chaos expectation was falsified

Artifacts: out/diffsim_baseline/ (validate_<m>.json, landscape_<m>.json and png,
fit_<m>.jsonl ledger, fit_<m>_inits.jsonl, refine_<m>.json, ls_<m>.json,
results_<m>.json, report.md). See artifact_dir below for which out/ tree that
resolves to.

Environment. jax is installed in the video2sim staging venv, not in the engine
venv, so the run command below uses that interpreter from this repository root.
Both venvs resolve warpmpm, ident and common to this repository's src tree, so
nothing else changes.

Run:
  ../.venv/bin/python -m experiments.diffsim validate  --material jelly
  ../.venv/bin/python -m experiments.diffsim fit       --material jelly
  ../.venv/bin/python -m experiments.diffsim landscape --material sand
  ../.venv/bin/python -m experiments.diffsim report    --material all
"""
from __future__ import annotations

import os
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = ENGINE_ROOT.parent          # video2sim, when the engine sits inside it


def artifact_dir(name: str = "diffsim_baseline") -> Path:
    """out/<name>, preferring the staging tree's copy while it holds the runs.

    Compatibility, deliberate: this campaign ran as sim/diffsim_identify.py from
    the video2sim root, so its results, resumable fit ledgers and the figures
    docs/four_method_comparison.md cites all live in video2sim/out/<name>. A
    stage that wrote to mpm_engine/out/<name> instead would lose the resume and
    orphan the report. So an existing non-empty staging directory wins; a fresh
    checkout with no such directory writes under this repository's own out/.
    DIFFSIM_OUT overrides both.
    """
    env = os.environ.get("DIFFSIM_OUT")
    if env:
        return Path(env)
    legacy = STAGING_ROOT / "out" / name
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return ENGINE_ROOT / "out" / name
