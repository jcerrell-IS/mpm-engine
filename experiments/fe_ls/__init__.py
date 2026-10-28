"""Least squares through a function-encoder basis: the third row of the NCLaw table.

Purpose. Take the convex weak-form identification of experiments.nclaw.suite and
replace the KNOWN constitutive form with a trained function-encoder basis, so
the unknown is a function (mu(I) for sand, W'(I1bar) for jelly,
eta_app(gamma_dot) for the two viscous surrogates) rather than a scalar. Same
dump, same masks, same grid-consistent time-weak assembly, same convex solve;
only the dictionary changes. The recovered curve is then baked into the engine's
tabulated material and re-simulated, which turns a curve error into a trajectory
error.

Layout: baseline.py holds the whole study (dictionaries, the regularized solve,
the three identification legs, the bake, the rollout legs, the report). It is
one protocol and it stays one module; the section banners inside are its map.

Key results (docs/four_method_comparison.md, campaign of 2026-08-21, grid-20
NCLaw-suite trajectories, NCLaw's position-MSE metric):

  sand      mu(I) through the K = 8 granular basis: curve
            relL2 0.164 on the realized support, 0.094 dissipation-weighted,
            matching the constant truth at the data's median to 0.2 percent and
            missing at the unvisited ends. Rolled out with the tabulated mu(I)
            material: 7.4e-4 reconstruction and 1.9e-4 generalization, a 12x
            gap against known-form least squares, which is what paying at the
            unvisited ends costs.
  jelly     one-invariant hyperelastic basis, K = 6 plus one volumetric column:
            shear modulus to 7.0 percent, volumetric to 0.9 percent.
            Identification only. The warp engine has no tabulated hyperelastic
            material, so the rollout cell is UNSUPPORTED and nothing was faked
            to fill it.
  plasticine, water
            REFUSED by the viscous family (negative apparent viscosity over 24
            and 65 percent of the rate support). The refusal is the result: this
            is the correct behavior for a basis facing the wrong material class.

Artifacts: out/fe_ls_baseline/ (identify_<m>.json, rollout_<m>.json, legs/,
dumps/, fe_curve_<m>.png, results_<m>.json, report.md, run.log). See
artifact_dir below for which out/ tree that resolves to.

Run:
  .venv/bin/python -m experiments.fe_ls identify --material sand
  .venv/bin/python -m experiments.fe_ls rollout  --material sand
  .venv/bin/python -m experiments.fe_ls report   --material all
"""
from __future__ import annotations

import os
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = ENGINE_ROOT.parent          # video2sim, when the engine sits inside it


def artifact_dir(name: str = "fe_ls_baseline") -> Path:
    """out/<name>, preferring the staging tree's copy while it holds the runs.

    Compatibility, deliberate: this campaign ran as sim/fe_ls_baseline.py from
    the video2sim root, so its identifications, rollout scores, prediction dumps
    and the figures docs/four_method_comparison.md cites all live in
    video2sim/out/<name>, and the report stage reads the diff-sim campaign's
    results out of video2sim/out/diffsim_baseline beside them. A stage that
    wrote to mpm_engine/out/<name> instead would re-run finished legs and orphan
    the report. So an existing non-empty staging directory wins; a fresh
    checkout with no such directory writes under this repository's own out/.
    FE_LS_OUT overrides both, and DIFFSIM_OUT overrides where the diff-sim legs
    are read from.
    """
    env = os.environ.get("FE_LS_OUT" if name == "fe_ls_baseline" else "DIFFSIM_OUT")
    if env:
        return Path(env)
    legacy = STAGING_ROOT / "out" / name
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return ENGINE_ROOT / "out" / name
