# experiments

Campaigns. Each one is a package with a stages CLI, artifacts under `out/<name>/`,
at least one test in `tests/`, and a row below. Run them from this directory's
parent with the project environment.

## What the campaigns established

nclaw, the cross-engine comparison. NCLaw's own protocol run with this engine and
a convex weak-form identification from one thrown-cube trajectory per material:
position MSE 1.7e-6 reconstruction and 7.8e-8 generalization on jelly, against
their published 2.4e-4 and 4.1e-4, with the simulator never differentiated.

    .venv/bin/python -m experiments.nclaw.suite report --material jelly

elastic, the drop family. Elastic moduli, a yield stress and nonlinear
hyperelastic coefficients from one gravity drop by the same convex solve. E to
0.069 percent on the cube through the grid-consistent route, which is the Step 0
acceptance gate; the earlier radial-window route left 13.4 percent there.

    .venv/bin/python -m experiments.elastic grid-gate

robotics, the Franka dough chain. A plate squeezes a dough blob, the grid-impulse
plate force identifies the law, and the identified law predicts and then plans.
Reference (tau_y, eta) = (200, 40), recovered (192, 55) from the grid impulse
against (384, 56) from the stress integral, and the recovered law transfers to
dough volumes it never saw.

    .venv/bin/python -m experiments.robotics.predict_volume_franka

diffsim, the differentiable-simulation baseline. Gradient descent through a
separate minimal JAX MPM, so warp stays non-differentiable. At equal budget on
jelly it reaches mu +0.4 percent in 52.4 minutes where the convex solve reaches
+0.8 percent in 5.0 seconds; run past that stop it wins on all seven parameters
at 774x to 3068x the wall time. The expected granular chaos did not appear: the
sand loss surface is unimodal and all five initializations converge.

    ../.venv/bin/python -m experiments.diffsim report --material all

fe_ls, least squares through a trained basis. The same convex solve with a
function-encoder dictionary in place of the known constitutive form, so the
unknown is a function. Sand mu(I) to curve relL2 0.164 on the realized support
and 0.094 dissipation-weighted, and the viscous basis refuses plasticine and
water rather than fitting them, which is the correct answer for the wrong
material class.

    .venv/bin/python -m experiments.fe_ls report --material all

pour, the recorded hardware pour. One side video and the Franka joint log of a
glycerol pour from the measured 500 mL cup give the receiver curve by the cup's own
graduations, a viscosity from the time-weak brink lubrication balance (ep0001:
3.03 Pa.s, 1.7 percent statistical and 14 percent for the fill tolerance), a twin at
that value at -2.1 percent of the real transfer, and a hold-time table for metering
100 to 160 mL. The episodes live outside the repository under `pouring_real_data/`.

    .venv/bin/python -m experiments.pour.pour_weakform_identify

## Stages, per campaign

| campaign | stages |
| --- | --- |
| `nclaw.suite` | gen, identify, rollout, report, cross, all |
| `elastic` | recover, shape, errors, sample-complexity, grid-gate, sequential, sequential-rollout, plastic, plastic-gate, plastic-gate-figure, plastic-sequential, hyperelastic, hyperelastic-fe, fe-basis, all |
| `robotics.<name>` | one module per leg, see `robotics/__init__.py` |
| `diffsim` | validate, landscape, fit, refine, ls, report |
| `fe_ls` | identify, rollout, report |
| `pour.<name>` | one module per stage: calibrate_geometry, perception, weakform_identify, validate, hold_sweep; see `pour/__init__.py` |

Each campaign's `__init__.py` carries its own numbers, artifact paths and the
limits of what it measured. `elastic/README.md` additionally records the
reproduction checks from its consolidation.

The diffsim campaign runs on the video2sim staging interpreter because jax is
installed there and not in the engine venv. Both interpreters resolve `warpmpm`,
`ident` and `common` to this repository's `src` tree.

## archive/

Finished exploration from the flat robotics-era directory: the shear-cell
rheology studies, the rollout figure and video variants, the perception probes,
the flood sweep. Kept for the writeup paths and not maintained. `archive/README.md`
indexes what each one did.

`weak_contrastive_pilot.py` sits at the top level, outside both directories. It is
a concluded pilot with a Pvol fix still queued.
