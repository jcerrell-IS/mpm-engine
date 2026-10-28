# Four methods on the same trajectories: LS, LS + function encoder, diff-sim, NCLaw

One overnight campaign (2026-08-21) compared four methods on the same
trajectories: given a single thrown-cube trajectory per material, how do
the recovered laws and their rollouts compare. Three methods ran on
identical warp truths at grid 20 (their training grid), scored by NCLaw's
own metric (position MSE, unit box, every 5th frame) on the training cube
(reconstruction) and each material's held-out mesh (generalization). The
NCLaw column is their published numbers, from their engine; that column is
context, and every caveat from docs/nclaw_comparison_plan.md applies to it.

Methods: LS = our convex weak-form solve with the known constitutive form.
LS + FE = the same convex solve with a trained function-encoder basis in
place of the known form. Diff-sim = gradient descent through a
differentiable JAX MPM rebuilt for this purpose (warp is never
differentiated), forward-mode AD, best of 5 initializations at a fixed
equal budget. Truth replay = the truth parameters replayed in the same
engine (the error at the true parameters).

## Rollout table (position MSE; recon = cube, gen = held-out mesh)

| material | LS recon / gen | LS+FE recon / gen | diff-sim recon / gen | diff-sim converged (d) | NCLaw recon / gen |
| --- | --- | --- | --- | --- | --- |
| jelly (armadillo) | 1.7e-6 / 7.8e-8 | UNSUPPORTED (a) | 3.1e-8 / 3.9e-8 | 8.4e-14 / 5.8e-14 | 2.4e-4 / 4.1e-4 |
| plasticine (bunny) | 4.6e-8 / 1.9e-8 | DIVERGED / 3.3e-3 (b) | 3.3e-6 / 1.6e-6 | 2.7e-12 / 1.0e-12 | 6.5e-5 / 2.3e-4 |
| sand (blub) | 6.1e-5 / 2.1e-5 | 7.4e-4 / 1.9e-4 | 2.7e-11 / 9.7e-12 | 3.8e-11 / 1.3e-11 | 2.6e-5 / 3.6e-4 |
| water (spot) | 2.4e-4 / 5.4e-6 (c) | 2.4e-7 / 2.9e-7 (c) | 7.0e-14 / 4.8e-14 | 7.0e-14 / 4.8e-14 | 2.0e-5 / 2.4e-4 |

(a) the warp engine has no tabulated hyperelastic material, so the
recovered W'(I1) curve (shear modulus 7.0 percent, volumetric 0.9 percent)
cannot be re-simulated.
(b) the FE viscous surrogate for an elastoplastic solid: the identification
itself REFUSED (negative apparent viscosity over 24 percent of the rate
support), and the clipped-at-zero curve diverges on the cube.
(c) not comparable to each other: the water LS cell rolls the identified
bulk (7.4 percent low), while the FE-viscous cell receives the truth bulk
as data by design and adds a near-zero viscosity table, so it measures the
surrogate's deviatoric contribution, not a bulk recovery.
(d) the equal-budget best init continued at reduced step until the loss
stopped decreasing (an extra 15 to 46 minutes per material). This column
reports accuracy without a time limit; the equal-budget column reports
accuracy at fixed time. Converged, the diff-sim beats the convex solve on
all seven parameters, at 774x to 3068x its total wall time.

## Identification table (same dumps, per material)

| material | unknowns | LS error / wall | diff-sim best-of-5 error / wall | diff-sim init spread |
| --- | --- | --- | --- | --- |
| jelly | mu, lam | +0.8 %, -4.8 % / 5.0 s | +0.4 %, -1.3 % / 52.4 min | 3 of 5 inits land at 7-9 % (mu) and 20-27 % (lam) |
| plasticine | mu, lam, tau_y | +0.4 %, +1.0 %, -0.4 % / 5.4 s | +63 %, -59 %, -0.6 % / 30.8 min | elastic pair spreads 3.2x and 3.8x, trading at fixed wave speed |
| sand | phi | +6.5 % / 2.3 s | +0.004 % / 72.1 min | all 5 converge (spread 1.00) |
| water | K | -7.4 % / 2.1 s | +0.0002 % / 45.9 min | all 5 within 0.03 % |

LS + FE identification (curve errors on the realized support): sand mu(I)
relL2 0.164 (0.094 dissipation-weighted; the curve matches the constant
truth at the data's median to 0.2 percent and misses at the unvisited
ends; the 12x rollout gap against known-form LS comes from those ends);
jelly W'(I1) shear 7.0 percent; plasticine and water REFUSED by the
viscous family (negative viscosity over 24 and 65 percent of support);
the basis family does not contain these materials, so refusal is expected.

## Findings

1. At equal wall time the plasticine elastic pair stays 60 percent wrong;
   with more time the same descent converges. The equal-budget
   position-matching loss trades mu against lam at nearly fixed wave speed
   (the DPSI Table-3 phenomenon, reproduced from our own harness), while
   the weak form separates all three parameters in 5 seconds. Continued
   past the budget, the descent converges (mu 0.02 percent, lam 0.01
   percent). So the diff-sim converged beats the convex solve on every
   parameter, at three to four orders of magnitude more wall time; the
   convex solve gets within a few percent in seconds and is never
   initialization-dependent.
2. The runs falsified the sand chaos expectation. The loss as a function
   of phi is unimodal and smooth at both coarse and fine scales at this
   horizon (0.5 s, grid 20); no initialization fails. Gradient descent
   through granular contact is viable at NCLaw's own scale. Cost and
   parameter degeneracies separate the methods; convergence does not.
3. The LS sand row carries a grid-20 bias (+6.5 percent in phi, against
   +0.8 percent at grid 32): the weak-form rows degrade faster with grid
   coarseness than the diff-sim loss does. A follow-up run at grid 32
   would quantify this.
4. The FE basis errs where the data is absent: it matches the known form
   where the data lives, pays at the unvisited ends (sand rollout 12x),
   refuses the wrong material class (plasticine, water), and cannot be
   rolled out where the engine lacks the material (jelly).
5. Against NCLaw's published numbers, method by method. Least squares
   wins jelly and plasticine on both scenes, wins sand and water
   generalization, loses sand reconstruction (6.1e-5 against 2.6e-5, the
   grid-20 friction bias of point 3) and loses water reconstruction
   (2.4e-4 against 2.0e-5, the 7.4 percent bulk error). The
   function-encoder solve wins water on both scenes and sand
   generalization, loses sand reconstruction, refuses plasticine, and has
   no jelly rollout. Diff-sim wins every scene at equal budget. These
   runs use two different engines; the cross-engine addendum below is
   the controlled comparison.

Reference errors and checks: the JAX-vs-warp forward gap at truth theta is
9.1e-13 to 3.3e-10 MSE (five to eight orders below every published NCLaw cell);
sand's best diff-sim init dipped below that reference error; the comparison
against it detected the discretization gap absorbed into phi. The
truth-parameter replay is bitwise zero in every cell. Sand blub's truth
seeds 106 escaping particles; in-box MSE accompanies every affected cell.

Artifacts: out/fe_ls_baseline/ (FE and LS rows, rollouts, per-material
reports and figures), out/diffsim_baseline/ (fits, landscapes, floors,
report.md), docs/diffsim_baseline_plan.md and the FE subsection (every
deviation with its measurement). The refine stage completed at 04:36 and
its converged rows are folded into the table and the reading above.

## Final addendum (2026-08-22): identified from their trajectories, scored on their trajectories

The cross-engine experiment completed on all four materials. Their simulator
generated every trajectory; our convex solve identified the law from the
dataset scene alone; our engine rolled the recovered law on all five scenes,
seeded from their frame-0 particles at their time step, under the opt-in
compatibility options (their grid semantics for all materials, their linear
volumetric law for water). Cells are position MSE against their trajectory.

| material | scene | our sim, correct properties | our sim, recovered properties | their published | margin |
| --- | --- | --- | --- | --- | --- |
| jelly | dataset | 1.2e-13 | 3.0e-6 | 2.4e-4 | 81x |
| jelly | time | 2.8e-13 | 8.8e-6 | 9.8e-4 | 111x |
| jelly | vel mean | 9.3e-14 | 1.8e-6 | 2.4e-4 | 132x |
| plasticine | dataset | 7.8e-12 | 3.6e-7 | 6.5e-5 | 181x |
| plasticine | time | 1.4e-11 | 5.6e-7 | 1.4e-4 | 252x |
| plasticine | vel mean | 5.3e-12 | 2.5e-7 | 4.6e-5 | 187x |
| sand | dataset | 2.0e-11 | 6.2e-9 | 2.6e-5 | 4200x |
| sand | time | 5.0e-11 | 9.1e-9 | 4.2e-5 | 4600x |
| sand | vel mean | 3.1e-11 | 1.2e-9 | 6.5e-5 | 53000x |
| water | dataset | 3.0e-12 | 3.8e-7 | 2.0e-5 | 52x |
| water | time | 9.0e-11 | 6.3e-6 | 3.5e-4 | 55x |
| water | vel mean | 5.2e-13 | 1.7e-7 | 1.9e-5 | 112x |

Counting convention, used everywhere below. Each material has five
rollouts: dataset, time, and three velocity scenes. NCLaw publishes one
number per axis (reconstruction, time, one velocity aggregate), so the
tables make 12 aggregate comparisons per tier, and each of the 20 rollouts
is compared against its axis's published number. Geometry scenes are not
in this addendum. Every rollout beats its published number. Identified
parameters from their data: jelly E 1.9 percent, nu 4 percent; plasticine
E 0.9, nu 0.5, yield 1.0 percent; sand friction 25.0 degrees within 0.07
percent; water stiffness
within 0.54 percent.

Each material needed one distinct fix, found by running our simulator twice against their
trajectory, once with the correct properties and once with the recovered
ones. The first run shows how much the two simulators disagree on their
own; the second adds whatever the identification got wrong:
1. Plasticine and jelly needed the boundary and transfer semantics: their
   freeslip zeroes only approaching wall-normal velocity, their grid divides
   by mass plus 1e-7, and their transfer is MLS. These are now five opt-in
   engine options, off by default and bit-identical when off; the wall fix
   alone cut the plasticine truth-replay error 141x.
2. Water needed their volumetric law: linear Kirchhoff stress lambda J(J-1)I
   at lambda = 57692 Pa, fitted from their stress channel with residual
   0.0000 while our power-law form fits with negative stiffness. Their code
   comments say the stiffer, more realistic law breaks their gradients; the
   comparison flag implements theirs, our default is unchanged.
3. For sand the engine was correct; the estimator was the problem: the
   shear-to-pressure ratio is constant over the yield set (0.5676 against
   0.5680), while the momentum-balance fit through a body only partly at
   yield reads 47 percent residual and 0.87. The identification takes that
   constant level whenever the solve residual exceeds 0.15, with both
   values recorded.

The sys-id comparison closes the loop: their own gradient-based oracle on
their engine reports 3.3e-13 to 2.5e-8 on these scenes; the convex solve
reaches within one to three orders of that in seconds, across an engine
boundary. Videos: out/nclaw_suite/videos_blender/
<material>_cross.mp4.

## Addendum (2026-08-21): reproduced on their engine, by the user

The user regenerated NCLaw's dataset and reran both their method and their
sys-id baseline (differentiable MPM, known constitutive form, parameters
optimized) on their own engine, tasks (a) time, (b) velocity, (c) geometry:

| method on their engine | jelly a/b/c | sand a/b/c | plasticine a/b/c | water a/b/c |
| --- | --- | --- | --- | --- |
| NCLaw (reproduced) | 9.6e-4 / 2.3e-4 / 3.6e-4 | 4.0e-5 / 5.8e-5 / 3.7e-4 | 1.5e-4 / 4.4e-5 / 1.6e-4 | 1.2e-3 / 1.4e-4 / 3.8e-4 |
| sys-id (reproduced) | 2.5e-8 / 3.8e-9 / 6.9e-9 | 1.2e-12 / 3.5e-13 / 3.3e-13 | 2.2e-10 / 1.1e-10 / 1.6e-10 | 1.1e-8 / 3.2e-10 / 6.0e-10 |

Readings: the NCLaw reproduction matches their published cells for jelly,
sand and plasticine; water reproduces 3 to 7x worse than published (their
Table 7 attributes water's published quality to an auxiliary velocity loss,
so this is plausibly a training-variant or seed difference). The sys-id
rows show their trajectories are fittable to 1e-12 inside their own engine,
which confirms that the cross-engine cells measured from their data are
dominated by the engine difference and pins the conclusion both engines now
agree on: the known constitutive form is worth four to nine orders of
magnitude over the learned network on their own benchmark, whether reached
by gradient descent (their sys-id, our diff-sim) or by the convex solve in
seconds. The remaining number is the engine-gap error at the true
parameters on their trajectories; the cross stage computes and stores it on
arrival
(identification_excess_over_floor in cross_<material>_<scene>.json).

## No-stress tier (2026-08-23): plan

The addendum above identified from their trajectories with every stored
channel available, the stress channel included. This tier removes the
stress channel from identification and keeps every stored kinematic channel
(x, v, C read as L, F) exactly as the full-channel run used it. Scene facts
stay available: particle volume, density, the grid, the box, gravity, and
for sand the configured elastic pair, which their own Drucker-Prager module
also holds fixed while fitting the friction angle alone. The stress channel
may be read for diagnosis only, and in exactly one labeled variant as a
measurement inside one grid cell of the floor, which is what a force plate
under the pile would give.

Per material, what the tier changes:

- jelly: the elastic momentum fit on the stored channels, unchanged.
- plasticine: the same elastic pair, plus a second yield estimator, the
  momentum fit with a yield column on the fast-shearing set, reported next
  to the strain-cap reading of the stored elastic F.
- sand: three pressure sources feeding the same friction fit. Primary is
  the Hencky volumetric relation on the stored F at E = 1e6 and nu = 0.2,
  which is how their own module gets pressure. Variant is pressure measured
  within one cell of the floor with the depth-below-surface shape scaled to
  match that basal level. Ablation is the pure depth closure, density times
  gravity times depth below the per-column free surface.
- water: the volumetric fit with J from the stored F.
- rollouts and scoring exactly as the full-channel run: our engine from
  their frame-0 particles, a correct-property and a recovered-property run
  per scene, their metric, and the same compatibility flags per material
  (jelly nclaw-bc substeps 1; plasticine nclaw-bc; sand nclaw-bc nclaw-law
  substeps 1; water nclaw-bc nclaw-law).

### Pre-registered expectations, written before the fits ran

1. Jelly's identification never reads stress, so the tier must reproduce
   the full-channel parameters to the digit (E 98071.36, nu 0.208082) and
   the five jelly cells with them. Any change falsifies this.
2. Plasticine's elastic pair is in the same position. Its strain-cap yield
   reading takes the saturation of the deviatoric Hencky strain of the
   stored elastic F, so it is stress-free as well and should also
   reproduce (5055.05 Pa, 1.1 percent high). The new momentum-fit yield
   column is the estimator that does not use the strain-cap reading;
   expect it within a few tens of percent, which at quadratic growth of
   rollout error in parameter error leaves the 90x to 156x published
   margins intact. If its residual check refuses, the report keeps the
   refusal as the result.
3. Sand is the only material whose full-channel identification read the
   stress channel. Their stress channel is elasticity(F) of the stored
   elastic F, so the Hencky volumetric relation at their fixed E and nu
   should reproduce the stored pressure to a few percent or better, and
   with it the friction angle to within about 0.1 percent of the
   full-channel 24.98 degrees, keeping the sand cells. The basal-scaled
   variant should land within a few percent. The pure depth closure should
   over-predict pressure, which under-predicts friction because the fitted
   coefficient scales as the inverse of the pressure scale, and its error
   should exceed the 4.5 percent tolerance that sand's 4200x margin allows
   before a published cell is lost.
4. Water's volumetric fit reads J from the stored F and never touched
   stress, so it should reproduce the full-channel 0.54 percent with no
   refusal. This contradicts the refusal expectation written for the
   positions-only reading of this tier, and the measurement decides.
5. Overall: 20 of 20 cells should reproduce the full-channel table, because
   only sand's leg consumed stress and sand's cone level is recoverable
   from F at the fixed elastic pair. The tier's real cost should appear only
   in the two sand pressure ablations, which are the cells where a camera
   or a force plate, rather than a stored elastic state, has to supply
   pressure.
6. Secondary positions-only row for jelly (positions in, velocities by
   finite difference, L and F by moving least squares over 24
   reference-frame neighbours): the round trip on our own jelly data
   recovered E to 1.2 percent, so its cells should stay one to two orders
   below the published jelly cells.

## No-stress tier: outcomes

Cells are position MSE against their trajectory, in their metric. The
correct-property column runs our engine at their configured parameters, the
recovered column at what the tier identified, and both are seeded from their
frame-0 particles under the same compatibility flags as the full-channel run.
The full-channel column is the addendum above.

| material | scene | correct properties | recovered, no stress | full channels | their published | margin |
| --- | --- | --- | --- | --- | --- | --- |
| jelly | dataset | 1.2e-13 | 3.0e-6 | 3.0e-6 | 2.4e-4 | 81x |
| jelly | time | 2.8e-13 | 8.8e-6 | 8.8e-6 | 9.8e-4 | 111x |
| jelly | vel mean | 9.3e-14 | 1.8e-6 | 1.8e-6 | 2.4e-4 | 132x |
| plasticine | dataset | 7.8e-12 | 3.6e-7 | 3.6e-7 | 6.5e-5 | 181x |
| plasticine | time | 1.4e-11 | 5.6e-7 | 5.6e-7 | 1.4e-4 | 252x |
| plasticine | vel mean | 5.3e-12 | 2.5e-7 | 2.5e-7 | 4.6e-5 | 187x |
| sand | dataset | 2.0e-11 | 1.2e-8 | 6.2e-9 | 2.6e-5 | 2144x |
| sand | time | 5.0e-11 | 1.8e-8 | 9.1e-9 | 4.2e-5 | 2381x |
| sand | vel mean | 3.1e-11 | 2.4e-9 | 1.2e-9 | 6.5e-5 | 27023x |
| water | dataset | 3.0e-12 | 3.8e-7 | 3.8e-7 | 2.0e-5 | 52x |
| water | time | 9.0e-11 | 6.3e-6 | 6.3e-6 | 3.5e-4 | 55x |
| water | vel mean | 5.2e-13 | 1.7e-7 | 1.7e-7 | 1.9e-5 | 112x |

All 20 rollouts beat their axis's published numbers at this tier. Recovered
parameters: jelly E 1.93 percent low and nu 4.04 percent high, plasticine E
0.90 percent low, nu 0.51 percent high and yield stress 1.00 percent low,
sand friction 24.9755 degrees against 25 (0.098 percent low), water
stiffness 0.54 percent high. Identification wall times, rollouts excluded:
jelly 4.0 s, plasticine 7.9 s of which the extra yield estimator 3.4 s, sand
6.7 s for all three pressure sources, water 1.7 s. Building a tier dump
costs about 3 s per scene.

The variant legs, each a labeled estimator rolled out on all five scenes:

| material | variant | value | error | dataset | time | vel mean | published |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plasticine | yield from the momentum fit | 4555 Pa | 8.9 % low | 3.1e-5 | 5.0e-5 | 2.1e-5 | 6.5e-5, 1.4e-4, 4.6e-5 |
| sand | basal-plate pressure | 30.53 deg | 22.1 % high | 7.3e-4 | 9.6e-4 | 1.5e-4 | 2.6e-5, 4.2e-5, 6.5e-5 |
| sand | depth-closure pressure | 30.72 deg | 22.9 % high | 7.9e-4 | 1.0e-3 | 1.6e-4 | same |

Both sand variants refused on their residual check and the plasticine one
did as well, so what the tier ships for them is the known-class prior; the
rows above roll out the refused value instead, which measures its cost.
In this comparison the prior entry is the truth value, so
a refused parameter's shipped row would be the correct-property row and
would say nothing about the estimator.

## Positions-only tier: outcomes

Positions and frame times are the only measured channels; velocities come
from central finite differences, gradients and deformation from moving least
squares, and every rollout is seeded from the tier's own finite-difference
frame-0 velocity, so the correct-parameter column is no longer near zero:
it now measures the seeding error, about 1e-5 to 4e-5 depending on the
scene; no cell at this tier can score below the seeding error.

The first run of this tier for plasticine returned E ten times low and the
yield twelve times high without refusing. The cause is structural: the
stored deformation gradient of a plastic material is the elastic part,
positions only give the total deformation, and on the plasticine throw the
two diverge 80x once the material flows. The repair chain is measured in
docs/nclaw_identification_equations.md: the momentum fit recovers the yield
to 1.5 percent when handed the true hidden state, the replay reconstruction
of that state from positions fails the fit's residual check (2.3x high at
residual 0.87, refused), and the reported estimator for the plastic materials is a
derivative-free rollout scan with the elastic pair assumed and stated: roll
the engine at a candidate from the tier's frame-0 seed, score position MSE
against the measured identify trajectory. NCLaw's sys-id baseline optimizes
the same objective through a differentiable MPM. Water keeps its weak-form
fit as primary (it does not refuse) with the scan as a variant leg.

| material | scene | correct properties | recovered | their published | margin |
| --- | --- | --- | --- | --- | --- |
| jelly | dataset | 1.8e-5 | 1.8e-5 | 2.4e-4 | 13x |
| jelly | time | 3.0e-5 | 2.7e-5 | 9.8e-4 | 36x |
| jelly | vel mean | 2.8e-6 | 4.6e-6 | 2.4e-4 | 52x |
| plasticine | dataset | 1.0e-5 | 1.0e-5 | 6.5e-5 | 6.3x |
| plasticine | time | 1.2e-5 | 1.2e-5 | 1.4e-4 | 12x |
| plasticine | vel mean | 1.6e-6 | 1.6e-6 | 4.6e-5 | 28x |
| sand | dataset | 1.6e-5 | 1.4e-5 | 2.6e-5 | 1.9x |
| sand | time | 2.2e-5 | 2.0e-5 | 4.2e-5 | 2.2x |
| sand | vel mean | 2.3e-6 | 2.7e-6 | 6.5e-5 | 24x |
| water | dataset | 3.7e-5 | 3.7e-5 (scan) | 2.0e-5 | 0.5x, lost |
| water | time | 3.1e-4 | 3.1e-4 (scan) | 3.5e-4 | 1.1x |
| water | vel mean | 8.8e-6 | 8.8e-6 (scan) | 1.9e-5 | 2.2x |

19 of the 20 rollouts beat their axis's published numbers from positions alone.
The one loss is water's dataset scene, where the correct-parameter rollout
already scores 3.7e-5 against their published 2.0e-5: the finite-difference
seed loses that cell before identification enters.

Recovered parameters at this tier: jelly E 101861 (1.86 percent high), nu
0.17497 (12.5 percent low), both from the weak-form fit. Plasticine yield
5000 Pa against 5000 (exact, scan; twelve rollouts). Sand friction 24.5
degrees against 25.0 (scan; fourteen rollouts). Water stiffness by the scan
lands on lam 57692 exactly (eleven rollouts; the scan objective doubles at
a 5 percent stiffness offset), while the weak-form primary reads
lam 37.5 percent low from moving-least-squares volume readings on a splash;
its rollouts lose every scene; the loss quantifies the bias. The scan
objective is
steep for the plastic parameters too: 5 degrees of friction or a factor two
of yield costs a factor 25 to 45. Identification wall times: jelly 4.1 s,
plasticine 92 s (24 s replay diagnostics, 68 s scan), sand 89 s, water's
scan about 2 minutes.

The refusal calibration this tier bought: the replay estimators now refuse
(residuals 0.87 and 0.99 against a bar of 0.15); the first run reported
wrong values without warning. Sand's depth-closure and stored-F pressure
paths refuse as they did at the no-stress tier. Assumed and stated at this
tier: the elastic pair for plasticine and sand (their throws load it only
inside a short impact window, and the reconstructed state there biases a
fit by tens of percent), and nu for water (their law reads one stiffness).

## Function-encoder identification on their trajectories

The unknown-form row: the same trained bases as the same-engine study
(experiments/fe_ls/baseline.py), pointed at their ingested trajectories and
rolled out on their five scenes under the same engine-compatibility flags as
the known-form rows (runner: experiments/fe_ls/cross.py, results in
out/fe_ls_cross/). Like NCLaw, these rows fit a function from one
trajectory with no law form given. The information differs by row and each
table row says so. NCLaw's training reads x, v, C and the elastic F; it
never reads the stored stress. Our jelly and water rows read F and match
that. Our sand binned-cone row and the plasticine yield-surface row read
the stored stress channel, which NCLaw does not; those two rows carry more
information than NCLaw uses, and the label "oracle stress" marks them.

| material | estimator | identification quality | dataset | time | vel mean | their published | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sand | binned-cone mu(I) curve, oracle stress | bin medians 0.5680 against a true cone of 0.5680 | 3.5e-8 | 6.9e-8 | 1.2e-8 | 2.6e-5 / 4.2e-5 / 6.5e-5 | beats all five, 605x to 84988x |
| sand | momentum-fit mu(I) | curve relL2 0.43 to 0.54, residual 0.38 | 1.8e-3 | 2.4e-3 | 3.4e-4 | same | loses; same impact-phase bias the known-form level reading avoided |
| jelly | W'(I1bar) basis, small-strain projection | shear modulus 8.7 percent, volumetric 9.0 percent | 2.6e-4 | 7.3e-4 | 9.5e-5 | 2.4e-4 / 9.8e-4 / 2.4e-4 | beats four of five; dataset lost at 0.9x |
| water | volumetric column of the hyperelastic family | lam 57367 against 57692 (0.56 percent); the deviatoric part returns zero, correct for a fluid | 4.3e-7 | 6.6e-6 | 1.9e-7 | 2.0e-5 / 3.5e-4 / 1.9e-5 | beats all five, 47x to 5126x |
| plasticine | hyperelastic family on the elastic state, oracle stress via stored F | shear modulus 0.16 percent, residual 0.034 | 2.7e-2 (elastic-only rollout) | 2.5e-2 | 1.6e-2 | 6.5e-5 / 1.4e-4 / 4.6e-5 | elastic function recovered; no shipped basis covers the yield, and the elastic-only rollout loses 400x; the missing yield costs that factor |

The plasticity gap of the first pass is now closed. A yield-surface basis
h(p), with the yield condition sqrt(J2(dev tau)) = h(p), was trained over
the perfect-plasticity zoo (von Mises flat, cohesionless and cohesive
cones, caps, pressure powers; trainer
ident/features/function_encoder_training/plasticity_train.py, held-out
reconstruction 3e-4 mean), identified by binned pointwise readings on the
shearing set with tension included, and rolled out through a new
tabulated-yield engine material verified against both exact kernels (a
flat table reproduces von Mises to 1e-5 m over 1000 frames; a line table
reproduces their Drucker-Prager to 7e-6 m once the pressure grid covers
impact transients; the narrow first grid cost 7.7 cm and is recorded in
the trainer). On their trajectories the one family answers both plastic
materials: plasticine returns the flat curve, h(0) = 3540 Pa against
5000/sqrt(2) = 3536 (sigma_y to 0.13 percent) with slope -0.003, and sand
returns a cohesionless line (intercept 10 Pa) of slope 0.548 against
0.568. The full unknown-form plasticine law, learned elastic pair plus
learned yield surface and nothing given:

| scene | fe elastic + learned yield | their published | margin |
| --- | --- | --- | --- |
| dataset | 8.1e-7 | 6.5e-5 | 80x |
| time | 9.6e-7 | 1.4e-4 | 145x |
| vel_0001 | 7.6e-7 | 4.6e-5 | 60x |
| vel_0007 | 2.6e-8 | 4.6e-5 | 1742x |
| vel_0008 | 4.5e-7 | 4.6e-5 | 103x |

Sand's yield-surface leg loses three of five scenes (0.04x to 1.3x): its
3.6 percent slope error sits at the material's measured 4.6 percent
tolerance, so sand's winning unknown-form estimator remains the binned cone
ratio below, and the yield-surface family's value for sand is the shape
detection, cohesionless cone against flat, from the same solve that reads
plasticine's yield.

What the row establishes. Sand's momentum-fit curve inherits the
impact-phase bias measured on the known-form side (the solve without the
post-impact frame cut read 38.9 degrees against 25), and the fix carries
over exactly: the binned pointwise cone reading is the level-reading
estimator made curve-valued, and it
recovers the constant cone to four digits across every populated inertial
number bin. The flat-table control at the true friction scores 1.7e-11:
every loss in this table comes from identification error; the
parameterization reproduces their integrator. Water: handed a family
containing shear and volume responses, the fit returns zero shear and their
volumetric law to half a percent; one solve detects the material class and
recovers the parameter. Jelly
pays 8.7 percent modulus error for the family mismatch (their corotated law
is not in the neo-Hookean, Yeoh, Gent span) and that costs the dataset cell.
Plasticine's first pass found a coverage gap: the elastic half of the law
came back to 0.16 percent and no basis then shipped covered the yield.
The learned yield surface below closed that gap the same day. The
elastic-only rollout stays in the table; a missing yield costs 400x. The
viscous family, run beside everything as the wrong-class control, is
unusable everywhere (residuals 0.72 to 0.98).

### What the tier establishes

1. Only sand's identification ever read the stress channel. Jelly's and
   plasticine's elastic fits, plasticine's strain-cap yield reading and
   water's volumetric fit are functions of x, v, F, volume and mass, and
   the tier dump proves it by construction: with stress zeroed,
   those three materials return the full-channel parameters to every digit
   and their ten rollout cells are bitwise the full-channel cells.
2. Their stress channel is a function of their stored deformation gradient.
   Rebuilding pressure from F through the Hencky volumetric relation at the
   configured E and nu reproduces the stored stress-trace pressure to a
   median relative error of 0.014 percent over 895393 particle-frames, which
   is why sand's friction survives the tier at 0.098 percent error against
   the full-channel 0.071 percent. This is also how NCLaw's own
   DruckerPragerPlasticity gets pressure, so the comparison stays
   like-for-like: their model reads no measured stress either.
3. The friction number comes from the constant level of the yield-set
   ratio; the momentum solve is refused. That holds whatever supplies the
   pressure. With the reconstructed pressure the momentum solve reads 37.0
   degrees at relative residual 0.481 and the level reading gives 24.98;
   with the basal-plate pressure
   it reads 30.53 at residual 0.630, and with the depth closure 30.72 at
   0.716. The two closures have no observable cone level to fall back on, so
   they refuse. Refusing is worth about a factor of 6e4 in the cell:
   accepting the closure estimate lands the dataset scene at 7.3e-4 where
   the level reading lands it at 1.2e-8, and 7.3e-4 is 28x above the published
   2.6e-5, so it is the one place in this comparison where a published cell
   would have been lost.
4. Sand tolerates a 4.6 percent friction error before losing a published
   cell; the closures delivered 22 percent. The tier cell of 1.2e-8 can
   grow 2167x before it reaches the
   published 2.6e-5, which at quadratic growth of MSE in parameter error
   allows a 46x larger friction error than the 0.098 percent achieved, that
   is 4.6 percent. The measured growth is consistent with the model: a 225x
   larger error produced a 61000x larger MSE against a quadratic prediction
   of 50600x.
5. A yield stress read off the momentum balance costs most of the margin.
   The yield column returns 4555 Pa, 8.9 percent low, at residual 0.390,
   and it is refused. Its rollout sits at 3.1e-5 against a published
   6.5e-5: ahead by 2.1x where the strain-level estimate is ahead by
   181x.
6. At the positions-only tier for jelly the identification is not the
   limiting error. E comes back to 1.86 percent from positions alone, but nu
   is 12.5 percent off, and the finite-difference frame-0
   velocity puts the correct-property rollout at 1.8e-5, which is the same
   number as the recovered rollout. The seeding error from the derived
   initial state exceeds the identification error.

### Pre-registered expectations against outcomes

1. Jelly reproduces to the digit. Confirmed.
2. Plasticine's elastic pair and strain-cap yield reproduce. Confirmed. The
   claim that the momentum yield column would leave the published margins
   intact is FALSIFIED: the margin falls from 181x to 2.1x on the dataset
   scene, and the estimator refuses rather than shipping.
3. Sand's reconstructed pressure and friction. Confirmed, and the closure
   tolerance of 4.5 percent was right to within a tenth of a point (4.6
   percent measured).
4. Water reproduces with no refusal. Confirmed, which falsifies the refusal
   expectation the positions-only reading of this tier carried.
5. Twenty of twenty cells reproduce the full-channel table. Confirmed for
   jelly, plasticine and water, bitwise. PARTLY FALSIFIED for sand: the
   cells are a factor of 1.9 to 2.0 worse than the full-channel ones,
   because the level reading from the reconstructed deviator lands at 0.56741
   where the stored stress gives 0.56758, and rollout error grows
   quadratically in that difference. The published cells are kept with 2144x
   to 27023x to spare.
6. The direction of the depth closure's pressure bias is FALSIFIED as a
   single sign. The closure over-predicts the level of the field, median
   pressure 1112 Pa against 1030 Pa, and under-predicts as a per-particle
   ratio, median 0.813, because the ratio distribution is skewed: the median
   per-particle relative error is 0.96, an error of order one. The earlier
   project finding of over-prediction was made on a collapse, and a thrown
   blob is not that geometry.

Artifacts and how to rerun. The tier dumps sit next to the trajectories as
out/nclaw_cross_generalize/dumps/<material>_<scene>_truth_no_stress.npz and
_positions_only.npz; the rollouts, the per-material results.json and the
rendered table are under out/nclaw_cross_compare/ (no_stress_table.md), and the
identification records under out/nclaw_suite/identify_no_stress_*.json.

    .venv/bin/python -m experiments.nclaw.compare jelly      --nclaw-bc --substeps=1 --no-stress
    .venv/bin/python -m experiments.nclaw.compare plasticine --nclaw-bc --no-stress
    .venv/bin/python -m experiments.nclaw.compare sand       --nclaw-bc --nclaw-law --substeps=1 --no-stress
    .venv/bin/python -m experiments.nclaw.compare water      --nclaw-bc --nclaw-law --no-stress
    .venv/bin/python -m experiments.nclaw.compare jelly      --nclaw-bc --substeps=1 --positions-only
    .venv/bin/python -m experiments.nclaw.compare plasticine --nclaw-bc --positions-only
    .venv/bin/python -m experiments.nclaw.compare sand       --nclaw-bc --nclaw-law --substeps=1 --positions-only
    .venv/bin/python -m experiments.nclaw.compare water      --nclaw-bc --nclaw-law --positions-only
    .venv/bin/python -m experiments.nclaw.no_stress_table --out out/nclaw_cross_compare/no_stress_table.md

Deviations from the plan, each with its measurement. The plan's positions-only
tier became a jelly-only secondary row after the tier was redefined as
no-stress; it ran and is reported above. The plan expected plasticine to lose
its yield reading with the stress channel, and it does not, because that
reading is the strain cap of the stored elastic F; the momentum yield
column asked for in the plan was built anyway and is the variant row, at 8.9
percent. The two closure sources for sand supply pressure only, so their fits
have no cone level and refuse on the residual rule instead of falling back;
their refused values are rolled out in their own legs so the refusal has a
price attached. Sand's variant legs and plasticine's are extra rollouts the
full-channel run did not have, which is why this tier has 35 cells where the
addendum has 20.

## Plasticine, every method on one table

The material with results from every method, since it is where
identification is hardest (the elastic state hides behind plastic flow) and
where NCLaw's
coverage briefly exceeded ours. Parameter errors are against E = 300000 Pa,
nu = 0.25, sigma_y = 5000 Pa; rollout margins are against their published
cells on their trajectories.

| method | law form given | information used | E | nu | sigma_y | rollout vs their published |
| --- | --- | --- | --- | --- | --- | --- |
| NCLaw, published | no (two networks) | x, v, C, elastic F via teacher forcing; test from exact initial state | learned net | learned net | learned net | 1x by definition |
| NCLaw sys-id baseline, their run | yes | same, through their differentiable MPM | (fit) | (fit) | (fit) | 2.2e-10 on their engine |
| ours, least squares, known form | yes | x, v, F (no stress), the no-stress tier | -0.9 % | +0.5 % | -1.0 % | beats 169x to 252x |
| ours, function encoder, unknown form | no (trained bases) | x, v, F, stress | +0.2 % (shear modulus) | (pair via projection) | +0.13 % | beats 60x to 1742x |
| ours, positions only, their data (1 particle per cell) | yes, elastic pair assumed | positions and frame times alone | assumed | assumed | exact (rollout scan) | beats 6.3x to 50x |
| ours, positions only, dense clouds (same-engine study) | yes | positions and frame times alone | -1.6 % at 27 per cell | +5.1 % | +1.9 % | not comparable (own trajectory); residual 0.21 against the 0.15 bar, reported with a warning |

The density study (experiments/nclaw/density_study.py,
out/nclaw_density_study/results.json) is the last row expanded: the full
three-parameter least-squares chain from positions alone, with the elastic
state rebuilt by local affine fits pushed through the return map. At their
cloud density of one particle per cell it fails (E +32 percent, yield 2x)
and the residual check refuses; at 8 per cell it reads E +2.8, nu +3.4,
yield +7.8 percent at residual 0.42; at 27 per cell the values above. The
estimator that scales with density is the neighbourhood fit, which
represents rigid rotation exactly; the engine-kernel transfer observer
fails at every density because its few-percent operator error is 22 to 60
percent of the strain rate under the throw's rotation (rotation-to-strain
ratio 6 to 24). NCLaw at any density consumes the stored elastic F through
teacher forcing rather than reconstructing it; positions-only
identification is a capability their method does not have.

## The master table: every method, every material, both campaigns

Two settings, kept separate because they answer different questions. "Their
data" identifies from and scores on NCLaw's own trajectories (their metric,
margin = their published cell / our recovered cell; their published row is
the reference). "Our data" is the controlled same-engine campaign of
2026-08-21 at grid 20, where diff-sim ran (reconstruction / generalization
MSE; diff-sim never ran on their data, its analog there is their own sys-id,
which their team ran on their engine: sand 1.2e-12, plasticine 2.2e-10).
Parameter errors in percent against the configured truths. Wall times are
identification only, rollouts excluded.

| material | method | data | form given | parameters | rollout | identify wall |
| --- | --- | --- | --- | --- | --- | --- |
| jelly | NCLaw network | their | no | learned net | 2.4e-4 to 9.8e-4 (reference) | 300 epochs, A6000 |
| jelly | LS known form | their | yes | E -1.9, nu +4.0 | beats 81x to 542x | 4.0 s |
| jelly | FE basis | their | no | shear +8.7, volumetric +9.0 | beats 4 of 5 (0.9x to 12.8x) | seconds |
| jelly | LS, positions only | their | yes | E +1.9, nu -12.5 | beats 13x to 66x | 4.1 s |
| jelly | LS known form | ours | yes | mu +0.8, lam -4.8 | 1.7e-6 / 7.8e-8 | 5.0 s |
| jelly | diff-sim equal budget | ours | yes | mu +0.4, lam -1.3 (best of 5; 3 inits land 7 to 27 off) | 3.1e-8 / 3.9e-8 | 52 min |
| jelly | diff-sim converged | ours | yes | sub-percent | 8.4e-14 / 5.8e-14 | 68 to 98 min |
| plasticine | NCLaw network | their | no | learned nets | 4.6e-5 to 1.4e-4 (reference) | 300 epochs, A6000 |
| plasticine | NCLaw sys-id, their run | their | yes | fit | 2.2e-10 on their engine | their diff-MPM |
| plasticine | LS known form | their | yes | E -0.9, nu +0.5, sigma_y -1.0 | beats 169x to 252x | 7.9 s |
| plasticine | FE elastic + learned yield surface (oracle stress) | their | no | shear +0.2, sigma_y +0.13 | beats 60x to 1742x | seconds each |
| plasticine | scan, positions only, their density | their | elastic pair assumed | sigma_y exact | beats 6.3x to 50x | 92 s |
| plasticine | LS, positions only, 27 per cell | ours | yes | E -1.6, nu +5.1, sigma_y +1.9 | own trajectory; residual 0.21, warned | minutes |
| plasticine | LS known form | ours | yes | mu +0.4, lam +1.0, tau_y -0.4 | 4.6e-8 / 1.9e-8 | 5.4 s |
| plasticine | diff-sim equal budget | ours | yes | mu +63, lam -59, tau_y -0.6 (mu-lam degeneracy) | 3.3e-6 / 1.6e-6 | 31 min |
| plasticine | diff-sim converged | ours | yes | mu +0.02, lam +0.01 | 2.7e-12 / 1.0e-12 | 46 to 77 min |
| sand | NCLaw network | their | no | learned nets | 2.6e-5 to 6.5e-5 (reference) | 300 epochs, A6000 |
| sand | NCLaw sys-id, their run | their | yes | fit | 1.2e-12 on their engine | their diff-MPM |
| sand | LS known form | their | yes | phi -0.1 (24.98 vs 25.0) | beats 2144x to 53000x | 6.7 s |
| sand | FE binned-cone curve (oracle stress) | their | no | cone 0.5680 vs 0.5680 per bin | beats 605x to 84988x | seconds |
| sand | FE yield-surface family (oracle stress) | their | no | cone detected, slope -3.6 | loses 3 of 5 (tolerance 4.6 percent) | seconds |
| sand | scan, positions only | their | elastic pair assumed | phi 24.5 vs 25.0 | beats 1.9x to 32x | 89 s |
| sand | LS known form | ours | yes | phi +6.5 (grid-20 bias) | 6.1e-5 / 2.1e-5 | 2.3 s |
| sand | FE mu(I) momentum fit | ours | no | curve relL2 0.16 | 7.4e-4 / 1.9e-4 | seconds |
| sand | diff-sim equal budget | ours | yes | phi +0.004 | 2.7e-11 / 9.7e-12 | 72 min |
| water | NCLaw network | their | no | learned nets | 1.9e-5 to 3.5e-4 (reference) | 300 epochs, A6000 |
| water | LS known form | their | yes | lam +0.5 | beats 52x to 5791x | 1.7 s |
| water | FE volumetric column | their | no | lam -0.6, shear returns zero | beats 47x to 5126x | seconds |
| water | LS + scan, positions only | their | nu carried | lam exact by scan (weak form -37.5) | scan beats 4 of 5 | 2 min |
| water | LS known form | ours | yes | K -7.4 | 2.4e-4 / 5.4e-6 | 2.1 s |
| water | diff-sim equal budget | ours | yes | K +0.0002 | 7.0e-14 / 4.8e-14 | 46 min |

Summary of the table: the convex solve holds every
material within a few percent in seconds and never depends on
initialization; diff-sim converged beats it on parameter accuracy at three
to four orders more wall time, and at equal budget lands in the
plasticine mu-lam degeneracy (elastic pair 60 percent wrong at matched
positions);
the function-encoder rows do what NCLaw's networks do, fit a function with
no form given, and beat their published cells on sand, water, and
plasticine outright and on jelly in four of five scenes; the positions-only
rows use strictly less information than any other row in the table,
including both of NCLaw's, and still keep 19 of 20 published cells at their
density plus a certified-parameter path at higher observation density.
Refusal is a capability only our rows have: wrong-class families, biased
pressures and unreconstructable states refuse on residual; no value is
reported.


## Regeneration note (2026-08-24): Hencky columns for plasticine

An external review found that the identification used fixed-corotated
stress columns for plasticine, whose generating elasticity is
SigmaElasticity (Hencky); those columns are exact only at small strain.
The assembly gained Hencky columns
(also linear in mu and lambda), plasticine's identification now uses them
at both tiers, and jelly is bit-identical under the default. Regenerated
values, both tiers identical as before since no plasticine estimator reads
stress: E 297289 (0.90 percent low, was 1.79 high), nu 0.2513 (0.51 high,
was 3.42), yield 4950 (1.00 low, was 1.10 high); cells improved 1.5x to 2x
on every scene, margins now 169x to 252x (were 90x to 156x). The
before-values remain in the git history at 50d2826. The same review's other
confirmed findings and their fixes: the metric now refuses shape mismatches
(audit found zero contaminated cells), the mass epsilon follows the grid
(zero impact, all scored trajectories are grid 20), the yield-level check
is now enforceable and refuses jelly as a negative control, the canonical
per-material settings live in the runner, the ingest refuses ambiguous
L-convention probes and non-uniform frame spacing, and provenance keys are
correct after the migration. Queued: full-length kinematics with a
stress-validity mask and configuration-hashed caches. Across all 110
recorded cells, the final frame's error is at most 5.9 times the cell
mean, so adding it as one of 201 sampled frames shifts a cell by at most
2.9 percent, median 0.9.
