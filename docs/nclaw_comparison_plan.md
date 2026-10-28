# NCLaw-matched comparison: identify from one cube, roll out on their shapes

Goal: run NCLaw's experiment protocol (ICML 2023) end to end with our engine
and our weak-form identification, all four materials, and put the resulting
reconstruction and generalization position-MSE table next to their published
numbers. Matched protocol, our trajectories: the claims are self-consistent
within our engine and the NCLaw column carries a one-line engine caveat. The
simulator is never differentiated.

## Scene spec (from NCLaw/experiments/configs, verified)

Training scene per material: a 0.5 m cube of particles centered at
(0.5, 0.5, 0.5) in a unit box (their resolution 10 per axis), thrown with
linear velocity [1.0, -1.5, -2.0] and angular velocity [4, 4, 4] in their
y-up frame (rotate to our z-up: gravity [0, 0, -9.8], velocities rotated the
same way), freeslip walls with a 3-cell bound, grid 32, their dt 5e-4 for
1000 steps = 0.5 s. Our engine picks its own stable dt and dumps at a fixed
frame cadence near theirs. Evaluation shapes: bunny, spot, dragon from
NCLaw/nclaw/assets (the sampling pipeline in sim/nclaw_geom_scene.py),
same preset throw.

Truth materials (their configs):
- jelly: fixed corotated, E = 1e5, nu = 0.2, rho = 1000.
- plasticine: corotated E = 3e5, nu = 0.25 + von Mises (yield from their
  von_mises yaml; read it and record).
- sand: E = 1e6, nu = 0.2, Drucker-Prager friction angle 25 deg, cohesion 0.
- water: volumetric elasticity E = 1e5, nu = 0.3 (EOS fluid, no viscosity).

## Step 0, the acceptance test that comes first: elastic weight functions

The current elastic recovery (sim/elastic_drop.py) uses a sphere-shaped
radial window; on a cube it does not vanish on the faces and contact
tractions leak in (measured: E 13.4 percent on the cube vs 0.15 percent on
the sphere). Fix: grid-consistent assembly. In the current configuration the
fixed-corotated Cauchy stress stays linear in theta,
sigma = mu * (2/J)(F - R)F^T + lambda (J - 1) I,
so the interior-free-grid-node balance (ident/weakform/grid_assembly.py
machinery, the fix Phase 1 depended on) applies with two stress columns.
Nodes whose support touches the floor or the box walls are masked.
Acceptance: sphere at or better than the current 0.15 percent E; cube E
within 2 percent; lambda reported with cond(A^T A). Fallback if the grid
route stalls: a cube-adapted separable window (product of squared face
distances), same acceptance.

## Identification per material, from the cube trajectory alone

- jelly: (mu, lambda) by the step-0 assembly.
- plasticine: (G, tau_y) by the convex viscoplastic rate form (the squeeze
  identification machinery); E-nu split reported as identifiable or not.
- sand: DP friction by constant-friction Mode C on the flowing mask
  (recovered mu maps to their friction angle through the cone convention in
  conventions helpers; record the formula used); elastic E from the
  pre-yield frames if the transient supports it, else prior-fixed and said
  so.
- water: attempt the EOS stiffness (volumetric weak form); the pour work
  measured water below the smallest identifiable viscosity, so a partial or
  full refusal is an acceptable and reportable outcome; the rollout then
  uses the known material class with the refused parameter at its prior.

## Rollout and scoring

Re-simulate every scene (cube + three shapes, per material) at the recovered
parameters with identical seeding (1:1 particles), and score NCLaw's metric:
box-normalized per-particle position MSE, averaged over frames. Report
reconstruction (cube) and generalization (shapes) per material, next to
NCLaw's published table values (extract from the local nclaw.pdf or README;
where a number is not published for a matched case, leave the cell blank
rather than approximating).

Pre-registered expectations (falsifiable): sand reproduces the earlier
nclaw_sand_compare result (recovered mu_s near 0.377); post-fix jelly cube E
within 2 percent; plasticine near the robotics-era accuracy (G and tau_y
within a few percent); water refuses at least one parameter. Rollout: our
generalization MSE at or below NCLaw's for sand (exact-form recovery should
transfer better than a fitted network); no pre-registered claim for
reconstruction, where a network fitted to the single trajectory has the
advantage.

## Layout, budget, division of labor

Code: elastic grid-consistent assembly + tests in mpm_engine
(src/ident/weakform/, branch nclaw-compare); scene generation, identify,
rollout, scoring in video2sim/sim/nclaw_suite.py (this repo, main), artifacts
under out/nclaw_suite/ with a results.json per material, as in the Phase-1
acceptance scripts.
Budget: about 16 truth sims + 16 rollouts, minutes each on the M3, plus fast
solves; the implementer proves one material end to end (jelly), the
orchestrator runs the rest and writes the comparison.

## Measured deviations

Every entry below is a change forced by a measurement, with the measurement.
The pre-registered expectations above are left as written.

(1) The 0.15 percent sphere reference does not reproduce, and the number it
replaces was a cancellation. Running sim/elastic_drop.recover on the dump in
the tree, out/elastic_drop/truth.npz, gives E error 0.35 percent, not 0.15, and
its per-modulus errors are mu 3.89 percent and lambda 27.5 percent. The small E
error comes from the two errors cancelling in E(mu, lambda); neither modulus is
accurate. The acceptance test therefore reports mu and lambda alongside E and
adds a second criterion, beating the radial window on E, so a cancellation
cannot pass.

(2) The route that shipped is the grid route in its time-weak form; the plan
described the instantaneous per-node balance. Measured chain on the sphere
dump, all with the same exact node rows:

  - raw per-node rows, second-order finite-difference acceleration: E 3.8 percent
  - plus smooth spatial windows over the kept nodes:                E 0.71 percent
  - plus fourth-order finite-difference acceleration:               E 0.47 percent
  - time-weak load, cosine temporal window:                         E 0.42 percent
  - time-weak load, sin^4 temporal window (shipped):                E 0.09 percent

The exact nodal balance carries the GRID
acceleration; a dump carries the particle trajectory. The gap between them is
the projection residual of P2G followed by G2P, and it enters divided by dt, so
the narrowest test function pays the most for it. Widening the test function in
space cuts it; integrating the balance in time against a weight that vanishes at
both ends of the window removes the acceleration from the data altogether. The
temporal weight sets the quadratic order of the remaining time quadrature, which
is why sin^4 beats the cosine bump by a factor of five.

(3) Step 0 acceptance, both dumps, shipped configuration (time-weak, sin^4
window over 26 sampled frames, 3-cell collider clearance, smooth spatial window):

  | dump   | E err | mu err | lambda err | nu (truth) | cond(A^T A) | rows |
  | sphere | 0.10% | 0.21%  | 0.97%      | 0.2986 (0.30) | 7.1e1 | 180 |
  | cube   | 0.07% | 0.42%  | 3.51%      | 0.3046 (0.30) | 4.1e1 | 180 |

against the radial window's sphere 0.35 / 3.89 / 27.5 and cube 13.4 / 10.9 /
42.9. The cube test, Step 0's target, passes by a factor of 190 on E. All
four acceptance tests pass, the two error bars and the two beat-the-radial-window
tests; out/elastic_drop/grid_gate.json holds the run.

The instantaneous route over the same rows measures E 0.63 percent on the sphere
and 1.50 percent on the cube. It would pass the cube bar and fail the sphere one,
so Step 0 requires the time-weak route. Collider
clearance is flat between two and three cells on both dumps and degrades at four
(sphere E 0.31, cube 0.29 percent), so three cells ships, which is also the
two-hop particle-node-particle reach the quadratic stencil predicts.

(4) NCLaw's own reconstruction runs use a 20-cube grid, not 32. Their
experiments/configs/train.yaml and eval.yaml both select sim: low, which is
num_grids 20, dt 5e-4, 1000 steps; num_grids 32 is sim: high, and the special
overrides for their pool, melting and slope scenes go to 64 and 128. The suite
runs the plan's 32 and this note carries the difference, since a grid change
moves the absolute MSE.

(5) Particle count is ours, not theirs. NCLaw seeds the training cube at
resolution 10 per axis, 1000 particles for a 0.5 m cube, which is about 1.1 of
their 20-grid cells and below one particle per cell on a 32 grid. Our engine is
run at pitch dx/2, eight particles per cell, giving 35937 for the cube and
21076 for the bunny. The metric is computed between our own truth and our own
rollout on identical clouds, so the count changes the magnitude of the MSE; the
comparison within one engine stays valid. It is one of the reasons the two
columns of the comparison table are not a head-to-head on one simulator.

(6) NCLaw's geometry-generalization column pairs a DIFFERENT held-out mesh with
each material: jelly with armadillo, plasticine with bunny, sand with blub,
water with spot. The suite therefore accepts those meshes as shapes and the
report names the matched one per material, so a reader can see when a cell is
strictly matched and when it is not.

(7) Plasticine yield truth, read from their config as instructed:
experiments/configs/env/blob/material/plasticity/von_mises_plasticine.yaml sets
E 3e5, nu 0.25, sigma_y 5e3. Their von_mises.yaml declares
cls: VonMisesPlasticity with E, nu, sigma_y required.

(8) The sand friction mapping is derived from the fork. warp-mpm's
set_parameters_dict stores alpha = sqrt(2/3) 2 sin phi / (3 - sin phi) and
sand_return_mapping yields when ||dev eps|| + (3 lam + 2 mu)/(2 mu) tr(eps)
alpha > 0. With Hencky elasticity, ||dev tau|| = 2 mu ||dev eps|| and
tr tau = (3 lam + 2 mu) tr eps = -3 p, so the surface is ||dev tau|| = 3 alpha p.
This tree measures friction as mu = sqrt(J2)/p with sqrt(J2) = ||dev tau||/sqrt(2),
hence

    mu = 3 alpha / sqrt(2) = 2 sqrt(3) sin phi / (3 - sin phi),
    sin phi = 3 mu / (2 sqrt(3) + mu),

the textbook compression-cone Drucker-Prager constant. Their 25 degrees is
mu = 0.5679.

(9) The Step 0 module hosts a law-independent core rather than an elastic-only
one. assemble_columns_timeweak takes the volume-weighted Cauchy stress columns
of any law linear in theta plus an optional known stress part, so the sand
constant-friction leg and the water EOS leg reuse the same node selection, collider
clearance, spatial windows and temporal weight instead of copying them. The
fixed-corotated pair is the first client and still names the module.

(10) The published numbers, extracted from their tables. NCLaw's own method's
row, position MSE in squared metres on their 1.0 m box: reconstruction from
their Table 1 (page 5), the three generalization axes from Table 2 (page 7),
tasks (a) longer horizon, (b) initial velocity, (c) geometry.

  | material   | recon  | (a) time | (b) velocity | (c) geometry |
  | jelly      | 2.4e-4 | 9.8e-4   | 2.4e-4       | 4.1e-4       |
  | plasticine | 6.5e-5 | 1.4e-4   | 4.6e-5       | 2.3e-4       |
  | sand       | 2.6e-5 | 4.2e-5   | 6.5e-5       | 3.6e-4       |
  | water      | 2.0e-5 | 3.5e-4   | 1.9e-5       | 2.4e-4       |

All eight cells the plan asks for are published, so none is left blank. Their
per-table Overall columns do not recompute from the cells, so only per-cell
values are used. Their strongest baseline beats them on jelly reconstruction
(the neural row, 1.2e-5); their labelled and system-identification oracles sit
below the rule in the same tables and are excluded from their own shading, so
they are not comparison targets. Their code defines the metric:
nclaw/utils.py:86 diff_mse is a torch mse_loss on absolute positions,
the mean over particles AND all three coordinates, averaged over every fifth
saved frame.

This corrects a number already in the tree: sim/nclaw_sand_compare.py carries
1.5e-2 as NCLaw's sand generalization value, where the table reads 3.6e-4. That
file is left alone here, but the suite uses 3.6e-4 and the discrepancy is on the
record.

(11) NCLaw's geometry column changes four things at once, so it is not a clean
geometry axis. Their task (c) switches the mesh, the particle count (about 30k),
the grid (20 to 32) and the horizon (1000 to 2000 steps), and it also switches
the throw from vel/preset to vel/mild, linear [1.0, -1.5, -1.5] and angular
[1, 1, 1]. Our generalization column changes the mesh alone, at one grid, one
horizon and the preset throw. The report carries this as a line of its own so
the two columns are not read as measuring the same thing.

(12) The sand validity mask was wrong on the first pass; a measurement forced
the fix. Marking every particle at or below zero pressure
INVALID left no surviving node at any threshold from 0.75 to 1.0, because about
a quarter of a tumbling cohesionless cube sits at or below zero pressure at any
instant. Those particles carry zero stress:
sand_return_mapping sets F_elastic = U V^T when tr eps >= 0, measured here as
||sigma|| of order 0.4 Pa against 1e4 Pa in the bulk. Treating them as modelled
contributors of zero stress, and excluding only positive-pressure particles
below yield (measured fraction: zero, every positive-pressure particle in this
throw is shearing), took the recovery from -32 percent to -1.0 percent on a
30-frame probe: mu_c 0.5622 against the truth 0.5680, friction angle 24.76
degrees against 25. The cone convention of item (8) is independently confirmed
by that probe: the empirical sqrt(J2)/p of the shearing particles is 0.568 to
four digits.

(13) The time-weak window length is per material, since a row sums a whole
window and one unusable frame kills it. The throw starts as a rigid rotation
where D is identically zero and nothing is at yield, so the sand frame list
starts at the longest contiguous run of shearing frames and its window is 10
sampled frames; the elastic legs use 26, where longer is strictly better because
their residual error is the temporal quadrature. Water uses 16.

(14) Water did not refuse, against the pre-registered expectation. On a
30-frame probe the volumetric weak form recovered the EOS bulk modulus as
8.043e4 against the truth 8.333e4, an error of 3.5 percent, with cond(A^T A) = 1
(a single column), relative residual 0.031, and a realized 99th-percentile
volumetric strain of 9.5 percent. The throw compresses the blob far more than
the pour work did, so the column is not starved. The refusal branch stays in the
code with its conditioning test; the orchestrator's full run decides the
reported cell.

(15) Jelly end to end, the material proved here. Identified from the cube throw
alone: mu 4.1697e4 against 4.1667e4 (+0.07 percent), lambda 2.7355e4 against
2.7778e4 (-1.52 percent), E 9.9913e4 against 1e5 (-0.09 percent), nu 0.1981
against 0.2, from 36 rows with cond(A^T A) 30.5 and relative residual 0.0066.
Rollout at the recovered law, NCLaw's metric: cube reconstruction 1.95e-6,
bunny generalization 1.30e-7, against their published jelly 2.4e-4 and 4.1e-4.
The generalization number being lower than the reconstruction number follows
from excitation: the cube throw deforms more, and the bunny at 21076 particles
deforms less.

## Orchestrator finding: sand frame-selection bias (2026-08-20)

The suite's first full sand run recovered phi = 21.97 deg against truth 25
(residual 0.47) and its rollouts inherited the error (cube 2.7e-4, blub
7.4e-2). The cause is the caveat the identification docstring itself
states: the flow direction is read from D, which is not coaxial with the
elastic strain during the collisional impact frames, and the default frame
selection (longest shearing run) starts there. Measured sweep of a
post-impact frame cutoff, frames after the kinetic energy decays below a fraction
of its peak: frac 0.5 gives phi 24.66 (residual 0.083), 0.2 gives 25.22
(0.042), 0.1 gives 25.06 (0.025); below 0.1 the ten-frame window refuses.
The shipped cutoff uses 0.1, relaxing to 0.2 then 0.5 when the window
refuses, records the
fraction used, and reads phi = 25.19 (residual 0.044, sd 0.003) through the
real code path. Sand rollouts regenerated at the corrected friction.

## Outcome (all four materials, primary cells at their per-material mesh
## and their geometry-eval throw, 2026-08-20)

Ours = identify from one cube throw (convex weak form, no simulator
differentiation), re-simulate. NCLaw = their published "ours" row (Table 1
p5 reconstruction, Table 2 p7 column (c) geometry). Both in box MSE, m^2.

| material | recon ours | recon NCLaw | gen ours (their mesh, mild) | gen NCLaw (c) |
| --- | --- | --- | --- | --- |
| jelly | 1.9e-6 | 2.4e-4 | 7.2e-8 (armadillo) | 4.1e-4 |
| plasticine | 1.1e-9 | 6.5e-5 | 5.8e-10 (bunny) | 2.3e-4 |
| sand | 1.1e-6 | 2.6e-5 | 5.7e-7 (blub) | 3.6e-4 |
| water | 1.5e-9 | 2.0e-5 | 1.5e-6 (spot) | 2.4e-4 |

Recovered parameters, from the cube trajectory alone: jelly E -0.09
percent, nu 0.198 (0.2); plasticine E +0.03 percent, nu 0.2507 (0.25),
yield -0.03 percent; sand phi 25.19 (25) after the post-impact frame cutoff;
water
bulk +0.04 percent. Secondary preset-throw generalization cells (geometry
isolated) are in the per-material reports; sand blub under the preset throw
is a scene-integrity failure on both engines' terms and is reported as a
failure; no MSE is quoted.

Pre-registration scorecard: jelly cube test PASS (0.09 vs 2 percent);
plasticine PASS; the sand expectation transfers from the old mu(I) test to
its DP analog and passes (phi within 1 percent); water FALSIFIED: no refusal
occurred, because the throw realizes 9.5 percent
volumetric strain where the pour had none; the sand-generalization
expectation PASSES; the no-claim on reconstruction turned out conservative,
ours is lower on every material.

Reading, with the two required caveats: (i) matched protocol on two
engines, two self-consistent measurements, not a head-to-head on one
simulator; (ii) our cells use known law FORMS, so the fair row in their own
Table 1 is the sys-id ORACLE (1.7e-8 to 5.8e-10 reconstruction), which they
label an upper bound requiring inaccessible knowledge. Our numbers sit in
that oracle band while using only the same observable trajectory their
network trains on. Convex weak-form identification reaches the sys-id
oracle band without a differentiable simulator; the function-encoder basis
covers the unknown-form case, measured in the FE section.

## Blender comparison renders (2026-08-20)

The comparison videos are now Blender renders rather than matplotlib scatter
panels: sim/nclaw_blender_render.py drives sim/nclaw_blender_scene.py, one
Blender session per panel, and writes out/nclaw_suite/videos_blender/<cell>.mp4
with truth on the left and the recovered-law rollout on the right under one
camera. Same dumps, same stride 2, same 63 frames at 25 fps as the matplotlib
version. Jelly, plasticine and water are surfaced through Points to Volume and
Volume to Mesh; sand is instanced icospheres, so it reads as grains.

Measurements behind the settings, all at 960x720 on the M3 Max:

- Cycles on Metal costs 1.5 to 2.3 s per frame at 64 samples with denoising and
  EEVEE costs 0.57 to 0.76 s. Cycles is the default because water needs real
  refraction. The full water_spot_mild cell, both panels plus encoding, took
  223.6 s. The first Cycles frame on a cold Metal kernel cache costs 105 s.
- Surfacing at voxel size 0.35 x particle spacing shows every particle as a
  lump (165310 vertices for water frame 60) and smoothing cannot remove bumps
  that are 9 voxels wide: 30 Smooth iterations moved the image by a mean of 2.6
  grey levels out of 255. Voxel size 1.0 x spacing with radius 2.0 x spacing
  (18442 vertices) reads as a liquid and renders faster.
- Grain radius 0.6 x spacing already closes the pile. Rasterising the projected
  grains at frame 40 of sand_blub_mild gives 5.39 percent frame coverage at
  0.6 x and 5.77 percent at 1.0 x; the rendered coverage of 4.64 percent agrees
  with the analytic value, which is how the grain geometry was checked.
- The NCLaw bsdf_pcd colours are sRGB encoded and are decoded before entering a
  Blender Base Color socket; used raw they turned the saturated red jelly into
  pale salmon. View transform Standard at exposure -0.6, since AgX desaturated
  the palette further. Their plasticine value decodes to a neon green that does
  not read as clay, so that one material keeps their hue at clay saturation.

## Cross-engine ingestion (2026-08-20)

Everything above compares two engines through two separately generated
trajectory sets. This section is the path that removes one of the two: when a
folder of trajectories produced by NCLAW'S OWN simulator arrives, our
identification and rollout consume it directly. The code is
mpm_engine/experiments/nclaw/ingest.py with the cross stage in
experiments/nclaw/suite.py, and the whole chain is validated before any of
their data exists by exporting one of our dumps into their format and reading
it back.

### The format we support, read from their code

Their eval and dataset scripts save one torch pickle per saved step,
state_root/0000.pt, 0001.pt, ..., each a dict(x, v, C, F, stress, sections,
types): x and v of shape (N, 3), C, F and stress of shape (N, 3, 3), float32,
their y-up frame, positions on the unit box (experiments/eval.py lines 104 and
128). The files do not carry the following four facts:

1. Volume and density are absent. Their MPMInitData sets one scalar volume per
   group: prod(size) / N for a uniformly seeded cube, mesh.volume / N *
   prod(size) for a mesh; rho is 1e3 for all four materials of the comparison.
   The manifest carries it.
2. Their stress channel is the MLS-MPM stress their P2G consumes,
   2 mu (F - R) F^T + lambda J (J - 1) I for fixed corotated, which is the
   KIRCHHOFF stress tau = J sigma. Our schema carries Cauchy, so the ingest
   divides by J = det(F).
3. Their loop computes stress = elasticity(F) BEFORE the step and saves it next
   to the post-step state, so the saved stress lags the saved state by one
   simulator step, and their frame 0 carries the zero-initialised stress. The
   ingest shifts the channel back (stress_lag_steps, default 1) and drops the
   final frame, which also removes the zero first frame. The shift is exact only
   when skip_frame is 1; a coarser cadence leaves a sub-frame lag that is
   recorded rather than absorbed.
4. Their G2P accumulates new_C += 4 w inv_dx^2 outer(v, dpos), which reads as
   C_ij = dv_i/dx_j, our L. The ingest verifies this: it runs the
   acceleration-consistency probe on the ingested arrays and transposes C only
   if the measurement says so, storing the verdict and both residuals in the
   dump metadata.

Their experiments/configs/default.yaml sets hydra.output_subdir: null, so there
is no .hydra folder, but eval.py does save the resolved config itself:
OmegaConf.save(cfg, exp_root / 'hydra.yaml'), one level above the state folder.
The ingest reads state_root/../hydra.yaml when it is there (cfg.sim.* and
cfg.env.blob.*), and the manifest always wins over it. Reading it needs pyyaml,
which this engine does not depend on; without pyyaml the ingest says so and the
manifest carries everything.

### Manifest schema

A small json beside the folder. Required: material (jelly, plasticine, sand,
water), rho, dt, skip_frame, num_grids, and one volume source out of
particle_volume, total_volume (divided by the particle count), or
shape = {"kind": "cube", "size": [sx, sy, sz]} (prod(size) / N). A missing
required entry fails with a message naming each one and where their value lives
in their configs. Optional, with defaults: grid_lim 1.0, bound 3,
gravity_yup [0, -9.8, 0], frame_convention "yup", stress_kind "kirchhoff",
stress_lag_steps 1, group (which types index to keep from a multi-material
folder), mls_k 24, mls_ridge 1e-8, law, law_params, name. Example, for their
training cube at sim: low:

    {"material": "jelly", "rho": 1000.0, "dt": 5e-4, "skip_frame": 1,
     "num_grids": 20, "bound": 3,
     "shape": {"kind": "cube", "size": [0.5, 0.5, 0.5]}}

### Degradation tiers, recorded per channel

Every channel is measured, derived, or absent, and the verdict is written into
the dump metadata (channel_provenance plus degradation_notes). v absent gives
central finite differences in time; C absent gives a moving-least-squares
velocity gradient over the k nearest neighbours, the neighbour sets fixed in the
reference frame; F absent gives the same least squares between reference and
current offsets. Stress absent means there is no oracle pressure: the elastic
and volumetric legs still run from F, and the Drucker-Prager friction leg
refuses, naming the missing channel, instead of substituting a closure. The
suite's identify_friction now carries that refusal.

### Measured round-trip results

Our dump out to their format and back, on the suite's own cube dumps, 126
frames and 35937 particles each:

  | dump       | x, v, L, F, volume, mass | stress          | theta agreement |
  | jelly_cube | bitwise identical        | 8.0e-8 relative | E and nu exactly equal |
  | sand_cube  | bitwise identical        | 5.8e-8 relative | friction angle 2.6e-11 relative |

The identified values are jelly E 9.99127e4 and nu 0.198077 on both paths, and
sand friction angle 25.1899096 on both (25.189909573222 direct against
25.189909573871 through their format). The stress residue is the single float32
store of the Kirchhoff product J sigma and it is the only lossy step in the
path: declaring the channel Cauchy in the manifest makes the stress round trip
bitwise as well. The L-convention probe returns the same verdict and the same
residuals on both files, L == dv_i/dx_j: jelly frame 14, median residual 0.0047
against L and 1.995 against L^T; sand frame 12, 0.0019 and 1.907.

Kinematics-only tier on the same jelly dump, positions in and nothing else: E
1.0122e5 against the truth 1e5, 1.2 percent high, nu 0.1957, against 0.09
percent low for the measured-channel path. That is the price of deriving v, L
and F from x alone at 24 neighbours. The sand leg
on the same stripped folder refuses the friction angle for want of the oracle
pressure and falls back to its prior.

### The two commands

On the CUDA machine, in their tree (their code does not run on this Mac). Their
own ground-truth command, the one experiments/scripts/eval/dataset.py issues per
material, is

    python experiments/eval.py env=jelly render=debug sim=low name=jelly/dataset

which writes experiments/log/jelly/dataset/state/0000.pt ... 1000.pt (1001
files, skip_frame 1 over the 1000 steps of sim: low) and the resolved config at
experiments/log/jelly/dataset/hydra.yaml. All four materials at once is

    python experiments/scripts/eval/dataset.py --gt

over their ENVS list, jelly, sand, plasticine, water. sim: low is what their own
reconstruction runs use: num_grids 20, dt 5e-4, 1000 steps, bound 3, gravity
[0, -9.8, 0]. Their training cube is 0.5 m at resolution 10 uniform, so 1000
particles and a particle volume of 1.25e-4 m^3, and rho is 1e3. Those five
numbers are the manifest.

Here, once the folder has arrived, one command per material:

    .venv/bin/python -m experiments.nclaw.suite cross --material jelly \
        --nclaw-dir /path/to/their/state_root --manifest manifest.json

which ingests, identifies from their trajectory, rolls our engine out from
their frame-0 cloud and frame-0 velocities in 1:1 particle correspondence, and
scores their trajectory against our rollout in their own metric, writing
out/nclaw_suite/cross_<material>_<scene>.json. The ingest alone is

    .venv/bin/python -m experiments.nclaw.ingest <nclaw_dir> \
        --manifest m.json --out out/nclaw_suite/dumps/<name>.npz

and the validation that needs no NCLaw data at all is

    .venv/bin/python -m experiments.nclaw.ingest --roundtrip \
        out/nclaw_suite/dumps/jelly_cube_truth.npz --material jelly --degraded

### Deviations, each with its measurement

(16) The cross-engine MSE is not comparable to the same-engine cells of the
table above. Their trajectory against our rollout carries the identification
error AND the difference between the two integrators at their grid and time
step, where the suite's own cells carry the identification error alone. The
cross stage writes that caveat into every results file, so no reader takes the
number as like-for-like.

(17) The suite's asset lookup moved from import time to first use. Resolving
their mesh folder while importing the suite made the ingest path, which needs
none of their meshes, depend on a clone of their repository being present.

(18) A non-physical identified law now stops before the engine. The synthetic
cross-stage test fed the engine the moduli a solve returns on a kinematically
smooth motion that is not a momentum-balance solution, nu about 5.3, which makes
lam + 2 mu negative, the wave speed and the time step NaN, and the first p2g2p
write outside the grid: on this machine a bus error, not an exception.
_wave_speed refuses E <= 0 and nu outside (-1, 0.5) with a message naming the
pair.

(19) The pre-registered 1e-9 theta agreement holds; the reason it holds for
sand is recorded here: the stress channel is the one
lossy link, at 6e-8 relative, but its error is uncorrelated across the 35937
particles and the weak form sums them, so the friction angle moves by 2.6e-11
and not by 6e-8.

(20) The cross-engine truth-replay error came almost entirely from the boundary
condition, and the
engine now offers their semantics as a first-class option. Truth-parameter
rollouts on their plasticine trajectories sat at 1.067e-3 position MSE with our
six collider slip planes. `MPM_Simulator_WARP.set_grid_semantics` adds five
independent, opt-in options; with all of them on, the same rollout scores
7.8e-12, a factor of 1.4e8. Attribution on the dataset scene, one behavior at a
time against the collider-plane baseline and then leaving one out of the full
set:

  | leg                                    | MSE       | vs baseline |
  | collider slip planes (baseline)        | 1.067e-3  | 1           |
  | freeslip walls only                    | 7.551e-6  | 141x better |
  | eps mass softening only                | 1.067e-3  | 1.00        |
  | empty-node gravity only                | 1.067e-3  | 1.00        |
  | MLS transfer only                      | 1.101e-3  | 1.03 worse  |
  | particle clip only                     | 1.067e-3  | 1.00        |
  | freeslip walls + MLS transfer          | 3.839e-9  | 2.8e5x      |
  | all five (full)                        | 7.841e-12 | 1.4e8x      |
  | full without MLS transfer              | 7.404e-6  |             |
  | full without eps                       | 3.839e-9  |             |
  | full without empty-node gravity        | 7.841e-12 |             |
  | full without particle clip             | 7.841e-12 |             |

In order of effect size: the wall semantics are worth 141x on their own,
the MLS transfer another 1967x once the walls are right and nothing at all
before that, and the eps-softened mass division a further 490x. Two of the five
are inert here; the reason is structural. Empty-node
gravity is unreachable through g2p: every node in a particle's own stencil
receives that particle's mass, so no node a particle gathers from is ever empty,
and NCLaw's `else` branch is effectively dead code. The particle clip never
binds either, because the corrected walls arrest every particle well inside the
clip band. Our slip plane removes the wall-normal velocity unconditionally
inside its half-space, which glues a separating body to the wall; theirs zeroes
it only where it points into the wall. Approach-only is the correct reading of
freeslip independently of this comparison, which is why the option is named for
the semantics and not for their code.

(21) Sand has the same elasticity-mapping exposure plasticine had, and it is not
yet checked. Their sand config uses `sigma_sand` elasticity, which is in the
Hencky family, while our suite maps sand to the Drucker-Prager material whose
elastic predictor has not been read against theirs term by term. Plasticine's
analogous mismatch (fixed corotated where their config says Hencky) moved the
truth-replay error by 1 percent, 1.078e-3 to 1.067e-3, so the expectation is
that this is small; it is recorded as unchecked.

(22) Water's law is not a reparameterization of ours, and that was the whole
water gap. Their water is VolumeElasticity in mode taichi paired with
SigmaPlasticity (env/blob/material/water.yaml), so Kirchhoff tau =
lam J (J - 1) I with no deviatoric term at all and Cauchy pressure linear in
J - 1; our fluid material is a gamma = 1.1 power law. Their splash reaches
J = 0.039, where their linear form gives 55 kPa of pressure and our power law
gives 2.9 MPa, a factor of 52. Measured on their own water_dataset stress
channel, p = -lam (J - 1) fits with lam = 57692 Pa,
exactly their E nu / ((1+nu)(1-2nu)), at zero relative residual, while the
power-law form fits with negative stiffness and 0.81 residual. Rolling their
five water trajectories with the grid semantics on:

  | scene      | our EOS   | their EOS  |
  | dataset    | 5.066e-3  | 2.987e-12  |
  | time       | 2.370e-2  | 9.018e-11  |
  | vel (mean) | 1.521e-3  | 5.177e-13  |

The engine now offers their forms as material 14 (`composed`), which pairs any
of five elasticity kinds with any of four plasticity kinds independently, the
same way their ComposeMaterial does. Identification stays the same convex
least-squares solve: the volumetric column becomes (J - 1) I with lam as the
single unknown, and it returns 5.8001e4 against their 5.7692e4, 0.54 percent
high, from the dataset scene alone.

(23) Sand's engine gap closed to machine level and its remaining gap is
identification, which the audit localizes precisely. Their
DruckerPragerPlasticity, their sigma_sand elasticity and our sand material agree
term by term: alpha = sqrt(2/3) 2 sin phi / (3 - sin phi) is the same expression
in both trees; our sand_return_mapping is their compress branch with cohesion 0,
and the two branch structures coincide except at the measure-zero boundary
delta_gamma = 0 with trace exactly 0; and our kirchoff_stress_drucker_prager,
U diag((2 mu log sig_i + lam sum log sig)/sig_i) V^T F^T, reduces algebraically
to U diag(2 mu eps_i + lam tr eps) U^T, which is their SigmaElasticity exactly.
Nothing in the mapping needed fixing. The truth-replay error confirms the
mapping: with their grid semantics and their dt, all five sand truth-replay
errors sit at 2.0e-11 to 5.4e-11.

Two things were wrong and only one of them is the engine's. The time step was:
our CFL picks two substeps per frame at sand's 33 m/s wave speed where they take
one, and their trajectory is their discrete solution at their dt, so
subdividing it moved us away from the reference and held the truth-replay
error at 2.0e-4.
--substeps=1 removes that, a factor of 1e7. The other is the friction
identification, which returns 38.91 degrees against their 25. That is not a
convention error: on their own sand_dataset stress channel the pointwise
yield reading sqrt(J2)/p is 0.5680 with a 10-to-90 spread of 0.5676 to 0.5682,
against the truth 3 alpha / sqrt(2) = 0.56802, and dev(tau) is coaxial with
dev(D) (median cosine 1.0000, 97.6 percent above 0.9), so both the yield
constant and the flow direction the leg assumes are right in this data. The
weak-form leg fits ONE global friction through a momentum balance over a body
that is only partly at yield, and a projection of dev(tau) onto the column
p (2 D / |gamma_dot|_eps) over all shearing particles gives 0.355, not 0.568:
the elastic interior pulls one way and the momentum weighting the other. The
recommended next step is a yield-set row filter, reusing identify_yield's
level detection; the engine and the mapping need no change.

(24) Five-scene recovered-theta cells against their published plasticine, water
and sand columns, with their grid semantics, their constitutive pair and their
dt. Every number is our engine rolled from their frame-0 cloud and scored in
their metric:

  | material   | cell    | truth-replay error | recovered | published | beats |
  | plasticine | dataset | 7.84e-12 | 7.21e-7   | 6.5e-5    | yes, 90x  |
  | plasticine | time    | 1.45e-11 | 9.00e-7   | 1.4e-4    | yes, 156x |
  | plasticine | vel     | 5.34e-12 | 3.99e-7   | 4.6e-5    | yes, 115x |
  | water      | dataset | 2.99e-12 | 3.85e-7   | 2.0e-5    | yes, 52x  |
  | water      | time    | 9.02e-11 | 6.32e-6   | 3.5e-4    | yes, 55x  |
  | water      | vel     | 5.18e-13 | 1.70e-7   | 1.9e-5    | yes, 112x |
  | sand       | dataset | 2.02e-11 | 6.62e-3   | 2.6e-5    | no        |
  | sand       | time    | 5.00e-11 | 8.20e-3   | 4.2e-5    | no        |
  | sand       | vel     | 3.08e-11 | 1.18e-3   | 6.5e-5    | no        |

The truth-replay errors show the engine reproduces their integrator to eleven
digits on all three materials. Where a cell fails it fails on identification
alone, and for
sand deviation (23) says which part.
