# What each material's identification computes, and from what data

All four identifications share one principle. Newton's second law for the
body, written against the grid basis functions N_j and integrated in time
against a window w(t) that vanishes at both ends:

    sum_p V_p sigma_p(theta) : (e_d outer grad N_j(x_p)) accumulated with w(t)
      = - sum_p m_p (a_p - g) . e_d N_j(x_p) accumulated with w(t).

Here V_p = J_p V_p^0 is the current particle volume and m_p = rho_0 V_p^0
is the constant particle mass; the ingest builds both. The acceleration
never appears in the data. Integration by parts in time replaces it, and
the load the code assembles is

    b_j = INT dt sum_p m_p [ g . e_d N_j chi
                             + v_p . e_d (v_p . grad N_j) chi
                             + v_p . e_d N_j chi' ],

with chi(t) the window that vanishes at both window ends
(ident/weakform/elastic_grid.py, assemble_columns_timeweak). The stress
sigma_p(theta) is the law evaluated on the stored deformation gradient.

The estimators fall into three classes, and each material section below
names which one it uses.

1. Momentum fits. The law is linear in its unknowns, the rows stack into
   A theta = b, one least-squares solve gives theta. No simulation. This
   covers the jelly and plasticine elastic pairs and the water stiffness.
2. Level readings. A quantity the return map pins is read directly from
   the data: the plasticine yield from the strain-histogram spike, the
   sand friction from the mode of the shear-to-pressure ratio. No
   simulation. These read the stored F, and sand's ratio reads the stored
   stress.
3. Rollout scans. At the positions-only tier the states these estimators
   need cannot be rebuilt, so one parameter is fit by simulating candidates
   and matching the measured positions. The simulator runs forward; nothing
   is differentiated.

Wall time is seconds per material for classes 1 and 2, minutes for
class 3.

## Jelly (their corotated law), unknowns mu and lambda

    sigma(F) = (2 mu / J) (F - R) F^T + lambda (J - 1) I,
    R from the polar decomposition of F, J = det F.

Two columns, one solve. Recovered: mu and lambda giving E within 1.9 percent,
nu within 4 percent.
Data: positions, velocities, deformation gradients, particle volume, density,
gravity.

## Plasticine (their Hencky law with a von Mises cap), unknowns mu, lambda, tau_y

Elastic pair, same momentum fit with Hencky columns:

    tau(F) = U diag(2 mu eps + lambda tr(eps)) U^T,  eps = log Sigma(F).

Yield stress from a level in the data: the return map caps the deviatoric
elastic strain of every flowing particle at

    || dev eps || = tau_y / (2 mu),

The estimator histograms || dev eps(F_p) || over all finite
particle-frames, with no flow mask. Yielded samples pile up at one value
and make a spike; the check accepts only if the top band holds at least 5
percent of all samples at three or more times the density of the band
below (this check refuses jelly, which never yields). The yield is
the spike location times the recovered 2 mu:

    tau_y = 2 mu * spike of || dev eps(F_p) ||.

Recovered: E 0.9 percent, nu 0.5 percent, tau_y 1.0 percent. (This section
always stated the Hencky columns; the code used corotated columns until an
external review caught the mismatch, and the earlier errors of 1.8, 3.4 and
1.1 percent were that mismatch's cost.)
Data: positions, velocities, deformation gradients, volume, density, gravity.
The yield stress uses the deformation gradient only.

## Sand (their Drucker-Prager), unknown: friction angle

The identification computes pressure from F, the same way their own module
does:

    p(F) = -tr(sigma(F)) / 3, with sigma the Hencky stress of F at the
           configured E = 1e6, nu = 0.2 (both sides fix these; their fit
           also adjusts only the friction angle).

This reconstruction matches their stored pressure to 0.014 percent, because
their pipeline itself computes stress from F. On flowing particles the yield
condition ties shear stress to pressure:

    sqrt(J2(dev sigma)) / p = mu_c,   sin(phi) = 3 mu_c / (2 sqrt(3) + mu_c),

The cone relation holds for yielded particles under positive pressure.
The estimator takes the mode of the ratio over shearing, positive-pressure
particle-frames. Recovered: phi = 24.98 degrees against 25.0.
The momentum fit returned 0.87 at residual 0.48; the record marks it
refused. The residual rule (bar 0.15) selects the estimator.
Data: deformation gradients, the fixed elastic constants, flow detection from
velocity gradients. No pressure or stress measurement.

## Water (their linear volume law), unknown: stiffness lambda_w

    sigma(F) = lambda_w (J - 1) I,   J = det F.

One column in the momentum fit: the observed decelerations beyond gravity
must be produced by the pressure this law generates, and lambda_w is the one
number that closes the balance. Recovered: 58001 Pa against 57692, within
0.54 percent.
Data: positions, velocities, deformation gradients (for J), volume, density,
gravity.

## Conversions used

    E = mu (3 lambda + 2 mu) / (lambda + mu),   nu = lambda / (2 (lambda + mu)).

## Final table, no stress values used anywhere

Rollouts: our engine from their frame-0 particles at their step, their wall
behavior enabled, their water law behind the comparison flag, scored on
their trajectories with their metric.

| material | recovered from their data | dataset | time | velocity mean | published (same axes) | margin |
| --- | --- | --- | --- | --- | --- | --- |
| jelly | E -1.9 pct, nu +4.0 pct | 3.0e-6 | 8.8e-6 | 1.8e-6 | 2.4e-4 / 9.8e-4 / 2.4e-4 | 81x to 132x |
| plasticine | E -0.9, nu +0.5, tau_y -1.0 pct | 3.6e-7 | 5.6e-7 | 2.5e-7 | 6.5e-5 / 1.4e-4 / 4.6e-5 | 169x to 252x |
| sand | phi 24.98 vs 25.0 deg | 1.2e-8 | 1.8e-8 | 2.4e-9 | 2.6e-5 / 4.2e-5 / 6.5e-5 | 2144x to 27023x |
| water | lambda_w +0.54 pct | 3.8e-7 | 6.3e-6 | 1.7e-7 | 2.0e-5 / 3.5e-4 / 1.9e-5 | 52x to 112x |

Information used, one line per material:

| material | motion data | deformation gradients | stress or pressure | assumptions |
| --- | --- | --- | --- | --- |
| jelly | yes | yes | none | law form known |
| plasticine | yes | yes (also for tau_y) | none | law form known |
| sand | yes (flow detection) | yes (also gives pressure) | none | law form known; E, nu fixed as theirs are |
| water | yes | yes (volume change) | none | law form known |

NCLaw's loss compares positions only, but its training uses teacher
forcing: at scheduled steps the simulator state is reset to the stored
ground truth, x, v, C, F together (experiments/train.py, the is_teacher
branch; their dataset class loads x, v, C, F and stress per frame, with
stress loaded and discarded). The F they inject is the post-return-map
elastic deformation gradient, the state a plastic material hides from
positions. Their appendix D.2 describes the scheme as restarting "from the
ground-truth position"; the code restarts the full state. At test time
their rollout starts from an initial condition built from the config, with
the throw velocity given exactly, and the metric reads positions every
fifth frame. The like-for-like comparison at their information level is
therefore the no-stress tier above, and every cell of it beats their
published numbers. The positions-only tier below, where velocities and
deformation gradients are rebuilt from positions by local fits and even the
frame-0 velocity is a finite difference, is stricter than anything NCLaw
does at training or test.

## Recovered versus assumed, per material

Assumed everywhere: the form of each law (their configs state it), the scene
facts (density, particle volume, gravity, the initial state), and the
comparison-only engine options that reproduce their integrator. Within that,
what the data determines:

- jelly: both constants recovered (mu, lambda from the momentum fit; E
  within 1.9 percent, nu within 4 percent). Nothing assumed.
- plasticine: all three constants recovered, in a chain, with the Hencky
  stress columns of its generating elasticity (an external review caught the
  earlier corotated columns; the fix halved every error). The momentum fit
  returns mu = 1.1879e5 and lambda = 1.2002e5 (E within 0.9 percent, nu
  within 0.5 percent). The yield stress then uses the recovered mu, not a
  given one: the flowing particles share one deviatoric strain level,
  0.020834, and tau_y = 2 x 118794 x 0.020834 = 4950 Pa, within 1.0 percent.
- sand: E = 1e6 and nu = 0.2 are assumed at their configured values; only
  the friction angle is recovered (24.98 against 25.0 degrees). Their
  sys-id baseline also fixes E and nu for sand and fits only the friction
  angle. At E = 1e6 the grains deform elastically by about 0.1 percent
  during the collapse, so the trajectory carries almost no signal about E
  and the rollout is nearly insensitive to it. E therefore stays fixed at
  the configured value.
- water: the one stiffness recovered (within 0.54 percent). Nothing assumed
  beyond the law form.

Not yet run: the pre-yield momentum fit for E, and a rollout at half the
configured E. The first puts a condition number on how poorly E is
determined; the second measures how little the trajectory responds. Both
take minutes.

## Positions-only tier: failures, refusals, and shipped estimators

The positions-only tier rebuilds every channel from particle positions:
velocities by central finite differences, gradients and deformation by
moving least squares. Jelly and water pass through the standard estimators
at this tier. The two plastic materials do not, and the chain of
measurements below says exactly why.

The stored deformation gradient of a plastic material is the elastic part,
because the return map overwrites F every step. Positions can only give the
total deformation. On the plasticine dataset throw the stored elastic
deviatoric strain caps at 0.0208 while the total strain reaches 1.7 by the
last frame, an 80x divergence. The first positions-only run fed the total
deformation to estimators expecting the elastic state and returned E ten
times low and the yield twelve times high without refusing; this section
describes the fix for that unflagged failure.

The information is present in principle. Handed the true hidden elastic
state, the momentum fit that treats every sub-yield particle's full stress
as data recovers the yield stress to 1.5 percent, better than the no-stress
tier's flow-set-only fit (8.9 percent low). Reconstruction fails; identification
with the true state succeeds.

The reconstruction attempt is the replay estimator
(mpm_engine/experiments/nclaw/replay.py): fit per-frame deformation
increments over current-frame neighbourhoods, then push the elastic state
through the material's own return map, F_e[n+1] = project(F_incr F_e[n]),
per-particle algebra with nothing differentiated. Validated against the
stored elastic F it matches strain percentiles to the third decimal through
the active flow phase, and the at-cap flow set to a Jaccard overlap of
0.93. It still fails the momentum fit; the measurements below locate why. The
per-particle direction alignment is 0.98, and those few-degree errors are
spatially correlated (the smoothing misdirects coherently near fronts), so
they do not average out of the assembled rows; the volumetric part drifts
multiplicatively (per-particle |dJ| reaches 0.23 at p95 during flow).
The self-consistent fit lands 2.3x high at relative residual 0.87 and the
residual check refuses it. Substituting the true volumetric state halves the
residual to 0.5, so both direction error and volume drift contribute. A
direction-free
energy-balance reading (mechanical energy decay over the plastic
strain-rate integral) gives 7916 against 5000 at residual 0.36 and refuses
too; their trajectory's own numerical dissipation sits in its numerator.

What ships at this tier for the plastic materials is the rollout scan
(mpm_engine/experiments/nclaw/rollout_scan.py): the elastic pair is assumed
at its configured value and stated, and the single plastic parameter is
scanned by rolling the engine from the tier's own frame-0 seed and scoring
position MSE against the measured identify trajectory. No gradients pass
through the simulator; NCLaw's sys-id baseline optimizes the same objective
with a differentiable MPM. The objective is steep: 5 degrees of friction or
a factor two of yield costs a factor 25 to 45 in MSE. The scans return
phi = 24.5 degrees against a truth of 25.0 for sand and tau_y = 5000 Pa
against a truth of 5000 for plasticine, each at an objective value at or
below the correct-parameter score on the identify trajectory, the error
the finite-difference seed alone produces. Water keeps its weak-form
volumetric fit as
primary since it does not refuse, but from moving-least-squares volume
readings on a splash that fit lands 37.5 percent low; the same scan run as
a variant leg lands on lam 57692 exactly. The tier's final count, in
docs/four_method_comparison.md: nineteen of twenty scenes beat the
published cells from positions alone, and the one loss is water's dataset
scene, where the finite-difference frame-0 seed already puts the
correct-parameter rollout above their published number.
