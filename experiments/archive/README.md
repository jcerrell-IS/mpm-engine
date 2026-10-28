# Archived experiments

Kept for writeup reference and git history. Not maintained, not expected to run.

These scripts produced numbers and figures that the writeups cite, so they stay
in the tree rather than being deleted. Their imports were fixed when they moved
here and their `out/` artifact paths are unchanged, but nothing here is covered
by a test and no one is keeping them working against engine changes. If you need
a result from this directory, read the script and expect to repair it.

The maintained campaigns are `../robotics/` (the Franka dough chain),
`../nclaw/`, `../diffsim/`, `../fe_ls/` and `../elastic/`.

Shared scaffolding still comes from `../robotics/common.py`, so these files
import from the maintained package rather than carrying their own copies.

## Shear-cell rheology studies

The 3D shear cell was the wide-shear-rate setting where a learned
function-encoder viscosity beat a two-parameter Bingham fit. `shear_cell_3d.py`
is the root of the group; the four scripts after it consume its
`out/shear_cell_3d/rollout_3d.npz` cache.

- `shear_cell_3d.py`: 3D lift of the 2D shear cell. Function-encoder recovery
  over roughly two decades of shear rate, then a held-out rollout at v=0.16 with
  the recovered curve re-simulated as a tabulated-viscosity material. Needs
  `fe-weights/viscous.npz`. Writes `out/shear_cell_3d/`.
- `strong_vs_weak.py`: strong form (pointwise stress oracle) against weak form
  (wall-force power balance) on the same sweeps. The FE basis is near-exact in
  the strong form and rides about 1.4x high in the weak form, which is the
  discrete closure factor; the misspecified Bingham fit fails in both, so the
  measurement limit and the model-class limit are separable. Writes
  `out/strong_vs_weak/`.
- `dough_fe_viscous.py`: FE against two-parameter Bingham recovery of a
  shear-thinning Herschel-Bulkley dough at three press speeds, from the plate
  force power balance. Writes `out/fe_viscous/`.
- `correct_model_check.py`: the control for the FE win. Fitting the true 3-term
  Herschel-Bulkley form by plain least squares gives a worse 30 percent rollout
  than FE's 11 percent, because the yield term and the power-law term are nearly
  collinear over a finite band and the unregularized fit collapses tau_y to zero.
  A ridge-regularized HB fit toward a generic dough prior recovers
  (eta=18.6, tau_y=68.6, pk=71.9, pn=0.20) at 25 percent. So the effect is
  identifiability and regularization, not model class.
- `rollout_snapshots.py`: snapshot grid of the held-out rollout, rows truth / FE
  / HB-ridge / Bingham. The shapes track across all four laws because the motion
  is displacement-controlled, which is why the wall force and not the deformation
  is the discriminator (FE 11, HB-ridge 25, Bingham 34 percent).
- `rollout_error_contours.py`: where each model departs from truth, as
  per-particle deviation binned over the x-z plane. Confirms the deformation
  errors stay small while the force carries the model difference.
- `shear_rollout_video.py`: renders the reference, FE and Bingham held-out
  rollouts above their wall-force traces, plus a rotating view of the reference
  block. Writes `shear_rollout_3d.mp4` and `shear_3d_view.mp4`.

## Rollout figures and videos from the Franka squeeze

- `rollout_franka_cotracker.py`: arm squeeze rendered as warm speckle in one
  camera, deformation extracted with CoTracker3, rollout error normalized by how
  far the material actually moves. Arm motion is identical across laws because
  the press is displacement-controlled, so the difference isolates the dough.
  Writes `out/rollout_arm/`.
- `rollout_force_video.py`: side-by-side truth and learned renders above a
  force-versus-strain trace with a current-frame marker. Reuses the frames from
  `rollout_franka_cotracker.py`, so that has to run first.

## Perception probes

- `speckle_particle_videos.py`: the grey-speckle particle videos that CoTracker
  actually tracks, for the 1x and 1.5x squeeze, as opposed to the
  marching-cubes surface render. Writes `out/speckle_particles/`.
- `surface_track_test.py`: smooth against material-textured dough surfaces
  through CoTracker3. A smooth reconstructed surface is nearly featureless and is
  re-triangulated every frame, so tracked points have no persistent texture; the
  textured render pins each surface vertex to its nearest material particle.
  Writes `out/surface_track/`.

## Experiment design

- `pressure_covariance_sweep.py`: information matrix A^T A and
  cov(theta_hat) for theta = (tau_y, eta) across press loads. Its own docstring
  carries the caveat that this is the scalar power-balance diagnostic and not the
  full divergence-free tensor weak form, which would need the contact impulse
  distributed against the same test fields. Writes
  `out/pressure_covariance_sweep/`.

## Floods

- `flood_sweep.py`: truck displacement and yaw over a grid of flood depths and
  surge velocities, using `warpmpm.vehicle.FloodScene`. Final displacement
  saturating near 0.83 m means the truck reached the downstream wall, so the
  domain ran out rather than the surge ending. Model scale is 1.45 m; read
  results at full size by Froude scaling. Writes `out/flood_sweep/`. Archived
  because it is a vehicle study with no robotics or identification content, not
  because the result is in doubt. The maintained equivalent is
  `examples/flood_vehicle.py`.

## Still active, not archived

- `../weak_contrastive_pilot.py` sits flat in `experiments/` and is untracked
  work in progress. It is a concluded pilot on contrastive weak-form training
  whose Pvol fix is still queued, so it is left exactly where it is.
