"""Franka dough campaign: identify a rheology from force, predict, then plan a shape.

One chain, run left to right. A plate on the MuJoCo Franka squeezes a dough blob;
the grid-impulse plate force identifies the von-Mises or Bingham law by the convex
weak-form solve; the identified law then predicts force and deformation on dough
volumes it never saw; and the same law drives CEM shape planning, including a
transfer to a different-sized object. The engine is never differentiated.

The identify leg itself lives outside this package, in
``examples/vonmises_identify.py`` (probe and identify), because it doubles as a
maintained demo. transfer_identify_plan imports it directly.

Run any stage from the engine root with the project environment:

    .venv/bin/python -m experiments.robotics.<name>

Shared scaffolding is in common.py: the path anchors, the sticky-floor press
scene, the 3-fold lobedness metric and the CoTracker wrapper. Artifact paths are
unchanged by the fold; each script writes the out/ directory in its own docstring.

Prediction legs, force and deformation.

  predict_volume_franka   Law identified from one squeeze, applied to a larger
                          flatter blob. Reference (tau_y=200, eta=40),
                          grid-impulse recovered (192, 55), stress-integral
                          recovered (384, 56). The stress-integral law
                          over-predicts, because that estimator biased the fit.
                          Writes out/predict_volume_franka.png.
  predict_volume_rollout  Predicts the unseen volume's deformation and measures
                          it through render plus CoTracker3, the real perception
                          path. The plate descent is displacement-controlled, so
                          the rheology-dependent signal is the lateral extrusion
                          rather than the vertical compression. Writes out/rollout.
  realdata_pipeline       The same squeeze delivered the way a real rig would:
                          textured video plus a plate-force CSV plus calibration,
                          nothing from simulator internals. Quasi-2D plane strain,
                          identified at 1x and 1.5x volume against truth
                          (tau_y, eta) = (200, 40). Writes out/realdata.
  volume_holdout_check    The full 2x2 cross-volume matrix, because
                          realdata_pipeline's own numbers are per-volume
                          self-consistency and not held-out generalization. Laws
                          under test: identified on 1x (276.7, 16.1), on 1.5x
                          (101.6, 59.7). Imports realdata_pipeline for the
                          force series, so run that first.

Planning legs, and the baseline they are measured against.

  shape_planning          CEM shape planning over the engine with Chamfer or EMD
                          targets, about 0.9 s per measured rollout, using the
                          identified parameters. The von-Mises plasticine keeps
                          97 percent of the imposed compression after release.
                          Plate coupling is sticky with no tangential slip model,
                          so actions are vertical presses only.
  transfer_identify_plan  Identify (G, yield) on a small block, plan the shaping
                          press on a held-out larger one. The convex-identified
                          law is a material property and so size-independent; the
                          A-law plan should reach about the oracle Chamfer.
  three_prong             Three-jaw gripper on dough: three near-cube fingertips
                          at 0, 120 and 240 degrees close radially and form a
                          three-lobed cross-section, scored by common.lobedness.
  dough_franka_threeprong The same tool carried by the Franka, with the arm posed
                          by inverting its EE kinematics, and a composite
                          arm-plus-dough render.
  gripper_render_dough    PyVista render of the two-finger gripper plan from
                          examples/gripper_shape.py: per-frame marching-cubes
                          dough surface, target beside achieved.

Everything else from the flat robotics-era directory moved to
../archive/, which has its own index. That includes the shear-cell rheology
studies, the rollout figure and video variants, and the perception probes.
"""
