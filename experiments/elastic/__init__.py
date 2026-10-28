"""Elastic identification campaign: stiffness, yield, and nonlinear hyperelastic laws
recovered from a gravity drop by one convex solve, with the simulator never differentiated.

Every stage drops a blob, reads the trajectory, and solves a weak-form momentum
residual that is linear in the material coefficients.

  drop.py           E to 0.35 percent on a sphere by the radial-window route and 13.4
                    percent on a cube, because the window does not vanish on a flat
                    face; learn on a rectangle and predict a held-out star geometry;
                    and the sample-complexity check that the estimator spread follows
                    Gauss-Markov while the unexcited bulk mode stays flat in N
  grid_gate.py      the Step 0 acceptance gate: the grid-consistent route takes the
                    cube from 13.4 percent to 0.069 percent and the sphere to 0.102
                    percent, which is what let the NCLaw comparison proceed
  plastic.py        von-Mises plasticine: G and lambda from the same solve, yield from
                    the deviatoric-strain saturation, identified to about 4 percent
                    when the drop yields and refused when it does not
  sequential.py     the recursive posterior for both materials: confidence is flat in
                    free fall and contracts at impact, E crosses a 5 percent threshold
                    and nu does not, and the cheap online Fisher confidence tracks the
                    expensive rollout error closely enough to serve as the stopping rule
  hyperelastic.py   Mooney-Rivlin and Yeoh, nonlinear in F and linear in theta: one
                    gentle probe leaves C01 at 36 percent, three hard probes bring it
                    to 31 percent with C10 at 16 percent and K at 2.9 percent, so the
                    limit is the strain path rather than the noise level
  fe_basis.py       one function-encoder basis over 480 materials in four hyperelastic
                    families, with the held-out reconstruction tracking the
                    Eckart-Young singular-value tail
  core.py           the one copy of the drop scene, dump reader, interior test
                    functions, rotation-invariant validity filter, stress bases, row
                    assembly, and the grid-consistent recovery

Consolidated 2026-08-22 from eight scripts in video2sim/sim (elastic_drop,
elastic_grid_gate, elastic_identify_sequential, plastic_drop,
plastic_identify_sequential, hyperelastic, hyperelastic_fe, sample_complexity). See
README.md for the reproduction checks, the one behavior change, and the artifact-path
compatibility notes.

Run:  .venv/bin/python -m experiments.elastic <stage>
      .venv/bin/python -m experiments.elastic --help
"""
