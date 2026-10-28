"""The recorded hardware pour: from one side video and the robot's joint log to a
viscosity, a twin, and a hold-time table for metering a target volume.

The data is an episode of the 2026-07-31 Franka pours (pouring_real_data/<episode>/,
outside the repository): the 500 mL measuring cup of geometry/measuring_cup.py holding
300 mL of dyed glycerol, one 60 degree roll over about 2.5 s into an identical cup on
the table, a side RealSense at 60 Hz and the joint log at 15 Hz. No force sensor, flow
meter or tracker.

Stages, run from the engine root with the project environment:

    .venv/bin/python -m experiments.pour.pour_calibrate_geometry
        table height, receiver center and grasp shift from the pre-pour depth cloud
    .venv/bin/python -m experiments.pour.pour_perception
        receiver level V(t) by the cup's own graduations, cup poses, the onset
    .venv/bin/python -m experiments.pour.pour_weakform_identify
        eta from the time-weak brink lubrication balance
    .venv/bin/python examples/pour_recorded_twin.py --eta <eta_hat>
        the twin, run once at the identified value
    .venv/bin/python -m experiments.pour.pour_validate
        the twin and the archived comparison twins against the real curve
    .venv/bin/python -m experiments.pour.pour_hold_sweep --plan 130
        the dwell at the roll's end pose that delivers a target volume

Artifacts: out/pour_wf/<episode>/ (observations.npz, identify.json, validation.json,
calibration.json, hold_table.json and their figures) and out/pour_recorded_twin/.

What it measured on ep0001. The receiver curve settles at 97.2 mL. The brink balance
V(t) - V(t0) = (1/eta) int F dt, with the film head from volume conservation and the
caliper rim curve, gives eta = 3.03 Pa.s, plus or minus 1.7 percent statistical and
14 percent for the 10 mL fill tolerance, which is the dominant systematic; the fill
itself is not identifiable from the curve. The twin at that value transfers 95.2 mL
(-2.1 percent, rms 14.8 mL over the overlap) with its first landed liquid at +1.85 s
against +1.82 s real; at the handbook 1.41 Pa.s it transfers 107.5 mL (+10.5 percent).
Held at the roll's end pose, the twin delivers 100 to 160 mL for dwells of 0.13 to
4.48 s.

Limits. eta is an effective value: the closure takes the free surface at the
hydrostatic level and each rim strip as an independent thin film, so the brink
discharge coefficient is absorbed into it. The twin uses the same separable contact as
the honey pour, which constrains only the approach velocity at the cup wall, so its
agreement with the real curve is a consistency check between the closure and the
engine rather than an independent measurement. The curve gives one number per frame,
so the pour identifies a viscosity, not a constitutive law.

pour_franka_calibrate.py holds the two measured constants of the honey pour
(examples/pour_franka.py): the spout azimuth in PandaPour.CUP_TO_HAND and the
stream-landing receiver position. pour_note_figures.py draws the figures of a
walkthrough note kept outside the repository.
"""
