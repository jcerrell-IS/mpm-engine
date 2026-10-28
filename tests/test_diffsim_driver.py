"""Driver-level checks for the differentiable-simulation baseline.

Cheap tests only: parameter coordinates, the Drucker-Prager cone constant against
the warp expression, the CFL rule against the substep count the warp truth run
actually used, and the hand-written Adam (optax is not installed in this tree).
The rollout itself is validated in tests/test_diffmpm_forward.py.

jax lives in the video2sim staging venv rather than the engine venv, so under
the engine venv every test here skips at the importorskip below. Run them with
  ../.venv/bin/python -m pytest tests/test_diffsim_driver.py
from the repository root.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp
from experiments.diffsim.forward import (
    alpha_from_friction,
    mu_lam_from_E_nu,
)
from experiments.diffsim.identify import (
    CFL,
    SPECS,
    adam_fit,
    dump_for,
    prior_box,
    q_of_theta,
    theta_of_q,
    truth_q,
    unpacker,
    wave_speed,
)


def test_theta_coordinate_round_trip():
    for material, spec in SPECS.items():
        truth = {n: t for n, t, _ in spec["params"]}
        q = q_of_theta(material, truth)
        back = theta_of_q(material, q)
        for k, v in truth.items():
            assert abs(back[k] / v - 1.0) < 1e-12, (material, k)


def test_prior_box_brackets_truth():
    for material in SPECS:
        lo, hi = prior_box(material)
        qt = truth_q(material)
        assert np.all(lo < qt) and np.all(qt < hi), material


def test_unpacker_matches_the_engine_parameters():
    th = unpacker("jelly")(jnp.asarray(truth_q("jelly"), jnp.float32))
    mu, lam = mu_lam_from_E_nu(1.0e5, 0.2)
    assert abs(float(th.mu) / mu - 1.0) < 1e-5
    assert abs(float(th.lam) / lam - 1.0) < 1e-5
    th = unpacker("sand")(jnp.asarray([25.0], jnp.float32))
    # warp's set_parameters_dict: alpha = sqrt(2/3) 2 sin phi / (3 - sin phi)
    s = math.sin(math.radians(25.0))
    assert abs(float(th.alpha) / (math.sqrt(2 / 3) * 2 * s / (3 - s)) - 1.0) < 1e-5


def test_alpha_from_friction_is_the_warp_expression():
    for phi in (5.0, 25.0, 45.0):
        s = math.sin(phi / 180.0 * 3.14159265)
        assert abs(float(alpha_from_friction(jnp.float32(phi)))
                   - math.sqrt(2 / 3) * 2 * s / (3 - s)) < 1e-6


@pytest.mark.parametrize("material", ["jelly", "plasticine", "sand", "water"])
def test_cfl_rule_reproduces_the_truth_substeps(material):
    """The wave-speed and CFL transcription, checked against the dump metadata."""
    import json

    path = dump_for(material)
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    ex = meta.get("extra", meta)
    frame_dt = float(d["frame_dt"])
    dx = float(ex["grid_lim"]) / int(ex["n_grid"])
    v_max = float(np.abs(d["v"][0]).max())
    c = wave_speed(material, truth_q(material))
    sub = max(math.ceil(frame_dt / (CFL * dx / (c + v_max))), 1)
    assert sub == int(ex["substeps_per_frame"]), (material, sub, c)


def test_adam_finds_a_quadratic_minimum_inside_the_box():
    target = np.array([0.3, -0.7])

    def vg(q):
        q = np.asarray(q, dtype=np.float64)
        d = q - target
        return float(d @ d), 2.0 * d

    lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
    best, _n, stop = adam_fit(vg, np.array([1.5, 1.5]), lo, hi, 0.1, 200,
                              lambda *a, **k: None)
    assert np.allclose(best["q"], target, atol=2e-3), best
    assert stop in ("plateau", "iterations")


def test_adam_respects_the_box():
    def vg(q):
        q = np.asarray(q, dtype=np.float64)
        return float(q.sum()), np.ones_like(q)          # push straight down

    lo, hi = np.array([-0.5]), np.array([0.5])
    best, _, _ = adam_fit(vg, np.array([0.4]), lo, hi, 0.2, 30,
                          lambda *a, **k: None)
    assert best["q"][0] >= lo[0] - 1e-12
