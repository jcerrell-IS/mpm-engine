"""The tiers that drop channels: what they keep, what they derive, what refuses.

Two tiers are under test, both built by experiments/nclaw/strip_channels.py from
a schema-valid dump:

no_stress
    every stored kinematic channel kept bitwise, the stress channel excluded.

positions_only
    positions and times kept, velocities by finite difference and the two
    gradients by moving least squares. The reference for those numbers is the
    ingest path itself: the same dump exported to their frame format, stripped
    to positions there, and read back through ``read_nclaw_dir``. The two
    routes must agree, or the tier and the ingest have drifted.

The pressure models the no-stress tier states in place of the stress trace are
checked against an independent per-particle evaluation of the same Hencky
relation, and against a static column for the depth closure.

No simulator runs in this file; the trajectory is the manufactured one of
tests/test_nclaw_ingest.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from experiments.nclaw import identify_no_stress as ns
from experiments.nclaw import ingest as ing
from experiments.nclaw import strip_channels as sc
from experiments.nclaw import suite

from ident.io.schema import validate_dump_schema
from ident.weakform.elastic_grid import E_nu_to_moduli
from tests.test_nclaw_ingest import E_TRUE, NU_TRUE, synthetic_dump

KINEMATIC_KEYS = ("x", "v", "L", "F", "volume", "mass", "volume0", "times")


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    return synthetic_dump(tmp_path_factory.mktemp("no_stress") / "manufactured.npz")


@pytest.fixture(scope="module")
def no_stress(dump):
    return sc.write_no_stress_dump(dump, log=lambda *_: None)


@pytest.fixture(scope="module")
def positions_only(dump):
    return sc.write_positions_only_dump(dump, log=lambda *_: None)


# ---------------------------------------------------------------------------
# The no-stress tier
# ---------------------------------------------------------------------------

def test_no_stress_keeps_every_kinematic_channel_bitwise(dump, no_stress):
    a, b = np.load(dump), np.load(no_stress)
    for key in KINEMATIC_KEYS:
        assert np.array_equal(a[key], b[key]), key
    assert np.count_nonzero(b["stress"]) == 0


def test_no_stress_dump_declares_the_exclusion(no_stress):
    meta = validate_dump_schema(no_stress)
    assert meta.pressure_source == "absent"
    assert meta.has_pressure is False
    assert meta.extra["tier"] == "no_stress"
    prov = meta.extra["channel_provenance"]
    assert prov["stress"] == "excluded"
    for key in ("x", "v", "L", "F"):
        assert prov[key] == "measured", key
    notes = " ".join(meta.extra["degradation_notes"])
    assert "no_stress" in notes and "no oracle pressure" in notes


def test_friction_refuses_on_the_tier_dump_and_names_the_missing_channel(no_stress):
    arr = suite._load_arrays(no_stress)
    out = suite.identify_friction(arr, window_frames=8, frame_stride=2,
                                  log=lambda *_: None)
    assert out["refused"] is True
    assert "oracle pressure" in out["reason"]
    assert out["pressure_source"] == "absent"


def test_stage_identify_refuses_a_dump_that_still_has_pressure(dump):
    """The tier stage must be handed the tier dump, not the full-channel one."""
    with pytest.raises(ValueError, match="oracle pressure"):
        ns.stage_identify_no_stress("jelly", dump=dump, log=lambda *_: None)


def test_elastic_leg_is_unchanged_by_the_exclusion(dump, no_stress):
    """Jelly's leg never reads stress, so the tier must reproduce it exactly."""
    direct = suite.identify_elastic(suite._load_arrays(dump), window_frames=8,
                                    frame_stride=2, log=lambda *_: None)
    tier = suite.identify_elastic(suite._load_arrays(no_stress), window_frames=8,
                                  frame_stride=2, log=lambda *_: None)
    for key in ("mu", "lam", "E", "nu", "n_rows"):
        assert tier[key] == direct[key], key


# ---------------------------------------------------------------------------
# The positions-only tier against the ingest's own derivation
# ---------------------------------------------------------------------------

def test_positions_only_matches_the_ingest_derivation(dump, positions_only, tmp_path):
    """Same derivation, two routes: through this tier and through the ingest.

    The ingest route goes out to their frame format, is stripped to positions
    there, and comes back through ``read_nclaw_dir``, which is the path a
    position-only release of theirs would travel.
    """
    state = tmp_path / "state"
    ing.export_to_nclaw(dump, state, log=lambda *_: None)
    stripped = ing.strip_to_positions(state, tmp_path / "state_x_only",
                                      log=lambda *_: None)
    man = {**ing.manifest_for_export(dump, "jelly"), "stress_lag_steps": 0}
    back = tmp_path / "ingest_x_only.npz"
    ing.read_nclaw_dir(stripped, man, back, log=lambda *_: None)

    a, b = np.load(positions_only), np.load(back)
    assert np.array_equal(a["x"], b["x"]), "positions travel exactly on both routes"
    for key in ("v", "L", "F", "volume", "mass", "volume0"):
        u, w = a[key].astype(np.float64), b[key].astype(np.float64)
        scale = max(float(np.abs(u).max()), 1e-300)
        assert np.abs(u - w).max() / scale < 1e-6, key


def test_positions_only_records_the_derivations(positions_only):
    meta = validate_dump_schema(positions_only)
    prov = meta.extra["channel_provenance"]
    assert prov["x"] == "measured"
    for key in ("v", "L", "F"):
        assert prov[key] == "derived", key
    assert prov["stress"] == "excluded"
    assert prov["volume0"] == "scene_fact"
    assert meta.has_pressure is False
    notes = " ".join(meta.extra["degradation_notes"])
    assert "finite differences" in notes and "least squares" in notes


def test_tier_path_names_and_reuse(dump, tmp_path):
    assert sc.tier_path(dump, "no_stress").name.endswith("_no_stress.npz")
    assert sc.tier_path(dump, "positions_only").name.endswith("_positions_only.npz")
    with pytest.raises(ValueError, match="unknown tier"):
        sc.tier_path(dump, "no_velocity")
    first = sc.write_tier_dump(dump, "no_stress", log=lambda *_: None)
    stamp = first.stat().st_mtime_ns
    again = sc.write_tier_dump(dump, "no_stress", log=lambda *_: None)
    assert again == first and again.stat().st_mtime_ns == stamp, "reused, not rewritten"


# ---------------------------------------------------------------------------
# The pressure models that stand in for the stress trace
# ---------------------------------------------------------------------------

def _reference_hencky_cauchy(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """tau = U diag(2 mu eps_i + lam tr eps) U^T over J, one particle at a time.

    An independent evaluation of the relation ``hencky_stress_parts`` vectorizes,
    written as an explicit loop so a broadcasting mistake in the vectorized form
    cannot hide.
    """
    out = np.empty_like(F)
    for i in range(F.shape[0]):
        U, s, _ = np.linalg.svd(F[i])
        eps = np.log(s)
        tau = U @ np.diag(2.0 * mu * eps + lam * eps.sum()) @ U.T
        out[i] = tau / float(np.prod(s))
    return out


def test_hencky_pressure_and_deviator_match_an_independent_evaluation():
    rng = np.random.default_rng(3)
    F = np.eye(3) + 0.25 * rng.standard_normal((64, 3, 3))
    mu, lam = E_nu_to_moduli(E_TRUE, NU_TRUE)
    ref = _reference_hencky_cauchy(F, mu, lam)
    p_ref = -np.trace(ref, axis1=1, axis2=2) / 3.0
    dev_ref = ref + p_ref[:, None, None] * np.eye(3)[None]

    p, dev, J = ns.hencky_stress_parts(F, E_TRUE, NU_TRUE)
    assert np.abs(p - p_ref).max() / np.abs(p_ref).max() < 1e-12
    assert np.abs(dev - dev_ref).max() / np.abs(dev_ref).max() < 1e-12
    assert np.allclose(J, np.linalg.det(F))
    assert np.abs(np.trace(dev, axis1=1, axis2=2)).max() < 1e-9 * np.abs(dev).max()


def test_yield_column_reproduces_the_stress_of_a_particle_at_yield():
    """The one-unknown model is exact where the return map put the particle.

    The von Mises return map caps ||dev eps|| at y / (2 mu), so a particle at
    yield has y = 2 mu ||eps_hat||, and the model sigma_vol I + y N / J must then
    equal the full Hencky Cauchy stress. This is the algebra the yield column
    rests on, checked term by term.
    """
    rng = np.random.default_rng(5)
    F = np.eye(3) + 0.2 * rng.standard_normal((32, 3, 3))
    mu, lam = E_nu_to_moduli(3.0e5, 0.25)
    N, sigma_vol, hat_norm, J = ns.hencky_yield_parts(F, mu, lam)
    y = 2.0 * mu * hat_norm                      # the cap, particle by particle
    model = (sigma_vol[:, None, None] * np.eye(3)[None]
             + (y / J)[:, None, None] * N)
    ref = _reference_hencky_cauchy(F, mu, lam)
    assert np.abs(model - ref).max() / np.abs(ref).max() < 1e-12
    assert np.abs(np.linalg.norm(N, axis=(1, 2)) - 1.0).max() < 1e-12


def test_depth_closure_is_hydrostatic_on_a_static_column():
    """rho g (h - z) exactly, with h the top of the column: the P0 convention."""
    z = np.arange(10) * 0.02
    pts = np.stack([np.full_like(z, 0.31), np.full_like(z, 0.42), z], axis=1)
    x = pts[None].repeat(3, axis=0)
    p = ns.column_surface_pressure(x, rho=1000.0, g_z=-9.8, cell=0.05)
    expect = 1000.0 * 9.8 * (z.max() - z)
    assert np.abs(p[0] - expect).max() < 1e-9
    assert np.array_equal(p[0], p[2])


def test_basal_scaling_lifts_the_depth_shape_onto_the_measurement():
    """A measurement that is k times the closure must return k times the shape."""
    rng = np.random.default_rng(7)
    z = rng.uniform(0.15, 0.45, 400)
    pts = np.stack([rng.uniform(0.2, 0.8, 400), rng.uniform(0.2, 0.8, 400), z], axis=1)
    x = pts[None].repeat(4, axis=0)
    cell, floor_z, rho, g_z = 0.05, 0.15, 1000.0, -9.8
    shape = ns.column_surface_pressure(x, rho, g_z, cell)
    k = 2.5
    p, diag = ns.basal_scaled_pressure(x, k * shape, floor_z, cell, rho, g_z)
    assert diag["n_frames_with_band_measurement"] == 4
    assert abs(diag["scale_median"] / k - 1.0) < 1e-9
    assert np.abs(p - k * shape).max() < 1e-9 * np.abs(k * shape).max()


def test_pressure_agreement_reports_a_known_bias():
    truth = np.full((4, 50), 1000.0)
    model = 1.4 * truth
    got = ns.pressure_agreement(model, truth)
    assert got["n"] == 200
    assert abs(got["ratio_median"] - 1.4) < 1e-12
    assert abs(got["rel_err_median"] - 0.4) < 1e-12
