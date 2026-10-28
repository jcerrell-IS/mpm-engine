"""Forward validation of the JAX differentiable-simulation baseline against warp.

The baseline only means anything if its forward map is the same simulator the
truth came from, so these tests measure the cross-engine gap directly (deviation
list in experiments/diffsim/forward.py) and check the two AD guards: the Newton
polar rotation that replaces svd3 in the corotated stress, and the broadened SVD
JVP.

jax lives in the video2sim staging venv rather than the engine venv, so under
the engine venv every test here skips at the importorskip below. Run them with
  ../.venv/bin/python -m pytest tests/test_diffmpm_forward.py
from the repository root.

The per-material gap tests need the grid-20 warp truths
(out/nclaw_suite/dumps/<material>_cube_g20_truth.npz) and skip when they are
absent. Full-horizon numbers are produced by experiments.diffsim validate; these
run 20 frames to stay quick.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from experiments.diffsim.forward import (
    Theta,
    alpha_from_friction,
    det3,
    inv3,
    load_truth,
    make_loss,
    mu_lam_from_E_nu,
    polar_rotation,
    rollout,
    svd3,
)

DUMPS = ROOT / "out" / "nclaw_suite" / "dumps"
# Cube edge in metres (the NCLaw blob is a 0.5 m cube of particles), the scale a
# position gap has to be read against.
CUBE_MM = 500.0

TRUTH = {
    "jelly": dict(zip(("mu", "lam"), mu_lam_from_E_nu(1.0e5, 0.2), strict=True)),
    "plasticine": dict(zip(("mu", "lam"), mu_lam_from_E_nu(3.0e5, 0.25), strict=True),
                       yield_stress=5.0e3),
    "sand": dict(zip(("mu", "lam"), mu_lam_from_E_nu(1.0e6, 0.2), strict=True),
                 alpha=float(alpha_from_friction(25.0))),
    "water": {"bulk": 83333.33333333331},
}


def _theta(material: str) -> Theta:
    return Theta(**{k: jnp.float32(v) for k, v in TRUTH[material].items()})


def _rng_mats(n: int, seed: int = 0, spread: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.eye(3) + spread * rng.standard_normal((n, 3, 3))).astype(np.float32)


# ---------------------------------------------------------------------------
# linear algebra helpers and the two AD guards
# ---------------------------------------------------------------------------

def test_det_and_inv_match_numpy():
    A = _rng_mats(64, seed=1)
    assert np.allclose(np.asarray(det3(jnp.asarray(A))), np.linalg.det(A), rtol=2e-5)
    assert np.allclose(np.asarray(inv3(jnp.asarray(A))), np.linalg.inv(A),
                       rtol=2e-4, atol=1e-5)


def test_polar_matches_svd():
    """Deviation (a): the Newton polar factor reproduces U V^T from an SVD."""
    A = _rng_mats(256, seed=2)
    A = A[np.linalg.det(A) > 0.2]
    U, _, Vt = np.linalg.svd(A)
    R_svd = U @ Vt
    R = np.asarray(polar_rotation(jnp.asarray(A)))
    assert np.abs(R - R_svd).max() < 1e-5, np.abs(R - R_svd).max()
    # and it IS a rotation
    assert np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max() < 1e-5


def test_polar_differentiable_at_identity():
    """R(F) is analytic at F = I; the SVD route is not, which is why (a) exists."""
    def f(t):
        return polar_rotation(jnp.eye(3, dtype=jnp.float32)[None] * (1.0 + t))[0, 0, 0]

    d = jax.jacfwd(f)(jnp.float32(0.0))
    assert np.isfinite(float(d))


def test_svd3_jvp_matches_finite_differences():
    A = jnp.asarray(_rng_mats(16, seed=3))
    dA = jnp.asarray(_rng_mats(16, seed=4) * 0.01)

    def f(t):
        U, s, Vt = svd3(A + t * dA)
        # a scalar that mixes all three factors
        return jnp.sum(s) + jnp.sum(U * 1.0) + jnp.sum(Vt * 2.0)

    ad = float(jax.jacfwd(f)(jnp.float32(0.0)))
    h = 1e-3
    fd = (float(f(jnp.float32(h))) - float(f(jnp.float32(-h)))) / (2 * h)
    assert abs(ad - fd) <= 2e-3 * max(abs(fd), 1.0), (ad, fd)


def test_svd3_jvp_finite_at_repeated_singular_values():
    """The recorded guard: broadening keeps inf * 0 out of forward mode at F = I."""
    A = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (4, 3, 3))
    dA = jnp.asarray(_rng_mats(4, seed=5) * 0.01)
    U, s, Vt = jax.jvp(svd3, (A,), (dA,))[1]
    for arr in (U, s, Vt):
        assert np.isfinite(np.asarray(arr)).all()


def test_svd3_reconstructs():
    A = jnp.asarray(_rng_mats(32, seed=6))
    U, s, Vt = svd3(A)
    rec = U @ (s[..., None, :] * jnp.eye(3, dtype=jnp.float32)) @ Vt
    assert np.abs(np.asarray(rec) - np.asarray(A)).max() < 2e-5


# ---------------------------------------------------------------------------
# cross-engine forward gap, per material
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("material", ["jelly", "water", "plasticine", "sand"])
def test_forward_gap_against_warp_truth(material):
    path = DUMPS / f"{material}_cube_g20_truth.npz"
    if not path.exists():
        pytest.skip(f"missing {path.name}; run the suite gen stage at n_grid 20")
    cloud, xt, scene, _ = load_truth(path, material)
    nf = 20
    x = np.asarray(rollout(_theta(material), cloud, scene, nf))
    d = x - xt[:nf + 1]
    assert np.isfinite(d).all()
    rms_mm = 1e3 * np.sqrt((d[-1] ** 2).sum(-1).mean())
    # a faithful transcription sits far below the deformation scale; the measured
    # values at 20 frames are 2e-4 mm (jelly) to 8e-3 mm on a 500 mm cube
    assert rms_mm < 0.05 * CUBE_MM / 100.0, rms_mm


def test_gradient_matches_finite_differences_jelly():
    path = DUMPS / "jelly_cube_g20_truth.npz"
    if not path.exists():
        pytest.skip("missing jelly grid-20 truth")
    cloud, xt, scene, _ = load_truth(path, "jelly")
    scene = scene._replace(n_frames=20)
    def unpack(q):
        return Theta(mu=10.0 ** q[0], lam=10.0 ** q[1])

    loss = make_loss(cloud, xt, scene, list(range(0, 21, 5)), unpack)
    q = jnp.asarray(np.log10([TRUTH["jelly"]["mu"] * 1.5,
                              TRUTH["jelly"]["lam"] * 0.8]), jnp.float32)
    ad = np.asarray(jax.jacfwd(loss)(q))
    h = 2e-3
    fd = np.array([(float(loss(q.at[i].add(h))) - float(loss(q.at[i].add(-h)))) / (2 * h)
                   for i in range(2)])
    assert np.allclose(ad, fd, rtol=5e-2), (ad, fd)


def test_walls_hold_a_resting_block():
    """The freeslip box: nothing leaves the domain and the floor is not crossed."""
    from experiments.diffsim.forward import Scene

    G, lim = 20, 1.0
    dx = lim / G
    h = dx / 2
    ax = np.arange(0.35, 0.65 + 0.5 * h, h, dtype=np.float32)
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    cloud = {"x0": jnp.asarray(pts), "v0": jnp.zeros_like(jnp.asarray(pts)),
             "vol0": jnp.full((len(pts),), h ** 3, jnp.float32),
             "mass": jnp.full((len(pts),), 1000.0 * h ** 3, jnp.float32)}
    scene = Scene(material="jelly", n_grid=G, grid_lim=lim, bound_cells=3,
                  dt=1e-3, substeps=4, n_frames=40)
    x = np.asarray(rollout(_theta("jelly"), cloud, scene))
    assert np.isfinite(x).all()
    assert x.min() > 3 * dx - 0.5 * h, x.min()
    assert x.max() < lim - 3 * dx + 0.5 * h, x.max()
