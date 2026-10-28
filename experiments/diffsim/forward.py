"""Minimal differentiable MLS-MPM in JAX: the differentiable-simulation baseline.

This module exists for ONE purpose, stated in the video2sim plan doc
docs/diffsim_baseline_plan.md: to run the sys-id-by-backprop method (NCLaw's
oracle row, DPSI's method) against our convex weak-form identification on the
same trajectories. The TrackEUCLID invariant stands: warp-mpm is never made
differentiable, and nothing here is imported by src/ident or by the warp engine.
The truth trajectories and every rollout in the comparison still come from warp.

Forward fidelity, the number that makes the baseline mean anything: at truth
theta this engine and warp agree to 9.1e-13 to 3.3e-10 position MSE across the
four materials, five to eight orders below every published NCLaw cell, so no
part of a recovered parameter is a cross-engine artifact. Measured per material
by experiments.diffsim identify's validate stage and, at 20 frames, by
tests/test_diffmpm_forward.py.

The scheme is transcribed from src/warpmpm/kernels/mpm_utils.py
(p2g_particle, grid_normalization_and_gravity, g2p_particle,
stress_update_particle) and mpm_solver_warp.py (add_surface_collider, the "slip"
branch), substep for substep:

  1. stress:  F_e = returnMap(F_trial), tau = tau(F_e)         [particle]
  2. p2g:     m_i    += w m_p
              (mv)_i += w m_p (v_p + C_p dpos) - dt V0_p tau_p grad w
  3. grid:    v_i = (mv)_i / m_i + dt g            where m_i > 1e-15
  4. walls:   freeslip, v_i -= (v_i . n) n  on nodes behind each of six planes
  5. g2p:     v_p = sum w v_i,  C_p = 4 inv_dx sum w v_i (x) dpos_grid,
              L_p = sum v_i (x) grad w,  x_p += dt v_p,
              F_trial = (I + dt L_p) F_e

Deviations from the warp kernels, all measured in tests/test_diffmpm_forward.py:

  a. The rotation of the fixed-corotated stress comes from a scaled Newton polar
     iteration, not from svd3. Reason: the AD of an SVD is singular at repeated
     singular values, which is exactly the rest state F = I of every particle at
     frame 0, while the polar factor R(F) is analytic there. The iteration
     reproduces svd3's R to float32 round-off (test_polar_matches_svd).
  b. Where an SVD is unavoidable (the von Mises and Drucker-Prager Hencky return
     maps) it carries a custom JVP whose 1 / (s_j^2 - s_i^2) factors are
     Lorentz-broadened, safe_inv(x) = x / (x^2 + SVD_JVP_EPS^2). Unbroadened,
     coincident singular values give inf * 0 = NaN in forward mode. The
     broadening is the guard the plan asks to be recorded; the resulting
     gradient is checked against central finite differences of the loss.
  c. Out-of-grid stencil indices are clamped per axis instead of writing out of
     bounds. The freeslip walls sit three cells inside the domain, so no
     in-domain particle reaches the clamp.

Everything else (weight polynomials, the floor() base index, the dt placement,
the 1e-15 mass threshold, the EOS exponent 1.1, the return-map guards 0.01 and
1e-14) is the warp expression verbatim.

This module writes no artifacts; the driver does. Exercise it through the
driver, whose validate stage is the cheapest useful call:
  ../.venv/bin/python -m experiments.diffsim validate --material jelly
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

F32 = jnp.float32
EYE = jnp.eye(3, dtype=F32)

# Lorentz broadening of the SVD JVP denominators (deviation b above). float32
# carries about 1e-7 relative precision, so singular-value gaps below 1e-6 are
# round-off and the broadened factor caps at 1 / (2 eps) = 5e5.
SVD_JVP_EPS = 1.0e-6
POLAR_ITERS = 8

MATERIAL_IDS = {"jelly": 0, "metal": 1, "sand": 2, "plasticine": 5, "water": 6}


# ---------------------------------------------------------------------------
# small dense 3x3 helpers (analytic, to keep LAPACK out of the hot path)
# ---------------------------------------------------------------------------

def det3(A: jnp.ndarray) -> jnp.ndarray:
    return (A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
            - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
            + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0]))


def inv3(A: jnp.ndarray) -> jnp.ndarray:
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    adj = jnp.stack([
        jnp.stack([a11 * a22 - a12 * a21, a02 * a21 - a01 * a22, a01 * a12 - a02 * a11], -1),
        jnp.stack([a12 * a20 - a10 * a22, a00 * a22 - a02 * a20, a02 * a10 - a00 * a12], -1),
        jnp.stack([a10 * a21 - a11 * a20, a01 * a20 - a00 * a21, a00 * a11 - a01 * a10], -1),
    ], -2)
    return adj / det3(A)[..., None, None]


def _T(A: jnp.ndarray) -> jnp.ndarray:
    return jnp.swapaxes(A, -1, -2)


def polar_rotation(F: jnp.ndarray, iters: int = POLAR_ITERS) -> jnp.ndarray:
    """Orthogonal polar factor R of F by Newton iteration R <- (R + R^-T) / 2.

    Deviation (a): this replaces U V^T from svd3. R(F) is analytic for det F > 0,
    so unlike the SVD route it is differentiable at F = I.
    """
    R = F
    for _ in range(iters):
        R = 0.5 * (R + _T(inv3(R)))
    return R


# ---------------------------------------------------------------------------
# SVD with a broadened JVP (deviation b)
# ---------------------------------------------------------------------------

def _svd_raw(A):
    U, s, Vt = jnp.linalg.svd(A)
    # ssvd sign convention: keep U and V proper rotations and push a reflection
    # into the last singular value, which is what warp's svd3 does. The return
    # maps read log|s|, so the sign only matters for inverted particles.
    sign = jnp.where(det3(U) * det3(Vt) < 0.0, -1.0, 1.0).astype(A.dtype)
    U = U * jnp.stack([jnp.ones_like(sign), jnp.ones_like(sign), sign], -1)[..., None, :]
    s = s * jnp.stack([jnp.ones_like(sign), jnp.ones_like(sign), sign], -1)
    return U, s, Vt


def _safe_inv(x: jnp.ndarray) -> jnp.ndarray:
    return x / (x * x + SVD_JVP_EPS * SVD_JVP_EPS)


@jax.custom_jvp
def svd3(A):
    return _svd_raw(A)


@svd3.defjvp
def _svd3_jvp(primals, tangents):
    (A,), (dA,) = primals, tangents
    U, s, Vt = _svd_raw(A)
    V = _T(Vt)
    dP = _T(U) @ dA @ V
    s2 = s * s
    denom = s2[..., None, :] - s2[..., :, None]          # [i, j] = s_j^2 - s_i^2
    Fm = _safe_inv(denom) * (1.0 - jnp.eye(3, dtype=A.dtype))
    S = s[..., None, :] * jnp.eye(3, dtype=A.dtype)
    ds = jnp.diagonal(dP, axis1=-2, axis2=-1)
    dU = U @ (Fm * (dP @ S + S @ _T(dP)))
    dV = V @ (Fm * (S @ dP + _T(dP) @ S))
    return (U, s, Vt), (dU, ds, _T(dV))


# ---------------------------------------------------------------------------
# constitutive models, transcribed from mpm_utils.py
# ---------------------------------------------------------------------------

class Theta(NamedTuple):
    """Traced material parameters. Unused entries are ignored per material."""
    mu: jnp.ndarray = jnp.float32(0.0)
    lam: jnp.ndarray = jnp.float32(0.0)
    yield_stress: jnp.ndarray = jnp.float32(0.0)
    alpha: jnp.ndarray = jnp.float32(0.0)         # Drucker-Prager cone constant
    bulk: jnp.ndarray = jnp.float32(0.0)


def alpha_from_friction(phi_deg):
    """warp's set_parameters_dict: alpha = sqrt(2/3) 2 sin phi / (3 - sin phi)."""
    s = jnp.sin(phi_deg / 180.0 * 3.14159265)
    return jnp.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)


def mu_lam_from_E_nu(E, nu):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def _kirchoff_FCR(F, R, J, mu, lam):
    return 2.0 * mu * (F - R) @ _T(F) + lam * (J * (J - 1.0))[..., None, None] * EYE


def _jelly(F_trial, th):
    J = det3(F_trial)
    R = polar_rotation(F_trial)
    return F_trial, _kirchoff_FCR(F_trial, R, J, th.mu, th.lam)


def _water(F_trial, th):
    # stress_update_particle, mat 6: F <- J^(1/3) I, then J <- det F, and
    # kirchoff_stress_water with gamma = 1.1 (weakly compressible).
    J = det3(F_trial)
    Jc = jnp.cbrt(J)
    F = Jc[..., None, None] * EYE
    Jn = Jc * Jc * Jc
    pressure = -th.bulk * (jnp.power(Jn, -1.1) - 1.0)
    return F, (Jn * pressure)[..., None, None] * EYE


def _von_mises_return(F_trial, th):
    """von_mises_return_mapping: Hencky predictor, radial return, scalar tau_y."""
    U, s_raw, Vt = svd3(F_trial)
    s = jnp.maximum(s_raw, 0.01)                       # warp's NaN guard
    eps = jnp.log(s)
    tr = eps.sum(-1)
    mean = tr / 3.0
    eps_hat = eps - mean[..., None]
    # ||dev tau|| = 2 mu ||dev eps|| exactly for this Hencky pair, which is the
    # yield test warp writes out through tau.
    cond = 2.0 * th.mu * jnp.linalg.norm(eps_hat, axis=-1)
    n_hat = jnp.linalg.norm(eps_hat, axis=-1) + 1e-6
    dgamma = n_hat - th.yield_stress / (2.0 * th.mu)
    eps_new = eps - (dgamma / n_hat)[..., None] * eps_hat
    F_plastic = U @ (jnp.exp(eps_new)[..., None, :] * EYE) @ Vt
    yielded = (cond > th.yield_stress)[..., None, None]
    return jnp.where(yielded, F_plastic, F_trial), U, Vt


def _plasticine(F_trial, th):
    F, _, _ = _von_mises_return(F_trial, th)
    J = det3(F)
    R = polar_rotation(F)
    return F, _kirchoff_FCR(F, R, J, th.mu, th.lam)


def _sand(F_trial, th):
    """sand_return_mapping + kirchoff_stress_drucker_prager (cohesionless cone).

    The stress reads the singular triple of the RETURNED F. Reusing (U, s_new, V)
    from the return map instead of a second svd3 is exact: U diag(s_new) V^T is a
    singular value decomposition of the returned F, and the stress expression
    U center V^T F^T = U (center s_new) U^T is invariant to the residual sign and
    ordering freedom of that decomposition.
    """
    U, s_raw, Vt = svd3(F_trial)
    eps = jnp.log(jnp.maximum(jnp.abs(s_raw), 1e-14))
    tr = eps.sum(-1)
    eps_hat = eps - (tr / 3.0)[..., None]
    n_hat = jnp.linalg.norm(eps_hat, axis=-1)
    dgamma = n_hat + (3.0 * th.lam + 2.0 * th.mu) / (2.0 * th.mu) * tr * th.alpha
    n_safe = jnp.where(n_hat > 1e-20, n_hat, 1.0)
    H = eps - eps_hat * (dgamma / n_safe)[..., None]
    plastic = jnp.exp(H)                                   # dgamma > 0, tr <= 0
    elastic = jnp.exp(eps)                                 # dgamma <= 0
    apex = jnp.ones_like(elastic)                          # dgamma > 0, tr > 0
    s_new = jnp.where((dgamma > 0.0)[..., None],
                      jnp.where((tr > 0.0)[..., None], apex, plastic), elastic)
    F_rec = U @ (s_new[..., None, :] * EYE) @ Vt
    # the elastic branch returns F_trial itself in warp; keep it bitwise
    F = jnp.where((dgamma <= 0.0)[..., None, None], F_trial, F_rec)
    log_s = jnp.log(jnp.maximum(jnp.abs(s_new), 1e-14))
    center = (2.0 * th.mu * log_s + th.lam * log_s.sum(-1)[..., None]) / s_new
    stress = U @ (center[..., None, :] * EYE) @ Vt @ _T(F)
    return F, stress


_CONSTITUTIVE = {"jelly": _jelly, "water": _water, "plasticine": _plasticine,
                 "sand": _sand}


def constitutive(F_trial: jnp.ndarray, th: Theta, material: str):
    """Return-mapped F and the symmetrized Kirchhoff stress, as warp does."""
    F, stress = _CONSTITUTIVE[material](F_trial, th)
    return F, 0.5 * (stress + _T(stress))


# ---------------------------------------------------------------------------
# scene, transfers, walls
# ---------------------------------------------------------------------------

class Scene(NamedTuple):
    """Static (hashable) run configuration; matches the warp scene one to one."""
    material: str
    n_grid: int = 20
    grid_lim: float = 1.0
    bound_cells: int = 3
    dt: float = 1.0e-3
    substeps: int = 4
    n_frames: int = 125
    gravity: tuple = (0.0, 0.0, -9.8)

    @property
    def dx(self) -> float:
        return self.grid_lim / self.n_grid


def _bspline(x, inv_dx):
    xg = x * inv_dx
    base = jnp.floor(xg - 0.5)
    fx = xg - base
    wa, wb, wc = 1.5 - fx, fx - 1.0, fx - 0.5
    w = jnp.stack([0.5 * wa * wa, 0.75 - wb * wb, 0.5 * wc * wc], -1)   # [n, axis, node]
    dw = jnp.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], -1)
    return base.astype(jnp.int32), fx, w, dw


def _flat_index(bi, i, j, k, G):
    ix = jnp.clip(bi[:, 0] + i, 0, G - 1)
    iy = jnp.clip(bi[:, 1] + j, 0, G - 1)
    iz = jnp.clip(bi[:, 2] + k, 0, G - 1)
    return (ix * G + iy) * G + iz


def _wall_mask(scene: Scene) -> jnp.ndarray:
    """Per-axis node mask of the six freeslip planes, exactly warp's test.

    add_surface_collider projects the full normal component where
    dot(node - point, n) < 0, with the planes at bound_cells * dx and at
    grid_lim - bound_cells * dx.
    """
    G, dx = scene.n_grid, scene.dx
    pos = jnp.arange(G, dtype=F32) * F32(dx)
    pad = F32(scene.bound_cells * dx)
    low = (pos - pad) < 0.0
    high = -(pos - (F32(scene.grid_lim) - pad)) < 0.0
    return (low | high)                                   # (G,) shared by all axes


def substep(state, th: Theta, cloud, scene: Scene):
    x, v, C, F_trial = state
    vol0, mass = cloud["vol0"], cloud["mass"]
    G, dx = scene.n_grid, scene.dx
    inv_dx = F32(1.0 / dx)
    dt = F32(scene.dt)

    F_e, stress = constitutive(F_trial, th, scene.material)

    bi, fx, w, dw = _bspline(x, inv_dx)
    gv = jnp.zeros((G * G * G, 3), F32)
    gm = jnp.zeros((G * G * G,), F32)
    mv = mass[:, None] * v
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                dweight = jnp.stack([dw[:, 0, i] * w[:, 1, j] * w[:, 2, k],
                                     w[:, 0, i] * dw[:, 1, j] * w[:, 2, k],
                                     w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]], -1) * inv_dx
                dpos = (jnp.array([i, j, k], F32) - fx) * F32(dx)
                add = (weight[:, None] * (mv + mass[:, None]
                                          * jnp.einsum("nab,nb->na", C, dpos))
                       - dt * vol0[:, None] * jnp.einsum("nab,nb->na", stress, dweight))
                idx = _flat_index(bi, i, j, k, G)
                gv = gv.at[idx].add(add)
                gm = gm.at[idx].add(weight * mass)

    m_safe = jnp.where(gm > 1e-15, gm, 1.0)
    g = jnp.asarray(scene.gravity, F32)
    v_grid = jnp.where((gm > 1e-15)[:, None], gv / m_safe[:, None] + dt * g, 0.0)

    wall = _wall_mask(scene)
    v_grid = v_grid.reshape(G, G, G, 3)
    v_grid = v_grid.at[..., 0].set(jnp.where(wall[:, None, None], 0.0, v_grid[..., 0]))
    v_grid = v_grid.at[..., 1].set(jnp.where(wall[None, :, None], 0.0, v_grid[..., 1]))
    v_grid = v_grid.at[..., 2].set(jnp.where(wall[None, None, :], 0.0, v_grid[..., 2]))
    v_grid = v_grid.reshape(G * G * G, 3)

    new_v = jnp.zeros_like(v)
    new_C = jnp.zeros_like(C)
    new_L = jnp.zeros_like(C)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                dweight = jnp.stack([dw[:, 0, i] * w[:, 1, j] * w[:, 2, k],
                                     w[:, 0, i] * dw[:, 1, j] * w[:, 2, k],
                                     w[:, 0, i] * w[:, 1, j] * dw[:, 2, k]], -1) * inv_dx
                dpos_grid = jnp.array([i, j, k], F32) - fx      # grid units, as warp
                vi = v_grid[_flat_index(bi, i, j, k, G)]
                new_v = new_v + vi * weight[:, None]
                new_C = new_C + (vi[:, :, None] * dpos_grid[:, None, :]
                                 * (weight * inv_dx * 4.0)[:, None, None])
                new_L = new_L + vi[:, :, None] * dweight[:, None, :]

    x_new = x + dt * new_v
    F_next = (EYE + dt * new_L) @ F_e
    return (x_new, new_v, new_C, F_next)


def init_state(cloud):
    n = cloud["x0"].shape[0]
    return (cloud["x0"], cloud["v0"],
            jnp.zeros((n, 3, 3), F32),
            jnp.broadcast_to(EYE, (n, 3, 3)).astype(F32))


@functools.partial(jax.jit, static_argnums=(2, 3))
def rollout(th: Theta, cloud, scene: Scene, n_frames: int | None = None):
    """Frame-sampled positions, frame 0 first, matching the warp dump cadence."""
    nf = scene.n_frames if n_frames is None else n_frames

    def one_substep(state, _):
        return substep(state, th, cloud, scene), None

    def one_frame(state, _):
        state, _ = lax.scan(one_substep, state, None, length=scene.substeps)
        return state, state[0]

    state0 = init_state(cloud)
    _, xs = lax.scan(one_frame, state0, None, length=nf)
    return jnp.concatenate([cloud["x0"][None], xs], 0)


def position_mse(x_pred, x_truth, frames, grid_lim: float = 1.0):
    """NCLaw's metric on the selected frames: mean over particles and coordinates."""
    d = (x_pred[frames] - x_truth[frames]) / grid_lim
    return jnp.mean(d * d)


def load_truth(path, material: str, substeps: int | None = None):
    """Frame-0 cloud, truth positions and the scene of a warp grid-20 dump.

    The initial condition is taken from the dump itself, so the two engines start
    from bitwise identical particle states. ``substeps`` overrides the dump's own
    substep count (the fit runs at a fixed, prior-safe time step; see the driver).
    """
    import json

    import numpy as np

    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    ex = meta.get("extra", meta)
    frame_dt = float(d["frame_dt"])
    sub = int(ex["substeps_per_frame"]) if substeps is None else int(substeps)
    scene = Scene(material=material, n_grid=int(ex["n_grid"]),
                  grid_lim=float(ex["grid_lim"]), bound_cells=int(ex["bound_cells"]),
                  dt=frame_dt / sub, substeps=sub,
                  n_frames=int(d["x"].shape[0]) - 1,
                  gravity=tuple(float(g) for g in ex["gravity"]))
    cloud = {"x0": jnp.asarray(d["x"][0], F32), "v0": jnp.asarray(d["v"][0], F32),
             "vol0": jnp.asarray(d["volume0"], F32),
             "mass": jnp.asarray(d["mass"], F32)}
    return cloud, np.asarray(d["x"], np.float32), scene, meta


def make_loss(cloud, x_truth, scene: Scene, frames, unpack):
    """loss(q) with q the optimizer's coordinates; unpack(q) -> Theta."""
    fr = jnp.asarray(frames)
    xt = jnp.asarray(x_truth, F32)

    @jax.jit
    def loss(q):
        return position_mse(rollout(unpack(q), cloud, scene), xt, fr,
                            scene.grid_lim)

    return loss
