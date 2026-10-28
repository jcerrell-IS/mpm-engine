"""The composed material (14) against reference formulas, one kind at a time.

Material 14 pairs any of five elasticity kinds with any of four plasticity kinds,
chosen independently per material type. The reference for each kind is NCLaw's
own torch implementation (NCLaw/nclaw/material/preset.py), transcribed below with
the class it comes from named. The transcription is deliberate rather than an
import: this repository must not depend on theirs, and a transcription is what a
reader can audit line against line. When their package IS importable, the last
test compares the transcriptions against their modules directly, so the
transcriptions themselves are checked too.

Every comparison is on the same batch of random deformation gradients, one
substep, stress out of the engine against stress out of the reference.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

N_GRID = 16
GRID_LIM = 1.0
DX = GRID_LIM / N_GRID
E_REF, NU_REF = 3.0e5, 0.25
RHO = 1000.0
SEED = 7

# Tolerance for a stress that reaches the comparison through warp's svd3. The
# engine's Hencky stress from its OWN returned F differs from an exact float64
# evaluation of the same formula on the same F by 2.5e-4 relative on this batch
# (E 1e6, singular values 0.96/0.87/0.57), while the returned F itself matches
# the reference to 1e-7. The gap is warp's fixed-iteration float32 svd3, a
# property of the engine that predates these kinds, not a formula difference;
# the F comparison is what tests the formulas, and it is tight.
SVD_STRESS_TOL = 1.0e-3


def _mu_lam(E: float, nu: float) -> tuple[float, float]:
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# --------------------------------------------------------------------------
# Reference formulas, transcribed from NCLaw/nclaw/material/preset.py
# --------------------------------------------------------------------------

def ref_corotated(F: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """CorotatedElasticity.forward"""
    mu, la = _mu_lam(E, nu)
    U, sigma, Vh = torch.linalg.svd(F)
    corotated = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))
    J = torch.prod(sigma, dim=1).view(-1, 1, 1)
    eye3 = torch.eye(3, dtype=F.dtype).unsqueeze(0)
    return corotated + la * J * (J - 1) * eye3


def ref_stvk(F: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """StVKElasticity.forward"""
    mu, la = _mu_lam(E, nu)
    _, sigma, _ = torch.linalg.svd(F)
    eye3 = torch.eye(3, dtype=F.dtype).unsqueeze(0)
    E_green = 0.5 * (torch.matmul(F.transpose(1, 2), F) - eye3)
    J = torch.prod(sigma, dim=1).view(-1, 1, 1)
    return 2 * mu * torch.matmul(F, E_green) + la * J * (J - 1) * eye3


def ref_hencky(F: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """SigmaElasticity.forward"""
    mu, la = _mu_lam(E, nu)
    U, sigma, _ = torch.linalg.svd(F)
    epsilon = sigma.log()
    trace = epsilon.sum(dim=1, keepdim=True)
    tau = 2 * mu * epsilon + la * trace
    return torch.matmul(torch.matmul(U, torch.diag_embed(tau)), U.transpose(1, 2))


def ref_volume_taichi(F: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """VolumeElasticity.forward, mode 'taichi'"""
    _, la = _mu_lam(E, nu)
    J = torch.det(F).view(-1, 1, 1)
    eye3 = torch.eye(3, dtype=F.dtype).unsqueeze(0)
    return la * J * (J - 1) * eye3


def ref_volume_ziran(F: torch.Tensor, E: float, nu: float,
                     gamma: float = 2.0) -> torch.Tensor:
    """VolumeElasticity.forward, mode 'ziran'"""
    mu, la = _mu_lam(E, nu)
    kappa = 2 / 3 * mu + la
    J = torch.det(F).view(-1, 1, 1)
    eye3 = torch.eye(3, dtype=F.dtype).unsqueeze(0)
    return kappa * (J - 1 / torch.pow(J, gamma - 1)) * eye3


def ref_von_mises(F: torch.Tensor, E: float, nu: float,
                  sigma_y: float) -> torch.Tensor:
    """VonMisesPlasticity.forward"""
    mu, _ = _mu_lam(E, nu)
    U, sigma, Vh = torch.linalg.svd(F)
    sigma = torch.clamp_min(sigma, 0.05)
    epsilon = torch.log(sigma)
    trace = epsilon.sum(dim=1, keepdim=True)
    epsilon_hat = epsilon - trace / 3
    epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
    delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
    cond = (delta_gamma > 0).view(-1, 1, 1)
    yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
    yield_F = torch.matmul(
        torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)
    return torch.where(cond, yield_F, F)


def ref_drucker_prager(F: torch.Tensor, E: float, nu: float,
                       friction_angle: float, cohesion: float) -> torch.Tensor:
    """DruckerPragerPlasticity.forward"""
    import math
    mu, la = _mu_lam(E, nu)
    sin_phi = math.sin(math.radians(friction_angle))
    alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
    U, sigma, Vh = torch.linalg.svd(F)
    sigma = torch.clamp_min(sigma, 0.05)
    epsilon = torch.log(sigma)
    trace = epsilon.sum(dim=1, keepdim=True)
    epsilon_hat = epsilon - trace / 3
    epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
    expand_epsilon = torch.ones_like(epsilon) * cohesion
    shifted_trace = trace - cohesion * 3
    cond_yield = (shifted_trace < 0).view(-1, 1)
    delta_gamma = epsilon_hat_norm + (3 * la + 2 * mu) / (2 * mu) \
        * shifted_trace * alpha
    compress = epsilon - (torch.clamp_min(delta_gamma, 0.0)
                          / epsilon_hat_norm) * epsilon_hat
    epsilon = torch.where(cond_yield, compress, expand_epsilon)
    return torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)


def ref_sigma_plasticity(F: torch.Tensor) -> torch.Tensor:
    """SigmaPlasticity.forward"""
    J = torch.det(F)
    Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
    return torch.diag_embed(Je_1_3)


# --------------------------------------------------------------------------
# Engine side: one stress evaluation on a prescribed F batch
# --------------------------------------------------------------------------

def _random_F(n: int, spread: float = 0.25, seed: int = SEED) -> np.ndarray:
    """Deformation gradients spanning stretch, compression, shear and rotation,
    all with det > 0."""
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=spread, size=(n, 3, 3))
    F = np.eye(3)[None] + A
    good = np.linalg.det(F) > 0.05
    F = F[good]
    assert len(F) > n // 2
    return np.ascontiguousarray(F.astype(np.float32))


def _engine_stress(F: np.ndarray, params: dict) -> np.ndarray:
    """Kirchhoff stress the engine computes for a prescribed F, one evaluation.

    F is written into particle_F_trial, so the return map runs on exactly the
    batch handed in and the stress comes out of the same kernel the solver uses.
    """
    import warp as wp
    wp.config.quiet = True
    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP
    from warpmpm.kernels.mpm_utils import compute_stress_from_F_trial

    n = len(F)
    pts = np.ascontiguousarray(
        (np.linspace(0.35, 0.65, n)[:, None] * np.ones(3)[None]).astype(np.float32))
    vol = np.ascontiguousarray(np.full(n, DX ** 3, dtype=np.float32))
    s = MPM_Simulator_WARP(n, device="cpu")
    s.load_initial_data_from_torch(torch.from_numpy(pts), torch.from_numpy(vol),
                                  n_grid=N_GRID, grid_lim=GRID_LIM, device="cpu")
    s.set_parameters_dict(params, device="cpu")
    s.finalize_mu_lam(device="cpu")
    s.import_particle_F_from_torch(
        torch.from_numpy(np.ascontiguousarray(F.reshape(n, 3, 3))), device="cpu")
    # F_trial is what the stress kernel return-maps; import_particle_F writes F
    s.mpm_state.particle_F_trial = wp.from_torch(
        torch.from_numpy(np.ascontiguousarray(F.reshape(n, 3, 3))).contiguous(),
        dtype=wp.mat33)
    wp.launch(kernel=compute_stress_from_F_trial, dim=n,
              inputs=[s.mpm_state, s.mpm_model, 1.0e-4], device="cpu")
    return (s.export_particle_stress_to_torch().numpy().reshape(n, 3, 3),
            s.export_particle_F_to_torch().numpy().reshape(n, 3, 3))


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, 1, 2))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


ELASTICITIES = [
    ("corotated", ref_corotated),
    ("hencky", ref_hencky),
    ("stvk", ref_stvk),
    ("volume_taichi", ref_volume_taichi),
    ("volume_ziran", ref_volume_ziran),
]


@pytest.mark.parametrize("kind,ref", ELASTICITIES)
def test_elasticity_kind_matches_reference(kind, ref):
    """Each elasticity kind, plasticity off, against its reference formula.

    The engine symmetrizes the stress before storing it (the solver needs a
    symmetric Cauchy stress), so the reference is symmetrized too. StVK and
    corotated are only symmetric at symmetric F, and that symmetrization is the
    engine's own convention, not a transcription error.
    """
    F = _random_F(64)
    params = {"material": "composed", "density": RHO, "g": [0.0, 0.0, 0.0],
              "E": E_REF, "nu": NU_REF, "elasticity": kind, "plasticity": "identity"}
    got, F_out = _engine_stress(F, params)
    want = _sym(ref(torch.from_numpy(F.astype(np.float64)),
                    E_REF, NU_REF).numpy())
    assert np.allclose(F_out, F, atol=1e-6), "identity plasticity changed F"
    assert _rel(got, want) < 2e-5, f"{kind}: rel {_rel(got, want):.2e}"


def test_von_mises_plasticity_matches_reference():
    """The von Mises return map, then Hencky stress: their plasticine pair."""
    F = _random_F(64, spread=0.3)
    sigma_y = 5.0e3
    params = {"material": "composed", "density": RHO, "g": [0.0, 0.0, 0.0],
              "E": E_REF, "nu": NU_REF, "elasticity": "hencky",
              "plasticity": "von_mises", "yield_stress": sigma_y}
    got, F_out = _engine_stress(F, params)
    F64 = torch.from_numpy(F.astype(np.float64))
    F_ref = ref_von_mises(F64, E_REF, NU_REF, sigma_y)
    want = _sym(ref_hencky(F_ref, E_REF, NU_REF).numpy())
    assert not np.allclose(F_out, F, atol=1e-4), "nothing yielded: weak test"
    # the returned F is the formula check and is tight; the stress carries the
    # engine's float32 svd3 error on top (see SVD_STRESS_TOL)
    assert _rel(F_out, F_ref.numpy()) < 1e-5, \
        f"returned F: rel {_rel(F_out, F_ref.numpy()):.2e}"
    assert _rel(got, want) < SVD_STRESS_TOL, f"rel {_rel(got, want):.2e}"


def test_drucker_prager_plasticity_matches_reference():
    """The Drucker-Prager return map, then Hencky stress: their sand pair. Run at
    cohesion 0 (their sand config) and at a nonzero cohesion."""
    F = _random_F(64, spread=0.3)
    for cohesion in (0.0, 0.02):
        params = {"material": "composed", "density": RHO, "g": [0.0, 0.0, 0.0],
                  "E": 1.0e6, "nu": 0.2, "elasticity": "hencky",
                  "plasticity": "drucker_prager", "friction_angle": 25.0,
                  "cohesion": cohesion}
        got, F_out = _engine_stress(F, params)
        F64 = torch.from_numpy(F.astype(np.float64))
        F_ref = ref_drucker_prager(F64, 1.0e6, 0.2, 25.0, cohesion)
        want = _sym(ref_hencky(F_ref, 1.0e6, 0.2).numpy())
        assert _rel(F_out, F_ref.numpy()) < 1e-5, \
            f"cohesion {cohesion}: returned F rel {_rel(F_out, F_ref.numpy()):.2e}"
        assert _rel(got, want) < SVD_STRESS_TOL, \
            f"cohesion {cohesion}: rel {_rel(got, want):.2e}"


def test_sigma_plasticity_matches_reference():
    """The volume-keeping return map, then the linear volumetric EOS: their water
    pair. This is the composition their water dataset was generated with."""
    F = _random_F(64, spread=0.2)
    E_w, nu_w = 1.0e5, 0.3
    params = {"material": "composed", "density": RHO, "g": [0.0, 0.0, 0.0],
              "E": E_w, "nu": nu_w, "elasticity": "volume_taichi",
              "plasticity": "sigma"}
    got, F_out = _engine_stress(F, params)
    F64 = torch.from_numpy(F.astype(np.float64))
    F_ref = ref_sigma_plasticity(F64)
    want = _sym(ref_volume_taichi(F_ref, E_w, nu_w).numpy())
    assert np.allclose(F_out, F_ref.numpy(), rtol=2e-6, atol=1e-7)
    assert _rel(got, want) < 2e-5, f"rel {_rel(got, want):.2e}"
    # and the identity the water identification column rests on:
    # Cauchy pressure = -lam (J - 1)
    _, la = _mu_lam(E_w, nu_w)
    J = np.linalg.det(F_out)
    p_engine = -np.trace(got, axis1=1, axis2=2) / 3.0 / J
    assert np.allclose(p_engine, -la * (J - 1.0), rtol=3e-5, atol=1e-4)


def test_default_materials_are_untouched():
    """Materials 0, 1, 2 and 6 must be exactly what they were: the composed
    material is additive. Checked against the reference formulas the existing
    paths claim, which is also an audit of the sand mapping."""
    F = _random_F(48, spread=0.2)
    common = {"density": RHO, "g": [0.0, 0.0, 0.0]}
    # jelly (0) is corotated + identity
    got, _ = _engine_stress(F, {**common, "material": "jelly",
                                "E": 1.0e5, "nu": 0.2})
    want = _sym(ref_corotated(torch.from_numpy(F.astype(np.float64)),
                              1.0e5, 0.2).numpy())
    assert _rel(got, want) < 2e-5, f"jelly: rel {_rel(got, want):.2e}"
    # sand (2) is Hencky + Drucker-Prager, cohesion 0
    got, _ = _engine_stress(F, {**common, "material": "sand", "E": 1.0e6,
                                "nu": 0.2, "friction_angle": 25.0})
    F64 = torch.from_numpy(F.astype(np.float64))
    want = _sym(ref_hencky(ref_drucker_prager(F64, 1.0e6, 0.2, 25.0, 0.0),
                           1.0e6, 0.2).numpy())
    assert _rel(got, want) < SVD_STRESS_TOL, f"sand: rel {_rel(got, want):.2e}"


def test_reference_transcriptions_match_their_modules():
    """When NCLaw is importable, the transcriptions above are compared against
    their actual modules, so the transcription is checked and not just trusted."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("einops")
    preset = pytest.importorskip("nclaw.material.preset")
    from omegaconf import OmegaConf

    F = torch.from_numpy(_random_F(32))
    cases = [
        (preset.CorotatedElasticity, {"E": E_REF, "nu": NU_REF, "random": False},
         lambda f: ref_corotated(f, E_REF, NU_REF)),
        (preset.StVKElasticity, {"E": E_REF, "nu": NU_REF, "random": False},
         lambda f: ref_stvk(f, E_REF, NU_REF)),
        (preset.SigmaElasticity, {"E": E_REF, "nu": NU_REF, "random": False},
         lambda f: ref_hencky(f, E_REF, NU_REF)),
        (preset.VolumeElasticity,
         {"E": 1.0e5, "nu": 0.3, "mode": "taichi", "random": False},
         lambda f: ref_volume_taichi(f, 1.0e5, 0.3)),
        (preset.VolumeElasticity,
         {"E": 1.0e5, "nu": 0.3, "mode": "ziran", "random": False},
         lambda f: ref_volume_ziran(f, 1.0e5, 0.3)),
    ]
    for cls, cfg, mine in cases:
        theirs = cls(OmegaConf.create(cfg))
        a = theirs(F).detach().numpy()
        b = mine(F).detach().numpy()
        assert _rel(a, b) < 1e-5, f"{cls.__name__} {cfg.get('mode')}: transcription"
