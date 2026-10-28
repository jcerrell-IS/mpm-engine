"""The mu(I) return mapping (material 9) against an independent numpy reference.

This is the constitutive-correctness counterpart of the G0 gate: it does not
trust the warp kernel, it cross-checks it against a from-scratch numpy
transcription and against the physical yield condition the identifier assumes.
The transcription is deliberate rather than an import, so a reader can audit it
line against line; the kernel it mirrors is
warpmpm.kernels.mpm_utils.mu_i_return_mapping.

A batch of 400 random F_trial (biased toward compaction, so most samples land in
the meaningful tr(eps) < 0 branch) goes through the production stress path,
compute_stress_from_F_trial with material mu_i_sand, so the material-id dispatch
and the Hencky stress branch are exercised too, not just the return map.

What is measured, with the numbers this batch produces:

  engine against reference, relative to max|tau| = 4.39e5 Pa
      median 8.2e-07, p99 6.8e-06, max 6.7e-04
  plastic states on the yield surface tau_bar = mu(I) p, on the reference
      worst relative 2.0e-07 over 186 samples
  elastic states at or below the static surface tau_bar <= mu_s p, on the engine
      worst ratio 0.9977 over 177 samples
  the return is non dilatant (returned Cauchy pressure equals the elastic
  predictor pressure), on the reference
      worst relative 2.5e-07
  cohesionless expansion returns zero stress, on the engine
      1.89 Pa against a 4.39e5 Pa compressive scale, ratio 4.3e-06

The yield-surface and pressure claims are tight on the reference and loose on
the engine (1.3e-03 and 2.1e-03 respectively) for one reason: warp's
fixed-iteration float32 svd3, the same effect test_composed_material.py
documents. Pinning the engine to the reference (first test) and the reference to
the physics (the rest) keeps every tolerance at the value it can actually hold.

Migrated from video2sim/sim/verify_mu_i.py, which this replaces.

Run:  .venv/bin/python -m pytest tests/test_mu_i_return_map.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

G_MOD, LAM = 4.0e5, 6.0e5
# E and nu chosen so finalize_mu_lam produces exactly (G_MOD, LAM) above
E, NU = 1.04e6, 0.3
MU_S, DELTA_MU, I0 = 0.38, 0.26, 0.3
D, RHO_S = 1.0e-3, 2650.0
DT = 5.0e-5
N_SAMPLES, SEED = 400, 0
N_GRID, GRID_LIM = 16, 1.0


# --------------------------------------------------------------------------
# Independent numpy reference of the same math, transcribed from
# warpmpm/kernels/mpm_utils.py: mu_i_return_mapping and kirchoff_stress_hencky
# --------------------------------------------------------------------------

def _reference(F_trial: np.ndarray) -> tuple[np.ndarray, dict]:
    """Kirchhoff stress and regime for one F_trial, in float64."""
    U, s, _ = np.linalg.svd(F_trial)
    sig = np.maximum(np.abs(s), 1e-6)
    eps = np.log(sig)
    tr_eps = eps.sum()

    if tr_eps >= 0.0:
        # cohesionless free separation: F_el = U V^T, so eps = 0 and tau = 0
        return np.zeros((3, 3)), {"regime": "expansion"}

    J = float(np.prod(sig))
    mean_eps = tr_eps / 3.0
    dev = eps - mean_eps
    p_K = -(2.0 * G_MOD / 3.0 + LAM) * tr_eps          # > 0 in compression
    tau_bar_K = 2.0 * G_MOD * np.linalg.norm(dev) / np.sqrt(2.0)
    p_C = p_K / J

    if tau_bar_K <= MU_S * p_K:
        eps_new, regime, gdp = eps, "elastic", 0.0
    else:
        # bisect g(gdp) = tau_bar_K - G dt gdp - mu(I) p_K, strictly decreasing
        I_coef = D * np.sqrt(RHO_S / p_C)
        lo, hi = 0.0, tau_bar_K / (G_MOD * DT)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            I_mid = mid * I_coef
            mu_mid = MU_S + DELTA_MU * I_mid / (I_mid + I0)
            if tau_bar_K - G_MOD * DT * mid - mu_mid * p_K > 0.0:
                lo = mid
            else:
                hi = mid
        gdp = 0.5 * (lo + hi)
        scale = (tau_bar_K - G_MOD * DT * gdp) / max(tau_bar_K, 1e-20)
        eps_new, regime = dev * scale + mean_eps, "plastic"

    tau_diag = 2.0 * G_MOD * eps_new + LAM * eps_new.sum()
    tau = U @ np.diag(tau_diag) @ U.T
    return tau, {"regime": regime, "gdp": gdp, "p_C": p_C, "J": J}


def _random_F_trials(n: int, rng: np.random.Generator) -> np.ndarray:
    """Compression plus deviatoric shear, biased toward compaction."""
    F = np.stack([rng.uniform(0.90, 0.999) * np.eye(3) + rng.uniform(-0.08, 0.08, (3, 3))
                  for _ in range(n)])
    return np.ascontiguousarray(F.astype(np.float32))


def _engine(F_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kirchhoff stress and returned F out of the production stress kernel."""
    import torch
    import warp as wp

    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP
    from warpmpm.kernels.mpm_utils import compute_stress_from_F_trial

    n = len(F_trials)
    dx = GRID_LIM / N_GRID
    pts = np.ascontiguousarray(
        (np.linspace(0.35, 0.65, n)[:, None] * np.ones(3)[None]).astype(np.float32))
    vol = np.ascontiguousarray(np.full(n, dx ** 3, dtype=np.float32))

    s = MPM_Simulator_WARP(n, device="cpu")
    s.load_initial_data_from_torch(torch.from_numpy(pts), torch.from_numpy(vol),
                                   n_grid=N_GRID, grid_lim=GRID_LIM, device="cpu")
    s.set_parameters_dict({"material": "mu_i_sand", "E": E, "nu": NU, "density": 1590.0,
                           "mu_s": MU_S, "delta_mu": DELTA_MU, "I0": I0,
                           "grain_diameter": D, "grain_density": RHO_S,
                           "g": [0.0, 0.0, -9.81]}, device="cpu")
    s.finalize_mu_lam(device="cpu")
    assert abs(float(s.mpm_model.mu.numpy()[0]) - G_MOD) < 1.0
    assert abs(float(s.mpm_model.lam.numpy()[0]) - LAM) < 1.0

    F_t = torch.from_numpy(np.ascontiguousarray(F_trials.reshape(n, 3, 3))).contiguous()
    s.import_particle_F_from_torch(F_t, device="cpu")
    # F_trial is what the stress kernel return-maps; import_particle_F writes F
    s.mpm_state.particle_F_trial = wp.from_torch(F_t, dtype=wp.mat33)
    wp.launch(kernel=compute_stress_from_F_trial, dim=n,
              inputs=[s.mpm_state, s.mpm_model, DT], device="cpu")
    return (s.export_particle_stress_to_torch().numpy().reshape(n, 3, 3),
            s.export_particle_F_to_torch().numpy().reshape(n, 3, 3))


@pytest.fixture(scope="module")
def batch() -> dict:
    F_trials = _random_F_trials(N_SAMPLES, np.random.default_rng(SEED))
    pairs = [_reference(F) for F in F_trials]
    tau, F_out = _engine(F_trials)
    return {"F_trials": F_trials, "ref": np.stack([t for t, _ in pairs]),
            "info": [i for _, i in pairs], "tau": tau, "F_out": F_out}


def _cauchy_invariants(tau: np.ndarray, J: float) -> tuple[float, float]:
    """Cauchy pressure and tau_bar = |dev sigma| / sqrt(2), the identifier's convention."""
    sigma = tau / J
    p_C = -float(np.trace(sigma)) / 3.0
    return p_C, float(np.linalg.norm(sigma + p_C * np.eye(3)) / np.sqrt(2.0))


def test_engine_matches_the_numpy_reference(batch):
    """The warp kernel and the transcription agree to float32 round-off.

    Judged on the 99th percentile, which catches any systematic error, with a
    loose ceiling on the max: a sample sitting exactly on the elastic/plastic
    boundary can take the other branch under float32, and the stress is
    continuous there, so its deviation stays small.
    """
    ref, tau = batch["ref"], batch["tau"]
    scale = max(float(np.abs(ref).max()), 1.0)
    per = np.abs(ref - tau).reshape(len(ref), -1).max(axis=1) / scale
    assert np.percentile(per, 99) < 5e-5, (
        f"kernel disagrees with the reference systematically: p99 {np.percentile(per, 99):.2e}")
    assert per.max() < 2e-3, f"kernel disagrees with the reference grossly: max {per.max():.2e}"


def test_plastic_states_land_on_the_yield_surface(batch):
    """tau_bar = mu(I) p with I = gdp d sqrt(rho_s / p), the identifier's convention."""
    worst, n_plastic = 0.0, 0
    for tau, info in zip(batch["ref"], batch["info"], strict=True):
        if info["regime"] != "plastic":
            continue
        n_plastic += 1
        # the return is deviatoric, so the returned F keeps the predictor's J
        p_C, tau_bar = _cauchy_invariants(tau, info["J"])
        inertial = info["gdp"] * D * np.sqrt(RHO_S / info["p_C"])
        mu_I = MU_S + DELTA_MU * inertial / (inertial + I0)
        worst = max(worst, abs(tau_bar - mu_I * p_C) / max(mu_I * p_C, 1.0))
    assert n_plastic > 20, f"batch lacks plastic coverage (n={n_plastic})"
    assert worst < 1e-3, f"plastic states miss the yield surface: worst relative {worst:.2e}"


def test_elastic_states_stay_below_the_static_yield_surface(batch):
    """Sub-yield samples come back untouched, so tau_bar <= mu_s p on the engine."""
    worst, n_elastic = 0.0, 0
    for i, info in enumerate(batch["info"]):
        if info["regime"] != "elastic":
            continue
        n_elastic += 1
        p_C, tau_bar = _cauchy_invariants(batch["tau"][i],
                                          float(np.linalg.det(batch["F_out"][i])))
        worst = max(worst, tau_bar / max(MU_S * p_C, 1e-30))
    assert n_elastic > 5, f"batch lacks elastic coverage (n={n_elastic})"
    assert worst <= 1.0 + 1e-3, f"an elastic state sits above the static surface: {worst:.6f}"


def test_the_return_is_non_dilatant(batch):
    """The returned Cauchy pressure equals the elastic-predictor pressure.

    This is what lets the identifier read the pressure the update consumed off
    the dumped stress trace.
    """
    worst = 0.0
    for tau, info in zip(batch["ref"], batch["info"], strict=True):
        if info["regime"] == "expansion":
            continue
        p_C, _ = _cauchy_invariants(tau, info["J"])
        worst = max(worst, abs(p_C - info["p_C"]) / max(info["p_C"], 1.0))
    assert worst < 1e-4, f"the return is dilatant: pressure drifts by {worst:.2e} relative"


def test_cohesionless_expansion_returns_zero_stress(batch):
    """tr(eps) >= 0 separates freely: F_el = U V^T carries no tension memory."""
    is_exp = np.array([i["regime"] == "expansion" for i in batch["info"]])
    assert is_exp.sum() > 10, f"batch lacks expansion coverage (n={is_exp.sum()})"
    compressive = float(np.abs(batch["tau"][~is_exp]).max())
    expansion = float(np.abs(batch["tau"][is_exp]).max())
    assert expansion / compressive < 1e-4, (
        f"expanding particles carry stress: {expansion:.3e} Pa against a "
        f"{compressive:.3e} Pa compressive scale")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
