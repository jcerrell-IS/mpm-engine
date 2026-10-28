"""Lock the tabulated mu(I) material (fork id 13) to the Drucker-Prager kernel.

A recovered mu(I) CURVE is re-simulated through material 13 and scored against a
truth trajectory that material 2 (Drucker-Prager sand) generated, which only
means something if the two return maps agree when the curve is the constant the
Drucker-Prager cone represents. This measures that agreement instead of assuming
it, on a small column collapse: gravity only, a slip floor, and a runout that
friction alone sets.

Why they should agree, read off the kernels. Material 2 projects the Hencky
deviator to ||dev eps|| = -(3 lam + 2 mu) / (2 mu) tr(eps) alpha, which in stress
is tau_bar = 3 alpha p / sqrt(2) = mu_c p with mu_c = 3 alpha / sqrt(2), the
constant suite.friction_to_mu returns. Material 13 bisects
tau_bar_K - G dt gd - mu(I) p_K to zero and therefore lands on
tau_bar = mu(I) p_K, the same surface when mu(I) is that constant. The stress
branches coincide too: kirchoff_stress_drucker_prager forms
U diag(2 mu log sig + lam tr) V^T F^T, which for F = U diag(sig) V^T is exactly
kirchoff_stress_hencky.

The negative control carries as much weight as the match: halving the tabulated
friction must change the runout, or the test would pass on a material that never
reads its table.

The second pairing here is material 13 against material 9, the PARAMETRIC mu(I)
(Jop-Forterre-Pouliquen), fed the same curve on the same log10 I grid. That is
the pairing the re-simulation of a recovered mu(I) curve actually relies on, and
the two kernels differ only in where mu comes from: material 9 evaluates
mu_s + delta_mu I/(I + I0) inside the bisection, material 13 interpolates the
table. Migrated from video2sim/sim/check_tabulated_mu_i.py, which this replaces;
that probe ran the same comparison on a 48-cell grid with 10944 particles and
measured relative L2 6.1e-07, against 1.5e-06 on this smaller scene.
"""
from __future__ import annotations

import numpy as np
import pytest

FRICTION_DEG = 25.0
STEPS = 1500
DT = 2.0e-4
N_GRID = 24

# the parametric mu(I) curve both materials are fed in the second pairing
POULIQUEN = {"mu_s": 0.38, "delta_mu": 0.26, "I0": 0.30}
GRAIN = {"grain_diameter": 1.0e-3, "grain_density": 2650.0}
SMIN, SMAX, N_TABLE = -4.0, 0.0, 256


def _collapse(material: str, extra: dict) -> np.ndarray:
    """Final particle positions of a small column collapse on a slip floor."""
    import warp as wp

    wp.init()
    import torch

    from warpmpm.kernels import MPM_Simulator_WARP

    grid_lim = 1.0
    dx = grid_lim / N_GRID
    h = dx / 2.0
    axy = np.arange(-0.05, 0.05 + 0.5 * h, h)
    az = np.arange(0.0, 0.25 + 0.5 * h, h)
    pts = np.stack(np.meshgrid(axy, axy, az, indexing="ij"), -1).reshape(-1, 3)
    pts[:, 0] += 0.5
    pts[:, 1] += 0.5
    pts[:, 2] += 3.0 * dx + 0.01
    pts = np.ascontiguousarray(pts.astype(np.float32))
    vol0 = np.full(len(pts), h ** 3, dtype=np.float32)

    s = MPM_Simulator_WARP(len(pts), device="cpu")
    s.load_initial_data_from_torch(
        torch.from_numpy(pts), torch.from_numpy(vol0),
        n_grid=N_GRID, grid_lim=grid_lim, device="cpu")
    kw = {"material": material, "density": 1000.0, "E": 1.0e6, "nu": 0.2,
          "g": [0.0, 0.0, -9.8]}
    kw.update(extra)
    s.set_parameters_dict(kw, device="cpu")
    s.finalize_mu_lam(device="cpu")
    s.add_surface_collider((0.0, 0.0, 3.0 * dx), (0.0, 0.0, 1.0), "slip")
    for step in range(STEPS):
        s.p2g2p(step, DT, device="cpu")
    return s.export_particle_x_to_torch().cpu().numpy()


def _flat_table(mu: float) -> dict:
    """A constant mu(I) table on the dump-schema grid, log10 I in [-4, 0]."""
    return {"eta_table": np.full(256, float(mu)).tolist(),
            "eta_table_smin": -4.0, "eta_table_smax": 0.0,
            "grain_diameter": 1.0e-3, "grain_density": 1000.0}


def _runout(x: np.ndarray) -> float:
    return float(np.abs(x[:, :2] - 0.5).max())


@pytest.fixture(scope="module")
def drucker_prager() -> np.ndarray:
    return _collapse("sand", {"friction_angle": FRICTION_DEG})


def test_constant_table_matches_drucker_prager(drucker_prager):
    from experiments.nclaw.suite import friction_to_mu

    mu_c = friction_to_mu(FRICTION_DEG)
    x_tab = _collapse("tabulated_mu_i", _flat_table(mu_c))
    assert np.isfinite(x_tab).all() and np.isfinite(drucker_prager).all()
    spread = _runout(drucker_prager)
    assert spread > 0.1, f"the column barely collapsed (runout {spread:.3f} m)"
    gap = float(np.abs(drucker_prager - x_tab).max())
    assert gap < 1.0e-4, (
        f"tabulated mu(I) at the cone constant mu = {mu_c:.4f} drifts from "
        f"Drucker-Prager by {gap:.3e} m over a {spread:.3f} m runout")


def test_a_softer_table_changes_the_runout(drucker_prager):
    from experiments.nclaw.suite import friction_to_mu

    x_soft = _collapse("tabulated_mu_i", _flat_table(0.5 * friction_to_mu(FRICTION_DEG)))
    assert _runout(x_soft) > _runout(drucker_prager) + 0.01, (
        "halving the tabulated friction did not spread the column further, so "
        "the table is not being read"
    )


def _pouliquen_table() -> dict:
    """The parametric mu(I) curve sampled on the dump-schema grid, log10 I in [-4, 0]."""
    inertial = 10.0 ** np.linspace(SMIN, SMAX, N_TABLE)
    mu = POULIQUEN["mu_s"] + POULIQUEN["delta_mu"] * inertial / (inertial + POULIQUEN["I0"])
    return {"eta_table": mu.tolist(), "eta_table_smin": SMIN, "eta_table_smax": SMAX, **GRAIN}


@pytest.fixture(scope="module")
def parametric_mu_i() -> np.ndarray:
    return _collapse("mu_i_sand", {**POULIQUEN, **GRAIN})


def test_pouliquen_table_matches_the_parametric_mu_i(parametric_mu_i):
    x_tab = _collapse("tabulated_mu_i", _pouliquen_table())
    assert np.isfinite(x_tab).all() and np.isfinite(parametric_mu_i).all()
    spread = _runout(parametric_mu_i)
    assert spread > 0.1, f"the column barely collapsed (runout {spread:.3f} m)"
    rel = float(np.linalg.norm(x_tab - parametric_mu_i)
                / max(np.linalg.norm(parametric_mu_i), 1e-12))
    assert rel < 5.0e-3, (
        f"tabulated mu(I) drifts from the parametric kernel on the same curve: "
        f"relative L2 {rel:.2e} over a {spread:.3f} m runout")


def test_a_wrong_constant_table_differs_from_the_parametric_curve(parametric_mu_i):
    # not vacuous: a low constant mu spreads the column centimetres further.
    # Position relative L2 is dominated by the absolute coordinate (~0.5 m), so
    # this one is scored as an RMS displacement in millimetres.
    x_const = _collapse("tabulated_mu_i", {"eta_table": np.full(N_TABLE, 0.15).tolist(),
                                           "eta_table_smin": SMIN, "eta_table_smax": SMAX,
                                           **GRAIN})
    rms_mm = float(np.sqrt(((x_const - parametric_mu_i) ** 2).sum(-1).mean())) * 1e3
    assert rms_mm > 1.0, (
        f"a constant mu = 0.15 table moved the column only {rms_mm:.2f} mm from "
        "the Pouliquen curve, so the table is not being read")
