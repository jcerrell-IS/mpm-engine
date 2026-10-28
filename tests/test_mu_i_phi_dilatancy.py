"""Material 11 (compressible mu(I)-Phi(I)) relaxes to the critical state; material 9 does not.

Material 11 yields against a compaction pressure p = K(Jp_ref/J - 1)_+ whose
stress-free reference volume drifts toward J_cs(I) = 1/(1 - chi I/(I + I0)) at a
rate proportional to the plastic shear increment. The claim that makes the
material worth having is emergent, not prescribed: under genuine flow the solid
fraction Phi = Phi_init/J should follow the rate-dependent critical state
Phi_c(I) = Phi_init (1 - chi I/(I + I0)), so a denser packing at low inertial
number and a looser one at high. Material 9, the non-dilatant local mu(I), holds
Phi at Phi_init whatever I does.

A short aspect-2 column collapse on a sticky floor supplies the range of I. For
each flowing particle-frame the test measures I = |gamma_dot| d / sqrt(p/rho_s)
from L and the 3D stress trace, bins by I, and compares the median Phi per bin
against Phi_c at that bin's median I. Observed, with Phi_init = 0.6 and chi = 0.2:

  I bin          material 11                     material 9
                 Phi_med   Phi_c    deviation    Phi_med   deviation from Phi_c
  [0.00, 0.01]   0.5988    0.5971   +0.0017      0.6011    +0.0041
  [0.01, 0.03]   0.5951    0.5929   +0.0022      0.6010    +0.0080
  [0.03, 0.06]   0.5863    0.5856   +0.0008      0.6006    +0.0161
  [0.06, 0.10]   0.5784    0.5781   +0.0002      0.6003    +0.0245

Material 11 tracks Phi_c to 0.0022 across the range and ends 0.0216 below
Phi_init, which is the O(chi) density observable the TrackEUCLID gates read.
Material 9 stays within 0.0011 of Phi_init and misses Phi_c by 0.0245 in the
fastest bin.

One finding from the probe this test replaces (video2sim/sim/validate_mu_i_phi.py):
that probe scored corr(I, Phi) and expected roughly zero for material 9, but
material 9 gives -0.452 (its high-I particles sit near the free surface, where
low pressure and float32 J noise correlate with I on their own). The correlation
does not discriminate the two materials; the binned median against Phi_c does,
which is what this test asserts.

Run:  .venv/bin/python -m pytest tests/test_mu_i_phi_dilatancy.py -q
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

PHI_INIT, CHI, I0 = 0.6, 0.2, 0.3
D, RHO_S = 1.0e-3, 2650.0
MU_S, DELTA_MU = 0.38, 0.0
N_GRID, GRID_LIM = 40, 0.4
T_END, FRAME_DT = 0.3, 4.0e-3
EDGES = np.array([0.0, 0.01, 0.03, 0.06, 0.10])
MIN_PER_BIN = 200


def _phi_c(inertial: float) -> float:
    return PHI_INIT * (1.0 - CHI * inertial / (inertial + I0))


def _collapse(material: str) -> np.ndarray:
    """Flowing particle-frames of an aspect-2 column collapse, as (I, Phi) rows."""
    import torch
    import warp as wp

    wp.init()
    from warpmpm.kernels import MPM_Simulator_WARP

    dx = GRID_LIM / N_GRID
    floor = 3.0 * dx
    h = dx / 2.0
    xc = 0.5 * GRID_LIM
    width = 0.05
    xs = np.arange(xc - 0.5 * width, xc + 0.5 * width, h)
    ys = np.arange(xc - 0.025, xc + 0.025, h)
    zs = np.arange(floor + 0.5 * h, floor + 2.0 * width, h)
    pos = np.ascontiguousarray(
        np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).astype(np.float32))
    vol = np.ascontiguousarray(np.full(len(pos), h ** 3, dtype=np.float32))

    c_el = np.sqrt(9.0e5 / 1590.0)
    substeps = int(np.ceil(FRAME_DT / (0.4 * dx / c_el)))
    dt = FRAME_DT / substeps
    n_frames = round(T_END / FRAME_DT)

    s = MPM_Simulator_WARP(len(pos), device="cpu")
    s.load_initial_data_from_torch(torch.from_numpy(pos), torch.from_numpy(vol),
                                   n_grid=N_GRID, grid_lim=GRID_LIM, device="cpu")
    s.set_parameters_dict({"material": material, "E": 1.0e6, "nu": 0.3, "density": 1590.0,
                           "mu_s": MU_S, "delta_mu": DELTA_MU, "I0": I0,
                           "grain_diameter": D, "grain_density": RHO_S,
                           "phi_init": PHI_INIT, "phi_chi": CHI,
                           "g": [0.0, 0.0, -9.81]}, device="cpu")
    s.finalize_mu_lam(device="cpu")
    s.add_surface_collider((0.0, 0.0, floor), (0.0, 0.0, 1.0), "sticky")

    rows, step = [], 0
    for frame in range(n_frames + 1):
        v = s.export_particle_v_to_torch().numpy()
        L = s.export_particle_L_to_torch().numpy().reshape(-1, 3, 3)
        F = s.export_particle_F_to_torch().numpy().reshape(-1, 3, 3)
        tau = s.export_particle_stress_to_torch().numpy().reshape(-1, 3, 3)
        J = np.clip(np.abs(np.linalg.det(F)), 1e-6, None)
        cauchy = tau / J[:, None, None]
        p = -np.trace(cauchy, axis1=1, axis2=2) / 3.0
        strain_rate = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        gamma_dot = np.sqrt(2.0 * np.sum(strain_rate ** 2, axis=(1, 2)) + 1e-12)
        inertial = gamma_dot * D / np.sqrt(np.maximum(p, 1e-6) / RHO_S)
        # gate: moving, pressure-bearing, and past the gate-release transient
        flowing = ((np.linalg.norm(v, axis=1) > 0.02) & (p > 50.0)
                   & (frame * FRAME_DT > 0.05))
        if flowing.any():
            rows.append(np.column_stack([inertial[flowing], PHI_INIT / J[flowing]]))
        if frame == n_frames:
            break
        for _ in range(substeps):
            s.p2g2p(step, dt, device="cpu")
            step += 1
    assert rows, f"{material}: no flowing particle-frames, the column did not collapse"
    return np.vstack(rows)


def _binned(rows: np.ndarray) -> list[tuple[float, float, int]]:
    """Per I bin: median I, median Phi, sample count, for bins with enough samples."""
    inertial, phi = rows[:, 0], rows[:, 1]
    out = []
    for lo, hi in pairwise(EDGES):
        m = (inertial >= lo) & (inertial < hi)
        if m.sum() >= MIN_PER_BIN:
            out.append((float(np.median(inertial[m])), float(np.median(phi[m])), int(m.sum())))
    return out


@pytest.fixture(scope="module")
def dilatant() -> list[tuple[float, float, int]]:
    return _binned(_collapse("mu_i_phi"))


@pytest.fixture(scope="module")
def non_dilatant() -> list[tuple[float, float, int]]:
    return _binned(_collapse("mu_i_sand"))


@pytest.mark.slow
def test_the_collapse_covers_a_range_of_inertial_numbers(dilatant, non_dilatant):
    for name, bins in (("mu_i_phi", dilatant), ("mu_i_sand", non_dilatant)):
        assert len(bins) == len(EDGES) - 1, (
            f"{name}: only {len(bins)} of {len(EDGES) - 1} I bins reached "
            f"{MIN_PER_BIN} samples, so the comparison is not covered")


@pytest.mark.slow
def test_material_11_follows_the_critical_state(dilatant):
    worst = max(abs(phi - _phi_c(i_med)) for i_med, phi, _ in dilatant)
    assert worst <= 5.0e-3, (
        f"material 11's solid fraction misses Phi_c(I) by {worst:.4f}, so the "
        "reference volume is not relaxing to the critical state")


@pytest.mark.slow
def test_material_11_dilates_at_the_highest_inertial_number(dilatant):
    _, phi_fast, _ = dilatant[-1]
    assert phi_fast <= PHI_INIT - 1.0e-2, (
        f"material 11 held Phi at {phi_fast:.4f} in the fastest bin, within "
        f"{PHI_INIT - phi_fast:.4f} of Phi_init, so nothing dilated")


@pytest.mark.slow
def test_material_9_holds_the_initial_solid_fraction(non_dilatant):
    worst = max(abs(phi - PHI_INIT) for _, phi, _ in non_dilatant)
    assert worst <= 5.0e-3, (
        f"material 9's solid fraction drifted {worst:.4f} from Phi_init; the "
        "non-dilatant mu(I) return must preserve the volume")


@pytest.mark.slow
def test_material_9_does_not_track_the_critical_state(non_dilatant):
    i_med, phi_fast, _ = non_dilatant[-1]
    gap = phi_fast - _phi_c(i_med)
    assert gap >= 1.5e-2, (
        f"material 9 sits only {gap:.4f} above Phi_c({i_med:.3f}), so this "
        "collapse cannot tell the dilatant material from the non-dilatant one")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "slow"]))
