"""Unit tests for the squeeze power-balance identification (examples/squeeze_plate_franka.py):
the shear-rate convention and that the regression recovers a known (tau_y, eta) from
synthetic dissipation data. Locks in the identification math the verification workflow checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# examples/common.py and src/common/ share a name, so whichever is imported
# first wins for the whole session. The example needs its own; a test that ran
# earlier and imported ident (which pulls in common.conventions) would otherwise
# leave the package bound here and this import would fail. Shadow the binding for
# the duration of this one import and restore it, so the file works in any
# collection order and leaves nothing behind for later tests.
_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
_saved = {k: v for k, v in sys.modules.items()
          if k == "common" or k.startswith("common.")}
for _k in _saved:
    del sys.modules[_k]
sys.path.insert(0, _EXAMPLES)
try:
    from squeeze_plate_franka import equivalent_shear_rate, power_balance_identify
finally:
    # the example inserts its own directory on import too, so drop every copy
    sys.path[:] = [q for q in sys.path if q != _EXAMPLES]
    for _k in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_k]
    sys.modules.update(_saved)


def test_equivalent_shear_rate_simple_shear():
    # simple shear v_x = gamma*z -> L[0,2]=gamma; |gd| = sqrt(2 D:D + eps^2) returns gamma
    gamma = 2.0
    L = np.zeros((1, 3, 3)); L[0, 0, 2] = gamma
    gd = equivalent_shear_rate(L)
    assert abs(float(gd[0]) - gamma) < 1e-2  # eps=0.02 is small vs gamma=2


def test_power_balance_recovers_synthetic_law():
    # build frames whose measured dissipation is exactly tau_y*X1 + eta*X2 (P_grav=0, KE
    # constant so dKE=0); encode it through F_plate so P_plate = v_plate*F_plate = diss.
    n = 80
    X1 = np.linspace(1.0, 5.0, n)
    X2 = np.linspace(0.5, 8.0, n)
    tau_y, eta, v_plate = 150.0, 30.0, 0.08
    diss = tau_y * X1 + eta * X2
    rec = {"t": list(np.linspace(0.0, 1.0, n)), "F_plate": list(diss / v_plate),
           "P_grav": [0.0] * n, "KE": [0.0] * n, "X1": list(X1), "X2": list(X2),
           "Pvol": [0.0] * n}
    out = power_balance_identify(rec, v_plate, frame_dt=1.0 / n, t_lo=0.0, t_hi=1.0)
    assert abs(out["tau_y_hat"] - tau_y) < 1.0
    assert abs(out["eta_hat"] - eta) < 1.0
    assert out["fit_relres"] < 1e-6  # exact synthetic data -> near-perfect fit


def test_power_balance_eos_term_adds_volumetric_work():
    # with correct_eos, INT p div v is added to the dissipation -> changes the recovered law
    n = 60
    X1 = np.linspace(1.0, 4.0, n); X2 = np.linspace(0.5, 6.0, n)
    rec = {"t": list(np.linspace(0.0, 1.0, n)), "F_plate": list(np.ones(n) * 100.0),
           "P_grav": [0.0] * n, "KE": [0.0] * n, "X1": list(X1), "X2": list(X2),
           "Pvol": list(-np.ones(n) * 0.5)}
    base = power_balance_identify(rec, 0.08, 1.0 / n, 0.0, 1.0, correct_eos=False)
    eos = power_balance_identify(rec, 0.08, 1.0 / n, 0.0, 1.0, correct_eos=True)
    assert (base["tau_y_hat"], base["eta_hat"]) != (eos["tau_y_hat"], eos["eta_hat"])
