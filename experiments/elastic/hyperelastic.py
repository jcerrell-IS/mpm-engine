"""Nonlinear hyperelastic recovery: stress nonlinear in F, still linear in the coefficients.

"Nonlinear elastic" here means the stress P(F) is nonlinear in the deformation, while
the strain energy is a coefficient-linear sum of fixed basis energies, so the inverse
stays one convex solve:

    Yeoh   W = sum_i C_i (I1bar - 3)^i + (K/2)(J-1)^2,   I1bar = J^-2/3 tr(F F^T)
    tau_C1 = 2 dev(bbar)              tau_C2 = 4 (I1bar-3) dev(bbar)
    tau_C3 = 6 (I1bar-3)^2 dev(bbar)  tau_K  = J(J-1) I,   bbar = J^-2/3 F F^T
    P_k = tau_k F^-T, into the reference-configuration dynamic weak form.

    Mooney-Rivlin  W = C10 (I1bar-3) + C01 (I2bar-3) + vol
    tau_C10 = 2 dev(bbar),  tau_C01 = 2 [I1bar dev(bbar) - dev(bbar^2)]
The I1 and I2 invariants are distinct, so Mooney separates at moderate strain, whereas
the Yeoh coefficients are powers of the same invariant and stay collinear until the
strain path is wide.

A single gentle bounce visits a narrow strain path, so the higher-order coefficients
are weakly conditioned. Stacking hard multi-speed impacts widens the path and pins
them:

    Mooney, single gentle probe (strain p99 0.039, cond 713):
        C10 22.1 percent, C01 36.0 percent, K 8.4 percent
    Mooney, three hard probes (strain p99 0.510, cond 83):
        C10 15.9 percent, C01 30.6 percent, K 2.9 percent
    Same rows through the trained one-invariant function-encoder basis (K = 6):
        W1(x) reconstruction relative L2 0.226, W1(0) 6.90e4 against truth 8.0e4

Reproduced 2026-08-22 from the dumps in the tree: the single-probe numbers match
out/hyperelastic/hyperelastic_recover.json exactly.

FORWARD SIMULATION IS NOT RUNNABLE HERE. These dumps came from the JAX MPM engine
(jmpm), which is no longer in this tree, so run_drop raises with that explanation. The
recovery stages read the existing dumps in out/hyperelastic, which is where the numbers
above come from. Porting the forward Yeoh and Mooney materials to warp-mpm would make
this stage self-contained; that is not done.

Artifacts: out/elastic/hyperelastic_recover.json, fe_operator_recover.json. Dumps read
from out/hyperelastic/{mooney_gentle,mooney_hard_v*,yeoh_v*}.npz.

Run:  .venv/bin/python -m experiments.elastic hyperelastic
      .venv/bin/python -m experiments.elastic hyperelastic --which yeoh
      .venv/bin/python -m experiments.elastic hyperelastic-fe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import (
    ENGINE_ROOT,
    DropDump,
    WeakForm,
    artifact,
    find_artifact,
    lstsq,
)

# Mooney-Rivlin is the headline nonlinear case: C10 (I1bar) and C01 (I2bar) are
# distinct invariants, so they separate at moderate strain.
MOONEY_TRUTH = dict(C10=8.0e4, C01=2.0e4, Kbulk=2.0e5)
# Yeoh is the cautionary case: C1, C2, C3 are powers of the SAME (I1bar - 3).
YEOH_TRUTH = dict(C1=8.0e4, C2=2.0e4, C3=5.0e3, Kbulk=2.0e5)
KEYS = {"yeoh": ["C1", "C2", "C3", "Kbulk"], "mooney": ["C10", "C01", "Kbulk"]}
FE_WEIGHTS = ENGINE_ROOT / "fe-weights"
# the probe sets the recorded results came from
MOONEY_PROBES = ["mooney_gentle.npz", "mooney_hard_v12.npz", "mooney_hard_v20.npz",
                 "mooney_hard_v30.npz"]
YEOH_PROBES = ["yeoh_v3.npz", "yeoh_v7.npz", "yeoh_v12.npz"]


def run_drop(*_args, **_kwargs):
    """Forward simulation of the Yeoh and Mooney drops, not available in this tree.

    The dumps in out/hyperelastic were produced by the JAX MPM engine (jmpm) with
    init_state and make_recorder, a 0.16 m sphere on a 48^3 grid at frame_dt 2e-3 and
    CFL 0.25, with an initial downward velocity vz0 setting the impact strain: the
    gentle Mooney probe at -3 m/s on a slip floor, the hard probes at -12, -20 and -30
    m/s on a sticky floor with drop_gap 0.04, and the Yeoh probes at -3, -7 and -12 m/s.
    jmpm is no longer in the tree, so this cannot be re-run from here.
    """
    raise SystemExit(
        "the hyperelastic forward drops came from the JAX MPM engine (jmpm), which is "
        "no longer in this tree. The recovery stages read the existing dumps in "
        "out/hyperelastic; see this function's docstring for the settings they used."
    )


def yeoh_basis(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """The four Yeoh first-Piola columns P_k = tau_k F^-T. Aux is I1bar - 3."""
    J = np.linalg.det(F)
    b = F @ np.transpose(F, (0, 2, 1))
    Jm23 = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)
    bbar = Jm23[:, None, None] * b
    I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
    eye = np.eye(3)
    devb = bbar - (I1bar / 3.0)[:, None, None] * eye
    Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
    taus = (2.0 * devb,
            4.0 * (I1bar - 3.0)[:, None, None] * devb,
            6.0 * ((I1bar - 3.0) ** 2)[:, None, None] * devb,
            (J * (J - 1.0))[:, None, None] * eye)
    return [t @ Finvt for t in taus], I1bar - 3.0


def mooney_basis(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """The three Mooney-Rivlin first-Piola columns for theta = (C10, C01, K)."""
    J = np.linalg.det(F)
    b = F @ np.transpose(F, (0, 2, 1))
    Jm23 = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)
    bbar = Jm23[:, None, None] * b
    I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
    eye = np.eye(3)
    devb = bbar - (I1bar / 3.0)[:, None, None] * eye
    b2 = bbar @ bbar
    trb2 = b2[:, 0, 0] + b2[:, 1, 1] + b2[:, 2, 2]
    devb2 = b2 - (trb2 / 3.0)[:, None, None] * eye
    Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
    taus = (2.0 * devb,
            2.0 * (I1bar[:, None, None] * devb - devb2),
            (J * (J - 1.0))[:, None, None] * eye)
    return [t @ Finvt for t in taus], I1bar - 3.0


BASIS = {"yeoh": yeoh_basis, "mooney": mooney_basis}


def _stack(dump_paths, basis, n_modes: int = 4):
    """Stack the weak-form rows of several probes into one system."""
    if isinstance(dump_paths, (str, Path)):
        dump_paths = [dump_paths]
    As, bs, auxs, truth = [], [], [], {}
    for dp in dump_paths:
        dump = DropDump.load(dp)
        truth = truth or dump.truth
        A, b, aux = WeakForm(dump, basis, n_modes=n_modes).assemble()
        As.append(A)
        bs.append(b)
        auxs.append(aux)
    return np.vstack(As), np.concatenate(bs), np.concatenate(auxs), truth


def recover(material: str, dump_paths, log=print, tag: str = "") -> dict:
    """Convex weak-form recovery of a hyperelastic law, nonlinear in F, linear in theta.

    Pass one dump for a single probe or a list of dumps at different impact speeds; the
    multi-probe form stacks rows, so the wider strain path pins the higher-order
    coefficients.
    """
    keys = KEYS[material]
    A, b, aux, truth = _stack(dump_paths, BASIS[material])
    theta, cond = lstsq(A, b)
    theta_t = np.array([truth[k] for k in keys])
    n_probes = 1 if isinstance(dump_paths, (str, Path)) else len(dump_paths)
    log(f"[{material} recover {tag}] rows={A.shape[0]} ({n_probes} probe(s))  "
        f"cond={cond:.1e}  strain (I1bar-3) median={np.median(aux):.3f} "
        f"p99={np.percentile(aux, 99):.3f}")
    errs = {}
    for i, k in enumerate(keys):
        e = abs(theta[i] / theta_t[i] - 1) if abs(theta_t[i]) > 1e-9 else abs(theta[i])
        errs[k] = float(e)
        log(f"[{material} recover {tag}] {k:6s}: {theta[i]:.4e}  "
            f"(truth {theta_t[i]:.4e}, {100*e:.1f}% err)")
    return dict(material=material, theta=theta.tolist(), theta_true=theta_t.tolist(),
                cond=cond, errs=errs, strain_p99=float(np.percentile(aux, 99)),
                n_rows=int(A.shape[0]))


def _fe1_basis(x_grid, table):
    """One-invariant function-encoder basis as a WeakForm basis function.

    The trained phi_k(I1bar - 3) gives K deviatoric columns P_k = 2 phi_k dev(bbar) F^-T
    plus one fixed volumetric column J(J-1) F^-T. Since
    tau = (sum_k theta_k phi_k) 2 dev(bbar) is exactly 2 W1(I1bar) dev(bbar), recovering
    theta recovers the constitutive function W1.
    """
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(x_grid, table, axis=0)

    def basis(F):
        J = np.linalg.det(F)
        b = F @ np.transpose(F, (0, 2, 1))
        Jm23 = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)
        bbar = Jm23[:, None, None] * b
        I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
        eye = np.eye(3)
        devb = bbar - (I1bar / 3.0)[:, None, None] * eye
        Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
        phi = spl(np.clip(I1bar - 3.0, x_grid[0], x_grid[-1]))       # (M, K)
        Pk = [(2.0 * phi[:, k])[:, None, None] * devb @ Finvt for k in range(phi.shape[1])]
        Pk.append((J * (J - 1.0))[:, None, None] * eye @ Finvt)
        return Pk, I1bar - 3.0

    return basis, spl


def recover_fe(dump_paths, fe_npz=None, n_modes: int = 4, log=print):
    """Recover a held-out hyperelastic law THROUGH the trained one-invariant basis.

    Encoding and identification are the same operation here: the weak form is assembled
    in the learned basis, and W1(x) = sum_k theta_k phi_k(x) is compared to the truth W1
    over the realized strain range.
    """
    fe = np.load(fe_npz or FE_WEIGHTS / "hyperelastic_1inv.npz")
    xg, table = fe["x_grid"], fe["table"]
    K = table.shape[1]
    basis, spl = _fe1_basis(xg, table)
    A, b, aux, truth = _stack(dump_paths, basis, n_modes=n_modes)
    theta, cond = lstsq(A, b)
    theta_fe = theta[:K]
    xr = np.linspace(0.0, float(np.percentile(aux, 98)), 60)
    W1_rec = spl(np.clip(xr, xg[0], xg[-1])) @ theta_fe
    out = dict(K=K, cond=cond, n_rows=int(A.shape[0]),
               strain_p98=float(np.percentile(aux, 98)))
    if "C1" in truth:
        W1_true = truth["C1"] + 2 * truth["C2"] * xr + 3 * truth["C3"] * xr ** 2
        out["W1_relL2"] = float(np.linalg.norm(W1_rec - W1_true)
                                / (np.linalg.norm(W1_true) + 1e-30))
        out["W1_true_base"] = float(W1_true[0])
        out["W1_rec_base"] = float(W1_rec[0])
        log(f"[FE recover] K={K} rows={A.shape[0]} cond={cond:.1e} "
            f"strain_p98={out['strain_p98']:.2f}")
        log(f"[FE recover] W1(x) reconstruction relL2 = {out['W1_relL2']:.3f}  "
            f"(W1(0): rec {W1_rec[0]:.3e} vs truth {W1_true[0]:.3e}; "
            f"W1(top): rec {W1_rec[-1]:.3e} vs truth {W1_true[-1]:.3e})")
    return out, xr, W1_rec, truth


def _fe2_basis(x1g, x2g, table, K):
    """Two-channel (I1, I2) function-encoder basis as a WeakForm basis function.

    The learned phi1_k and phi2_k give
        P_k = 2 [phi1_k dev(bbar) + phi2_k (I1bar dev(bbar) - dev(bbar^2))] F^-T,
    so recovering theta recovers both response functions W1 and W2. Aux carries the
    flattened (I1bar - 3) so the caller can report strain coverage; the (I1, I2) query
    points are recomputed in the reconstruction step.
    """
    from scipy.interpolate import RegularGridInterpolator
    n = len(x1g)
    tab = table.reshape(n, n, 2 * K)
    interp = RegularGridInterpolator((x1g, x2g), tab, bounds_error=False, fill_value=None)

    def query(F):
        J = np.linalg.det(F)
        b = F @ np.transpose(F, (0, 2, 1))
        Jm23 = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)
        bbar = Jm23[:, None, None] * b
        I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
        b2 = bbar @ bbar
        trb2 = b2[:, 0, 0] + b2[:, 1, 1] + b2[:, 2, 2]
        I2bar = 0.5 * (I1bar ** 2 - trb2)
        eye = np.eye(3)
        devb = bbar - (I1bar / 3.0)[:, None, None] * eye
        devb2 = b2 - (trb2 / 3.0)[:, None, None] * eye
        q1 = np.clip(I1bar - 3.0, x1g[0], x1g[-1])
        q2 = np.clip(I2bar - 3.0, x2g[0], x2g[-1])
        return J, devb, devb2, I1bar, q1, q2, eye

    def basis(F):
        J, devb, devb2, I1bar, q1, q2, eye = query(F)
        Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
        phi = interp(np.stack([q1, q2], 1))                          # (M, 2K)
        base_I2 = I1bar[:, None, None] * devb - devb2
        Pk = []
        for k in range(K):
            tau_k = 2.0 * ((phi[:, k])[:, None, None] * devb
                           + (phi[:, K + k])[:, None, None] * base_I2)
            Pk.append(tau_k @ Finvt)
        Pk.append((J * (J - 1.0))[:, None, None] * eye @ Finvt)
        return Pk, np.stack([q1, q2], 1)

    return basis, interp


def recover_fe2(dump_paths, fe_npz=None, n_modes: int = 4, lam_rel: float = 0.05,
                log=print) -> dict:
    """Recover a held-out Mooney-Rivlin law through the two-channel (I1, I2) encoder.

    lam_rel is the ridge as a fraction of trace(A^T A) / K, the function-encoder prior
    taming the I1 and I2 collinearity under compression-dominated loading.
    """
    fe = np.load(fe_npz or FE_WEIGHTS / "hyperelastic_2inv.npz")
    x1g, x2g = np.unique(fe["x1"]), np.unique(fe["x2"])
    table, K = fe["table"], int(fe["K"])
    basis, interp = _fe2_basis(x1g, x2g, table, K)
    # aux here is the (I1bar-3, I2bar-3) query pair per particle, so it stacks to (M, 2)
    A, b, Q, truth = _stack(dump_paths, basis, n_modes=n_modes)
    AtA = A.T @ A
    lam = lam_rel * np.trace(AtA) / AtA.shape[0]
    theta = np.linalg.solve(AtA + lam * np.eye(AtA.shape[0]), A.T @ b)[:K]
    phi = interp(Q)
    W1 = phi[:, :K] @ theta
    W2 = phi[:, K:] @ theta
    out = dict(K=K, cond=float(np.linalg.cond(AtA)), n_rows=int(A.shape[0]),
               W1_mean=float(np.median(W1)), W2_mean=float(np.median(W2)))
    if "C10" in truth:
        out["C10_err"] = float(abs(np.median(W1) / truth["C10"] - 1))
        out["C01_err"] = float(abs(np.median(W2) / truth["C01"] - 1))
        log(f"[FE2 recover] K={K} rows={A.shape[0]} cond={out['cond']:.1e}")
        log(f"[FE2 recover] W1 (I1 response) median {np.median(W1):.3e} vs C10 "
            f"{truth['C10']:.3e} ({100*out['C10_err']:.0f}%);  W2 (I2 response) median "
            f"{np.median(W2):.3e} vs C01 {truth['C01']:.3e} ({100*out['C01_err']:.0f}%)")
    return out


def _probes(names: list[str]) -> list[Path]:
    found = [find_artifact(n) for n in names]
    missing = [n for n, p in zip(names, found, strict=True) if p is None]
    if missing:
        raise SystemExit(
            f"missing hyperelastic dumps {missing}; they came from the JAX MPM engine "
            "which is no longer in the tree (see run_drop's docstring)")
    return [p for p in found if p is not None]


def stage_recover(which: str = "mooney", log=print) -> dict:
    """Coefficient recovery for the hyperelastic families, single and multi-probe."""
    res: dict = {}
    if which in ("mooney", "all"):
        gentle, *hard = _probes(MOONEY_PROBES)
        log("\n--- Mooney-Rivlin SINGLE GENTLE probe (small strain, I1 and I2 collinear) ---")
        res["mooney_gentle"] = recover("mooney", gentle, tag="gentle", log=log)
        log("\n--- Mooney-Rivlin HARD MULTI-probe (about 50 percent strain, invariants "
            "separate) ---")
        res["mooney_hard_multi"] = recover("mooney", hard, tag="hard-multi", log=log)
    if which in ("yeoh", "all"):
        paths = _probes(YEOH_PROBES)
        log("\n--- Yeoh SINGLE probe (vz -3): higher-order terms unidentifiable ---")
        res["yeoh_single"] = recover("yeoh", paths[0], tag="single", log=log)
        log("\n--- Yeoh MULTI-probe (vz -3, -7, -12): wide strain path ---")
        res["yeoh_multi"] = recover("yeoh", paths, tag="multi", log=log)
    artifact("hyperelastic_recover.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"wrote {artifact('hyperelastic_recover.json')}")
    return res


def stage_recover_fe(log=print) -> dict:
    """Recovery through the trained function-encoder bases, one and two invariant."""
    yeoh = _probes(YEOH_PROBES)
    out, _, _, _ = recover_fe(yeoh, log=log)
    # the one-invariant result stays at the top level, which is the shape the
    # pre-consolidation out/hyperelastic/fe_operator_recover.json has
    res: dict = dict(out)
    _, *hard = _probes(MOONEY_PROBES)
    res["fe2_mooney_hard_multi"] = recover_fe2(hard, log=log)
    artifact("fe_operator_recover.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"wrote {artifact('fe_operator_recover.json')}")
    return res
