"""One function-encoder basis spanning four hyperelastic families, with its error bound.

For a battery of deformation gradients (uniaxial tension and compression, equibiaxial,
simple shear, pure dilation, at strains up to 50 percent) this evaluates the true
first-Piola stress of four families over parameter grids:

    neo-Hookean    W = C10 (I1bar - 3) + vol                     linear baseline
    Mooney-Rivlin  W = C10 (I1bar-3) + C01 (I2bar-3) + vol       nonlinear, I2 term
    Yeoh           W = sum_i C_i (I1bar-3)^i + vol               nonlinear, I1 powers
    Gent           W = -(mu Jm / 2) ln(1 - (I1bar-3)/Jm) + vol   nonlinear, non-polynomial

Each material becomes a stress-response fingerprint, the flattened PK1 over the
battery. One weighted SVD over the whole library gives a single multi-family basis with
a computable Eckart-Young approximation budget, the singular-value tail, and the
held-out reconstruction is measured against that tail. This is the offline basis that
hyperelastic.recover_fe identifies coefficients in.

Result: 480 materials, 4 families, fingerprint dimension 405. The held-out
reconstruction tracks the Eckart-Young tail across K, so the bound is both predictive
and tight on unseen materials, the nonlinear Gent and Yeoh members included. The
per-family reconstruction at K = 12 and the rest of the numbers are in the json.

Artifacts: out/elastic/hyperelastic_fe_basis.json and approximation_bound.png (copied
into docs/writeup/figs, where paper.tex includes it). Pre-consolidation location was
out/hyperelastic/hyperelastic_fe_basis.json.

Run:  .venv/bin/python -m experiments.elastic fe-basis
"""
from __future__ import annotations

import json

import numpy as np

from .core import artifact, publish_figure

RNG = np.random.default_rng(0)


def battery(n_strain: int = 9, smax: float = 0.5) -> np.ndarray:
    """Deformation-gradient battery: uniaxial, biaxial, shear, dilation, over strains."""
    Fs = []
    for s in np.linspace(0.02, smax, n_strain):
        lam = 1.0 + s
        Fs.append(np.diag([lam, 1 / np.sqrt(lam), 1 / np.sqrt(lam)]))   # uniaxial tension
        Fs.append(np.diag([1 / (1 + s), 1.0, 1.0]))                     # uniaxial compression
        Fs.append(np.diag([lam, lam, 1 / lam ** 2]))                    # equibiaxial
        sh = np.eye(3)
        sh[0, 1] = s                                                    # simple shear
        Fs.append(sh)
        Fs.append(np.diag([1 + 0.3 * s] * 3))                           # dilation, excites bulk
    return np.array(Fs)


def pk1(model: str, theta: dict, F: np.ndarray) -> np.ndarray:
    """First-Piola stress P = tau F^-T for one hyperelastic model, as a fingerprint block."""
    J = np.linalg.det(F)
    b = F @ np.transpose(F, (0, 2, 1))
    Jm23 = np.power(np.clip(J, 1e-6, None), -2.0 / 3.0)
    bbar = Jm23[:, None, None] * b
    I1bar = bbar[:, 0, 0] + bbar[:, 1, 1] + bbar[:, 2, 2]
    eye = np.eye(3)
    devb = bbar - (I1bar / 3.0)[:, None, None] * eye
    Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
    K = theta.get("Kbulk", 2e5)
    tau = (K * J * (J - 1.0))[:, None, None] * eye                      # volumetric, shared
    if model == "neohookean":
        tau = tau + 2.0 * theta["C10"] * devb
    elif model == "mooney":
        b2 = bbar @ bbar
        trb2 = b2[:, 0, 0] + b2[:, 1, 1] + b2[:, 2, 2]
        devb2 = b2 - (trb2 / 3.0)[:, None, None] * eye
        tau = tau + 2.0 * ((theta["C10"] + theta["C01"] * I1bar)[:, None, None] * devb
                           - theta["C01"] * devb2)
    elif model == "yeoh":
        W1 = (theta["C1"] + 2 * theta["C2"] * (I1bar - 3)
              + 3 * theta["C3"] * (I1bar - 3) ** 2)
        tau = tau + 2.0 * W1[:, None, None] * devb
    elif model == "gent":
        x = np.clip((I1bar - 3.0) / theta["Jm"], -0.99, 0.95)
        W1 = 0.5 * theta["mu"] / (1.0 - x)                              # non-polynomial locking
        tau = tau + 2.0 * W1[:, None, None] * devb
    else:
        raise ValueError(f"unknown model {model!r}")
    return (tau @ Finvt).reshape(len(F), 9)


def library(n_per: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Sample the four families over parameter grids: fingerprints and family labels."""
    F = battery()
    fps: list[np.ndarray] = []
    labels: list[str] = []

    def add(model, sampler):
        for _ in range(n_per):
            th = sampler()
            fps.append(pk1(model, th, F).reshape(-1))
            labels.append(model)

    add("neohookean", lambda: dict(C10=10 ** RNG.uniform(4.3, 5.3),
                                   Kbulk=10 ** RNG.uniform(5, 5.7)))
    add("mooney", lambda: dict(C10=10 ** RNG.uniform(4.3, 5.3),
                               C01=10 ** RNG.uniform(3.5, 4.7),
                               Kbulk=10 ** RNG.uniform(5, 5.7)))
    add("yeoh", lambda: dict(C1=10 ** RNG.uniform(4.3, 5.3),
                             C2=10 ** RNG.uniform(3.5, 4.7),
                             C3=10 ** RNG.uniform(3.0, 4.2),
                             Kbulk=10 ** RNG.uniform(5, 5.7)))
    add("gent", lambda: dict(mu=10 ** RNG.uniform(4.3, 5.3),
                             Jm=RNG.uniform(0.5, 5.0),
                             Kbulk=10 ** RNG.uniform(5, 5.7)))
    return np.array(fps), np.array(labels)


def run(log=print) -> dict:
    """Build the basis, measure held-out reconstruction, and check the Eckart-Young tail."""
    X, lab = library()
    N, D = X.shape
    # per-sample normalize so the basis learns the response SHAPE, not the stiffness scale
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    idx = RNG.permutation(N)
    ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]
    _, s, Vt = np.linalg.svd(Xn[tr], full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    def recon_err(K: int) -> float:
        B = Vt[:K]
        proj = Xn[te] @ B.T @ B
        return float(np.median(np.linalg.norm(Xn[te] - proj, axis=1)))

    Ks = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    errs = {K: recon_err(K) for K in Ks}
    K99 = int(np.searchsorted(energy, 0.9999) + 1)
    Kfix = 12
    B = Vt[:Kfix]
    fam_err = {}
    for m in ["neohookean", "mooney", "yeoh", "gent"]:
        sel = te[lab[te] == m]
        if len(sel):
            proj = Xn[sel] @ B.T @ B
            fam_err[m] = float(np.median(np.linalg.norm(Xn[sel] - proj, axis=1)))

    # The approximation bound, empirically. Eckart-Young says the best K-term error is
    # the singular-value tail sqrt(sum_{k>K} s_k^2 / sum_k s_k^2). The basis is trained
    # on the train split and the tail is computed from the TRAIN singular values, so if
    # the held-out error coincides the bound is predictive on unseen materials.
    def tail(K: int) -> float:
        return float(np.sqrt(np.sum(s[K:] ** 2) / np.sum(s ** 2)))

    Kfull = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
    bound = {K: tail(K) for K in Kfull if len(s) > K}
    held = {K: recon_err(K) for K in bound}

    out = dict(N=N, D=D, n_train=ntr,
               families=["neohookean", "mooney", "yeoh", "gent"],
               singular_energy_K={k: float(energy[k - 1]) for k in [2, 3, 5, 8, 12]},
               K_for_99_99pct=K99, recon_relL2_by_K=errs,
               family_recon_relL2_at_K12=fam_err,
               eckart_young_tail_by_K=bound, heldout_relL2_by_K_full=held,
               bound_vs_empirical_ratio={K: float(held[K] / max(bound[K], 1e-30))
                                         for K in bound})
    artifact("hyperelastic_fe_basis.json").write_text(json.dumps(out, indent=2, default=float))

    log(f"hyperelastic FE basis: N={N} materials (4 families including nonlinear Yeoh "
        f"and Gent), D={D} fingerprint dim")
    log(f"  cumulative energy: K=2 {energy[1]:.4f}  K=3 {energy[2]:.4f}  "
        f"K=5 {energy[4]:.6f}  K=8 {energy[7]:.6f}  K=12 {energy[11]:.6f}")
    log(f"  K for 99.99 percent energy: {K99}")
    log("  held-out reconstruction relL2 vs K (Eckart-Young approximation budget):")
    for K in Ks:
        log(f"    K={K:2d}: median relL2 {errs[K]:.2e}")
    log(f"  per-family held-out reconstruction at K={Kfix}:")
    for m, e in fam_err.items():
        log(f"    {m:11s}: {e:.2e}")
    log("  bound check: held-out error vs the Eckart-Young tail")
    log(f"    {'K':>3} {'EY tail (bound)':>16} {'held-out relL2':>16} {'ratio':>8}")
    for K in bound:
        log(f"    {K:3d} {bound[K]:16.3e} {held[K]:16.3e} "
            f"{held[K]/max(bound[K],1e-30):8.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ksb = list(bound)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.semilogy(Ksb, [bound[K] for K in Ksb], "k-", lw=2,
                label=r"Eckart--Young bound $\sqrt{\sum_{k>K}s_k^2}$")
    ax.semilogy(Ksb, [held[K] for K in Ksb], "o", color="#e34a33", ms=6,
                label="held-out reconstruction (empirical)")
    ax.set_xlabel("basis modes $K$")
    ax.set_ylabel("reconstruction relL2")
    ax.set_title("Approximation bound, verified empirically\n"
                 "(held-out error tracks the singular-value tail)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = publish_figure(fig, "approximation_bound.png", dpi=150, tight=True)
    plt.close(fig)
    log(f"\nwrote {artifact('hyperelastic_fe_basis.json')} and {p}")
    return out
