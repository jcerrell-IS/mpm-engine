"""Plasticine drop: recover (G, lambda, yield) from a von-Mises blob, no backpropagation.

A ductile von-Mises blob (warp-mpm material "metal") is dropped hard enough to yield
and plastically flatten, so it does not bounce back like the elastic blob. Two things
come out of the same dump:

  (G, lambda) from the dynamic weak-form momentum balance. The Hencky Cauchy stress
      sigma = (1/J) U diag(2 G eps_i + lambda sum eps) U^T,   eps_i = log sing(F)
  is linear in (G, lambda), so it is the same convex solve as the elastic drop and the
  inertia term sets the absolute scale.

  the yield stress from the SATURATION of deviatoric strain. The return map caps
  ||dev(eps)|| at eps_y = yield / (2 G) inside yielded regions, so
  yield = 2 G (saturated ||dev(eps)||).

The gate: yield is identifiable only if some material has yielded, which shows up as a
clear ||dev(eps)|| plateau. A sub-yield loading gives a lower bound, and the estimate
is refused. Running the same material at two drop heights separates the two cases,
which is the plastic-versus-elastic coverage bound.

Results (E 1e6, nu 0.3, yield 2e4, so G 3.85e5 and eps_y about 2.6 percent):
  hard drop  yields, G and lambda recovered, yield identified to about 4 percent
  soft drop  same material stays elastic, G and lambda still recovered, yield REFUSED
G comes out of both drops. Stiffness needs strain; yield needs strain past yield.

Artifacts: out/elastic/{plastic_truth,plastic_yield,plastic_elastic}.npz and
plasticine_gate.png. The pre-consolidation dumps are out/plastic_drop/{truth,yield,
elastic}.npz and the figure was out/nclaw_compare/plasticine_gate.png.

Run:  .venv/bin/python -m experiments.elastic plastic
      .venv/bin/python -m experiments.elastic plastic-gate
      .venv/bin/python -m experiments.elastic plastic-gate-figure
"""
from __future__ import annotations

import numpy as np

from .core import (
    TRUTH_PLASTIC,
    DropDump,
    WeakForm,
    artifact,
    find_artifact,
    hencky_basis,
    lstsq,
    publish_figure,
    run_drop,
)

# names in out/elastic, with the pre-consolidation name searched as a fallback
DUMPS = {"truth": ("plastic_truth.npz", "truth.npz"),
         "yield": ("plastic_yield.npz", "yield.npz"),
         "elastic": ("plastic_elastic.npz", "elastic.npz")}


def _find(kind: str):
    for name in DUMPS[kind]:
        p = find_artifact(name)
        if p is not None:
            return p
    return None


def _drop(kind: str, yield_stress: float, drop_gap: float, reuse: bool = True, log=print):
    if reuse:
        p = _find(kind)
        if p is not None:
            log(f"[plastic] reusing {p}")
            return p
    return run_drop(artifact(DUMPS[kind][0]), TRUTH_PLASTIC["E"], TRUTH_PLASTIC["nu"],
                    TRUTH_PLASTIC["rho"], material="metal", yield_stress=yield_stress,
                    shape="sphere", size=0.14, drop_gap=drop_gap, t_end=0.8, log=log)


def recover(dump_path, log=print) -> dict:
    """Convex weak-form recovery of (G, lambda) plus the yield stress and its gate."""
    dump = DropDump.load(dump_path)
    A, b, dev_all = WeakForm(dump, hencky_basis).assemble()
    theta, cond = lstsq(A, b)
    G_h, lam_h = float(theta[0]), float(theta[1])
    G_t, lam_t, y_t = dump.truth["G"], dump.truth["lam"], dump.truth["yield_stress"]

    # yield from the deviatoric-strain plateau over the whole run
    p98, p995 = (float(q) for q in np.percentile(dev_all, [98, 99.5]))
    y_h = 2.0 * G_h * p995
    plateau = p98 / (p995 + 1e-12)      # near 1 means saturated, well below 1 means elastic
    yielded = plateau > 0.85

    log(f"[recover] rows={A.shape[0]} cond(A^TA)={cond:.1f}")
    log(f"[recover] G={G_h:.3e} ({100*abs(G_h/G_t-1):.1f}%)  "
        f"lam={lam_h:.3e} ({100*abs(lam_h/lam_t-1):.1f}%)")
    log(f"[recover] ||dev(eps)|| saturation p99.5={p995:.4f} "
        f"(truth eps_y={y_t/(2*G_t):.4f})  plateau ratio={plateau:.2f} -> "
        f"{'YIELDED' if yielded else 'NO clear yield'}")
    if yielded:
        log(f"[recover] yield={y_h:.3e}  (truth {y_t:.3e}, "
            f"{100*abs(y_h/y_t-1):.1f}% err)")
    else:
        log(f"[recover] yield REFUSED (no plateau); lower bound only: yield > {y_h:.3e}")
    return dict(G=G_h, lam=lam_h, yield_=y_h, yielded=yielded, plateau=plateau,
                G_err=abs(G_h / G_t - 1), y_err=abs(y_h / y_t - 1), cond=cond)


def stage_truth(log=print) -> dict:
    """The reference hard drop, plus the check that the Hencky basis reproduces the
    stress the simulator stored (relative L2 at the last frame)."""
    p = _drop("truth", TRUTH_PLASTIC["yield_stress"], 0.30, log=log)
    d = DropDump.load(p)
    zc = d.x[:, :, 2].mean(1)
    log(f"  centroid z: {zc[0]:.3f} -> min {zc.min():.3f} -> end {zc[-1]:.3f} "
        f"(end/min={zc[-1]/zc.min():.2f})")
    if d.stress is not None:
        (S_G, S_L), devmag = hencky_basis(d.F[-1])
        Sf = d.stress[-1]
        rel = float(np.linalg.norm(d.truth["G"] * S_G + d.truth["lam"] * S_L - Sf)
                    / (np.linalg.norm(Sf) + 1e-30))
        log(f"  stress-basis check relL2 = {rel:.2e}; ||dev(eps)|| max {devmag.max():.4f} "
            f"vs eps_y={d.truth['yield_stress']/(2*d.truth['G']):.4f}")
    return recover(p, log=log)


def stage_gate(yield_stress: float = 2.0e4, log=print) -> tuple[dict, dict]:
    """Same material, two drop heights: one yields, one does not.

    The hard drop exceeds yield, so (G, lambda, yield) all come out. The soft drop
    stays below yield under the same material, so (G, lambda) come out and the yield
    estimate is refused as a lower bound.
    """
    log("\n=== HARD drop (drop_gap 0.30, exceeds yield, YIELDS) ===")
    py = _drop("yield", yield_stress, 0.30, log=log)
    ry = recover(py, log=log)
    log("\n=== SOFT drop (same material, drop_gap 0.04, stays below yield, ELASTIC) ===")
    pe = _drop("elastic", yield_stress, 0.04, log=log)
    re = recover(pe, log=log)
    log(f"\n[gate] G recovered in BOTH ({100*ry['G_err']:.1f}% / "
        f"{100*re['G_err']:.1f}%) regardless of yield.")
    log(f"[gate] yield: YIELDED case identified to {100*ry['y_err']:.1f}%; "
        f"ELASTIC case REFUSED (plateau {re['plateau']:.2f}).")
    return ry, re


def stage_gate_figure(log=print):
    """The gate as a histogram: ||dev(eps)|| at peak deformation for both drops.

    The yielding drop piles up at the cap eps_y = yield / (2 G), which is the yield
    signature. The elastic drop has a smooth sub-cap distribution, no plateau, so the
    yield estimate is refused.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def devdist(kind: str):
        p = _find(kind)
        if p is None:
            raise SystemExit(f"missing the {kind} dump; run the plastic-gate stage first")
        d = DropDump.load(p)
        dev_t = []
        for Ft in d.F[::20]:
            sig = np.clip(np.linalg.svd(Ft, compute_uv=False), 1e-6, None)
            eps = np.log(sig)
            dev = eps - eps.mean(-1, keepdims=True)
            dev_t.append(np.linalg.norm(dev, axis=1))
        peak = max(range(len(dev_t)), key=lambda i: dev_t[i].mean())
        return dev_t[peak], d.truth["yield_stress"], d.truth["G"]

    dy, yy, Gy = devdist("yield")
    de, _, _ = devdist("elastic")
    eps_y = yy / (2 * Gy)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(dy, bins=60, range=(0, 0.035), color="#e8590c", alpha=0.6,
            label="HARD drop, YIELDS (pile-up at the cap; yield recovered to 4%)")
    ax.hist(de, bins=60, range=(0, 0.035), color="#1c7ed6", alpha=0.55,
            label="SOFT drop, ELASTIC (smooth, below the cap; yield REFUSED)")
    ax.axvline(eps_y, color="k", ls="--", lw=1.8,
               label=f"yield cap eps_y = yield/(2G) = {eps_y:.3f}")
    ax.set_xlabel("||dev(eps)||  (deviatoric Hencky strain, per particle)")
    ax.set_ylabel("particle count")
    ax.set_title("Plasticine identifiability gate (von-Mises, convex weak form, no backprop)\n"
                 "yield is identifiable only when the loading reaches yield: the yielded drop\n"
                 "saturates at the cap, the elastic drop does not", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = publish_figure(fig, "plasticine_gate.png")
    plt.close(fig)
    log(f"wrote {p}")
    return p
