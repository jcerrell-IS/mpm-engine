"""Sequential identification: how confidence grows with measurements, and when to stop.

The same recursive estimator serves the elastic and the plasticine drop, so it lives
here once. Frames are streamed through the convex weak form; the Fisher information
M_t = sum_{tau <= t} A_tau^T A_tau / sigma^2 accumulates; the posterior is the
conjugate Gaussian

    Sigma_t = (M_t + prior_prec)^-1,
    theta_t = Sigma_t (rhs_t + prior_prec theta_prior).

Nothing is differentiated and no rollout is re-simulated inside the loop.

The two figures:

  elastic (columns mu, lambda, quantities of interest E and nu). The posterior standard
  deviation is flat during free fall, because no strain means A near zero means M does
  not grow, then drops sharply at impact. E crosses a 5 percent confidence threshold;
  nu does not, because it needs volumetric strain and a bounce barely compresses. So
  "is there enough deformation to extract this property" is answered per property:
  shear yes, bulk no, and the tight nu band is itself the warning, since posterior
  variance does not see the coverage bias.

  plasticine (columns G, lambda, plus the yield read off the deviatoric saturation).
  G contracts to truth at impact. The yield estimate is refused as a lower bound until
  the deviatoric strain saturates at the cap, then locks on. The histogram version of
  the same test is in plastic.py.

The stopping rule: rollout_vs_frames re-simulates the law recovered from the first N
frames and measures how well it predicts the whole trajectory. The expensive rollout
error and the cheap online posterior standard deviation of E collapse together once
impact is observed, so the online quantity is usable as the stopping rule without ever
re-simulating.

CHANGED IN CONSOLIDATION: the two pre-consolidation scripts filtered particles on
||F - I||, which is not frame-objective, while every other script in the family had
moved to the rotation-invariant ||log sigma(F)||. This module uses the rotation-invariant
filter, so the streamed numbers shift slightly from the figures in the tree. The
before-and-after numbers are recorded in README.md.

Artifacts: out/elastic/{elastic_bayesian_identify,elastic_rollout_vs_frames,
plasticine_rls_identify}.png, copied into docs/writeup/figs. The pre-consolidation
figures were written to out/nclaw_compare/.

Run:  .venv/bin/python -m experiments.elastic sequential
      .venv/bin/python -m experiments.elastic sequential-rollout
      .venv/bin/python -m experiments.elastic plastic-sequential
"""
from __future__ import annotations

import numpy as np

from .core import (
    DropDump,
    E_nu_to_moduli,
    WeakForm,
    artifact,
    fcr_basis,
    find_artifact,
    hencky_basis,
    lstsq,
    moduli_to_E_nu,
    noise_scale,
    position_error,
    publish_figure,
    run_drop,
)


def _qoi_jacobian(mu: float, lam: float, h: float = 1.0) -> np.ndarray:
    """Numerical Jacobian of (E, nu) with respect to (mu, lambda)."""
    f0 = np.array(moduli_to_E_nu(mu, lam))
    jm = (np.array(moduli_to_E_nu(mu + h, lam)) - f0) / h
    jl = (np.array(moduli_to_E_nu(mu, lam + h)) - f0) / h
    return np.stack([jm, jl], axis=1)


def rls_stream(wf: WeakForm, prior_mean: np.ndarray, prior_rel_std: float = 5.0):
    """Stream a weak form frame by frame, yielding (t, theta, Sigma, FrameRows or None).

    sigma^2 comes from the full-data fit, so it is the model's own residual scale.
    The prior is deliberately weak, prior_rel_std times the prior mean, so the
    posterior is data-dominated once informative frames arrive.
    """
    A, b, _ = wf.assemble()
    theta_full, _ = lstsq(A, b)
    sigma2 = noise_scale(A, b, theta_full) ** 2
    prior_prec = np.diag(1.0 / (prior_rel_std * prior_mean) ** 2)
    k = len(prior_mean)
    M = np.zeros((k, k))
    rhs = np.zeros(k)
    for t in wf.interior_frames():
        fr = wf.frame(t)
        if fr is not None:
            M += fr.A.T @ fr.A / sigma2
            rhs += fr.A.T @ fr.b / sigma2
        Sigma = np.linalg.inv(M + prior_prec)
        theta = Sigma @ (rhs + prior_prec @ prior_mean)
        yield t, theta, Sigma, fr


# ------------------------------------------------------------------------ elastic


def stream_elastic(dump_name: str = "truth.npz", prior_E: float = 1.2e5,
                   prior_nu: float = 0.30):
    """Records of (frame, time, E, E_std, nu, nu_std, dev strain, vol strain)."""
    p = find_artifact(dump_name)
    if p is None:
        raise SystemExit(f"missing {dump_name}; run the recover stage first")
    dump = DropDump.load(p)
    wf = WeakForm(dump, fcr_basis)
    prior_mean = np.array(E_nu_to_moduli(prior_E, prior_nu))
    eye = np.eye(3)
    rec = []
    for t, theta, Sigma, fr in rls_stream(wf, prior_mean):
        mu_h, lam_h = float(theta[0]), float(theta[1])
        if mu_h <= 0 or (lam_h + mu_h) <= 0:
            E_h = nu_h = E_std = nu_std = np.nan
        else:
            E_h, nu_h = moduli_to_E_nu(mu_h, lam_h)
            J = _qoi_jacobian(mu_h, lam_h)
            cov = J @ Sigma @ J.T
            E_std = float(np.sqrt(max(cov[0, 0], 0.0)))
            nu_std = float(np.sqrt(max(cov[1, 1], 0.0)))
        dev = vol = 0.0
        if fr is not None:
            Fm = fr.F
            Jm = np.linalg.det(Fm)
            meaneps = np.trace(Fm - eye, axis1=1, axis2=2) / 3.0
            dev = float(np.mean(np.linalg.norm(
                (Fm - eye) - meaneps[:, None, None] * eye, axis=(1, 2))))
            vol = float(np.mean(np.abs(Jm - 1.0)))
        rec.append((t, t * dump.frame_dt, E_h, E_std, nu_h, nu_std, dev, vol))
    return np.array(rec), (dump.truth["mu"], dump.truth["lam"])


def stage_elastic_figure(dump_name: str = "truth.npz", log=print):
    """Posterior mean and 95 percent credible interval of E and nu versus frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec, (mu_t, lam_t) = stream_elastic(dump_name)
    E_t, nu_t = moduli_to_E_nu(mu_t, lam_t)
    t = rec[:, 1]
    E_h, E_std, nu_h, nu_std, dev, vol = (rec[:, i] for i in (2, 3, 4, 5, 6, 7))
    below = np.where(100 * E_std / E_t < 5.0)[0]
    imp_t = float(rec[below[0], 1]) if len(below) else t[-1]

    fig, ax = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1.1], hspace=0.13))
    ax[0].fill_between(t, (E_h - 1.96 * E_std) / 1e3, (E_h + 1.96 * E_std) / 1e3,
                       color="#1c7ed6", alpha=0.25, label="95% credible interval")
    ax[0].plot(t, E_h / 1e3, color="#1c7ed6", lw=2.2, label="posterior mean E")
    ax[0].axhline(E_t / 1e3, color="k", ls="--", lw=1.5, label="truth E")
    ax[0].axvline(imp_t, color="0.7", lw=1.0)
    ax[0].set_ylim(0, 1.6 * E_t / 1e3)
    ax[0].set_ylabel("E  (kPa)")
    ax[0].set_title("Bayesian sequential identification (prior to conjugate posterior, "
                    "recursive)\ncredible band starts at the prior and contracts to truth "
                    "once impact is observed", fontsize=11)
    ax[0].legend(fontsize=9, loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(t, nu_h - 1.96 * nu_std, nu_h + 1.96 * nu_std,
                       color="#e8590c", alpha=0.25, label="95% credible interval")
    ax[1].plot(t, nu_h, color="#e8590c", lw=2.2, label="posterior mean nu")
    ax[1].axhline(nu_t, color="k", ls="--", lw=1.5, label="truth nu")
    ax[1].axvline(imp_t, color="0.7", lw=1.0)
    ax[1].set_ylim(0, 0.5)
    ax[1].set_ylabel("nu  (Poisson)")
    ax[1].legend(fontsize=9, loc="upper right")
    ax[1].grid(alpha=0.3)
    ax[1].text(0.02, 0.06, "bulk under-excited, so nu is biased low despite a TIGHT band "
               "(posterior variance misses the coverage bias)",
               transform=ax[1].transAxes, fontsize=8, color="#a33")
    ax[2].plot(t, dev, color="#1c7ed6", lw=2.0, label="deviatoric strain (informs E)")
    ax[2].plot(t, vol, color="#e8590c", lw=2.0, label="volumetric strain |J-1| (informs nu)")
    ax[2].axvline(imp_t, color="0.7", lw=1.0)
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("strain / frame")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p = publish_figure(fig, "elastic_bayesian_identify.png")
    plt.close(fig)
    log(f"wrote {p}")
    log(f"  E: prior {E_h[0]/1e3:.0f} kPa (+/-{1.96*E_std[0]/1e3:.0f}) -> posterior "
        f"{E_h[-1]/1e3:.1f} kPa (+/-{1.96*E_std[-1]/1e3:.2f}), truth {E_t/1e3:.0f}; "
        f"confident at t={imp_t:.2f}s")
    log(f"  nu: posterior {nu_h[-1]:.3f} +/-{1.96*nu_std[-1]:.3f} (truth {nu_t:.2f}), "
        f"wider and biased because the bulk mode is starved")
    return p


def stage_rollout_vs_frames(dump_name: str = "truth.npz",
                            checks=(60, 95, 105, 112, 122, 140, 200, 320, 450),
                            log=print):
    """Rollout prediction error versus frames observed, with the online surrogate.

    Re-simulates the law recovered from the first N frames and measures how well it
    predicts the whole trajectory. The cheap online posterior standard deviation of E
    is overlaid to show it tracks the expensive rollout error, so it can serve as the
    stopping rule.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = find_artifact(dump_name)
    if p is None:
        raise SystemExit(f"missing {dump_name}; run the recover stage first")
    dump = DropDump.load(p)
    rec, (mu_t, lam_t) = stream_elastic(dump_name)
    E_t, _ = moduli_to_E_nu(mu_t, lam_t)
    raw = np.load(p)
    size = float(raw["size"]) if "size" in raw.files else 0.14
    out = []
    for N in checks:
        rows = rec[rec[:, 0] <= N]
        if not len(rows):
            continue
        row = rows[-1]
        E_N, E_std_N, nu_N = float(row[2]), float(row[3]), float(row[4])
        if not (E_N > 1e3 and 0.0 < nu_N < 0.49):
            out.append((N, np.nan, 100 * E_std_N / E_t))
            continue
        tmp = artifact(f"_roll_{N}.npz")
        rp = run_drop(tmp, E_N, nu_N, dump.rho, material="jelly", shape=dump.shape,
                      size=size, log=lambda *_: None)
        rmse, _, _ = position_error(p, rp)
        out.append((N, rmse, 100 * E_std_N / E_t))
        tmp.unlink(missing_ok=True)
    out = np.array(out)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(out[:, 0], out[:, 1], "o-", color="#1c7ed6", lw=2.2,
            label="rollout prediction RMSE (mm)  [re-sim, expensive]")
    ax.set_xlabel("number of frames observed for identification (N)")
    ax.set_ylabel("rollout RMSE vs truth (mm)", color="#1c7ed6")
    ax.tick_params(axis="y", labelcolor="#1c7ed6")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(out[:, 0], out[:, 2], "s--", color="#e8590c", lw=2.0,
             label="posterior std of E (%)  [online Fisher surrogate, cheap]")
    ax2.set_ylabel("posterior std of E (%)", color="#e8590c")
    ax2.tick_params(axis="y", labelcolor="#e8590c")
    ax.set_title("Sequential elastic ID: rollout prediction error vs frames observed\n"
                 "free-fall frames carry no information; once impact is seen the rollout "
                 "error and\nthe cheap Fisher confidence collapse together, so the "
                 "surrogate is the stopping rule", fontsize=10)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper right")
    fig.tight_layout()
    fp = publish_figure(fig, "elastic_rollout_vs_frames.png")
    plt.close(fig)
    log(f"wrote {fp}")
    for N, rmse, estd in out:
        log(f"  N={int(N):3d} frames: rollout RMSE {rmse:6.2f} mm | "
            f"online E-std {estd:6.2f}%")
    return out


# ---------------------------------------------------------------------- plasticine


def stream_plastic(dump_name: str = "plastic_yield.npz", prior_G: float = 2.0e5,
                   prior_nu: float = 0.30):
    """Records of (time, G, G_std, yield, running saturation, yielded flag)."""
    p = find_artifact(dump_name) or find_artifact("yield.npz")
    if p is None:
        raise SystemExit("missing the yielding plasticine dump; run plastic-gate first")
    dump = DropDump.load(p)
    wf = WeakForm(dump, hencky_basis)
    lam_p = prior_G * 2.0 * prior_nu / (1.0 - 2.0 * prior_nu)
    prior_mean = np.array([prior_G, lam_p])
    rec = []
    sat_run = 0.0
    yielded_at = None
    for t, theta, Sigma, fr in rls_stream(wf, prior_mean):
        plateau = 0.0
        p995 = 0.0
        if fr is not None:
            p98, p995 = (float(q) for q in np.percentile(fr.aux, [98, 99.5]))
            plateau = p98 / (p995 + 1e-12)
            sat_run = max(sat_run, p995)
        G_h = float(theta[0])
        G_std = float(np.sqrt(max(Sigma[0, 0], 0.0)))
        is_yld = (fr is not None) and (plateau > 0.85) and (p995 > 0.01)
        if is_yld and yielded_at is None:
            yielded_at = t * dump.frame_dt
        rec.append((t * dump.frame_dt, G_h, G_std, 2.0 * G_h * sat_run, sat_run,
                    1.0 if is_yld else 0.0))
    truth = (dump.truth["G"], dump.truth["lam"], dump.truth["yield_stress"])
    return np.array(rec), truth, yielded_at


def stage_plastic_figure(log=print):
    """G posterior band and the yield estimate versus frame, with the yield onset marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec, (G_t, _lam_t, y_t), yat = stream_plastic()
    t, G_h, G_std, y_h = (rec[:, i] for i in range(4))
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1], hspace=0.13))
    ax[0].fill_between(t, (G_h - 1.96 * G_std) / 1e3, (G_h + 1.96 * G_std) / 1e3,
                       color="#1c7ed6", alpha=0.25, label="95% credible interval")
    ax[0].plot(t, G_h / 1e3, color="#1c7ed6", lw=2.2, label="posterior mean G")
    ax[0].axhline(G_t / 1e3, color="k", ls="--", lw=1.5, label="truth G")
    ax[0].set_ylim(0, 1.7 * G_t / 1e3)
    ax[0].set_ylabel("G  (kPa)")
    ax[0].set_title("Plasticine recursive (RLS) identification\n"
                    "G contracts to truth at impact (recursive, convex, no backprop)",
                    fontsize=10)
    ax[0].legend(fontsize=9, loc="upper right")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, y_h / 1e3, color="#e8590c", lw=2.2,
               label="recovered yield (= 2 G * saturated ||dev eps||)")
    ax[1].axhline(y_t / 1e3, color="k", ls="--", lw=1.5, label="truth yield")
    if yat is not None:
        for a_ in ax:
            a_.axvline(yat, color="#2f9e44", lw=1.4)
        ax[1].text(yat, 0.5 * y_t / 1e3,
                   f"  yield identifiable\n  (yields at t={yat:.2f}s)",
                   color="#2f9e44", fontsize=8.5)
    ax[1].set_ylim(0, 1.5 * y_t / 1e3)
    ax[1].set_xlabel("time (s) [streamed frame]")
    ax[1].set_ylabel("yield  (kPa)")
    ax[1].text(0.02, 0.06, "before yield onset: refused (lower bound); after: locks to truth",
               transform=ax[1].transAxes, fontsize=8.5, color="#a33")
    ax[1].legend(fontsize=9, loc="lower right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = publish_figure(fig, "plasticine_rls_identify.png")
    plt.close(fig)
    log(f"wrote {p}")
    log(f"  G: prior {G_h[0]/1e3:.0f} -> {G_h[-1]/1e3:.1f} kPa (truth {G_t/1e3:.0f}, "
        f"{100*abs(G_h[-1]/G_t-1):.1f}%); yield {y_h[-1]/1e3:.1f} kPa "
        f"(truth {y_t/1e3:.0f}, {100*abs(y_h[-1]/y_t-1):.1f}%); yields at t={yat}")
    return p
