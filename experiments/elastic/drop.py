"""Elastic gravity drop: recover the moduli from a bounce, with no backpropagation.

NCLaw learns an elastic law by differentiating an MPM rollout (BPTT on positions).
This is the elastic analogue of the granular weak-form recovery: drop a
fixed-corotated blob, observe the bounce (positions, velocities, F per particle per
frame), and recover (mu, lambda) by a convex linear-in-theta weak-form momentum
residual. The simulator is never differentiated.

    P = mu 2(F - R) + lambda J(J-1) F^-T  =  mu P_mu(F) + lambda P_lam(F)
    sum_p V0_p P : grad_X w_j  =  -sum_p V0_p rho0 (a_p - g) . w_j

Two unknowns, many (test function x frame) rows, one least-squares solve. The
inertia term supplies the absolute force scale, so no force sensor is needed.

Results, radial-window route, reproduced 2026-08-22 on the dumps in the tree:
  sphere (truth.npz):     E 2.0070e5 vs 2.0e5, 0.35 percent error, nu 0.256, cond 4.4
  cube   (box_truth.npz): E 2.2687e5 vs 2.0e5, 13.4 percent error, nu 0.329, cond 16.4
The cube error is the reason the grid-consistent route exists: the radial window
does not vanish on a flat face, so the floor contact traction leaks into the
residual. See grid_gate.py, which drives the cube to 0.07 percent.

Sample complexity, on the cube dump: the estimator spread follows the Gauss-Markov
prediction sigma / sqrt(lambda_k) and falls as 1/sqrt(N) for the shear mode, while the
bulk mode (Fisher eigenvalue ratio about 1e4) stays flat. More rows do not buy a mode
the motion never excited, so the estimator refuses it instead.

Artifacts: out/elastic/{truth,recovered,box_truth,box_pred,star_truth,star_pred}.npz,
sample_complexity.json, sample_complexity.png. Reads the pre-consolidation dumps in
out/elastic_drop when they are present rather than re-simulating.

Run:  .venv/bin/python -m experiments.elastic recover
      .venv/bin/python -m experiments.elastic shape
      .venv/bin/python -m experiments.elastic errors
      .venv/bin/python -m experiments.elastic sample-complexity
"""
from __future__ import annotations

import json

import numpy as np

from .core import (
    TRUTH_ELASTIC,
    DropDump,
    WeakForm,
    artifact,
    fcr_basis,
    find_artifact,
    lstsq,
    moduli_to_E_nu,
    noise_scale,
    position_error,
    publish_figure,
    run_drop,
)

RNG = np.random.default_rng(0)


def _drop(name: str, E: float, nu: float, shape: str, size: float,
          drop_gap: float = 0.18, reuse: bool = True, log=print):
    """Run a drop unless the dump already exists here or in a legacy directory."""
    if reuse:
        p = find_artifact(name)
        if p is not None:
            log(f"[drop] reusing {p}")
            return p
    return run_drop(artifact(name), E, nu, TRUTH_ELASTIC["rho"], material="jelly",
                    shape=shape, size=size, drop_gap=drop_gap, log=log)


def assemble(dump_path, n_modes: int = 4):
    """The FCR weak-form system for one dump: (A with columns (mu, lambda), b, aux)."""
    wf = WeakForm(DropDump.load(dump_path), fcr_basis, n_modes=n_modes)
    return wf.assemble()


def recover(dump_path, n_modes: int = 4, log=print) -> dict:
    """Convex weak-form recovery of (mu, lambda) from one bounce, radial-window route."""
    dump = DropDump.load(dump_path)
    A, b, aux = WeakForm(dump, fcr_basis, n_modes=n_modes).assemble()
    theta, cond = lstsq(A, b)
    mu_h, lam_h = float(theta[0]), float(theta[1])
    mu_t, lam_t = dump.truth["mu"], dump.truth["lam"]
    E_h, nu_h = moduli_to_E_nu(mu_h, lam_h)
    E_t, nu_t = moduli_to_E_nu(mu_t, lam_t)
    log(f"[recover] rows={A.shape[0]}  cond(A^T A)={cond:.1f}  Hencky strain coverage "
        f"[{aux.min():.3f}, {np.percentile(aux, 99):.3f}]")
    log(f"[recover] mu:  {mu_h:.4e}  (truth {mu_t:.4e},  {100*abs(mu_h/mu_t-1):.1f}% err)")
    log(f"[recover] lam: {lam_h:.4e}  (truth {lam_t:.4e},  {100*abs(lam_h/lam_t-1):.1f}% err)")
    log(f"[recover] E:   {E_h:.4e}  (truth {E_t:.4e},  {100*abs(E_h/E_t-1):.1f}% err)   "
        f"nu: {nu_h:.3f} (truth {nu_t:.3f})")
    return dict(mu=mu_h, lam=lam_h, E=E_h, nu=nu_h, E_err=abs(E_h / E_t - 1),
                mu_err=abs(mu_h / mu_t - 1), cond=cond)


def stage_recover(log=print) -> dict:
    """Sphere drop, recover, re-simulate with the recovered law, compare positions."""
    tp = _drop("truth.npz", TRUTH_ELASTIC["E"], TRUTH_ELASTIC["nu"], "sphere", 0.14, log=log)
    rec = recover(tp, log=log)
    rp = run_drop(artifact("recovered.npz"), rec["E"], rec["nu"], TRUTH_ELASTIC["rho"],
                  material="jelly", shape="sphere", size=0.14, log=log)
    rmse_mm, mse_box, nf = position_error(tp, rp)
    log(f"[learned bounce] truth vs recovered-law re-sim: {rmse_mm:.2f} mm RMS, "
        f"box-norm MSE {mse_box:.2e} over {nf} frames")
    return dict(rec=rec, rmse_mm=rmse_mm, mse_box=mse_box, frames=nf)


def stage_shape(log=print) -> dict:
    """Learn on a rectangular blob, predict a held-out star blob.

    The elastic analogue of the sand bunny and dragon generalization: the star is
    non-convex and qualitatively different from the training geometry, and the
    recovery that predicts it is a convex solve, not a differentiated rollout.
    """
    E, nu, rho = TRUTH_ELASTIC["E"], TRUTH_ELASTIC["nu"], TRUTH_ELASTIC["rho"]
    box = _drop("box_truth.npz", E, nu, "box", 0.16, log=log)
    rec = recover(box, log=log)
    star_t = _drop("star_truth.npz", E, nu, "star", 0.17, log=log)
    star_p = run_drop(artifact("star_pred.npz"), rec["E"], rec["nu"], rho,
                      material="jelly", shape="star", size=0.17, log=log)
    rmse_mm, mse_box, nf = position_error(star_t, star_p)
    log("\n[shape-gen] learn on RECTANGLE, predict STAR (held-out geometry):")
    log(f"  recovered E={rec['E']:.3e} ({100*rec['E_err']:.1f}% err), "
        f"mu={rec['mu']:.3e} ({100*rec['mu_err']:.1f}% err)")
    log(f"  star: truth-law vs recovered-law re-sim = {rmse_mm:.2f} mm RMS, "
        f"box-norm MSE {mse_box:.2e} over {nf} frames")
    return dict(rec=rec, rmse_mm=rmse_mm, mse_box=mse_box)


def stage_errors(log=print) -> dict:
    """Reconstruction versus generalization position error, the NCLaw metric family.

    Reconstruction re-simulates the recovered law on the training geometry (the
    rectangle); generalization uses the held-out star. Mirrors the sand Table 5 rows.
    """
    E, nu, rho = TRUTH_ELASTIC["E"], TRUTH_ELASTIC["nu"], TRUTH_ELASTIC["rho"]
    box = _drop("box_truth.npz", E, nu, "box", 0.16, log=log)
    rec = recover(box, log=log)
    boxp = run_drop(artifact("box_pred.npz"), rec["E"], rec["nu"], rho, material="jelly",
                    shape="box", size=0.16, drop_gap=0.18, log=log)
    rec_rmse, rec_mse, _ = position_error(box, boxp)
    st = _drop("star_truth.npz", E, nu, "star", 0.17, log=log)
    sp = find_artifact("star_pred.npz")
    if sp is None:
        sp = run_drop(artifact("star_pred.npz"), rec["E"], rec["nu"], rho,
                      material="jelly", shape="star", size=0.17, log=log)
    gen_rmse, gen_mse, _ = position_error(st, sp)
    log("\n[elastic: reconstruction vs generalization] per-particle position error")
    log(f"  recovered E={rec['E']:.3e} ({100*rec['E_err']:.1f}% err), "
        f"mu err {100*rec['mu_err']:.1f}%")
    log(f"  RECONSTRUCTION (rectangle, training geom): {rec_rmse:.2f} mm RMS, "
        f"box-norm MSE {rec_mse:.2e}")
    log(f"  GENERALIZATION (star, held-out geom):      {gen_rmse:.2f} mm RMS, "
        f"box-norm MSE {gen_mse:.2e}")
    tp, rp = find_artifact("truth.npz"), find_artifact("recovered.npz")
    if tp is not None and rp is not None:
        s_rmse, s_mse, _ = position_error(tp, rp)
        log(f"  (sphere reconstruction, for reference:     {s_rmse:.2f} mm RMS, "
            f"box-norm MSE {s_mse:.2e})")
    return dict(recon_rmse=rec_rmse, recon_mse=rec_mse, gen_rmse=gen_rmse,
                gen_mse=gen_mse, rec=rec)


def stage_sample_complexity(dump_name: str = "box_truth.npz", n_boot: int = 400,
                            n_subsets: int = 24, log=print) -> dict:
    """Recovery error versus the number of weak-form rows, against Gauss-Markov.

    The least-squares estimator of A theta = b has covariance sigma^2 (A^T A)^-1. In
    the eigenbasis of A^T A the per-mode standard deviation is sigma / sqrt(lambda_k),
    and lambda_k grows linearly with the row count N, so the per-mode error falls as
    1/sqrt(N) at a rate set by that mode's per-sample Fisher information. A mode the
    motion does not excite has lambda_k near zero and does not improve with N.

    The validation injects noise rather than resampling truth: take N rows, build a
    clean signal b0 = A_N theta_full, add iid N(0, sigma^2), recover, repeat. For the
    linear-Gaussian model Gauss-Markov is exact, so the empirical covariance must equal
    sigma^2 (A_N^T A_N)^-1 with no finite-population correction.
    """
    dump = find_artifact(dump_name)
    if dump is None:
        dump = _drop(dump_name, TRUTH_ELASTIC["E"], TRUTH_ELASTIC["nu"], "box", 0.16, log=log)
    d = DropDump.load(dump)
    A, b, _ = WeakForm(d, fcr_basis).assemble()
    theta_t = np.array([d.truth["mu"], d.truth["lam"]])
    Nfull = A.shape[0]
    th_full, _ = lstsq(A, b)
    sigma = noise_scale(A, b, th_full)

    Ns = np.unique(np.round(np.logspace(np.log10(60), np.log10(Nfull), 12)).astype(int))
    out: dict = {"N": [], "mu_err": [], "lam_err": [], "mu_std": [], "lam_std": [],
                 "mu_gm_std": [], "lam_gm_std": []}
    for N in Ns:
        emp_list, gm_list, err_list = [], [], []
        for _ in range(n_subsets):
            idx = RNG.choice(Nfull, size=N, replace=False)
            AN, bN = A[idx], b[idx]
            AtA = AN.T @ AN
            if np.linalg.cond(AtA) > 1e14:
                continue
            gm_list.append(np.sqrt(np.diag(sigma ** 2 * np.linalg.inv(AtA))))
            b0 = AN @ th_full
            ths = np.array([np.linalg.lstsq(AN, b0 + RNG.normal(0, sigma, N),
                                            rcond=None)[0]
                            for _ in range(n_boot // n_subsets + 1)])
            emp_list.append(ths.std(0))
            tN, *_ = np.linalg.lstsq(AN, bN, rcond=None)
            err_list.append([abs(tN[0] / theta_t[0] - 1), abs(tN[1] / theta_t[1] - 1)])
        emp = np.mean(emp_list, 0)
        gm = np.mean(gm_list, 0)
        err = np.mean(err_list, 0)
        out["N"].append(int(N))
        out["mu_err"].append(float(err[0]))
        out["lam_err"].append(float(err[1]))
        out["mu_std"].append(float(emp[0]))
        out["lam_std"].append(float(emp[1]))
        out["mu_gm_std"].append(float(gm[0]))
        out["lam_gm_std"].append(float(gm[1]))

    evals = np.linalg.eigvalsh(A.T @ A)
    out["fisher_eigs"] = sorted(evals.tolist())
    out["fisher_ratio"] = float(evals.max() / max(evals.min(), 1e-30))
    out["sigma"] = sigma
    out["Nfull"] = Nfull
    out["mu_truth"], out["lam_truth"] = float(theta_t[0]), float(theta_t[1])
    artifact("sample_complexity.json").write_text(json.dumps(out, indent=2))
    log(f"Nfull={Nfull} sigma={sigma:.3e} Fisher eig ratio (shear/bulk)="
        f"{out['fisher_ratio']:.1e}")
    log(f"{'N':>6} {'mu_err%':>8} {'mu_std':>10} {'mu_GMstd':>10} | "
        f"{'lam_err%':>8} {'lam_std':>10} {'lam_GMstd':>10}")
    for i, N in enumerate(out["N"]):
        log(f"{N:6d} {100*out['mu_err'][i]:8.1f} {out['mu_std'][i]:10.2e} "
            f"{out['mu_gm_std'][i]:10.2e} | {100*out['lam_err'][i]:8.1f} "
            f"{out['lam_std'][i]:10.2e} {out['lam_gm_std'][i]:10.2e}")
    _sample_complexity_figure(out, log=log)
    return out


def _sample_complexity_figure(out: dict, log=print) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = np.array(out["N"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(N, out["mu_std"], "o", color="#2b8cbe",
                 label=r"shear $\mu$ (empirical std)")
    ax[0].loglog(N, out["mu_gm_std"], "-", color="#2b8cbe",
                 label=r"Gauss-Markov $\sigma/\sqrt{\lambda_\mu}$")
    ax[0].loglog(N, out["lam_std"], "s", color="#e34a33",
                 label=r"bulk $\lambda$ (empirical std)")
    ax[0].loglog(N, out["lam_gm_std"], "-", color="#e34a33",
                 label=r"Gauss-Markov $\sigma/\sqrt{\lambda_\lambda}$")
    ax[0].set_xlabel("number of weak-form samples $N$")
    ax[0].set_ylabel("estimator standard deviation")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Estimator spread follows the Gauss-Markov bound\n"
                    "(empirical std vs prediction)", fontsize=10)
    ax[1].loglog(N, 100 * np.array(out["mu_err"]), "o-", color="#2b8cbe",
                 label=r"shear $\mu$")
    ax[1].loglog(N, 100 * np.array(out["lam_err"]), "s-", color="#e34a33",
                 label=r"bulk $\lambda$")
    ax[1].set_xlabel("number of weak-form samples $N$")
    ax[1].set_ylabel("error to truth (%)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].set_title("Excited mode improves with $N$; starved mode does not\n"
                    f"Fisher eigenvalue ratio "
                    f"$\\lambda_\\mu/\\lambda_\\lambda={out['fisher_ratio']:.0e}$",
                    fontsize=10)
    fig.tight_layout()
    p = publish_figure(fig, "sample_complexity.png", dpi=150, tight=True)
    plt.close(fig)
    log(f"wrote {p}")
