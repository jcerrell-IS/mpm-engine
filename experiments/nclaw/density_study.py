"""Identification from positions alone as a function of particle density.

The positions-only chain is exact except for one link. Velocities are exact
(both engines advect x by dt v). The elastic F is exact given the affine
increments C (replaying stored C reproduces stored F to 1e-5). The lossy
link is estimating C from positions, and its error is set by observation
density: the grid velocity field is only recoverable where several particles
fall in each cell. NCLaw's clouds have one per cell and the compounded
replay fails the momentum fit there.

This study measures the link directly. One plasticine throw per density
(particles per cell 1, 8, 27 by lattice pitch), our engine at the NCLaw
grid semantics, dumped at EVERY simulation step so the backward position
difference is the step velocity. The chain then runs from positions alone:

  positions -> velocities (backward difference)
            -> affine increments C (engine-kernel observer, replay.py)
            -> elastic F (return-map replay, self-consistent in tau_y)
            -> (mu, lambda) and tau_y by the convex least-squares momentum
               fit; no assumed parameters, no rollout scan, no differentiation

and is scored against the dump's own stored channels: per-step C error where
comparable, end-to-end F error, recovered parameters against truth, fit
residuals, and a rollout at the recovered parameters seeded from frame 1
(whose velocity the tier knows exactly), scored by position MSE against the
truth trajectory. Same engine on both sides, so the study isolates
observability at a given density.

Run from the engine root (about half an hour for the dense case):

    .venv/bin/python -m experiments.nclaw.density_study [ppc ...]

Writes out/nclaw_density_study/{dumps,results.json}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.nclaw import replay as rp
from experiments.nclaw import suite

OUT = suite.ROOT / "out" / "nclaw_density_study"
DUMPS = OUT / "dumps"
TRUTH = {"E": 3.0e5, "nu": 0.25, "yield_stress": 5000.0}
T_END, N_FRAMES = 0.4, 800          # dt 5e-4, every step dumped
N_GRID, GRID_LIM = 20, 1.0
PITCH_DIV = {1: 1, 8: 2, 27: 3}     # particles per cell -> lattice pitch dx/div


def cube_cloud(ppc: int) -> dict:
    dx = GRID_LIM / N_GRID
    h = dx / PITCH_DIV[ppc]
    half = 0.25
    ax = np.arange(-half, half + 0.5 * h, h)
    pts = (np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
           + suite.CENTER).astype(np.float32)
    return {"pts": pts, "vol0": np.full(len(pts), h ** 3, dtype=np.float32),
            "v0": suite.throw_velocity(pts), "n_frames": N_FRAMES,
            "t_end": T_END, "n_grid": N_GRID, "grid_lim": GRID_LIM,
            "source": f"cube_ppc{ppc}"}


def chain(ppc: int, increment_estimator: str = "knn", log=print) -> dict:
    """The chain at one particle density, with one increment estimator.

    ``increment_estimator`` selects how the per-step deformation increments
    are estimated from positions. "grid" scatters the measured velocities to
    the simulation grid with the engine's B-spline weights and reads the
    velocity gradient back (replay.grid_affine_increments); its error scales
    with the whole affine field, and the throw's rigid rotation is 6 to 24
    times the strain rate, so a 3 percent operator error is a 22 to 60
    percent strain-rate error at any density.
    "knn" fits, for each particle, the linear map that best carries its 16
    nearest neighbours from one frame to the next
    (replay.incremental_gradients); a linear fit reproduces rigid rotation
    exactly, so its error scales with the strain variation inside the
    neighbourhood, which shrinks with density.
    """
    dump = DUMPS / f"plasticine_ppc{ppc}_truth.npz"
    if not dump.exists():
        t0 = time.time()
        suite.run_scene("plasticine", "cube", dump, cloud=cube_cloud(ppc),
                        nclaw_bc=True, substeps=1, log=log)
        log(f"[density] ppc={ppc} truth generated in {time.time() - t0:.0f}s")
    arr = suite._load_arrays(dump)
    x = arr["x"]
    T, N = x.shape[:2]
    dt = arr["frame_dt"]
    res: dict = {"ppc": ppc, "increment_estimator": increment_estimator, "n_particles": N, "n_frames": T}

    # positions-only from here: velocities by the exact backward difference
    v_fd = np.zeros_like(arr["v"])
    v_fd[1:] = (x[1:] - x[:-1]) / dt
    v_fd[0] = v_fd[1]
    verr = np.linalg.norm(v_fd[1:] - arr["v"][1:], axis=-1)
    res["fd_velocity_median_abs_err"] = float(np.median(verr))
    arr_po = dict(arr)
    arr_po["v"] = v_fd

    t0 = time.time()
    if increment_estimator == "grid":
        Cinc = rp.grid_affine_increments(x, dt, N_GRID, GRID_LIM, arr["mass"])
        Finc = np.eye(3) + dt * Cinc
    else:
        Finc = rp.incremental_gradients(x, k=16)
    res["observer_seconds"] = round(time.time() - t0, 1)

    def replay(tau_y: float | None):
        Fe = np.tile(np.eye(3), (N, 1, 1))
        out = np.empty((T, N, 3, 3))
        out[0] = Fe
        flow = np.zeros((T, N), bool)
        for n in range(T - 1):
            Fe = Finc[n] @ Fe
            if tau_y is not None:
                Fe, cl = rp.project_von_mises(
                    Fe, TRUTH["E"], TRUTH["nu"], tau_y)
                flow[n + 1] = cl
            out[n + 1] = Fe
        return out, flow

    # elastic pair on the pre-yield window of the unprojected (total) replay
    F_tot, _ = replay(None)
    arr_el = dict(arr_po)
    arr_el["F"] = F_tot
    frames = rp.pre_yield_frames(arr_el)
    el = suite.identify_elastic(arr_el, window_frames=26, frames=frames, log=log)
    res["elastic_preyield"] = {k: el.get(k) for k in
                               ("E", "nu", "mu", "lam", "residual_rel",
                                "n_rows", "refused")}
    if el.get("refused", True):
        res["verdict"] = "elastic fit refused"
        return res
    mu_h, lam_h = float(el["mu"]), float(el["lam"])
    E_h = float(el["E"])
    nu_h = float(el["nu"])

    # yield stress, self-consistent: replay at a candidate, fit, repeat
    tau, path = 1.0e4, []
    fit: dict = {}
    for it in range(6):
        Fe, flow = replay(tau)
        fit = rp._fit_scale_on_flow_set(
            arr_po, Fe, flow, E_h, nu_h, np.ones((T, N)),
            window_frames=26, frame_stride=2, margin_cells=3.0,
            valid_frac_min=0.9, log=log)
        if fit.get("refused"):
            break
        tau_new = float(fit["theta"][0])
        path.append({"candidate": tau, "fit": tau_new,
                     "residual_rel": float(fit["residual_rel"])})
        log(f"[density] ppc={ppc} yield iter {it}: {tau:.4g} -> {tau_new:.4g} "
            f"resid {float(fit['residual_rel']):.3f}")
        done = abs(tau_new - tau) < 0.01 * tau
        tau = tau_new
        if done:
            break
    refused = bool(fit.get("refused", True)) or \
        float(fit.get("residual_rel", 1.0)) > 0.15
    res["yield_fit"] = {"yield_stress": tau, "iterations": path,
                        "residual_rel": fit.get("residual_rel"),
                        "refused": refused}

    # end-to-end reconstruction fidelity against the dump's stored channels
    Fe, _ = replay(tau)
    fid = {}
    for fr in (100, 300, 600):
        if fr >= T:
            continue
        ferr = (np.linalg.norm(Fe[fr] - arr["F"][fr], axis=(1, 2))
                / np.maximum(np.linalg.norm(arr["F"][fr], axis=(1, 2)), 1e-9))
        fid[f"frame_{fr}"] = {"F_rel_err_p50": float(np.percentile(ferr, 50)),
                              "F_rel_err_p95": float(np.percentile(ferr, 95))}
    res["reconstruction_vs_stored_F"] = fid

    res["theta_recovered"] = {"E": E_h, "nu": nu_h, "yield_stress": tau}
    res["theta_errors_pct"] = {
        "E": 100.0 * (E_h / TRUTH["E"] - 1.0),
        "nu": 100.0 * (nu_h / TRUTH["nu"] - 1.0),
        "yield_stress": 100.0 * (tau / TRUTH["yield_stress"] - 1.0)}

    # evaluation rollout, seeded from frame 1 whose velocity is known exactly
    if not refused:
        theta = {"E": E_h, "nu": nu_h, "yield_stress": tau}
        cloud = {"pts": np.ascontiguousarray(x[1].astype(np.float32)),
                 "vol0": np.ascontiguousarray(arr["vol0"].astype(np.float32)),
                 "v0": np.ascontiguousarray(v_fd[1].astype(np.float32)),
                 "n_frames": T - 2, "t_end": float(dt * (T - 2)),
                 "n_grid": N_GRID, "grid_lim": GRID_LIM,
                 "source": f"frame1_of_ppc{ppc}"}
        pred = DUMPS / f"plasticine_ppc{ppc}_{increment_estimator}_recovered.npz"
        if not pred.exists():
            suite.run_scene("plasticine", "cube", pred, theta=theta,
                            cloud=cloud, nclaw_bc=True, substeps=1, log=log)
        d_t = np.load(dump)["x"][1:T - 1]
        d_p = np.load(pred)["x"]
        n = min(len(d_t), len(d_p))
        mse = float(np.mean((d_t[:n:5] - d_p[:n:5]) ** 2))
        res["rollout_mse_recovered_vs_truth"] = mse
    return res


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    increment_estimator = "grid" if "--grid" in args else "knn"
    ppcs = [int(a) for a in args if not a.startswith("--")] or [1, 8, 27]
    DUMPS.mkdir(parents=True, exist_ok=True)
    results = {}
    path = OUT / "results.json"
    if path.exists():
        results = json.loads(path.read_text())
    for ppc in ppcs:
        results[f"{ppc}_{increment_estimator}"] = chain(ppc, increment_estimator=increment_estimator)
        path.write_text(json.dumps(results, indent=2, default=float))
        print(f"[density] ppc={ppc} {increment_estimator} recorded -> {path}")
    for key, r in sorted(results.items()):
        print(key, ":", r.get("theta_errors_pct"),
              "yield refused:", r.get("yield_fit", {}).get("refused"),
              "rollout MSE:", r.get("rollout_mse_recovered_vs_truth"))


if __name__ == "__main__":
    main()
