"""Differentiable-simulation baseline driver: Adam through a JAX MPM rollout.

The comparison method of the video2sim plan doc docs/diffsim_baseline_plan.md.
For each NCLaw-suite material it fits the unknown parameters by gradient descent
on the frame-subsampled particle-position MSE against the warp grid-20 truth,
with forward-mode gradients taken through the whole rollout (forward.py). Our
own identification is the convex weak-form solve in experiments.nclaw.suite;
this file is its alternative, not a part of it.

What the campaign found, in one line per claim: the trade is budget, not
correctness (jelly mu +0.4 percent in 52.4 min against +0.8 percent in 5.0 s for
the convex solve); the plasticine elastic degeneracy is a valley, since
continuing the best init reaches mu 0.02 percent and lam 0.01 percent; and the
sand chaos expectation was falsified, its phi landscape being unimodal with all
five inits converging. Full table in docs/four_method_comparison.md.

Stages, resumable, one material at a time:

  validate   cross-engine forward gap at truth theta, at the dump's own time
             step and at the fit's time step (the two floors of the comparison)
  landscape  forward-only loss sweep across the prior, coarse and fine
             (pre-registered for sand, run for water too since it is 1-D)
  fit        Adam from 5 prior draws, per-iteration jsonl ledger, finished
             inits skipped on a re-run
  refine     the best init continued past the equal-budget stop, at a third of
             the step, to separate "what one budget buys" from "where it lands"
  ls         the least-squares row on the SAME dump, timed
  report     per-material json plus out/diffsim_baseline/report.md

Time step: the substep count is fixed for the whole fit (a theta-dependent count
would change the traced graph between iterations) at the value the warp truth run
used, and the optimizer is clipped to the prior box. That is the most favourable
choice available to the baseline: the loss at truth theta then sits at the
cross-engine forward gap, 1e-12 to 3e-10, so nothing about the time step can bias
the recovered theta. Measured alternative, recorded rather than used: sizing the
substeps from the CFL at the stiff corner of the prior instead (6 rather than 4
for jelly, 5 rather than 4 for water) raises the floor to 1.5e-5 and 1.4e-4 by
time discretization alone, which would have swamped the fit. The whole prior box
is stable at the truth substep count, measured directly.

Artifacts, all under out/diffsim_baseline (see experiments.diffsim.artifact_dir
for which out/ tree that is): validate_<m>.json, landscape_<m>.json and .png,
fit_<m>.jsonl, fit_<m>_inits.jsonl, refine_<m>.json, ls_<m>.json,
results_<m>.json, fit_<m>.png, report.md.

Run (jax lives in the video2sim staging venv, not the engine venv; run from the
mpm_engine root):
  ../.venv/bin/python -m experiments.diffsim validate  --material jelly
  ../.venv/bin/python -m experiments.diffsim fit       --material jelly
  ../.venv/bin/python -m experiments.diffsim landscape --material sand
  ../.venv/bin/python -m experiments.diffsim report    --material all
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import ENGINE_ROOT, artifact_dir
from .forward import (
    Theta,
    alpha_from_friction,
    load_truth,
    make_loss,
    mu_lam_from_E_nu,
    rollout,
)

ENGINE = ENGINE_ROOT
OUT = artifact_dir("diffsim_baseline")
# the root the recorded figure paths are relative to, so results_<m>.json keeps
# saying "out/diffsim_baseline/fit_jelly.png" wherever OUT resolved
OUT_ROOT = OUT.parent.parent
DUMPS = ENGINE / "out" / "nclaw_suite" / "dumps"
RHO = 1000.0
CFL = 0.35                     # nclaw/suite.py run_scene
FRAME_STRIDE = 5               # NCLaw's own metric cadence
CUBE_MM = 500.0                # the cube of particles is 0.5 m on a side

# Per-material unknowns, priors and budgets. The prior is the plan's: log-uniform
# over a decade around truth for moduli, uniform 15 to 45 degrees for the
# friction angle. Iteration budgets are set from the measured per-iteration wall
# time so that one material's five inits land inside about 90 minutes.
SPECS: dict[str, dict] = {
    "jelly": {
        "params": [("mu", 41666.6667, "log10"), ("lam", 27777.7778, "log10")],
        "lr": 0.05, "iters": 120, "inits": 5,
        "truth_engine": {"E": 1.0e5, "nu": 0.2},
    },
    "plasticine": {
        "params": [("mu", 120000.0, "log10"), ("lam", 120000.0, "log10"),
                   ("yield_stress", 5000.0, "log10")],
        "lr": 0.05, "iters": 40, "inits": 5,
        "truth_engine": {"E": 3.0e5, "nu": 0.25, "yield_stress": 5.0e3},
    },
    "sand": {
        "params": [("friction_angle", 25.0, "deg")],
        "lr": 0.5, "iters": 80, "inits": 5,
        "fixed": {"E": 1.0e6, "nu": 0.2},
        "truth_engine": {"E": 1.0e6, "nu": 0.2, "friction_angle": 25.0},
    },
    "water": {
        "params": [("bulk_modulus", 83333.3333, "log10")],
        "lr": 0.05, "iters": 150, "inits": 5,
        "truth_engine": {"bulk_modulus": 83333.3333},
    },
}
PHI_PRIOR = (15.0, 45.0)
DECADE = 0.5                   # half width of the log-uniform prior, in dex


# ---------------------------------------------------------------------------
# parameter coordinates: q is what Adam moves, theta is physical
# ---------------------------------------------------------------------------

def prior_box(material: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for _, truth, kind in SPECS[material]["params"]:
        if kind == "log10":
            lo.append(math.log10(truth) - DECADE)
            hi.append(math.log10(truth) + DECADE)
        else:
            lo.append(PHI_PRIOR[0])
            hi.append(PHI_PRIOR[1])
    return np.array(lo), np.array(hi)


def q_of_theta(material: str, theta: dict) -> np.ndarray:
    q = []
    for name, _, kind in SPECS[material]["params"]:
        q.append(math.log10(theta[name]) if kind == "log10" else theta[name])
    return np.array(q, dtype=np.float64)


def theta_of_q(material: str, q) -> dict:
    out = {}
    for i, (name, _, kind) in enumerate(SPECS[material]["params"]):
        out[name] = float(10.0 ** q[i]) if kind == "log10" else float(q[i])
    return out


def truth_q(material: str) -> np.ndarray:
    return q_of_theta(material, {n: t for n, t, _ in SPECS[material]["params"]})


def unpacker(material: str):
    """q (jax array) -> Theta for the engine."""
    if material == "jelly":
        return lambda q: Theta(mu=10.0 ** q[0], lam=10.0 ** q[1])
    if material == "plasticine":
        return lambda q: Theta(mu=10.0 ** q[0], lam=10.0 ** q[1],
                               yield_stress=10.0 ** q[2])
    if material == "water":
        return lambda q: Theta(bulk=10.0 ** q[0])
    if material == "sand":
        fx = SPECS["sand"]["fixed"]
        mu, lam = mu_lam_from_E_nu(fx["E"], fx["nu"])
        return lambda q: Theta(mu=jnp.float32(mu), lam=jnp.float32(lam),
                               alpha=alpha_from_friction(q[0]))
    raise KeyError(material)


def truth_theta_obj(material: str) -> Theta:
    return unpacker(material)(jnp.asarray(truth_q(material), jnp.float32))


def wave_speed(material: str, q) -> float:
    """The p-wave speed run_scene sizes its time step from, at coordinates q."""
    th = theta_of_q(material, q)
    if material == "water":
        return math.sqrt(th["bulk_modulus"] / RHO)
    if material == "sand":
        fx = SPECS["sand"]["fixed"]
        mu, lam = mu_lam_from_E_nu(fx["E"], fx["nu"])
    else:
        mu, lam = th["mu"], th["lam"]
    return math.sqrt((lam + 2.0 * mu) / RHO)


def prior_cfl_substeps(material: str, frame_dt: float, dx: float, v_max: float) -> int:
    """The substep count run_scene's CFL rule would ask for at the stiff corner of
    the prior. Reported by the validate stage as the measured alternative; the fit
    uses the truth run's own count (see the module docstring)."""
    _, hi = prior_box(material)
    c = wave_speed(material, hi)
    dt_cfl = CFL * dx / max(c + v_max, 1e-9)
    return max(math.ceil(frame_dt / dt_cfl), 1)


# ---------------------------------------------------------------------------
# problem assembly
# ---------------------------------------------------------------------------

def dump_for(material: str) -> Path:
    p = DUMPS / f"{material}_cube_g20_truth.npz"
    if not p.exists():
        raise SystemExit(
            f"missing {p}. Generate it with:\n  cd {ENGINE} && .venv/bin/python "
            f"-m experiments.nclaw.suite gen --material {material} --shapes cube "
            "--n-grid 20")
    return p


def _figure_path(p: Path) -> str:
    """A figure path as the results json records it, relative where possible."""
    try:
        return str(p.relative_to(OUT_ROOT))
    except ValueError:
        return str(p)


def build(material: str, substeps: int | None = None, n_frames: int | None = None):
    """Cloud, truth, scene and loss for one material, at the truth time step."""
    path = dump_for(material)
    cloud0, xt, scene0, meta = load_truth(path, material)
    frame_dt = scene0.dt * scene0.substeps
    v_max = float(np.abs(np.asarray(cloud0["v0"])).max())
    sub = substeps if substeps is not None else scene0.substeps
    scene = scene0._replace(dt=frame_dt / sub, substeps=sub)
    if n_frames is not None:
        scene = scene._replace(n_frames=n_frames)
    frames = list(range(0, scene.n_frames + 1, FRAME_STRIDE))
    unpack = unpacker(material)
    loss = make_loss(cloud0, xt, scene, frames, unpack)
    return {"cloud": cloud0, "x_truth": xt, "scene": scene, "scene_truth": scene0,
            "loss": loss, "unpack": unpack, "frames": frames, "meta": meta,
            "frame_dt": frame_dt, "v_max": v_max, "dump": path.name}


def value_and_grad(loss, k: int):
    """Forward-mode value and gradient: k JVPs, vectorized over the tangents.

    jax.linearize is the wrong tool here even though it looks cheaper: its partial
    evaluation stores residuals for every substep of the rollout, the reverse-mode
    memory pattern, and the measured cost was worse than the whole fit budget.
    vmap over jvp keeps the pass tape free and hands back the primal for free.
    """
    basis = jnp.eye(k, dtype=jnp.float32)

    @jax.jit
    def vg(q):
        ys, gs = jax.vmap(lambda e: jax.jvp(loss, (q,), (e,)))(basis)
        return ys[0], gs

    return vg


# ---------------------------------------------------------------------------
# stage: validate (the cross-engine forward gap, both time steps)
# ---------------------------------------------------------------------------

def stage_validate(material: str, log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prob_t = build(material, substeps=None)
    scene_truth = prob_t["scene_truth"]
    th = truth_theta_obj(material)
    xt = prob_t["x_truth"]
    res = {"material": material, "dump": prob_t["dump"],
           "n_particles": int(xt.shape[1]), "n_frames": int(xt.shape[0]) - 1,
           "cube_mm": CUBE_MM, "truth_theta": theta_of_q(material, truth_q(material))}

    sub_prior = prior_cfl_substeps(material, prob_t["frame_dt"],
                                   scene_truth.dx, prob_t["v_max"])
    runs = {"warp_substeps": scene_truth,
            "prior_cfl_substeps": scene_truth._replace(
                dt=prob_t["frame_dt"] / sub_prior, substeps=sub_prior)}
    per = {}
    for tag, scene in runs.items():
        t0 = time.time()
        x = np.asarray(rollout(th, prob_t["cloud"], scene))
        wall = time.time() - t0
        d = x - xt
        rms_mm = 1e3 * np.sqrt((d ** 2).sum(-1).mean(-1))
        mse = (d ** 2).mean(axis=(1, 2))
        per[tag] = {
            "substeps": int(scene.substeps), "dt": float(scene.dt),
            "wall_seconds": wall,
            "rms_mm_per_frame": [float(v) for v in rms_mm],
            "rms_mm_final": float(rms_mm[-1]),
            "rms_mm_max": float(rms_mm.max()),
            "rms_mm_final_over_cube": float(rms_mm[-1] / CUBE_MM),
            "mse_metric": float(mse[::FRAME_STRIDE].mean()),
            "finite": bool(np.isfinite(d).all()),
        }
        log(f"[validate] {material} {tag}: sub={scene.substeps} "
            f"final RMS {rms_mm[-1]:.4f} mm ({100 * rms_mm[-1] / CUBE_MM:.5f} % of "
            f"the cube), loss floor {per[tag]['mse_metric']:.3e}, {wall:.1f} s")
    res["forward_gap"] = per
    res["loss_at_truth_fit_dt"] = per["warp_substeps"]["mse_metric"]
    res["note"] = (
        "warp_substeps is the fit's time step (the truth run's own) and its "
        "mse_metric is the loss floor the fit can reach. prior_cfl_substeps is the "
        "measured alternative the fit does not use: sizing the step from the CFL at "
        "the stiff corner of the prior, whose floor is time discretization only.")
    (OUT / f"validate_{material}.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# stage: landscape (forward-only sweep)
# ---------------------------------------------------------------------------

def stage_landscape(material: str, n_points: int = 21, log=print) -> dict:
    if len(SPECS[material]["params"]) != 1:
        raise SystemExit(f"the landscape stage is 1-D; {material} has "
                         f"{len(SPECS[material]['params'])} unknowns")
    OUT.mkdir(parents=True, exist_ok=True)
    prob = build(material)
    loss = prob["loss"]
    name, truth, kind = SPECS[material]["params"][0]
    lo, hi = prior_box(material)
    qt = truth_q(material)[0]

    scans = {"coarse": np.linspace(lo[0], hi[0], n_points)}
    fine_half = 1.0 if kind == "deg" else 0.02        # 1 degree, or 0.02 dex
    scans["fine"] = np.linspace(qt - fine_half, qt + fine_half, n_points)

    out = {"material": material, "param": name, "kind": kind, "truth": truth,
           "prior": [float(lo[0]), float(hi[0])], "scans": {}}
    for tag, qs in scans.items():
        vals, walls = [], []
        for q in qs:
            t0 = time.time()
            vals.append(float(loss(jnp.asarray([q], jnp.float32))))
            walls.append(time.time() - t0)
        theta = [float(10.0 ** q) if kind == "log10" else float(q) for q in qs]
        v = np.asarray(vals)
        # smoothness read-outs: the discrete second difference against the local
        # scale of the first difference. A smooth basin has |d2| of the same order
        # as the curvature; contact chaos shows up as second differences that do
        # not shrink with the step.
        d1 = np.diff(v)
        d2 = np.diff(v, 2)
        sign_flips = int(np.sum(np.diff(np.sign(d1)) != 0))
        out["scans"][tag] = {
            "q": [float(q) for q in qs], "theta": theta, "loss": vals,
            "argmin_theta": theta[int(np.argmin(v))],
            "min_loss": float(v.min()),
            "first_diff_max_abs": float(np.abs(d1).max()),
            "second_diff_max_abs": float(np.abs(d2).max()),
            "monotone_pieces_sign_flips": sign_flips,
            "unimodal": bool(sign_flips <= 1),
            "wall_seconds": float(np.sum(walls)),
        }
        log(f"[landscape] {material} {tag}: min at {name}="
            f"{out['scans'][tag]['argmin_theta']:.4g} (truth {truth:.4g}), "
            f"loss {v.min():.3e} to {v.max():.3e}, sign flips in the slope "
            f"{sign_flips}, unimodal {out['scans'][tag]['unimodal']}")
    (OUT / f"landscape_{material}.json").write_text(json.dumps(out, indent=2))
    _landscape_figure(material, out)
    return out


def _landscape_figure(material: str, res: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, tag in zip(axes, ("coarse", "fine"), strict=True):
        s = res["scans"][tag]
        ax.semilogy(s["theta"], s["loss"], "o-", lw=1.6, ms=4, color="#1c7ed6")
        ax.axvline(res["truth"], color="#e8590c", ls="--", lw=1.4, label="truth")
        ax.set_xlabel(res["param"])
        ax.set_ylabel("position MSE (unit box)")
        ax.set_title(f"({'a' if tag == 'coarse' else 'b'}) {material}: {tag} sweep")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / f"landscape_{material}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# stage: fit (Adam, resumable)
# ---------------------------------------------------------------------------

LR_FLOOR = 0.02                # cosine decay lands at this fraction of lr


def adam_fit(vg, q0, lo, hi, lr, n_iter, on_iter, plateau_rel=1e-4,
             plateau_len=10):
    """Plain Adam in the q coordinates, clipped to the prior box, cosine-decayed.

    optax is not installed in this tree, so this is the twenty-line version: the
    update is the standard bias-corrected Adam step, which is scale free in the
    gradient, so lr is read in the units of q (dex for a modulus, degrees for the
    friction angle) and the step length is about lr per iteration regardless of
    how steep the loss is. That is also why the schedule matters: measured on
    jelly at a constant lr of 0.05 dex, the iterate overshoots the basin and
    oscillates at a 12 percent radius in mu, which is the step length and not the
    identifiability. Cosine decay to LR_FLOOR * lr over the budget removes that
    floor.
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    q = np.array(q0, dtype=np.float64)
    m = np.zeros_like(q)
    v = np.zeros_like(q)
    best = {"loss": math.inf, "q": q.copy(), "iter": 0}
    hist: list[float] = []
    stop = "iterations"
    for t in range(1, n_iter + 1):
        y, g = vg(jnp.asarray(q, jnp.float32))
        y = float(y)
        g = np.asarray(g, dtype=np.float64)
        if not np.isfinite(y) or not np.isfinite(g).all():
            stop = "non-finite loss or gradient"
            on_iter(t, q, y, g, stop)
            break
        if y < best["loss"]:
            best = {"loss": y, "q": q.copy(), "iter": t}
        on_iter(t, q, y, g, None)
        hist.append(y)
        if len(hist) > plateau_len:
            window = hist[-(plateau_len + 1):]
            # spread over the window, not its endpoints: an oscillating iterate
            # can return to the same value and is not a plateau
            rel = (max(window) - min(window)) / max(abs(window[-1]), 1e-30)
            if rel < plateau_rel:
                stop = "plateau"
                break
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        lr_t = lr * (LR_FLOOR + (1.0 - LR_FLOOR)
                     * 0.5 * (1.0 + math.cos(math.pi * t / max(n_iter, 1))))
        q = np.clip(q - lr_t * mh / (np.sqrt(vh) + eps), lo, hi)
    return best, len(hist), stop


def stage_fit(material: str, inits: int | None = None, iters: int | None = None,
              log=print) -> dict:
    spec = SPECS[material]
    n_inits = spec["inits"] if inits is None else inits
    n_iter = spec["iters"] if iters is None else iters
    OUT.mkdir(parents=True, exist_ok=True)
    prob = build(material)
    k = len(spec["params"])
    vg = value_and_grad(prob["loss"], k)
    lo, hi = prior_box(material)
    rng = np.random.default_rng(20260820)
    q_inits = [lo + (hi - lo) * rng.random(k) for _ in range(n_inits)]

    ledger = OUT / f"fit_{material}.jsonl"
    done_path = OUT / f"fit_{material}_inits.jsonl"
    done: dict[int, dict] = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["init"])] = r
        log(f"[fit] {material}: resuming, {sorted(done)} already finished")

    for i, q0 in enumerate(q_inits):
        if i in done:
            continue
        t_start = time.time()
        with ledger.open("a") as fh:
            def on_iter(t, q, y, g, stop, fh=fh, i=i, t_start=t_start):
                rec = {"init": i, "iter": t, "loss": y,
                       "theta": theta_of_q(material, q), "q": list(map(float, q)),
                       "grad": [float(x) for x in np.atleast_1d(g)],
                       "wall_s": time.time() - t_start}
                if stop:
                    rec["stop"] = stop
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if t % 5 == 1 or t == 1:
                    log(f"[fit] {material} init {i} iter {t:3d} loss {y:.4e} "
                        f"theta {rec['theta']} ({rec['wall_s']:.0f} s)")

            best, n_done, stop = adam_fit(vg, q0, lo, hi, spec["lr"], n_iter,
                                          on_iter)
        rec = {"init": i, "q0": list(map(float, q0)),
               "theta0": theta_of_q(material, q0),
               "best_loss": best["loss"], "best_theta": theta_of_q(material, best["q"]),
               "best_q": list(map(float, best["q"])), "best_iter": best["iter"],
               "iterations": n_done, "stop": stop,
               "wall_seconds": time.time() - t_start,
               "seconds_per_iteration": (time.time() - t_start) / max(n_done, 1)}
        with done_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[i] = rec
        log(f"[fit] {material} init {i} done: loss {best['loss']:.4e} theta "
            f"{rec['best_theta']} after {n_done} iterations "
            f"({rec['wall_seconds']:.0f} s, {rec['seconds_per_iteration']:.1f} s/iter, "
            f"stop: {stop})")

    return {"material": material, "inits": [done[i] for i in sorted(done)],
            "substeps": int(prob["scene"].substeps), "n_unknowns": k,
            "iteration_budget": n_iter, "lr": spec["lr"],
            "prior_box": {"lo": list(map(float, lo)), "hi": list(map(float, hi))}}


def stage_refine(material: str, iters: int = 250, lr_scale: float = 0.3,
                 log=print) -> dict:
    """Continue the best init past the equal-budget stop, from its own iterate.

    The fit stage answers "what does a 90 minute diff-sim budget buy". Measured on
    jelly, the answer was an iterate still descending at the last iteration, which
    would understate the method. This stage answers the other question, "where
    does it land if allowed to converge", by restarting the best init at a smaller
    step and running until it plateaus. The two rows are reported separately
    because they are different claims.
    """
    inits_path = OUT / f"fit_{material}_inits.jsonl"
    if not inits_path.exists():
        raise SystemExit(f"missing {inits_path}; run the fit stage first")
    recs = [json.loads(ln) for ln in inits_path.read_text().splitlines()
            if ln.strip()]
    best = min(recs, key=lambda r: r["best_loss"])
    spec = SPECS[material]
    prob = build(material)
    vg = value_and_grad(prob["loss"], len(spec["params"]))
    lo, hi = prior_box(material)
    ledger = OUT / f"refine_{material}.jsonl"
    t_start = time.time()
    with ledger.open("a") as fh:
        def on_iter(t, q, y, g, stop, fh=fh):
            rec = {"init": best["init"], "iter": t, "loss": y,
                   "theta": theta_of_q(material, q),
                   "q": list(map(float, q)), "wall_s": time.time() - t_start}
            if stop:
                rec["stop"] = stop
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if t % 10 == 1:
                log(f"[refine] {material} iter {t:3d} loss {y:.4e} "
                    f"theta {rec['theta']}")

        out, n_done, stop = adam_fit(vg, np.asarray(best["best_q"]), lo, hi,
                                     spec["lr"] * lr_scale, iters, on_iter)
    res = {"material": material, "from_init": best["init"],
           "start_theta": best["best_theta"], "start_loss": best["best_loss"],
           "lr": spec["lr"] * lr_scale, "iterations": n_done, "stop": stop,
           "best_loss": out["loss"], "best_theta": theta_of_q(material, out["q"]),
           "best_iter": out["iter"], "wall_seconds": time.time() - t_start,
           "seconds_per_iteration": (time.time() - t_start) / max(n_done, 1)}
    (OUT / f"refine_{material}.json").write_text(json.dumps(res, indent=2))
    log(f"[refine] {material}: {res['start_loss']:.3e} -> {res['best_loss']:.3e}, "
        f"theta {res['best_theta']} after {n_done} iterations "
        f"({res['wall_seconds'] / 60:.1f} min, stop: {stop})")
    return res


# ---------------------------------------------------------------------------
# stage: ls (the least-squares row on the same dump)
# ---------------------------------------------------------------------------

def stage_ls(material: str, log=print) -> dict:
    """Run the weak-form identification on the SAME grid-20 dump and time it."""
    OUT.mkdir(parents=True, exist_ok=True)
    dump = dump_for(material)
    cmd = [str(ENGINE / ".venv" / "bin" / "python"), "-m",
           "experiments.nclaw.suite", "identify", "--material", material,
           "--dump", str(dump), "--tag", "g20"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ENGINE, capture_output=True, text=True)
    wall = time.time() - t0
    res = {"material": material, "dump": dump.name, "wall_seconds": wall,
           "cmd": " ".join(cmd), "returncode": proc.returncode,
           "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    ident_path = ENGINE / "out" / "nclaw_suite" / f"identify_{material}_g20.json"
    if ident_path.exists():
        res["identify"] = json.loads(ident_path.read_text())
    (OUT / f"ls_{material}.json").write_text(json.dumps(res, indent=2))
    log(f"[ls] {material}: rc={proc.returncode} {wall:.1f} s "
        f"theta={res.get('identify', {}).get('theta_engine')}")
    return res


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------

def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def theta_compare(material: str, theta: dict) -> dict:
    """Recovered against truth in the same physical parameters, per unknown."""
    out = {}
    for name, truth, _ in SPECS[material]["params"]:
        got = theta.get(name)
        out[name] = {"recovered": got, "truth": truth,
                     "rel_err": (None if got is None else abs(got / truth - 1.0))}
    return out


def theta_engine_from_fit(material: str, theta: dict | None) -> dict | None:
    """The fitted theta in warp-mpm's own arguments, so the recovered law can be
    rolled out in the truth engine without a second conversion step."""
    if not theta:
        return None
    if material == "sand":
        fx = SPECS["sand"]["fixed"]
        return {"E": fx["E"], "nu": fx["nu"],
                "friction_angle": theta["friction_angle"]}
    if material == "water":
        return {"bulk_modulus": theta["bulk_modulus"],
                "E": 1.0e5, "nu": 0.3}
    mu, lam = theta["mu"], theta["lam"]
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    out = {"E": E, "nu": nu}
    if material == "plasticine":
        out["yield_stress"] = theta["yield_stress"]
        out["softening"] = 0.0
    return out


def ls_theta_in_our_params(material: str, ident: dict | None) -> dict | None:
    """The LS row's theta expressed in the unknowns the diff-sim fits."""
    if not ident:
        return None
    te = ident.get("theta_engine", {})
    if material in ("jelly", "plasticine"):
        el = ident.get("elastic", {})
        out = {"mu": el.get("mu"), "lam": el.get("lam")}
        if material == "plasticine":
            out["yield_stress"] = te.get("yield_stress")
        return out
    if material == "sand":
        return {"friction_angle": te.get("friction_angle")}
    if material == "water":
        return {"bulk_modulus": te.get("bulk_modulus")}
    return None


def _fit_figure(material: str) -> Path | None:
    """Loss and parameter traces per init, straight from the iteration ledger."""
    ledger = OUT / f"fit_{material}.jsonl"
    if not ledger.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    inits = sorted({r["init"] for r in recs})
    names = [n for n, _, _ in SPECS[material]["params"]]
    fig, axes = plt.subplots(1, 1 + len(names), figsize=(5.0 * (1 + len(names)), 4.2),
                             squeeze=False)
    ax = axes[0][0]
    for i in inits:
        rs = [r for r in recs if r["init"] == i]
        ax.semilogy([r["iter"] for r in rs],
                    np.maximum([r["loss"] for r in rs], 1e-14), lw=1.5,
                    label=f"init {i}")
    val = _read(OUT / f"validate_{material}.json")
    if val:
        ax.axhline(val["loss_at_truth_fit_dt"], color="k", ls=":", lw=1.3,
                   label="loss at truth (floor)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("position MSE (unit box)")
    ax.set_title(f"(a) {material}: Adam through the rollout")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    for j, (name, truth, _) in enumerate(SPECS[material]["params"]):
        ax = axes[0][j + 1]
        for i in inits:
            rs = [r for r in recs if r["init"] == i]
            ax.plot([r["iter"] for r in rs], [r["theta"][name] for r in rs], lw=1.5,
                    label=f"init {i}")
        ax.axhline(truth, color="#e8590c", ls="--", lw=1.4, label="truth")
        ax.set_xlabel("iteration")
        ax.set_ylabel(name)
        if SPECS[material]["params"][j][2] == "log10":
            ax.set_yscale("log")
        ax.set_title(f"({chr(98 + j)}) {name}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = OUT / f"fit_{material}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def stage_report(materials: list[str], log=print) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for material in materials:
        val = _read(OUT / f"validate_{material}.json")
        land = _read(OUT / f"landscape_{material}.json")
        ls = _read(OUT / f"ls_{material}.json")
        refine = _read(OUT / f"refine_{material}.json")
        inits_path = OUT / f"fit_{material}_inits.jsonl"
        inits = [json.loads(ln) for ln in inits_path.read_text().splitlines()
                 if ln.strip()] if inits_path.exists() else []
        best = min(inits, key=lambda r: r["best_loss"]) if inits else None
        spread = {}
        if inits:
            for name, truth, _ in SPECS[material]["params"]:
                vals = np.array([r["best_theta"][name] for r in inits])
                spread[name] = {
                    "min": float(vals.min()), "max": float(vals.max()),
                    "median": float(np.median(vals)),
                    "spread_factor": float(vals.max() / max(vals.min(), 1e-30)),
                    "rel_err_per_init": [float(abs(v / truth - 1.0)) for v in vals],
                }
        res = {
            "schema_version": "diffsim-baseline-1.0",
            "material": material,
            "unknowns": [n for n, _, _ in SPECS[material]["params"]],
            "prior": {"kind": "log-uniform over a decade (moduli), "
                              "uniform 15 to 45 deg (friction angle)",
                      "box_q": dict(zip(("lo", "hi"),
                                        [list(map(float, a)) for a in prior_box(material)],
                                        strict=True))},
            "forward_gap": val,
            "diffsim": {
                "n_inits": len(inits),
                "best": None if not best else {
                    "loss": best["best_loss"],
                    "theta": theta_compare(material, best["best_theta"]),
                    "init": best["init"], "iterations": best["iterations"],
                    "wall_seconds": best["wall_seconds"],
                    "seconds_per_iteration": best["seconds_per_iteration"],
                    "stop": best["stop"]},
                "theta_engine": (None if not best
                                 else theta_engine_from_fit(material, best["best_theta"])),
                "refined_best_init": None if not refine else {
                    "from_init": refine["from_init"],
                    "loss": refine["best_loss"],
                    "theta": theta_compare(material, refine["best_theta"]),
                    "theta_engine": theta_engine_from_fit(material,
                                                          refine["best_theta"]),
                    "iterations": refine["iterations"],
                    "wall_seconds": refine["wall_seconds"],
                    "stop": refine["stop"]},
                "init_spread": spread,
                "per_init": [{"init": r["init"], "theta0": r["theta0"],
                              "best_theta": r["best_theta"], "loss": r["best_loss"],
                              "iterations": r["iterations"], "stop": r["stop"],
                              "wall_seconds": r["wall_seconds"]} for r in inits],
                "total_wall_seconds": float(sum(r["wall_seconds"] for r in inits)),
            },
            "least_squares": None if not ls else {
                "wall_seconds": ls["wall_seconds"],
                "theta_engine": (ls.get("identify") or {}).get("theta_engine"),
                "theta": theta_compare(
                    material, ls_theta_in_our_params(material, ls.get("identify")) or {}),
                "refused_parameters": (ls.get("identify") or {}).get(
                    "refused_parameters", []),
            },
            "landscape": None if not land else {
                tag: {k: s[k] for k in ("argmin_theta", "min_loss", "unimodal",
                                        "monotone_pieces_sign_flips",
                                        "second_diff_max_abs")}
                for tag, s in land["scans"].items()},
        }
        fig = _fit_figure(material)
        res["figures"] = [_figure_path(p) for p in
                          [fig, OUT / f"landscape_{material}.png"] if p and p.exists()]
        (OUT / f"results_{material}.json").write_text(json.dumps(res, indent=2))
        all_res[material] = res
    _write_report(all_res, log=log)
    return all_res


def _fmt(x, sig=4):
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    return f"{x:.{sig}g}"


def _write_report(all_res: dict, log=print) -> Path:
    lines = [
        "# Differentiable-simulation baseline against the convex weak-form solve",
        "",
        "Both methods see the same warp grid-20 cube-throw truths (126 frames,",
        "9261 particles, dx 0.05, freeslip box). The baseline differentiates a",
        "separate minimal JAX MPM (experiments/diffsim/forward.py); warp-mpm is",
        "never made differentiable. Loss: NCLaw's position MSE on every fifth",
        "frame.",
        "",
        "## Cross-engine forward gap at truth theta (the floor)",
        "",
        "| material | substeps | final-frame RMS gap | as % of the 500 mm cube "
        "| loss floor | floor if the step were CFL-sized on the prior |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m, r in all_res.items():
        fg = r.get("forward_gap")
        if not fg:
            continue
        w = fg["forward_gap"]["warp_substeps"]
        f = fg["forward_gap"]["prior_cfl_substeps"]
        lines.append(
            f"| {m} | {w['substeps']} | {w['rms_mm_final']:.4f} mm "
            f"| {100 * w['rms_mm_final_over_cube']:.5f} % | {w['mse_metric']:.2e} "
            f"| {f['mse_metric']:.2e} (sub {f['substeps']}) |")

    lines += ["", "## Recovered parameters", "",
              "| material | parameter | truth | diff-sim best | rel err "
              "| least squares | rel err |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, r in all_res.items():
        best = (r["diffsim"] or {}).get("best")
        lsq = r.get("least_squares")
        for name, truth, _ in SPECS[m]["params"]:
            d = (best or {}).get("theta", {}).get(name, {})
            q = (lsq or {}).get("theta", {}).get(name, {})
            lines.append(
                f"| {m} | {name} | {_fmt(truth)} | {_fmt(d.get('recovered'))} | "
                f"{'' if d.get('rel_err') is None else f'{100 * d['rel_err']:.2f} %'} | "
                f"{_fmt(q.get('recovered'))} | "
                f"{'' if q.get('rel_err') is None else f'{100 * q['rel_err']:.2f} %'} |")

    ref = {m: r["diffsim"]["refined_best_init"] for m, r in all_res.items()
           if (r["diffsim"] or {}).get("refined_best_init")}
    if ref:
        lines += ["", "## Best init continued past the equal-budget stop", "",
                  "The fit rows above are what one fixed budget buys. These rows",
                  "restart the best init at a third of the step and run to a",
                  "plateau, which is where the method lands when cost is no object.",
                  "",
                  "| material | parameter | truth | continued | rel err | loss "
                  "| iterations | extra wall |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, rr in ref.items():
            for name, truth, _ in SPECS[m]["params"]:
                d = rr["theta"].get(name, {})
                rel = ("" if d.get("rel_err") is None
                       else f"{100 * d['rel_err']:.2f} %")
                lines.append(
                    f"| {m} | {name} | {_fmt(truth)} | {_fmt(d.get('recovered'))} | "
                    f"{rel} | {rr['loss']:.3e} | {rr['iterations']} | "
                    f"{rr['wall_seconds'] / 60:.1f} min |")

    lines += ["", "## Cost and init sensitivity", "",
              "| material | unknowns | diff-sim wall (all inits) | s / iteration "
              "| iterations (best init) | least-squares wall "
              "| ratio (all inits / LS) | init spread (max/min) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, r in all_res.items():
        ds = r["diffsim"]
        best = ds.get("best")
        lsq = r.get("least_squares")
        ratio = ("" if not (best and lsq and lsq["wall_seconds"])
                 else f"{ds['total_wall_seconds'] / lsq['wall_seconds']:.0f} x")
        sp = ds.get("init_spread") or {}
        spread_txt = ", ".join(f"{k} {v['spread_factor']:.2f}" for k, v in sp.items())
        total_min = f"{ds['total_wall_seconds'] / 60:.1f} min"
        lines.append(
            f"| {m} | {len(r['unknowns'])} | "
            f"{'' if not ds['total_wall_seconds'] else total_min} | "
            f"{'' if not best else f'{best['seconds_per_iteration']:.1f}'} | "
            f"{'' if not best else best['iterations']} | "
            f"{'' if not lsq else f'{lsq['wall_seconds']:.1f} s'} | {ratio} | {spread_txt} |")

    lines += ["", "## Recovered laws in the engine's own arguments", "",
              "For the warp rollout leg of the comparison.", "",
              "| material | diff-sim theta_engine |", "| --- | --- |"]
    for m, r in all_res.items():
        te = (r["diffsim"] or {}).get("theta_engine")
        if te:
            lines.append(f"| {m} | " + ", ".join(
                f"{k} {v:.6g}" for k, v in te.items()) + " |")

    land = {m: r["landscape"] for m, r in all_res.items() if r.get("landscape")}
    if land:
        lines += ["", "## Loss landscape (forward-only sweeps)", "",
                  "| material | sweep | argmin | min loss | unimodal | slope sign flips |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for m, sc in land.items():
            for tag, s in sc.items():
                lines.append(f"| {m} | {tag} | {_fmt(s['argmin_theta'])} | "
                             f"{s['min_loss']:.3e} | {s['unimodal']} | "
                             f"{s['monotone_pieces_sign_flips']} |")

    lines += ["", "## Caveats", "",
              "- The diff-sim's own forward map is not the truth engine. The",
              "  cross-engine gap above is the floor of its loss; where it is",
              "  orders of magnitude below the achieved loss it cannot bias theta.",
              "- The fit runs at the truth run's own substep count, the most",
              "  favourable choice available to it, so no part of the recovered",
              "  theta is a time-discretization artefact. The last column above",
              "  is what the floor would have been had the step been sized by the",
              "  CFL rule at the stiff corner of the prior instead.",
              "- Adam is clipped to the prior box, so an init spread of one means",
              "  agreement inside the box, not global uniqueness.",
              "- The iteration budgets differ per material (120, 80, 150 and 40",
              "  for jelly, sand, water and plasticine) because each was sized to",
              "  one wall-clock budget, not to convergence. Plasticine's 40",
              "  iterations over three unknowns is the least converged row of the",
              "  table, and its elastic pair trades mu against lam at nearly fixed",
              "  wave speed; read its continued row for where the method lands.",
              ""]
    p = OUT / "report.md"
    p.write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"[report] wrote {p}")
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["validate", "landscape", "fit", "refine",
                                     "ls", "report"])
    ap.add_argument("--material", default="jelly")
    ap.add_argument("--inits", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--points", type=int, default=21)
    a = ap.parse_args(argv)

    mats = sorted(SPECS) if a.material == "all" else [a.material]
    for m in mats:
        if m not in SPECS:
            raise SystemExit(f"unknown material {m}; known: {sorted(SPECS)}")
    if a.stage == "report":
        stage_report(mats)
        return
    for m in mats:
        if a.stage == "validate":
            stage_validate(m)
        elif a.stage == "landscape":
            stage_landscape(m, n_points=a.points)
        elif a.stage == "fit":
            stage_fit(m, inits=a.inits, iters=a.iters)
        elif a.stage == "refine":
            stage_refine(m, iters=a.iters or 250)
        elif a.stage == "ls":
            stage_ls(m)


if __name__ == "__main__":
    main()
