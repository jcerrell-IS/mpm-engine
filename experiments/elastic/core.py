"""Shared core of the elastic identification campaign: scene, dump IO, weak form, recovery.

Every stage in this package drops a blob, reads the dump, and solves one convex
linear-in-theta weak-form system. The pieces below were copy-pasted across eight
scripts in video2sim/sim before the consolidation: the drop-scene runner appeared
twice, and the radial interior-bump construction, the validity filter and the row
assembly loop appeared eight times each, with drift between the copies. One version
of each lives here now:

  run_drop            the warp-mpm gravity-drop scene, fixed-corotated ("jelly") or
                      von-Mises ("metal"), dumping x, v, F, and stress for metal
  DropDump            the dump reader, with the truth parameters it carries
  radial_bumps        interior reference test functions: a radial window that
                      vanishes on the reference surface, times low-order modes
  validity_mask       the ROTATION-INVARIANT validity filter, ||log sigma(F)||, not
                      ||F - I||. The latter is not frame-objective, so a rotated but
                      barely strained element was being dropped, and the SVD branch
                      used to store F differs between backends. The two sequential
                      scripts still used the old measure at consolidation time; they
                      now use this one (see README for the resulting number changes).
  fcr_basis           P(F) = mu 2(F-R) + lambda J(J-1) F^-T, the two FCR columns
  hencky_basis        sigma = G S_G + lambda S_L, the von-Mises Hencky columns
  WeakForm            per-frame rows and the stacked system for any of those bases
  grid_recover        the grid-consistent recovery through ident.weakform.elastic_grid,
                      which is the accurate route on a non-spherical blob and the one
                      the Step 0 gate ships

Artifacts: out/elastic/ under the mpm_engine root. Dumps written before the
consolidation stay where they were (video2sim/out/elastic_drop, out/plastic_drop,
out/hyperelastic and the mpm_engine copies of those directories); find_artifact
searches them, so a stage reuses an existing 676 MB dump instead of re-simulating.
Figures land in out/elastic/ and are copied into video2sim/docs/writeup/figs when
that tree is present, which is where the LaTeX includes them from.

Run:  .venv/bin/python -m experiments.elastic --help
"""
from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[2]        # mpm_engine
STAGING_ROOT = ENGINE_ROOT.parent                        # video2sim research staging tree
OUT = ENGINE_ROOT / "out" / "elastic"

# read-only fallbacks: where the pre-consolidation scripts wrote their dumps
LEGACY_DIRS: tuple[Path, ...] = tuple(
    root / "out" / name
    for root in (ENGINE_ROOT, STAGING_ROOT)
    for name in ("elastic", "elastic_drop", "plastic_drop", "hyperelastic")
)
WRITEUP_FIGS = STAGING_ROOT / "docs" / "writeup" / "figs"

DEVICE = "cpu"
TRUTH_ELASTIC = dict(E=2.0e5, nu=0.30, rho=1000.0)
# G = 3.85e5, eps_y ~ 1.04 percent, so this one clearly yields
TRUTH_PLASTIC = dict(E=1.0e6, nu=0.30, rho=1000.0, yield_stress=8.0e3)


# --------------------------------------------------------------------------- paths


def artifact(name: str) -> Path:
    """Path to write an artifact to (out/elastic/<name>)."""
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def find_artifact(name: str) -> Path | None:
    """Path to read an artifact from: the campaign directory first, then the
    pre-consolidation locations. Returns None when nothing has produced it yet."""
    for d in (OUT, *LEGACY_DIRS):
        p = d / name
        if p.exists():
            return p
    return None


def publish_figure(fig, name: str, dpi: int = 140, tight: bool = False) -> Path:
    """Save a figure to out/elastic/<name> and copy it into the writeup figure
    directory when the staging tree is present (the LaTeX includes it from there)."""
    p = artifact(name)
    fig.savefig(p, dpi=dpi, **({"bbox_inches": "tight"} if tight else {}))
    if WRITEUP_FIGS.is_dir():
        shutil.copyfile(p, WRITEUP_FIGS / name)
    return p


# ------------------------------------------------------------------- forward scene


def raw_points(shape: str, size: float, h: float) -> np.ndarray:
    """Particle cloud for a shape, centred near the origin (placed by run_drop).

    size is the characteristic half-extent. sphere and box are the training
    geometries; star is a five-point star in the x-z plane extruded thin in y, a
    non-convex held-out geometry for the shape-generalization stage.
    """
    g = np.arange(-size - h, size + h, h)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    if shape == "sphere":
        keep = np.linalg.norm(P, axis=1) <= size
    elif shape == "box":
        a = np.array([size, size, 0.85 * size])            # rectangular, not cubic
        keep = np.all(np.abs(P) <= a, axis=1)
    elif shape == "star":
        from matplotlib.path import Path as MplPath
        R, rin, ty = size, 0.42 * size, 0.55 * size        # outer, inner radius, y half-thickness
        ang = np.pi / 2 + np.arange(10) * np.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, R, rin)
        verts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        inside = MplPath(verts).contains_points(P[:, [0, 2]])
        keep = inside & (np.abs(P[:, 1]) <= ty)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return P[keep]


def run_drop(out_path, E, nu, rho, *, material="jelly", yield_stress=None, hardening=0.0,
             shape="sphere", size=0.14, drop_gap=0.18, n_grid=48, grid_lim=1.0,
             t_end=0.9, frame_dt=2.0e-3, floor_bc="slip", seed=0, log=print) -> Path:
    """Gravity-drop a blob onto a floor plane and dump the trajectory.

    material "jelly" is fixed corotated (elastic, bounces); material "metal" is
    von-Mises elastoplastic (yields and stays flat) and needs yield_stress. The
    metal dump carries the per-particle Cauchy stress as well, which the plastic
    stage uses to check its stress basis against the simulator.

    The substep count comes from the elastic wave speed sqrt(E/rho) at CFL 0.3.
    """
    import warp as wp
    wp.config.quiet = True
    wp.init()
    import torch

    from warpmpm.kernels import MPM_Simulator_WARP

    if material == "metal" and yield_stress is None:
        raise ValueError("material 'metal' needs yield_stress")

    dx = grid_lim / n_grid
    h = dx / 2
    floor = 3 * dx
    cx = cy = grid_lim * 0.5
    pos = raw_points(shape, size, h)
    pos += np.random.default_rng(seed).uniform(-0.15 * h, 0.15 * h, pos.shape)
    pos[:, 0] += cx - pos[:, 0].mean()                    # centre x and y
    pos[:, 1] += cy - pos[:, 1].mean()
    pos[:, 2] += (floor + drop_gap) - pos[:, 2].min()     # base drop_gap above the floor
    pos = pos.astype(np.float32)
    vol = np.full(len(pos), h ** 3, dtype=np.float32)
    X_ref = pos.copy()

    c_el = float(np.sqrt(E / rho))
    substeps = int(np.ceil(frame_dt / (0.3 * dx / c_el)))
    dt = frame_dt / substeps
    n_frames = round(t_end / frame_dt)
    log(f"[{material}] {shape} N={len(pos)} grid={n_grid}^3 dx={dx*1e3:.1f}mm c={c_el:.0f}m/s "
        f"dt={dt:.1e} sub={substeps} frames={n_frames} E={E:.2e} nu={nu}"
        + (f" yield={yield_stress:.1e}" if yield_stress is not None else ""))

    s = MPM_Simulator_WARP(len(pos), device=DEVICE)
    s.load_initial_data_from_torch(
        torch.from_numpy(np.ascontiguousarray(pos)),
        torch.from_numpy(np.ascontiguousarray(vol)),
        n_grid=n_grid, grid_lim=grid_lim, device=DEVICE)
    params = {"material": material, "E": E, "nu": nu, "density": rho,
              "g": [0.0, 0.0, -9.81]}
    if material == "metal":
        params |= {"yield_stress": yield_stress, "hardening": hardening}
    s.set_parameters_dict(params, device=DEVICE)
    s.finalize_mu_lam(device=DEVICE)
    s.add_surface_collider((0.0, 0.0, floor), (0.0, 0.0, 1.0), floor_bc)

    X, V, F, S = [], [], [], []
    t0 = time.time()
    step = 0
    for frame in range(n_frames + 1):
        x = s.export_particle_x_to_torch().detach().cpu().numpy().copy()
        if not np.isfinite(x).all():
            log(f"[{material}] NaN at frame {frame}")
            break
        X.append(x)
        V.append(s.export_particle_v_to_torch().detach().cpu().numpy().copy())
        F.append(s.export_particle_F_to_torch().detach().cpu().numpy().copy())
        if material == "metal":
            S.append(s.export_particle_stress_to_torch().detach().cpu().numpy().copy())
        if frame == n_frames:
            break
        for _ in range(substeps):
            s.p2g2p(step, dt, device=DEVICE)
            step += 1

    mu, lam = E_nu_to_moduli(E, nu)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if material == "metal":
        extra = dict(stress=np.asarray(S, np.float32), yield_stress=yield_stress)
    # mu and G are the same number; both keys are written so a reader of either the
    # old elastic dumps (mu) or the old plastic dumps (G) keeps working
    np.savez(out_path, x=np.asarray(X, np.float32), v=np.asarray(V, np.float32),
             F=np.asarray(F, np.float32), X_ref=X_ref.astype(np.float32),
             vol=vol, frame_dt=frame_dt, rho=rho, g=np.array([0.0, 0.0, -9.81]),
             floor=floor, E=E, nu=nu, mu=mu, G=mu, lam=lam, material=material,
             shape=shape, size=size, grid_lim=grid_lim, **extra)
    log(f"[{material}] wrote {out_path} ({len(X)} frames, {time.time()-t0:.0f}s)  "
        f"mu={mu:.3e} lam={lam:.3e}")
    return out_path


# ------------------------------------------------------------------ moduli algebra


def moduli_to_E_nu(mu: float, lam: float) -> tuple[float, float]:
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    return float(E), float(nu)


def E_nu_to_moduli(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return float(mu), float(lam)


def polar_R(F: np.ndarray) -> np.ndarray:
    """Rotation from the polar decomposition of a batch of F, det > 0 enforced."""
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    bad = np.linalg.det(R) < 0
    if bad.any():
        U[bad, :, -1] *= -1.0
        R = U @ Vt
    return R


# ------------------------------------------------------------------- acceleration


def central_difference_accel(v: np.ndarray, frame_dt: float) -> np.ndarray:
    """Material acceleration by trajectory finite differences, the tree default.

    Interior frames use the second-order central difference, so a[t] here equals the
    (v[t+1] - v[t-1]) / (2 dt) that every pre-consolidation script computed; the
    endpoints get one-sided differences so the array keeps the frame indexing.
    """
    a = np.zeros_like(v)
    if v.shape[0] >= 3:
        a[1:-1] = (v[2:] - v[:-2]) / (2.0 * frame_dt)
    if v.shape[0] >= 2:
        a[0] = (v[1] - v[0]) / frame_dt
        a[-1] = (v[-1] - v[-2]) / frame_dt
    return a


def fd_accel(v: np.ndarray, frame_dt: float, order: int = 4) -> np.ndarray:
    """Fourth-order central difference where the stencil fits, second otherwise.

    The second-order difference underestimates an oscillatory acceleration by
    (omega h)^2 / 6, which on this bounce measured a quarter of a percent on E.
    """
    a = central_difference_accel(v, frame_dt)
    if order >= 4 and v.shape[0] >= 5:
        a[2:-2] = (-v[4:] + 8.0 * v[3:-1] - 8.0 * v[1:-3] + v[:-4]) / (12.0 * frame_dt)
    return a


# ------------------------------------------------------------------- test functions


def radial_bumps(X_ref: np.ndarray, n_modes: int = 4) -> tuple[list, list]:
    """Interior reference test functions and their reference gradients.

    W(r) = (1 - (r/Rb)^2)^2 with Rb slightly outside the cloud, so W and grad W
    vanish on the reference surface and the boundary traction drops out of the weak
    form. The first mode is W itself; modes 2 to 4 are W times the centred
    coordinates, which add the linear response. Returns (phis, gphis), each a list
    of n_modes arrays shaped (N,) and (N, 3).
    """
    c = X_ref.mean(0)
    r = np.linalg.norm(X_ref - c, axis=1)
    Rb = r.max() * 1.02
    u = r / Rb
    W = np.clip(1.0 - u ** 2, 0.0, None) ** 2
    dWdr = -4.0 * u / Rb * np.clip(1.0 - u ** 2, 0.0, None)
    rsafe = np.where(r > 1e-9, r, 1.0)
    gradW = (dWdr / rsafe)[:, None] * (X_ref - c)
    dX = (X_ref - c) / Rb
    n = len(X_ref)
    modes = [np.ones(n), *[dX[:, i] for i in range(3)][: max(0, n_modes - 1)]]
    gmodes = [np.zeros((n, 3)),
              *[np.eye(3)[i][None, :] / Rb * np.ones((n, 1)) for i in range(3)][
                  : max(0, n_modes - 1)]]
    phis = [W * m for m in modes]
    gphis = [gradW * m[:, None] + W[:, None] * gm for m, gm in zip(modes, gmodes, strict=True)]
    return phis, gphis


def validity_mask(F: np.ndarray, j_lo: float = 0.3, j_hi: float = 2.0,
                  hencky_max: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-invariant validity filter for one frame of F, shape (N, 3, 3).

    Returns (mask, hencky) with hencky = ||log sigma(F)||, the Hencky strain
    magnitude. The filter is on the singular values, not on ||F - I||: the latter is
    not frame-objective, so an element that has rotated but barely strained gets
    dropped, and the SVD branch used to STORE F differs between warp svd3 and numpy
    by a benign rotation that leaves the stress identical. The Hencky measure passes
    exactly the same physical elements whichever branch was stored.
    """
    sig = np.clip(np.linalg.svd(F, compute_uv=False), 1e-9, None)
    J = sig.prod(-1)
    hencky = np.linalg.norm(np.log(sig), axis=1)
    mask = (j_lo < J) & (j_hi > J) & (hencky < hencky_max) & np.isfinite(J)
    return mask, hencky


# ------------------------------------------------------------------- stress bases
# A basis maps one frame of masked F to (list of K stress tensors, per-particle aux
# scalar). The stress is whatever the weak form wants contracted against grad_X w:
# a first Piola for the reference-configuration forms, the Cauchy-like Hencky
# tensors for the von-Mises form (which match the dumped stress to 5e-5).


def fcr_basis(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Fixed-corotated first-Piola columns, P = mu P_mu + lambda P_lam.

    P_mu = 2 (F - R) with R from polar(F); P_lam = J (J - 1) F^-T. Aux is the
    Hencky strain magnitude, used as the strain-coverage report.
    """
    J = np.linalg.det(F)
    R = polar_R(F)
    P_mu = 2.0 * (F - R)
    Finvt = np.transpose(np.linalg.inv(F), (0, 2, 1))
    P_lam = (J * (J - 1.0))[:, None, None] * Finvt
    sig = np.clip(np.linalg.svd(F, compute_uv=False), 1e-9, None)
    return [P_mu, P_lam], np.linalg.norm(np.log(sig), axis=1)


def hencky_basis(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """von-Mises Hencky stress columns, sigma = G S_G + lambda S_L.

    sigma = (1/J) U diag(2 G eps_i + lambda sum eps) U^T with eps_i = log sing(F).
    The tau_i sigma_i convention here matches the dumped warp-mpm stress to 5e-5
    when combined with the reference volume in the weak form. Aux is ||dev(eps)||,
    which the return map caps at yield / (2 G) once the material yields, so the
    plastic stage reads the yield stress off its saturation.
    """
    U, sig, _ = np.linalg.svd(F)
    sig = np.clip(sig, 1e-6, None)
    eps = np.log(sig)
    tr = eps.sum(-1)
    Ut = np.transpose(U, (0, 2, 1))

    def spec(diag):
        return (U * diag[:, None, :]) @ Ut

    S_G = spec(2.0 * eps * sig)
    S_L = spec(tr[:, None] * sig)
    dev = eps - tr[:, None] / 3.0
    return [S_G, S_L], np.linalg.norm(dev, axis=1)


# ----------------------------------------------------------------------- dump IO


@dataclass
class DropDump:
    """One drop trajectory: the fields the weak form needs plus the truth parameters."""
    path: Path
    x: np.ndarray                    # (T, N, 3) current positions
    v: np.ndarray                    # (T, N, 3)
    F: np.ndarray                    # (T, N, 3, 3)
    X_ref: np.ndarray                # (N, 3) reference configuration
    vol: np.ndarray                  # (N,) reference volume
    rho: float
    g: np.ndarray                    # (3,)
    frame_dt: float
    floor: float
    grid_lim: float
    truth: dict
    shape: str
    stress: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.x.shape[0]

    @classmethod
    def load(cls, path) -> DropDump:
        path = Path(path)
        d = np.load(path)
        x = d["x"].astype(np.float64)
        T, N = x.shape[0], x.shape[1]
        truth: dict = {}
        for k in ("E", "nu", "mu", "lam", "G", "yield_stress", "C1", "C2", "C3",
                  "C10", "C01", "Kbulk"):
            if k in d.files:
                truth[k] = float(d[k])
        truth.setdefault("G", truth.get("mu", np.nan))
        truth.setdefault("mu", truth.get("G", np.nan))
        return cls(
            path=path,
            x=x,
            v=d["v"].astype(np.float64),
            F=d["F"].astype(np.float64).reshape(T, N, 3, 3),
            X_ref=d["X_ref"].astype(np.float64),
            vol=d["vol"].astype(np.float64),
            rho=float(d["rho"]),
            g=np.asarray(d["g"], dtype=float).reshape(3),
            frame_dt=float(d["frame_dt"]),
            floor=float(d["floor"]) if "floor" in d.files else 0.0,
            grid_lim=float(d["grid_lim"]) if "grid_lim" in d.files else 1.0,
            truth=truth,
            shape=str(d["shape"]) if "shape" in d.files else "sphere",
            stress=(d["stress"].astype(np.float64).reshape(T, N, 3, 3)
                    if "stress" in d.files else None),
        )


def position_error(truth_path, pred_path) -> tuple[float, float, int]:
    """Per-particle position error between two dumps of the same cloud: RMS in mm,
    the box-normalized MSE that the NCLaw tables use, and the frames compared."""
    t = np.load(truth_path)
    r = np.load(pred_path)
    nf = min(t["x"].shape[0], r["x"].shape[0])
    n = min(t["x"].shape[1], r["x"].shape[1])
    diff = t["x"][:nf, :n] - r["x"][:nf, :n]
    rmse_mm = float(np.sqrt((diff ** 2).sum(-1).mean()) * 1e3)
    gl = float(t["grid_lim"])
    mse_box = float(((diff / gl) ** 2).sum(-1).mean())
    return rmse_mm, mse_box, nf


# --------------------------------------------------------------------- weak form


@dataclass
class FrameRows:
    """Rows contributed by one frame: A (n_rows, K), b (n_rows,), and the masked state."""
    A: np.ndarray
    b: np.ndarray
    F: np.ndarray                    # masked F for this frame, (M, 3, 3)
    aux: np.ndarray                  # per-particle basis aux, (M,)


BasisFn = Callable[[np.ndarray], tuple[list[np.ndarray], np.ndarray]]


class WeakForm:
    """The dynamic reference-configuration weak form for one dump and one basis.

    Row j of one frame, for direction dirn and test function phi:

        A[j, k] = sum_p V0_p  P_k(F_p) e_dirn . grad_X phi(X_p)
        b[j]    = -sum_p V0_p rho0 (a_p - g)_dirn phi(X_p)

    The interior bumps make the surface traction vanish; the inertia term rho0 a
    supplies the absolute force scale, so the moduli are pinned with no force
    sensor. Linear in theta, so the inverse is one least-squares solve and the
    simulator is never differentiated.
    """

    def __init__(self, dump: DropDump, basis: BasisFn, n_modes: int = 4,
                 min_particles: int = 50, accel_order: int = 2):
        self.dump = dump
        self.basis = basis
        self.min_particles = min_particles
        self.phis, self.gphis = radial_bumps(dump.X_ref, n_modes)
        self.accel = (central_difference_accel(dump.v, dump.frame_dt) if accel_order < 4
                      else fd_accel(dump.v, dump.frame_dt, accel_order))

    def interior_frames(self) -> range:
        return range(1, self.dump.n_frames - 1)

    def frame(self, t: int) -> FrameRows | None:
        """Rows for one frame, or None when too few particles survive the filter."""
        Ft = self.dump.F[t]
        mask, _ = validity_mask(Ft)
        if mask.sum() < self.min_particles:
            return None
        Fm = Ft[mask]
        Pk, aux = self.basis(Fm)
        Vm = self.dump.vol[mask]
        at = self.accel[t][mask]
        rho0, g = self.dump.rho, self.dump.g
        rows_A, rows_b = [], []
        for dirn in range(3):
            for phi, gphi in zip(self.phis, self.gphis, strict=True):
                gp = gphi[mask]
                ph = phi[mask]
                rows_A.append([np.sum(Vm * np.einsum("pj,pj->p", P[:, dirn, :], gp))
                               for P in Pk])
                rows_b.append(-np.sum(Vm * rho0 * (at[:, dirn] - g[dirn]) * ph))
        return FrameRows(np.asarray(rows_A), np.asarray(rows_b), Fm, aux)

    def assemble(self, frames: Sequence[int] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack every usable frame: (A, b, concatenated aux)."""
        As, bs, auxs = [], [], []
        for t in (self.interior_frames() if frames is None else frames):
            fr = self.frame(t)
            if fr is None:
                continue
            As.append(fr.A)
            bs.append(fr.b)
            auxs.append(fr.aux)
        if not As:
            raise SystemExit(f"{self.dump.path}: no frame passed the validity filter")
        return np.vstack(As), np.concatenate(bs), np.concatenate(auxs)


def lstsq(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Unregularized least squares plus cond(A^T A), the campaign's default solve."""
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return theta, float(np.linalg.cond(A.T @ A))


def noise_scale(A: np.ndarray, b: np.ndarray, theta: np.ndarray) -> float:
    """Residual noise standard deviation, the sigma of the Gauss-Markov covariance."""
    r = A @ theta - b
    return float(np.sqrt(r @ r / max(len(b) - A.shape[1], 1)))


# ------------------------------------------------- grid-consistent recovery route


def grid_recover(dump_path, n_grid: int = 48, frame_stride: int = 4,
                 margin_cells: float = 3.0, min_support_mass_frac: float = 0.25,
                 route: str = "timeweak", window_frames: int = 26, time_power: int = 2,
                 taper_cells: float = 1.0, window_modes: int = 4, fd_order: int = 4,
                 log=print) -> dict:
    """Grid-consistent (mu, lambda) recovery, the route the Step 0 gate ships.

    The radial-window route above vanishes on a sphere surface, so it is accurate on
    the sphere drop and biased on the cube drop, where the window is nonzero on the
    faces and the floor contact traction leaks into the residual. This route uses the
    grid node rows of ident.weakform.elastic_grid instead, which is the Bubnov-Galerkin
    form the identification stack settled on.

    route "timeweak" uses the time-integrated load and needs no acceleration data;
    route "instant" uses the per-frame balance with a finite-difference acceleration.
    Both read the same exact node rows. n_grid is a run_drop setting that the dump
    does not record, so it is passed in.
    """
    from ident.weakform.elastic_grid import (
        assemble_elastic_grid,
        assemble_elastic_timeweak,
        solve_elastic_grid,
    )

    dump_path = Path(dump_path)
    d = np.load(dump_path)
    # slice to the frames used BEFORE widening to float64: the cube dump is
    # 676 MB of float32 and its F alone reaches 800 MB widened, which is what killed
    # an earlier full-sweep run on this machine
    n_all = d["x"].shape[0]
    keep = np.arange(0, n_all, frame_stride)
    x = d["x"][keep].astype(np.float64)
    v = d["v"][keep].astype(np.float64)
    F = d["F"][keep].astype(np.float64).reshape(x.shape[0], x.shape[1], 3, 3)
    vol0 = d["vol"].astype(np.float64)
    rho0 = float(d["rho"])
    mass = rho0 * vol0
    g = np.asarray(d["g"], dtype=float).reshape(3)
    frame_dt = float(d["frame_dt"])
    grid_lim = float(d["grid_lim"])
    floor = float(d["floor"])
    mu_t, lam_t = float(d["mu"]), float(d["lam"])
    E_t, nu_t = moduli_to_E_nu(mu_t, lam_t)

    frame_dt_used = frame_dt * frame_stride
    frames = list(range(x.shape[0]))
    planes = [((0.0, 0.0, floor), (0.0, 0.0, 1.0))]
    t0 = time.time()
    if route == "timeweak":
        sysm = assemble_elastic_timeweak(
            x, F, v, vol0, mass, g, frame_dt_used, n_grid, grid_lim,
            frames=frames, window_frames=window_frames, time_power=time_power,
            collider_planes=planes, collider_margin_cells=margin_cells,
            min_support_mass_frac=min_support_mass_frac,
            window_taper_cells=taper_cells, window_modes=window_modes,
        )
    else:
        sysm = assemble_elastic_grid(
            x, F, vol0, mass, fd_accel(v, frame_dt_used, fd_order), g,
            n_grid, grid_lim, frames=frames,
            collider_planes=planes, collider_margin_cells=margin_cells,
            min_support_mass_frac=min_support_mass_frac,
            window_taper_cells=taper_cells, window_modes=window_modes,
        )
    if sysm.n_rows < 10:
        return {"n_rows": sysm.n_rows, "status": "no rows"}
    out = solve_elastic_grid(sysm)
    E_h, nu_h = out["E"], out["nu"]
    res = {
        "dump": dump_path.name,
        "shape": str(d["shape"]),
        "route": route,
        "n_particles": int(x.shape[1]),
        "n_frames_dumped": int(n_all),
        "frame_stride": frame_stride,
        "n_frames_used": len(sysm.frames_used),
        "window_frames": window_frames if route == "timeweak" else None,
        "time_power": time_power if route == "timeweak" else None,
        "taper_cells": taper_cells,
        "window_modes": window_modes,
        "margin_cells": margin_cells,
        "min_support_mass_frac": min_support_mass_frac,
        "n_rows": sysm.n_rows,
        "n_rows_before_gating": sysm.n_rows_before_gating,
        "row_survival": sysm.row_survival,
        "strain_coverage": list(sysm.strain_coverage),
        "mu_hat": out["mu"], "mu_true": mu_t,
        "mu_err": abs(out["mu"] / mu_t - 1.0),
        "lam_hat": out["lam"], "lam_true": lam_t,
        "lam_err": abs(out["lam"] / lam_t - 1.0),
        "E_hat": E_h, "E_true": E_t, "E_err": abs(E_h / E_t - 1.0),
        "nu_hat": nu_h, "nu_true": nu_t,
        "cond_AtA": out["cond_AtA"],
        "cond_AtA_scaled": out["cond_AtA_scaled"],
        "residual_rel": out["residual_rel"],
        "mu_sd": out["mu_sd"], "lam_sd": out["lam_sd"],
        "assembly_seconds": time.time() - t0,
    }
    log(f"[grid:{route:8s}] {res['shape']:7s} margin={margin_cells:.1f}c "
        f"rows={res['n_rows']:6d} (survival {100*res['row_survival']:.1f}%)  "
        f"E err {100*res['E_err']:.3f}%  mu err {100*res['mu_err']:.3f}%  "
        f"lam err {100*res['lam_err']:.3f}%  nu {res['nu_hat']:.4f}  "
        f"cond {res['cond_AtA']:.3e}  [{res['assembly_seconds']:.0f}s]")
    return res
