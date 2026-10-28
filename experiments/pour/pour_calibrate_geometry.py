"""Scene geometry of a recorded pour episode, measured from the side camera's depth
stream: the table height, where the receiver stands, and how far the held cup sits from
the nominal handle grasp. The cup SHAPE is the caliper spec (MeasuringCupSpec); the depth
cloud only says WHERE that known shape is, so each fit has two free numbers, a center.

Pre-pour window (arm at rest with the filled cup, before the roll):
  1. per-pixel median depth over the window -> one denoised depth map -> base-frame
     cloud through the intrinsics and the embedded AprilTag extrinsics
  2. table height: robust height of the table points around the receiver
  3. receiver: points in the rim band (z = table + cup height) fitted with the cup's
     outer top ellipse (49.80 x 48.91 mm semi-axes, orientation Q_RCV), spout and handle
     sectors excluded, MAD-clipped; the center is RECEIVER_XY
  4. held cup: the same fit at the rim height the grasp chain predicts with zero shift;
     fitted minus nominal center is CUP_SHIFT_XY, and the fitted rim height checks the
     grasp height
  5. liquid surface: median cloud height inside the held cup's footprint, read against
     the measured rim -> fill depth -> the cavity-volume curve. Indicative only: the
     RealSense's IR pattern penetrates the translucent glycerol and refracts at its
     surface, so the return sits several mm deep (on ep0001 about 5 mm below where the
     metered 300 mL puts the surface). It bounds gross fill errors, not the +-10 mL that
     matter for eta_hat.

Prints each constant next to the value baked into pour_recorded_twin.py. Only ep0001 has
been processed; another episode needs the same window check (arm at rest, cup filled).

Run:
  python experiments/pour/pour_calibrate_geometry.py                 # ep0001
  python experiments/pour/pour_calibrate_geometry.py --episode pouring_real_data/ep0002

Outputs (out/pour_wf/<episode>/):
  calibration.json   measured constants, residuals, point counts
  calibration.png    rim points with the fitted ellipses, the liquid surface
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))

from pour_recorded_twin import (
    CUP_SHIFT_XY,
    HOLD_SECONDS,
    PRE_ROLL,
    RECEIVER_XY,
    SPEC,
    TABLE_Z,
    RecordedPanda,
    load_episode,
)
from warpmpm.geometry.measuring_cup import write_cup_obj

OUT_ROOT = REPO / "out" / "pour_wf"
RIM_BAND = 0.005          # m; |z - rim height| accepted into a rim fit
RIM_R = (0.040, 0.062)    # m; horizontal distance from the center accepted into a rim fit
SPOUT_HALF_W = 0.022      # m; |y| of the excluded spout sector (spout width 37.7 mm)
HANDLE_HALF_W = 0.015     # m; |y| of the excluded handle sector (handle width 20 mm)
CLIP = 0.003              # m; inlier cut on the radial residual
SURFACE_R = 0.020         # m; footprint radius read for the liquid surface (away from walls)
SURFACE_BAND = (0.010, 0.060)  # m below the measured rim where the surface may sit
N_BOOT = 200              # bootstrap resamples for the center uncertainties


def pour_send_time(ep_dir: Path) -> float:
    acts = [json.loads(ln) for ln in (ep_dir / "actions.jsonl").read_text().splitlines()
            if ln]
    moves = [a for a in acts if a.get("type") == "go_to_pose"]
    return next(a["t_send"] for a in moves
                if a["target_quat_xyzw"] != [0.5, 0.5, -0.5, 0.5])


def fuse_depth(ep_dir: Path, meta: dict, t_lo: float, t_hi: float, stride: int):
    """Per-pixel median depth over the side frames in [t_lo, t_hi] (every `stride`),
    back-projected into the robot base frame. Returns (points (N,3), n_frames)."""
    cm = meta["cameras"]["side"]
    intr = cm["intrinsics"]
    scale = float(cm["depth_scale"])
    T = np.asarray(meta["extrinsics"]["cameras"]["side"]["T_base_cam"]["matrix"])
    rows = [json.loads(ln) for ln in (ep_dir / "frames_side.jsonl").read_text().splitlines()
            if ln]
    sel = [r for r in rows if t_lo <= r["t_host"] <= t_hi][::stride]
    if len(sel) < 5:
        raise SystemExit(f"only {len(sel)} side frames in the window [{t_lo}, {t_hi}]")
    stack = np.stack([np.array(Image.open(ep_dir / "side_depth" / f"{r['frame_idx']:06d}.png"),
                               dtype=np.float32) for r in sel])
    valid = stack > 0
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    med = np.where(valid.mean(axis=0) >= 0.5, med, 0.0) * scale     # metres
    h, w = med.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = med > 0
    cam = np.stack([(u[ok] - intr["ppx"]) / intr["fx"] * med[ok],
                    (v[ok] - intr["ppy"]) / intr["fy"] * med[ok], med[ok]], axis=1)
    return cam @ T[:3, :3].T + T[:3, 3], len(sel)


def table_height(pw: np.ndarray, around_xy, r_in: float = 0.10, r_out: float = 0.20):
    """Robust table height from the annulus around a cup (its own footprint excluded)."""
    d = np.hypot(pw[:, 0] - around_xy[0], pw[:, 1] - around_xy[1])
    z = pw[(d > r_in) & (d < r_out) & (pw[:, 2] > -0.12) & (pw[:, 2] < -0.02), 2]
    for _ in range(3):
        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med))
        z = z[np.abs(z - med) < 3.0 * max(mad, 1e-4)]
    return float(np.median(z)), len(z), float(1.4826 * np.median(np.abs(z - np.median(z))))


def rim_points(pw: np.ndarray, center_xy, z_rim: float) -> np.ndarray:
    """Mask of rim-band points around a nominal center, without the spout sector (toward
    world -x for both cups: Q_RCV and R_CUP_REF) and the handle sector (toward +x)."""
    dx, dy = pw[:, 0] - center_xy[0], pw[:, 1] - center_xy[1]
    r = np.hypot(dx, dy)
    m = (np.abs(pw[:, 2] - z_rim) < RIM_BAND) & (r > RIM_R[0]) & (r < RIM_R[1])
    m &= ~((dx < -0.030) & (np.abs(dy) < SPOUT_HALF_W))
    m &= ~((dx > 0.035) & (np.abs(dy) < HANDLE_HALF_W))
    return m


def _ell_res(c, p, a, b):
    """Scaled radial distance of points p to the axis-aligned ellipse (a, b) at c: the
    cup module's own ellipse metric."""
    rho = np.sqrt(((p[:, 0] - c[0]) / a) ** 2 + ((p[:, 1] - c[1]) / b) ** 2)
    return (rho - 1.0) * min(a, b)


def fit_center(pts_xy: np.ndarray, a: float, b: float, c0):
    """Center of an axis-aligned ellipse of KNOWN semi-axes through noisy rim points: a
    soft-L1 start, then two plain least-squares passes on the |residual| < CLIP inliers.
    Returns (center, residuals of all points, inlier mask, bootstrap sd of the center).
    The sd matters when the visible arc is short: it constrains the center well along
    the arc's normal and poorly along its tangent."""
    c = least_squares(_ell_res, np.asarray(c0, dtype=float), args=(pts_xy, a, b),
                      loss="soft_l1", f_scale=CLIP).x
    keep = np.abs(_ell_res(c, pts_xy, a, b)) < CLIP
    for _ in range(2):
        if keep.sum() < 10:
            break
        c = least_squares(_ell_res, c, args=(pts_xy[keep], a, b)).x
        keep = np.abs(_ell_res(c, pts_xy, a, b)) < CLIP
    rng = np.random.default_rng(0)
    inl = pts_xy[keep]
    boots = np.array([least_squares(_ell_res, c, args=(inl[rng.integers(0, len(inl), len(inl))],
                                                       a, b)).x for _ in range(N_BOOT)])
    return c, _ell_res(c, pts_xy, a, b), keep, boots.std(axis=0)


def fit_rim(pw: np.ndarray, c0, z_rim_nominal: float):
    """Two selection/fit rounds (the second around the refined center). Returns a dict
    with the center, the measured rim height, residual stats, and the point sets."""
    c = np.asarray(c0, dtype=float)
    for _ in range(2):
        m = rim_points(pw, c, z_rim_nominal)
        if m.sum() < 20:
            raise SystemExit(f"only {m.sum()} rim-band points near {c}; check the window")
        c, r, keep, sd = fit_center(pw[m][:, :2], SPEC.a_top_outer, SPEC.b_top_outer, c)
    pts = pw[m]
    z_rim = float(np.median(pts[keep, 2]))
    return dict(center=c, center_sd=sd, z_rim=z_rim, n_band=int(m.sum()),
                n_inlier=int(keep.sum()),
                median_abs_res=float(np.median(np.abs(r[keep]))),
                p90_abs_res=float(np.quantile(np.abs(r[keep]), 0.9)),
                pts=pts, res=r, keep=keep)


def ellipse_xy(c, a, b, n=200):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return c[0] + a * np.cos(th), c[1] + b * np.sin(th)


def run(ep_dir: Path, window, stride: int) -> dict:
    out = OUT_ROOT / ep_dir.name
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ep_dir / "meta.json").read_text())
    t_send = pour_send_time(ep_dir)
    pw, n_frames = fuse_depth(ep_dir, meta, t_send + window[0], t_send + window[1], stride)
    print(f"{ep_dir.name}: {n_frames} pre-pour depth frames fused -> {len(pw)} points")

    # 2. table
    z_tab, n_tab, mad_tab = table_height(pw, RECEIVER_XY)
    print(f"table height   {z_tab*1e3:+8.1f} mm  (baked {TABLE_Z*1e3:+.1f}; {n_tab} pts, "
          f"MAD {mad_tab*1e3:.1f} mm)")

    # 3. receiver
    rcv = fit_rim(pw, RECEIVER_XY, z_tab + SPEC.rim_z)
    print(f"receiver xy    ({rcv['center'][0]:.4f}, {rcv['center'][1]:.4f}) "
          f"+- ({rcv['center_sd'][0]*1e3:.1f}, {rcv['center_sd'][1]*1e3:.1f}) mm  "
          f"(baked ({RECEIVER_XY[0]:.4f}, {RECEIVER_XY[1]:.4f}); "
          f"{rcv['n_inlier']}/{rcv['n_band']} pts, "
          f"median |res| {rcv['median_abs_res']*1e3:.1f} mm, "
          f"p90 {rcv['p90_abs_res']*1e3:.1f} mm)")
    print(f"receiver rim z {rcv['z_rim']*1e3:+8.1f} mm -> table "
          f"{(rcv['z_rim']-SPEC.rim_z)*1e3:+.1f} mm "
          f"through the cup height (plane fit {z_tab*1e3:+.1f})")

    # 4. held cup at the pour-start reference pose, nominal grasp (zero shift)
    ep = load_episode(ep_dir, PRE_ROLL, HOLD_SECONDS)
    arm = RecordedPanda(ep, write_cup_obj(SPEC, out / "cup_render.obj"), height=64, width=64,
                        max_geom=4000, cup_shift_xy=(0.0, 0.0))
    p_nom, _q_nom = arm.cup_pose_at(ep["t_ref"])
    arm.close()
    cup = fit_rim(pw, p_nom[:2], p_nom[2] + SPEC.rim_z)
    shift = cup["center"] - p_nom[:2]
    dz_rim = cup["z_rim"] - (p_nom[2] + SPEC.rim_z)
    print(f"held-cup shift ({shift[0]:+.4f}, {shift[1]:+.4f}) "
          f"+- ({cup['center_sd'][0]*1e3:.1f}, {cup['center_sd'][1]*1e3:.1f}) mm  "
          f"(baked ({CUP_SHIFT_XY[0]:+.4f}, {CUP_SHIFT_XY[1]:+.4f}); "
          f"{cup['n_inlier']}/{cup['n_band']} pts, "
          f"median |res| {cup['median_abs_res']*1e3:.1f} mm, "
          f"p90 {cup['p90_abs_res']*1e3:.1f} mm)")
    print(f"held-cup rim z {cup['z_rim']*1e3:+8.1f} mm, {dz_rim*1e3:+.1f} mm vs the grasp chain")

    # 5. liquid surface -> fill volume (indicative only, see the module docstring)
    d = np.hypot(pw[:, 0] - cup["center"][0], pw[:, 1] - cup["center"][1])
    ms = ((d < SURFACE_R) & (pw[:, 2] < cup["z_rim"] - SURFACE_BAND[0])
          & (pw[:, 2] > cup["z_rim"] - SURFACE_BAND[1]))
    zs = pw[ms, 2]
    for _ in range(3):
        med = np.median(zs)
        mad = 1.4826 * np.median(np.abs(zs - med))
        zs = zs[np.abs(zs - med) < 3.0 * max(mad, 1e-4)]
    z_surf = float(np.median(zs))
    mad_surf = float(1.4826 * np.median(np.abs(zs - z_surf)))
    fill_depth = z_surf - (cup["z_rim"] - SPEC.rim_z + SPEC.floor_z)
    v0 = SPEC.cavity_volume(fill_depth)
    area = float(SPEC.cavity_area(SPEC.floor_z + fill_depth))
    # 1 mm of surface height is `area` m^3; quote the MAD-based spread in mL
    v0_pm = area * mad_surf
    depths = np.linspace(0.0, 0.09, 400)
    depth300 = float(np.interp(300e-6, [SPEC.cavity_volume(x) for x in depths], depths))
    print(f"liquid surface {z_surf*1e3:+8.1f} mm ({len(zs)} pts, MAD {mad_surf*1e3:.1f} mm): "
          f"{(cup['z_rim']-z_surf)*1e3:.1f} mm below the rim, fill depth {fill_depth*1e3:.1f} mm "
          f"-> {v0*1e6:.0f} mL at face value (+-{v0_pm*1e6:.0f} at 1 MAD; {area*1e6*1e-3:.1f} mL "
          f"per mm); the metered 300 mL would put it "
          f"{(SPEC.rim_z-SPEC.floor_z-depth300)*1e3:.1f} mm "
          f"below the rim. IR depth through the translucent liquid reads deep: indicative only.")

    res = dict(
        episode=ep_dir.name, window_s=list(window), n_frames=n_frames, n_points=len(pw),
        table_z=z_tab, table_z_mad=mad_tab, table_z_from_receiver_rim=rcv["z_rim"] - SPEC.rim_z,
        receiver_xy=rcv["center"].tolist(), receiver_rim_z=rcv["z_rim"],
        receiver_fit=dict(n_band=rcv["n_band"], n_inlier=rcv["n_inlier"],
                          median_abs_res=rcv["median_abs_res"], p90_abs_res=rcv["p90_abs_res"]),
        held_cup_nominal_xyz=p_nom.tolist(), held_cup_xy=cup["center"].tolist(),
        cup_shift_xy=shift.tolist(), held_cup_rim_z=cup["z_rim"], held_cup_rim_dz=dz_rim,
        held_cup_fit=dict(n_band=cup["n_band"], n_inlier=cup["n_inlier"],
                          median_abs_res=cup["median_abs_res"], p90_abs_res=cup["p90_abs_res"]),
        surface_z=z_surf, surface_mad=mad_surf, surface_n=len(zs),
        fill_depth=fill_depth, v0_ml_face_value=v0 * 1e6, v0_ml_per_mm=area * 1e3,
        surface_note="IR depth through the translucent glycerol reads deep; indicative only",
        receiver_xy_sd=rcv["center_sd"].tolist(), cup_shift_sd=cup["center_sd"].tolist(),
        baked=dict(table_z=TABLE_Z, receiver_xy=list(RECEIVER_XY), cup_shift_xy=list(CUP_SHIFT_XY)),
    )
    (out / "calibration.json").write_text(json.dumps(res, indent=2))

    # ---- figure ---------------------------------------------------------------------
    mm = 1e3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, fit, c0, lab in ((axes[0], rcv, np.asarray(RECEIVER_XY), "receiver"),
                             (axes[1], cup, p_nom[:2], "held cup")):
        pts, keep = fit["pts"], fit["keep"]
        ax.plot(pts[~keep, 0] * mm, pts[~keep, 1] * mm, ".", ms=3, color="0.6",
                label="rim band, clipped")
        ax.plot(pts[keep, 0] * mm, pts[keep, 1] * mm, ".", ms=3, color="tab:blue", label="inliers")
        ex, ey = ellipse_xy(fit["center"], SPEC.a_top_outer, SPEC.b_top_outer)
        ax.plot(ex * mm, ey * mm, "k-", lw=1, label="outer top ellipse (calipers)")
        ax.plot(c0[0] * mm, c0[1] * mm, "r+", ms=12, mew=1.5, label="start / nominal center")
        ax.plot(fit["center"][0] * mm, fit["center"][1] * mm, "kx", ms=9, mew=1.5,
                label="fitted center")
        ax.set_aspect("equal")
        ax.set_title(f"{lab}: {fit['n_inlier']} inliers, "
                     f"median |res| {fit['median_abs_res']*mm:.1f} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)
    ax = axes[2]
    near = d < 0.075
    ax.scatter(pw[near, 0] * mm, pw[near, 2] * mm, s=2, c=pw[near, 1] * mm, cmap="coolwarm")
    ax.axhline(cup["z_rim"] * mm, color="k", lw=0.8,
               label=f"measured rim z {cup['z_rim']*mm:.1f} mm")
    ax.axhline(z_surf * mm, color="tab:orange", lw=1.2, label=f"liquid surface {z_surf*mm:.1f} mm")
    ax.axhline((cup["z_rim"] - SPEC.rim_z) * mm, color="k", lw=0.8, ls=":",
               label="cup base (rim - 99.06)")
    ax.set_title(f"held cup, side view: surface {(cup['z_rim']-z_surf)*mm:.0f} mm below the rim "
                 f"({v0*1e6:.0f} mL at face value; IR reads deep)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=120)
    plt.close(fig)
    print("wrote", out / "calibration.json", "and", out / "calibration.png")
    print("twin flags for this episode (the grasp roll is NOT measured here; it stays the "
          f"onset-tuned default):\n  --table-z {z_tab:.4f} --receiver-xy {rcv['center'][0]:.4f} "
          f"{rcv['center'][1]:.4f} --cup-shift {shift[0]:.4f} {shift[1]:.4f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, default=REPO / "pouring_real_data" / "ep0001")
    ap.add_argument("--window", type=float, nargs=2, default=(-3.0, -1.2),
                    help="pre-pour window, s relative to the pour command (arm at rest)")
    ap.add_argument("--stride", type=int, default=4, help="use every Nth side frame")
    args = ap.parse_args()
    run(args.episode.resolve(), tuple(args.window), args.stride)
