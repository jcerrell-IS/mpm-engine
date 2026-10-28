"""The two scene constants of the honey pour (examples/pour_franka.py) that were measured
rather than taken from calipers, and how to re-measure them.

  azimuth   phi*, the cup-frame azimuth of the pour's downhill direction at peak tilt.
            The Genesis grasp does not know about the spout, so PandaPour.CUP_TO_HAND is
            the Genesis grasp composed with Rz(-phi*), which points the spout (+x cup)
            where the pour goes. Prints phi*, the composed matrix, and how far the
            downhill azimuth wanders while the cup is tilted more than 30 deg.
  receiver  where the stream lands: runs the honey pour at 96^3 with NO receiver and
            records every particle crossing the receiver-rim plane downward. The
            crossing centroid is RECEIVER_POS's (x, y); the spread says whether the
            inner opening catches the stream. Re-run after any change to the action,
            the fill, or the cup.

Run:
  python experiments/pour/pour_franka_calibrate.py azimuth
  python experiments/pour/pour_franka_calibrate.py receiver [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))


def azimuth() -> None:
    from warpmpm.adapters.mujoco_adapter import PandaPour
    from warpmpm.colliders.glass import quat_to_mat

    arm = PandaPour(height=64, width=64)
    print("action duration", arm.duration, "s")
    for t in np.linspace(0.0, arm.TILT_SECONDS, 13):
        p, q = arm.cup_pose_at(float(t))
        R = quat_to_mat(q)
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        phi = np.degrees(np.arctan2(-R[2, 1], -R[2, 0]))
        print(f"t={t:5.2f}s tilt={tilt:6.2f} deg  downhill cup-azimuth phi={phi:8.2f} deg  "
              f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
    # with the current CUP_TO_HAND the residual azimuth at peak should be ~0: the spout
    # already points downhill. The matrix that would zero it exactly is printed anyway.
    p, q = arm.cup_pose_at(arm.TILT_SECONDS)
    R = quat_to_mat(q)
    phi = np.arctan2(-R[2, 1], -R[2, 0])
    print(f"\nresidual downhill azimuth at max tilt in the CURRENT cup frame: "
          f"{np.degrees(phi):.3f} deg")
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0],
                   [0.0, 0.0, 1.0]])
    np.set_printoptions(precision=8, suppress=False)
    print("CUP_TO_HAND that zeroes it = Rz(-phi) @ CUP_TO_HAND =\n", repr(Rz.T @ arm.CUP_TO_HAND))
    worst = 0.0
    for t in np.linspace(0.0, arm.duration, 120):
        _p2, q2 = arm.cup_pose_at(float(t))
        R2 = quat_to_mat(q2) @ Rz
        if np.degrees(np.arccos(np.clip(R2[2, 2], -1, 1))) > 30.0:
            worst = max(worst, abs(np.degrees(np.arctan2(-R2[2, 1], -R2[2, 0]))))
    print(f"max |downhill azimuth| while tilt > 30 deg over the whole action: {worst:.2f} deg")
    arm.close()


def receiver(device: str = "auto", n_grid: int = 96) -> None:
    from pour_franka import (
        FILL_FRACTION,
        FPS,
        GRID_LIM,
        HONEY,
        SPEC,
        WORLD_TO_MPM,
        collision_extras,
        q_xyzw,
        substeps_per_tick,
    )
    from warpmpm import GridConfig, Solver, newtonian
    from warpmpm.adapters.mujoco_adapter import PandaPour
    from warpmpm.colliders.glass import angular_velocity_between, quat_to_mat
    from warpmpm.geometry.measuring_cup import build_cup_sdf, cup_fill, project_out_of_solid

    arm = PandaPour(height=64, width=64)
    grid = GridConfig(n_grid=n_grid, grid_lim=GRID_LIM)
    h = grid.dx / 2
    extras = collision_extras(grid.dx)
    sdf = build_cup_sdf(SPEC, res=160, margin=0.010, extra_wall=extras[0], extra_base=extras[1])
    cup_pos0, cup_quat0 = arm.cup_pose_at(0.0)
    pos_local, vol = cup_fill(SPEC, h, fill_fraction=FILL_FRACTION)
    pos = (cup_pos0 + pos_local @ quat_to_mat(cup_quat0).T + WORLD_TO_MPM).astype(np.float32)
    print(f"N={len(pos)} h={h*1e3:.2f}mm extras=({extras[0]*1e3:.1f},{extras[1]*1e3:.1f})mm "
          f"cup0={np.round(cup_pos0, 3)}")
    s = Solver(grid=grid, device=device).load_particles(pos, vol)
    s.set_material(newtonian(**HONEY))
    s.add_plane((0, 0, WORLD_TO_MPM[2]), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    src = s.add_sdf_collider(sdf, center=cup_pos0 + WORLD_TO_MPM, quat=q_xyzw(cup_quat0),
                             band=0.5 * grid.dx, surface="separable", friction=0.05)
    dt_tick = 1.0 / FPS
    substeps = substeps_per_tick(HONEY, grid.dx, dt_tick)
    dt = dt_tick / substeps
    for _ in range(round(0.4 * FPS)):
        s.step(dt, substeps)
    x, v, npx = project_out_of_solid(s.x(), s.v(), cup_pos0 + WORLD_TO_MPM, cup_quat0, SPEC,
                                     clearance=0.35 * h, extra_wall=extras[0],
                                     extra_base=extras[1])
    s.set_x(x)
    s.set_v(v)
    print(f"settled ({substeps} substeps/frame), projected {npx}")

    plane_z = SPEC.rim_z + WORLD_TO_MPM[2]      # receiver rim height with its base on z=0
    cross, cross_t = [], []
    z_prev = s.x()[:, 2].copy()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    n_frames = round(arm.duration * FPS) + 30
    t0 = time.time()
    for frame in range(n_frames):
        t = frame * dt_tick
        p0, q0 = arm.cup_pose_at(t)
        p1, q1 = arm.cup_pose_at(t + dt_tick)
        s.set_sdf_pose(src, center=p0 + WORLD_TO_MPM, quat=q_xyzw(q0),
                       velocity=(p1 - p0) / dt_tick,
                       omega=angular_velocity_between(q0, q1, dt_tick))
        s.step(dt, substeps)
        xn = s.x()
        crossed = (z_prev > plane_z) & (xn[:, 2] <= plane_z)
        if crossed.any():
            cross.append(xn[crossed, :2] - WORLD_TO_MPM[:2])
            cross_t.extend([t] * int(crossed.sum()))
        z_prev = xn[:, 2].copy()
        lo, hi = np.minimum(lo, xn.min(axis=0)), np.maximum(hi, xn.max(axis=0))
        if frame % 60 == 0:
            print(f"frame {frame}/{n_frames} t={t:.2f} crossings={sum(len(c) for c in cross)} "
                  f"[{(time.time() - t0) / (frame + 1) * 1000:.0f} ms/f]")
    arm.close()
    if not cross:
        print("no rim-plane crossings: the stream never reached the receiver height")
        return
    c = np.concatenate(cross)
    ct = np.asarray(cross_t)
    print(f"\ncrossings at the rim plane (world z={SPEC.rim_z:.3f}): {len(c)}, "
          f"{ct.min():.2f}..{ct.max():.2f} s")
    print(f"centroid  x={c[:, 0].mean():+.4f}  y={c[:, 1].mean():+.4f}   "
          f"median x={np.median(c[:, 0]):+.4f} y={np.median(c[:, 1]):+.4f}   "
          f"std ({c[:, 0].std()*1e3:.1f}, {c[:, 1].std()*1e3:.1f}) mm")
    for cx, cy, tag in ((c[:, 0].mean(), c[:, 1].mean(), "centroid"),
                        (np.median(c[:, 0]), np.median(c[:, 1]), "median")):
        inside = (((c[:, 0] - cx) / SPEC.a_top_inner) ** 2
                  + ((c[:, 1] - cy) / SPEC.b_top_inner) ** 2) < 1.0
        print(f"caught by the inner opening centered at the {tag}: {inside.mean() * 100:.1f}%")
    print(f"particle bounds (MPM frame): lo={np.round(lo, 3)} hi={np.round(hi, 3)} "
          f"(domain [0, {GRID_LIM}])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["azimuth", "receiver"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-grid", type=int, default=96)
    args = ap.parse_args()
    if args.what == "azimuth":
        azimuth()
    else:
        receiver(device=args.device, n_grid=args.n_grid)
