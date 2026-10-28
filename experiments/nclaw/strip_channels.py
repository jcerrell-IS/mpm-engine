"""Write a reduced-channel copy of a dump, for identification under a tier.

Two tiers, each a copy of a schema-valid dump with fewer measured channels and
a per-channel provenance record, so a run at a tier cannot read a channel the
tier does not have:

no_stress
    every stored kinematic channel is kept exactly (x, v, L, F, volume, mass,
    volume0), the stress channel is zeroed and ``pressure_source`` becomes
    "absent". This is the tier of a comparison where only the other engine's
    motion is observable. What a leg then
    needs instead of the stress trace is a stated pressure model, and
    ``validate_dump_schema(...).has_pressure`` is False so a leg that silently
    read the trace refuses instead.

positions_only
    positions and times are kept; velocities come from central finite
    differences, the velocity gradient and the deformation gradient from the
    moving least squares of ``ingest.py`` over k reference-frame neighbours,
    the volume from det(F) times the reference volume, and the mass from the
    scene's density. The scene facts a camera plus a scene description gives
    are kept: particle reference volume, density, gravity, the grid, the box
    and the frame times. Stress is absent here too.

Both derivations are the ones ``experiments/nclaw/ingest.py`` applies to a
folder of theirs that is missing those channels; this module calls those
functions, so the two paths cannot drift.

CLI:
  .venv/bin/python -m experiments.nclaw.strip_channels <dump.npz> \\
      [--tier no_stress|positions_only] [--out path.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nclaw.ingest import (
    INGEST_VERSION,
    _neighbors,
    fd_velocity,
    mls_deformation_gradient,
    mls_velocity_gradient,
)
from ident.io.schema import validate_dump_schema

TIER_SUFFIX = {"no_stress": "_no_stress", "positions_only": "_positions_only"}
TIERS = tuple(TIER_SUFFIX)


def tier_path(src: str | Path, tier: str) -> Path:
    """Where the tier copy of ``src`` goes: next to it, with the tier in the name."""
    src = Path(src)
    if tier not in TIER_SUFFIX:
        raise ValueError(f"unknown tier {tier!r}; known: {TIERS}")
    return src.with_name(f"{src.stem}{TIER_SUFFIX[tier]}.npz")


def _copy_globals(d) -> dict[str, np.ndarray]:
    """Every global and array of the source except the ones a tier rewrites."""
    rewritten = {"stress", "pressure_source", "meta_json", "v", "L", "F",
                 "volume", "mass", "volume0"}
    return {k: d[k] for k in d.files if k not in rewritten}


def write_no_stress_dump(src: str | Path, out: str | Path | None = None,
                         log=print) -> Path:
    """Copy a dump with the stress channel excluded and everything else kept.

    Positions, velocities, L, F, volumes and masses are written back bitwise, so
    a leg that reads only kinematics gets exactly the numbers the full-channel
    run gave it, and any difference in a result at this tier is attributable to
    the missing stress channel alone.
    """
    src = Path(src)
    validate_dump_schema(src)                      # never touch raw keys unchecked
    d = np.load(src)
    out = Path(out) if out is not None else tier_path(src, "no_stress")
    meta = json.loads(str(d["meta_json"]))
    prov = dict(meta.get("channel_provenance") or {})
    prov.update({k: prov.get(k, "measured") for k in ("x", "v", "L", "F")})
    prov["stress"] = "excluded"
    notes = list(meta.get("degradation_notes") or [])
    notes.append("tier no_stress: every stored kinematic channel is kept bitwise "
                 "and the stress channel is excluded, so there is no oracle "
                 "pressure. A leg that needs pressure must state its model.")
    meta.update({
        "channel_provenance": prov,
        "degradation_notes": notes,
        "has_oracle_pressure": False,
        "tier": "no_stress",
        "tier_source_dump": src.name,
        "tier_version": INGEST_VERSION,
    })
    arrays = _copy_globals(d)
    arrays.update({
        "v": d["v"], "L": d["L"], "F": d["F"],
        "volume": d["volume"], "mass": d["mass"], "volume0": d["volume0"],
        "stress": np.zeros_like(d["stress"]),
        "pressure_source": np.array("absent"),
        "meta_json": np.array(json.dumps(meta, default=float)),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    log(f"[tier] {src.name} -> {out.name}: no_stress, provenance={prov}")
    return out


def write_positions_only_dump(src: str | Path, out: str | Path | None = None,
                              mls_k: int = 24, mls_ridge: float = 1.0e-8,
                              log=print) -> Path:
    """Copy a dump keeping positions, times and the scene facts, deriving the rest.

    The derivations are ``ingest.fd_velocity``, ``ingest.mls_velocity_gradient``
    and ``ingest.mls_deformation_gradient``, the same functions a folder of
    theirs missing those channels goes through, over neighbour sets fixed in the
    reference frame.
    """
    src = Path(src)
    validate_dump_schema(src)
    d = np.load(src)
    out = Path(out) if out is not None else tier_path(src, "positions_only")
    meta = json.loads(str(d["meta_json"]))
    x = d["x"].astype(np.float64)
    T, P = x.shape[0], x.shape[1]
    frame_dt = float(d["frame_dt"])
    rho = float(d["rho_s"])

    nbr = _neighbors(x[0], int(mls_k))
    v = fd_velocity(x, frame_dt)
    L = mls_velocity_gradient(x, v, nbr, float(mls_ridge))
    F = mls_deformation_gradient(x, nbr, float(mls_ridge))
    J = np.linalg.det(F)
    J = np.where(np.abs(J) < 1e-12, 1.0, J)

    vol0 = d["volume0"].astype(np.float64)         # scene fact: their seeding volume
    volume = J * vol0[None, :]
    mass = (rho * vol0).astype(np.float32)

    prov = {"x": "measured", "v": "derived", "L": "derived", "F": "derived",
            "volume": "derived", "mass": "derived", "volume0": "scene_fact",
            "stress": "excluded"}
    notes = list(meta.get("degradation_notes") or [])
    notes += [
        "tier positions_only: positions and frame times are measured; v is "
        "central finite differences of x in time",
        f"L and F are moving least squares over k={int(mls_k)} reference-frame "
        "neighbours, the ingest's own derivation",
        "volume is det(F) times the scene's reference volume and mass is the "
        "scene's density times that volume",
        "stress excluded: no oracle pressure. A leg that needs pressure must "
        "state its model.",
    ]
    meta.update({
        "channel_provenance": prov,
        "degradation_notes": notes,
        "has_oracle_pressure": False,
        "tier": "positions_only",
        "tier_source_dump": src.name,
        "tier_version": INGEST_VERSION,
        "mls_k": int(mls_k), "mls_ridge": float(mls_ridge),
    })
    arrays = _copy_globals(d)
    arrays.update({
        "v": v.astype(np.float32),
        "L": L.astype(np.float32).reshape(T, P, 9),
        "F": F.astype(np.float32).reshape(T, P, 9),
        "volume": volume.astype(np.float32),
        "mass": mass,
        "volume0": vol0.astype(np.float32),
        "stress": np.zeros((T, P, 9), dtype=np.float32),
        "pressure_source": np.array("absent"),
        "meta_json": np.array(json.dumps(meta, default=float)),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    log(f"[tier] {src.name} -> {out.name}: positions_only, k={int(mls_k)}, "
        f"provenance={prov}")
    return out


def write_tier_dump(src: str | Path, tier: str, out: str | Path | None = None,
                    log=print, **kw: Any) -> Path:
    """The tier copy of one dump, built if absent and reused if present."""
    dest = Path(out) if out is not None else tier_path(src, tier)
    if dest.exists():
        log(f"[tier] reuse {dest.name} (exists)")
        return dest
    if tier == "no_stress":
        return write_no_stress_dump(src, dest, log=log, **kw)
    if tier == "positions_only":
        return write_positions_only_dump(src, dest, log=log, **kw)
    raise ValueError(f"unknown tier {tier!r}; known: {TIERS}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump", help="a schema-valid dump npz")
    ap.add_argument("--tier", default="no_stress", choices=TIERS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mls-k", type=int, default=24)
    a = ap.parse_args(argv)
    kw = {"mls_k": a.mls_k} if a.tier == "positions_only" else {}
    path = write_tier_dump(a.dump, a.tier, a.out, **kw)
    print(path)


if __name__ == "__main__":
    main()
