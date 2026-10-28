"""Figures of the recorded-pour walkthrough note (docs/pour_pipeline_ep0001.tex).

All four read the pipeline's own outputs for one episode and write docs/figs/*.pdf:
  tilt        Stage-0 cup tilt on the pour clock, with the pour command, the measured
              onset, and the return marked                         -> ep0001_tilt.pdf
  extraction  a Stage-1 overlay frame beside the receiver curve V(t) and its IoU
              (needs the probe stills of pour_perception.py --probe) -> ep0001_extraction.pdf
  head        the film head at one instant: the section through the spout and the level
              over the rim curve, from the real pose and measured V(t) -> ep0001_head.pdf
  film        the lubrication-film schematic (no data)              -> film_sketch.pdf
  hold        the hold-time lookup table of pour_hold_sweep.py, twin at eta_hat and at
              the handbook value, targets read off                 -> ep0001_hold_table.pdf

Run (after pour_perception.py, and --probe for the extraction still):
  python experiments/pour/pour_note_figures.py            # all four
  python experiments/pour/pour_note_figures.py head tilt  # a subset
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
from matplotlib import gridspec
from matplotlib.patches import Arc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "experiments" / "pour"))

FIGS = REPO / "docs" / "figs"
T_FRAME = 3.20          # the instant shown in the extraction and head figures (pour clock)
V0_ML = 300.0           # the protocol fill the head figure assumes (identification prior)


def load_obs(episode: str) -> dict:
    return dict(np.load(REPO / "out" / "pour_wf" / episode / "observations.npz"))


def fig_tilt(episode: str) -> None:
    d = load_obs(episode)
    t, tilt, t_ret = d["t"], d["tilt_deg"], float(d["t_ret_done"])
    onset = float(d["rcv_onset_s"]) if "rcv_onset_s" in d else np.nan
    m = (t >= -1.2) & (t <= t_ret + 2.5)
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.plot(t[m], tilt[m], color="k", lw=1.4)
    marks = [(0.0, "pour command ($t_\\mathrm{send}$)"), (t_ret, "return done")]
    if np.isfinite(onset):
        marks.insert(1, (onset, "onset: first liquid lands"))
    for x, lab in marks:
        ax.axvline(x, color="grey", lw=0.9, ls="--")
        ax.text(x + 0.09, 65, lab, rotation=90, va="top", ha="left", fontsize=8, color="0.25")
    ax.set_xlabel(r"$t - t_\mathrm{send}$  [s]")
    ax.set_ylabel(r"cup tilt $\alpha$  [deg]")
    ax.set_ylim(-3, 68)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / f"{episode}_tilt.pdf")
    plt.close(fig)
    print(f"tilt: range {tilt[m].min():.1f}..{tilt[m].max():.1f} deg, onset {onset:.2f} s, "
          f"return done {t_ret:.2f} s")


def fig_extraction(episode: str) -> None:
    import imageio.v2 as imageio

    ep = REPO / "out" / "pour_wf" / episode
    d = load_obs(episode)
    t, V, iou, t_ret = d["t"], d["rcv_vol"] * 1e6, d["rcv_iou"], float(d["t_ret_done"])
    stills = sorted(ep.glob(f"probe_*_t+{T_FRAME:.2f}.png"))
    if not stills:
        raise SystemExit(f"no probe still at t=+{T_FRAME:.2f}; run pour_perception.py --probe")
    crop = imageio.imread(stills[0])[320:620, 70:410]
    v_final = float(np.nanmedian(V[t > t[np.isfinite(V)].max() - 1.0]))

    fig = plt.figure(figsize=(6.6, 3.0))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.8], height_ratios=[2.4, 1.0],
                           hspace=0.10, wspace=0.24, left=0.005, right=0.985, top=0.90,
                           bottom=0.16)
    axi = fig.add_subplot(gs[:, 0])
    axi.imshow(crop)
    axi.set_axis_off()
    axi.set_title(rf"overlay frame, $t = +{T_FRAME:.1f}$ s", fontsize=9)

    m = (t >= -1.2) & (t <= 7.9)
    axv = fig.add_subplot(gs[0, 1])
    axv.plot(t[m], V[m], color="k", lw=1.3)
    for x in (0.0, t_ret):
        axv.axvline(x, color="grey", lw=0.8, ls="--")
    axv.axvline(T_FRAME, color="tab:orange", lw=1.0, ls=":")
    axv.set_ylabel(r"$V(t)$  [mL]", fontsize=9)
    axv.set_ylim(-6, 118)
    axv.grid(alpha=0.25)
    axv.tick_params(labelbottom=False, labelsize=8)
    axv.annotate(f"settles at {v_final:.1f} mL", xy=(7.75, v_final), xytext=(5.6, 62),
                 fontsize=8, color="0.25",
                 arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    axq = fig.add_subplot(gs[1, 1], sharex=axv)
    axq.plot(t[m], iou[m], color="0.45", lw=1.0)
    for x in (0.0, t_ret):
        axq.axvline(x, color="grey", lw=0.8, ls="--")
    axq.axvline(T_FRAME, color="tab:orange", lw=1.0, ls=":")
    axq.set_ylabel("IoU", fontsize=9)
    axq.set_ylim(0, 1.05)
    axq.set_xlabel(r"$t - t_\mathrm{send}$  [s]", fontsize=9)
    axq.grid(alpha=0.25)
    axq.tick_params(labelsize=8)
    fig.savefig(FIGS / f"{episode}_extraction.pdf", dpi=200)
    plt.close(fig)
    print(f"extraction: V settles at {v_final:.1f} mL, IoU max {np.nanmax(iou[m]):.2f}")


def fig_head(episode: str) -> None:
    from pour_perception import SPEC, build_cavity_lattice, rim_curve_local

    from warpmpm.colliders.glass import quat_to_mat

    d = load_obs(episode)
    i = int(np.argmin(np.abs(d["t"] - T_FRAME)))
    p, q = d["cup_pos"][i], d["cup_quat"][i]
    R = quat_to_mat(q)
    tilt = float(d["tilt_deg"][i])
    v_rcv = d["rcv_vol"][i]
    if not np.isfinite(v_rcv):
        ok = np.isfinite(d["rcv_vol"])
        v_rcv = np.interp(T_FRAME, d["t"][ok], d["rcv_vol"][ok])
    v_src = V0_ML * 1e-6 - v_rcv
    lattice, cell = build_cavity_lattice()
    wz = np.sort(lattice @ R[2] + p[2])
    L = float(wz[int(np.clip(round(v_src / cell), 1, len(wz) - 1))])

    mm = 1e3
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(6.6, 2.9),
        gridspec_kw=dict(width_ratios=[1.25, 1.0], left=0.075, right=0.985, top=0.90,
                         bottom=0.17, wspace=0.30))
    # (a) section through the spout (local y = 0), world x-z plane
    slab = lattice[np.abs(lattice[:, 1]) < 0.0015]
    w = slab @ R.T + p
    liq = w[w[:, 2] < L]
    axa.plot(liq[:, 0] * mm, liq[:, 2] * mm, ".", ms=1.2, color="#f39c2c", alpha=0.5,
             rasterized=True)
    zs = np.linspace(SPEC.floor_z, SPEC.rim_z, 120)
    a_i, _ = SPEC.inner_semi_axes(zs)
    xplus = a_i + SPEC.spout_dx(np.zeros_like(zs), zs)
    for xs in (xplus, -a_i):
        pts = np.stack([xs, np.zeros_like(zs), zs], 1) @ R.T + p
        axa.plot(pts[:, 0] * mm, pts[:, 2] * mm, "k-", lw=1.4)
    zf = SPEC.floor_z
    a_f, _ = SPEC.inner_semi_axes(zf)
    fl = np.stack([np.linspace(-a_f, a_f + SPEC.spout_dx(0.0, zf), 30), np.zeros(30),
                   np.full(30, zf)], 1) @ R.T + p
    axa.plot(fl[:, 0] * mm, fl[:, 2] * mm, "k-", lw=1.4)
    tip = np.array([SPEC.tip_x, 0.0, SPEC.rim_z]) @ R.T + p
    x_lo = min(liq[:, 0].min(), fl[:, 0].min()) * mm - 12
    x_hi = tip[0] * mm + 30
    axa.hlines(L * mm, x_lo, x_hi, color="tab:blue", lw=1.2, ls="--")
    axa.text(x_lo + 2, L * mm + 2, "level $L(t)$ (horizontal)", fontsize=8, color="tab:blue")
    axa.plot(*(tip[[0, 2]] * mm), "o", ms=4, color="k")
    axa.text(tip[0] * mm + 3, tip[2] * mm - 6, "brink\n(spout edge)", fontsize=8, va="top")
    axa.annotate("", xy=(tip[0] * mm + 14, L * mm), xytext=(tip[0] * mm + 14, tip[2] * mm),
                 arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.2))
    axa.text(tip[0] * mm + 17, (L + tip[2]) / 2 * mm, "head", fontsize=8, color="tab:red",
             va="center")
    axa.set_aspect("equal")
    axa.set_xlabel("world $x$  [mm]", fontsize=9)
    axa.set_ylabel("world $z$  [mm]", fontsize=9)
    axa.set_title(f"(a) section through the spout, $t=+{T_FRAME:.1f}$ s "
                  f"(tilt {tilt:.0f}$^\\circ$)", fontsize=9)
    axa.tick_params(labelsize=8)
    # (b) the rim curve across the spout: the head h(y)
    rim = rim_curve_local()
    ys = rim[:, 1]
    keep = np.abs(ys) <= 0.019
    rim, ys = rim[keep], ys[keep]
    zr = (rim @ R.T + p)[:, 2]
    axb.plot(ys * mm, zr * mm, "k-", lw=1.4, label="rim curve $z_\\mathrm{rim}(y)$")
    axb.hlines(L * mm, ys[0] * mm, ys[-1] * mm, color="tab:blue", lw=1.2, ls="--",
               label="level $L(t)$")
    axb.fill_between(ys * mm, zr * mm, np.minimum(L, np.maximum(zr, L)) * mm, where=zr < L,
                     color="#f39c2c", alpha=0.55, label="gap $L - z_\\mathrm{rim}(y)$")
    j = len(ys) // 2
    axb.annotate("", xy=(0, L * mm), xytext=(0, zr[j] * mm),
                 arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.2))
    axb.text(1.2, (L + zr[j]) / 2 * mm, "$L{-}z_\\mathrm{rim}(0)$", fontsize=8,
             color="tab:red", va="center")
    axb.set_xlabel("transverse $y$ across the spout  [mm]", fontsize=9)
    axb.set_ylabel("world $z$  [mm]", fontsize=9)
    axb.set_title("(b) the level over the rim curve", fontsize=9)
    axb.legend(fontsize=7, loc="lower center")
    axb.tick_params(labelsize=8)
    fig.savefig(FIGS / f"{episode}_head.pdf", dpi=200)
    plt.close(fig)
    cosa = np.cos(np.radians(tilt))
    print(f"head: t={d['t'][i]:.2f} tilt={tilt:.1f} v_rcv={v_rcv*1e6:.1f} mL "
          f"v_src={v_src*1e6:.1f} mL head at the tip {(L - tip[2]) * cosa * 1e3:.1f} mm")


def fig_film() -> None:
    alpha = 50.0
    a = np.radians(alpha)
    es = np.array([np.cos(a), -np.sin(a)])
    en = np.array([np.sin(a), np.cos(a)])
    S, H = 1.0, 0.24

    def P(s, n):
        return s * es + n * en

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    poly = np.array([P(0.02, 0), P(S, 0), P(S, H), P(0.02, H)])
    ax.fill(poly[:, 0], poly[:, 1], color="#f7b955", alpha=0.55, zorder=1)
    w0, w1 = P(-0.16, 0), P(S + 0.10, 0)
    ax.plot([w0[0], w1[0]], [w0[1], w1[1]], "k-", lw=2.0, zorder=3)
    for s in np.linspace(-0.10, S + 0.06, 26):
        b, e = P(s, 0), P(s - 0.035, -0.045)
        ax.plot([b[0], e[0]], [b[1], e[1]], "k-", lw=0.6, alpha=0.55)
    f0, f1 = P(0.02, H), P(S, H)
    ax.plot([f0[0], f1[0]], [f0[1], f1[1]], color="#c87f0a", lw=1.4, zorder=3)
    ax.plot([w0[0], w0[0] - 0.40], [w0[1], w0[1]], color="grey", lw=0.8, ls="--")
    ax.add_patch(Arc(w0, 0.62, 0.62, theta1=180, theta2=180 + alpha, color="grey", lw=0.9))
    am = np.radians(180 + alpha / 2)
    ax.text(w0[0] + 0.40 * np.cos(am), w0[1] + 0.40 * np.sin(am), r"$\alpha$", fontsize=11,
            ha="center", va="center")
    s0 = 0.52
    ns = np.linspace(0, H, 60)
    u = (H * ns - ns ** 2 / 2) / (H ** 2 / 2) * 0.30
    tips = np.array([P(s0 + ui, ni) for ui, ni in zip(u, ns, strict=True)])
    ax.plot(tips[:, 0], tips[:, 1], color="tab:red", lw=1.5, zorder=4)
    base = np.array([P(s0, 0), P(s0, H)])
    ax.plot(base[:, 0], base[:, 1], color="tab:red", lw=0.8, ls=":", zorder=4)
    for ni in np.linspace(H / 6, H, 5):
        ui = (H * ni - ni ** 2 / 2) / (H ** 2 / 2) * 0.30
        ax.annotate("", xy=P(s0 + ui, ni), xytext=P(s0, ni),
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
    ut = P(s0 + 0.34, H * 0.30)
    ax.text(ut[0] + 0.02, ut[1] - 0.02, r"$u(n)$", color="tab:red", fontsize=10, va="top")
    ne = P(s0, H * 1.55)
    ax.annotate("", xy=ne, xytext=P(s0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.text(ne[0] - 0.005, ne[1] + 0.03, r"$n$", fontsize=10, ha="center")
    h0, h1 = P(0.14, 0), P(0.14, H)
    ax.annotate("", xy=h1, xytext=h0, arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
    hm = P(0.075, H / 2)
    ax.text(hm[0] - 0.01, hm[1] + 0.05, r"$h$", fontsize=11, ha="center")
    g0 = np.array([1.28, 0.14])
    ax.annotate("", xy=g0 + np.array([0.0, -0.20]), xytext=g0,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
    ax.text(g0[0] + 0.03, g0[1] - 0.10, r"$g$", fontsize=11)
    ax.annotate("free surface: nothing\ndrags from above", xy=P(0.80, H), xytext=(0.97, -0.16),
                fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    ax.annotate("wall: liquid sticks, $u=0$", xy=P(0.72, 0), xytext=(-0.42, -0.66), fontsize=8,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
    ax.text(0.82, -0.74, "flux $q$ = area under the profile\n"
            "$= \\rho g \\sin\\alpha\\, h^3 / 3\\eta$", fontsize=8, color="tab:red", ha="left",
            va="top")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.60, 1.62)
    ax.set_ylim(-1.04, 0.40)
    fig.tight_layout(pad=0.15)
    fig.savefig(FIGS / "film_sketch.pdf")
    plt.close(fig)
    print("film: sketch written")


def fig_hold(episode: str) -> None:
    tab = json.loads((REPO / "out" / "pour_wf" / episode / "hold_table.json").read_text())
    eta_hat = tab["eta_hat"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for key, run in tab["runs"].items():
        eta = float(key)
        ident = abs(eta - eta_hat) < 5e-3
        ax.plot(run["hold_s"], run["final_ml"], "o-" if ident else "s--",
                color="k" if ident else "0.55", lw=1.4 if ident else 1.0, ms=4,
                label=(rf"twin at $\hat\eta = {eta:.2f}\,\mathrm{{Pa\,s}}$" if ident
                       else rf"twin at the handbook ${eta:.2f}\,\mathrm{{Pa\,s}}$"))
    for t_str, h in tab["hold_for_target_s"][f"{eta_hat:.2f}"].items():
        t = float(t_str)
        ax.axhline(t, color="0.8", lw=0.6, ls=":")
        if h is not None:
            ax.plot([h, h], [0, t], color="tab:blue", lw=0.7, ls="--")
            ax.text(h + 0.05, t - 9, f"{h:.2f} s", fontsize=7, color="tab:blue")
    ax.axhline(97.2, color="tab:red", lw=0.8, ls="-.", label="real ep0001, no hold")
    ax.set_xlabel("hold at the roll's end pose  [s]")
    ax.set_ylabel("transferred  [mL]")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / f"{episode}_hold_table.pdf")
    plt.close(fig)
    print("hold: table figure written;",
          {k: (None if v is None else round(v, 2))
           for k, v in tab["hold_for_target_s"][f"{eta_hat:.2f}"].items()})


FIGURES = {"tilt": fig_tilt, "extraction": fig_extraction, "head": fig_head, "film": fig_film,
           "hold": fig_hold}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", choices=list(FIGURES), help="default: all four")
    ap.add_argument("--episode", default="ep0001")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    for name in args.which or list(FIGURES):
        fn = FIGURES[name]
        fn() if name == "film" else fn(args.episode)
