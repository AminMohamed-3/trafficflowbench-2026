"""F1 (pipeline and data access) and F14 (date-bootstrap distributions of the last state pieces) for the report (worker F2, 2026-10-21).

Called by make_figures.py (jobs F1, F14) with the make_figures module as `mf`, so style, save() and CAPTIONS are shared. Standalone:
    .venv_tfb/Scripts/python.exe final_solution/report/figures/make_f1_f14.py            # F1 + F14 only (CAPTIONS.md is
                                                                                        # written by a full make_figures.py run)
F14 reads out/F14_bootstrap_draws.csv, made by f14_draws.py (regenerated draws, parity with the stored instrument JSONs asserted);
if that file is missing it runs f14_draws.py first (about 2 min, CPU, < 1 GB).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ F1
# data-access classes (Okabe-Ito): what a block may read
ACC = {
    "train": ("train months", "#56B4E9", "k"),
    "month": ("scored month, released cells", "#E69F00", "k"),
    "online": ("stamped <= T, earlier dates", "#009E73", "w"),
    "mar4apr": ("March for April (declared)", "#CC79A7", "k"),
}
ACC_LONG = {
    "train": "train months (June 2030 to February 2031)",
    "month": "released cells of the scored month, all dates (offline Tasks 1, 3, 4 only)",
    "online": "data stamped <= T on the window's date, earlier dates of the month, train (online Task 2)",
    "mar4apr": "March released data or a March board read of a METHOD, used for April rows (declared)",
}
TAG = {"train": "train", "month": "month", "online": "≤T", "mar4apr": "Mar→Apr"}

# one entry per block: id, lane, x, y, w, h, title, body, access tags
BLOCKS = [
    # Task 1 state, main chain (row a)
    ("s0", "T1", 10.0, 62.5, 11.4, 11.0, "Stage 0 + S08",
     "pooled LightGBM\nresiduals, station\nlevels; S08 split-local\nLightGBM on pseudo-\ntargets", ["train", "month"]),
    ("phys", "T1", 22.9, 62.5, 11.4, 11.0, "Physical stages",
     "common-mode shock,\nclass hedge, log-flow\nkriging, projection\nto the published FD", ["month"]),
    ("s2", "T1", 35.8, 62.5, 11.4, 11.0, "Stage 2",
     "LightGBM residuals on\nspeed and flow, then\nthe FD projection", ["train", "month"]),
    ("b6", "T1", 48.7, 62.5, 11.4, 11.0, "B6 blend",
     "U-Net members\nblended with the\nstack per corridor\n(March fit also\nserves April)", ["month", "mar4apr"]),
    ("late", "T1", 61.6, 62.5, 11.4, 11.0, "Late stages",
     "polish; stage 3 (S7b\n+60 own-month\ntrees); occupancy;\nflow-only stage 4", ["train", "month"]),
    ("gap", "T1", 74.5, 62.5, 11.4, 11.0, "Gap rows + clip",
     "90-min blackouts\naround Task 2\nwindows; speed\n≤ 1.15 x free speed", ["month", "mar4apr"]),
    # Task 1 state, learned on the scored month (row b)
    ("lab", "T1", 10.0, 50.5, 24.3, 9.6, "Label sources of the scored month",
     "own view: released cells hidden to make labels;\nofficial view: every released cell (18.4% the own\nview never showed); INEL: released, ineligible,\nnever scored (labels only, never inputs)",
     ["month"]),
    ("mem", "T1", 35.8, 50.5, 24.3, 9.6, "B6 U-Net members, per month",
     "3 seeds x 10 corridors; own view 48k it,\nramp and occupancy channels; FV-FT +6k\nand FV2 +6k on the official view; FV-INEL\nW1.0 (INEL labels). April: same methods\non April's own cells",
     ["month", "mar4apr"]),
    ("s7c", "T1", 61.6, 50.5, 11.4, 9.6, "S7c-INEL",
     "+60 stage-3 trees\nfitted on INEL cells\n(date-parity gate)", ["month", "mar4apr"]),
    ("gapm", "T1", 74.5, 50.5, 11.4, 9.6, "Gap models",
     "specialist, gap U-Net,\nweights refit by date\ncross-fit, edge taper", ["month", "mar4apr"]),
    # Task 3 physics
    ("phy3", "T3", 10.0, 42.9, 75.9, 5.9, "Physics is scored on our Task 1 rows",
     "S_phys = S_FD/3 + 2 S_LWR/3 on the submitted speed and flow; no rows of its own, so every state piece is also a\n"
     "physics piece; pieces were ranked with .88 x (.35 dState + .45 dPhys)", []),
    # Task 2 onset
    ("q03", "T2o", 10.0, 32.0, 17.5, 9.2, "Q03 site classifier",
     "LightGBM on online precursors,\nfitted on train months", ["train", "online"]),
    ("q0436", "T2o", 29.4, 32.0, 17.5, 9.2, "Q0436 switches",
     "two calibrated online switches\nand an agreement rule on the\nwindow's own history", ["online"]),
    ("dec_o", "T2o", 48.8, 32.0, 17.5, 9.2, "Onset decode",
     "cell set with the highest expected\nIoU; top-two union if the\nprobability ratio is ≥ .8", ["online"]),
    # Task 2 ongoing
    ("mem_q", "T2g", 10.0, 17.5, 14.2, 12.4, "Ongoing members",
     "1-D CNN and sccnn\n(type A), 5 seeds each,\n16 epochs; ConvNeXt-\nTiny at .2 (ImageNet-\n1k init)", ["train"]),
    ("ada", "T2g", 25.9, 17.5, 14.2, 12.4, "Per-date adaptation",
     "fine-tune on pseudo\nwindows of earlier\ndates (5 ep, lr 6e-4);\n.75 adapted +\n.25 unadapted", ["online"]),
    ("cold", "T2g", 41.8, 17.5, 14.2, 12.4, "Cold dates",
     "≤ 2 earlier dates:\nXSD two-stage pool\nwith a speed term (A3);\nTiny member adapted\nper date (A1)", ["online", "mar4apr"]),
    ("dec_g", "T2g", 57.7, 17.5, 14.2, 12.4, "Ongoing decode",
     "expected-IoU top-k over\n6 steps; persistence\ncap .92; type A\nraised to ≥ .8", ["online"]),
    ("qacc", "T2g", 73.6, 17.5, 12.3, 12.4, "Q-ACCUM",
     "three per-date\nmember deltas\nsummed; 11 cells", ["online", "mar4apr"]),
    # Task 4 ODME
    ("odme", "T4", 10.0, 8.0, 50.0, 7.6, "ODME: the organiser's NNLS",
     "min ||A f - c||² + .05 ||f - b||², f ≥ 0, on the split's own link counts c and prior b;\n"
     "every move away from this projection lost on the board", ["month"]),
]
LANES = [("T1", "Task 1\nstate", 49.8, 74.6), ("T3", "Task 3\nphysics", 42.3, 49.3), ("T2o", "Task 2\nonset", 31.2, 41.8),
         ("T2g", "Task 2\nongoing", 16.7, 30.6), ("T4", "Task 4\nODME", 7.3, 16.3)]
ARROWS = [("s0", "phys"), ("phys", "s2"), ("s2", "b6"), ("b6", "late"), ("late", "gap"),
          ("lab", "mem"), ("q03", "q0436"), ("q0436", "dec_o"), ("mem_q", "ada"), ("ada", "cold"), ("cold", "dec_g"),
          ("dec_g", "qacc")]
UP = [("mem", "b6"), ("s7c", "late"), ("gapm", "gap")]   # row b feeds row a


def _check_access():
    """The diagram's own rule check: Task 2 blocks never read the month offline; Task 1/3/4 blocks never use the online class."""
    bad = []
    for bid, lane, *_, tags in BLOCKS:
        if lane.startswith("T2") and "month" in tags:
            bad.append(bid)
        if not lane.startswith("T2") and "online" in tags:
            bad.append(bid)
    assert not bad, bad
    return True


def fig_F1(mf):
    _check_access()
    info = json.loads((mf.ROOT / "artifacts/claude_ref/l_isinel/cand/c80/c_s7c_both_aprw10_c80.json").read_text())
    sha8 = info["full_sha256"][:8]
    fig = plt.figure(figsize=(7.2, 5.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 80); ax.axis("off")
    pos = {}
    edge, txt, faint = "#333333", "#111111", "#F4F4F4"
    for lane, lab, y0, y1 in LANES:
        ax.add_patch(FancyBboxPatch((.6, y0), 99.0, y1 - y0, boxstyle="round,pad=0,rounding_size=.6", fc=faint, ec="none", zorder=0))
        ax.text(1.6, (y0 + y1) / 2, lab, ha="left", va="center", fontsize=6.8, fontweight="bold", color=txt)
    for bid, lane, x, y, w, h, title, body, tags in BLOCKS:
        pos[bid] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.5", fc="white", ec=edge, lw=.6, zorder=2))
        ax.text(x + .6, y + h - .7, title, ha="left", va="top", fontsize=5.9, fontweight="bold", color=txt, zorder=3)
        wrap = bid == "phy3"
        ax.text(x + .6, y + h - 2.55, body, ha="left", va="top", fontsize=4.85, color=txt, zorder=3, linespacing=1.18,
                wrap=wrap)
        # access chips along the bottom edge, right-aligned
        cx = x + w - .5
        for t in reversed(tags):
            name, col, tc = ACC[t]
            cw = 1.15 + .78 * len(TAG[t])
            cx -= cw
            ax.add_patch(FancyBboxPatch((cx, y + .45), cw, 1.75, boxstyle="round,pad=0,rounding_size=.35", fc=col, ec="none", zorder=4))
            ax.text(cx + cw / 2, y + .45 + .9, TAG[t], ha="center", va="center", fontsize=4.6, color=tc, fontweight="bold", zorder=5)
            cx -= .4

    def arrow(p, q, style="-|>", ls="-", col=edge, lw=.7, rad=0.):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                                        mutation_scale=6, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    for a, b in ARROWS:
        xa, ya, wa, ha = pos[a]; xb, yb, wb, hb = pos[b]
        arrow((xa + wa, ya + ha / 2), (xb, yb + hb / 2))
    for a, b in UP:
        xa, ya, wa, ha = pos[a]; xb, yb, wb, hb = pos[b]
        xm = max(xa, xb) + min(wa, wb) / 2 if a != "mem" else xb + wb / 2
        arrow((xm, ya + ha), (xm, yb))
    # C80 box
    X0, W0 = 89.0, 10.0
    ax.add_patch(FancyBboxPatch((X0, 8.0), W0, 66.0, boxstyle="round,pad=0,rounding_size=.6", fc="#E8E8E8", ec=edge, lw=.9, zorder=2))
    xc = X0 + W0 / 2
    for yy, s, kw in ((61.5, "C80", dict(fontsize=8.0, fontweight="bold")), (58.8, "final build", {}), (55.8, "sha", {}),
                      (54.0, sha8, {}), (51.0, "public", {}), (49.2, ".90088", {}), (45.6, "March and", {}), (43.8, "April rows", {}),
                      (42.0, "built separately", {})):
        ax.text(xc, yy, s, ha="center", va="center", color=txt, zorder=3, **({"fontsize": 5.6} | kw))
    outs = [("gap", "state rows"), ("dec_o", "onset rows"), ("qacc", "ongoing rows"), ("odme", "ODME rows")]
    for bid, lab in outs:
        x, y, w, h = pos[bid]
        arrow((x + w, y + h / 2), (X0, y + h / 2), lw=.9)
        ax.text(X0 + .7, y + h / 2 + .5, lab, ha="left", va="bottom", fontsize=5.0, fontweight="bold", color="#333333", zorder=3)
    # physics reads the state rows (dashed)
    xg, yg, wg, hg = pos["gap"]; xp, yp, wp, hp = pos["phy3"]
    ax.annotate("", xy=(xp + wp - 1.0, yp + hp), xytext=(87.6, 68.0),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=.6, ls=(0, (2, 1.5)), mutation_scale=6,
                                connectionstyle="angle,angleA=-90,angleB=0,rad=0"), zorder=1)
    # legend
    hs = [Patch(fc=ACC[k][1], ec="none", label=f"{TAG[k]}: {ACC_LONG[k]}") for k in ACC]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(.006, .0), ncol=2, fontsize=4.9, frameon=False, handlelength=1.4,
              handleheight=.9, columnspacing=1.2, borderaxespad=.2)
    ax.text(1.0, 79.2, "Each scored month is built alone from its own released view; Tasks 1, 3 and 4 are offline, Task 2 is online. "
            "Chips under a block = the data it may read.", ha="left", va="top", fontsize=5.6, color=txt)
    counts = {k: sum(k in b[-1] for b in BLOCKS) for k in ACC}
    data = pd.DataFrame([dict(block=b[0], lane=b[1], title=b[6], body=b[7].replace("\n", " "),
                              **{f"reads_{k}": int(k in b[-1]) for k in ACC}) for b in BLOCKS])
    mf.save(fig, "F1_pipeline", data)
    n2 = sum(b[1].startswith("T2") for b in BLOCKS)
    m4a_t1 = sum("mar4apr" in b[-1] for b in BLOCKS if b[1] == "T1"); m4a_t2 = sum("mar4apr" in b[-1] for b in BLOCKS if b[1].startswith("T2"))
    mf.CAPTIONS["F1_pipeline"] = dict(
        title="The final pipeline and the data each block may read",
        claim=(f"C80 (sha {sha8}) is built separately for each scored month from {len(BLOCKS)} blocks. The {counts['month']} blocks "
               f"that read the month's released cells all sit in the offline Tasks 1, 3 and 4; the {n2} Task 2 blocks read only train "
               f"months and data stamped at or before T; {counts['mar4apr']} blocks ({m4a_t1} state, {m4a_t2} ongoing) take a March "
               f"fit or a March board read of a method into April rows by a declared rule."),
        number=(f"{len(BLOCKS)} blocks: train {counts['train']}, month (offline) {counts['month']}, online <= T {counts['online']}, "
                f"March for April {counts['mar4apr']}; Task 2 blocks with offline month access: 0 (asserted)"),
        intervals="none (diagram)",
        source="REPORT.md section 4; docs/report/ASSEMBLY.md section 1; docs/report/FINAL_BUILD_PLAN.md section 1; "
               "docs/report/ADOPTED.md rows 1 to 35; C80 sha from artifacts/claude_ref/l_isinel/cand/c80/c_s7c_both_aprw10_c80.json.",
        notes=("Access tags are a declaration drawn from those documents, checked by a rule in make_f1_f14.py (no Task 2 block has "
               "offline month access, no Task 1/3/4 block is tagged online); they are not a code audit (T5 holds the audits). "
               "Blocks are simplified: occupancy, the S lane and stage-4 details are merged into 'Late stages'."))


# ------------------------------------------------------------------------------------------------ F14
F14_ORDER = [("s2b->fv", "FV-FT", "I-S-INEL", "28-29"), ("fv->fv2", "FV2", "I-S-INEL", "30"),
             ("fv2->fold", "FV-INEL W1.0", "INEL-FOLD", "31+34"), ("S->B0", "gap edge taper", "gap bench", "29"),
             ("fv2->s7c_x", "S7c-INEL", "I-S-INEL", "35"), ("fold->s7c_fv2x", "S7c-INEL, fold chain", "S7C-FOLD", "35")]


def fig_F14(mf):
    f = mf.OUT / "F14_bootstrap_draws.csv"
    if not f.exists():
        subprocess.run([sys.executable, str(HERE / "f14_draws.py"), "all"], cwd=str(mf.ROOT), check=True)
    D = pd.read_csv(f)
    par = pd.read_csv(mf.OUT / "F14_bootstrap_parity.csv")
    assert (par.parity == "EXACT").all(), par[par.parity != "EXACT"]
    rows = []
    for pair, lab, inst, row in F14_ORDER:
        for sp, mon in (("validation", "March"), ("private", "April")):
            s = D[(D.pair == pair) & (D.split == sp)]
            x = s[s.draw >= 0].value.to_numpy(); pt = float(s[s.draw == -1].value.iloc[0])
            rows.append(dict(piece=lab, pair=pair, instrument=inst, adopted_row=row, month=mon, n_draws=len(x), point=pt,
                             median=float(np.median(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)),
                             p_gt0=float((x > 0).mean())))
    S = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=.32, hspace=.62))
    cols = {"March": mf.OK["blue"], "April": mf.OK["orange"]}
    for k, (pair, lab, inst, row) in enumerate(F14_ORDER):
        ax = axes.flat[k]
        for j, mon in enumerate(("March", "April")):
            sp = "validation" if mon == "March" else "private"
            x = D[(D.pair == pair) & (D.split == sp) & (D.draw >= 0)].value.to_numpy() * 1e5
            r = S[(S.pair == pair) & (S.month == mon)].iloc[0]
            yy = 1 - j
            vp = ax.violinplot([x], positions=[yy], vert=False, widths=.8, showextrema=False, points=200, bw_method="scott")
            for b in vp["bodies"]:
                b.set_facecolor(cols[mon]); b.set_edgecolor("none"); b.set_alpha(.45)
            ax.plot([r.lo95 * 1e5, r.hi95 * 1e5], [yy, yy], color=cols[mon], lw=1.1, solid_capstyle="butt")
            ax.plot([r.point * 1e5], [yy], marker="o", ms=2.8, color=cols[mon], mec="white", mew=.4, zorder=4)
            ax.text(1.0, yy + .3, f"P>0 {r.p_gt0:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=5.0, color=cols[mon])
        ax.axvline(0, color=mf.OK["grey"], lw=.6, zorder=0)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["March", "April"])
        ax.set_ylim(-.6, 1.75)
        ax.set_xlabel("S_total change (x 1e-5)", fontsize=5.8)
        ax.set_title(f"{'abcdef'[k]}  {lab}  (row {row}, {inst})", fontsize=6.2)
        lo, hi = ax.get_xlim(); lo = min(0., lo); ax.set_xlim(lo, hi + .22 * (hi - lo))
    fig.legend(handles=[Line2D([], [], color="k", marker="o", ms=2.8, lw=1.1, label="point (all dates once) and 95% interval"),
                        Patch(fc=mf.OK["grey"], alpha=.45, label="density of 500 joint date-bootstrap draws"),
                        Patch(fc=cols["March"], alpha=.6, label="March (public split)"),
                        Patch(fc=cols["April"], alpha=.6, label="April (private split)")],
               fontsize=5.4, loc="lower center", bbox_to_anchor=(.5, -.06), ncol=4, frameon=False)
    mf.save(fig, "F14_bootstrap", {"summary": S})   # draws and parity CSVs are written by f14_draws.py
    g = lambda pair, mon: S[(S.pair == pair) & (S.month == mon)].iloc[0]
    fv2m, fv2a = g("fv->fv2", "March"), g("fv->fv2", "April")
    s7m, s7a = g("fv2->s7c_x", "March"), g("fv2->s7c_x", "April")
    fim, fia = g("fv2->fold", "March"), g("fv2->fold", "April")
    pp = lambda p: f"{p:.2f}" if p >= .995 else f"{p:.2f}"[1:]
    kfc = json.loads(mf.P["isinel_gate"].read_text())["instrument"]["k_fc"]
    pos_all = [r for _, r in S.iterrows() if r.lo95 > 0]
    exc = S[S.lo95 <= 0]
    exc_txt = ", ".join(f"{r.piece} {r.month}" for _, r in exc.iterrows()) or "none"
    mf.CAPTIONS["F14_bootstrap"] = dict(
        title="Date-bootstrap distributions of the last state pieces on their own instruments",
        claim=(f"On their own instruments the last state pieces are above zero at 95% in {len(pos_all)} of {len(S)} piece-months; "
               f"the exceptions ({exc_txt}) straddle zero (FV2 March {mf.f5(fv2m.point, 6)}, P>0 {pp(fv2m.p_gt0)}; April "
               f"{mf.f5(fv2a.point, 6)}, P>0 {pp(fv2a.p_gt0)}). S7c-INEL is small but sure ({mf.f5(s7m.point, 6)} "
               f"[{mf.f5(s7m.lo95, 6)}, {mf.f5(s7m.hi95, 6)}] March, {mf.f5(s7a.point, 6)} April), and FV-INEL W1.0 on held-out "
               f"folds gives {mf.f5(fim.point, 6)} [{mf.f5(fim.lo95, 6)}, {mf.f5(fim.hi95, 6)}] (March) and "
               f"{mf.f5(fia.point, 6)} (April)."),
        number=(f"FV2 P>0 {pp(fv2m.p_gt0)} / {pp(fv2a.p_gt0)}; S7c-INEL P>0 {pp(s7m.p_gt0)} / {pp(s7a.p_gt0)}; "
                f"piece-months with 95% low > 0: {len(pos_all)} of {len(S)}"),
        intervals="95% interval = 2.5% and 97.5% quantiles of 500 joint date-bootstrap draws (seed 4701), all panels resampled "
                  "together by date; one draw set per instrument and month.",
        source="draws regenerated by figures/f14_draws.py from the stored chain arms (artifacts/claude_ref/l_isinel, l_inelfold/isi, "
               "l_isinel/s7cf/isi, l_gap3 + l_gap2/g1) with the instruments' own code; point and interval equal the stored JSONs "
               f"in {int((par.parity == 'EXACT').sum())} of {len(par)} parts (F14_bootstrap_parity.csv).",
        notes=("Units: I-S-INEL, INEL-FOLD and S7C-FOLD show fc = .88 (.35 dState + .45 dPhys); the gap bench shows .308 dState, its "
               f"board rule. Both are S_total forecast units, not board reads (I-S-INEL board/fc k = {kfc:.2f}, l_isinel/GATE.json). FV-FT on I-S-INEL "
               "is a retrospective read (the instrument came after its adoption). Missing: ADOPTED row 32 (GAP-ON-FV, board read "
               "only, no offline draws) and row 34 W1.0 alone (only inside the INEL-FOLD pair, which carries FV-INEL and W1.0 "
               "together). Pieces before row 28 are not drawn."))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import make_figures as mf
    fig_F1(mf); fig_F14(mf)
    print("MADE", mf.MADE)
