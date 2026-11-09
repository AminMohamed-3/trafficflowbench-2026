"""Figures and tables for the TrafficFlowBench final report (worker F1, 2026-10-21).

One script, deterministic, CPU only, reads only files inside the repository (no network, no Kaggle, no truth of scored
cells). Every figure is written as PNG (200 dpi) + PDF, together with the CSV of the numbers it draws, into
final_solution/report/figures/out/. Tables are written as CSV + Markdown. CAPTIONS.md (next to this script) is
regenerated on every run with the claim and the number each figure proves; the numbers in it are computed here.

Final file = C80 (public .90088, sha 3e27d66c).

Usage (repo root, laptop venv):
    .venv_tfb/Scripts/python.exe final_solution/report/figures/make_figures.py            # everything
    .venv_tfb/Scripts/python.exe final_solution/report/figures/make_figures.py --only F2 T2

Data rules (also printed into CAPTIONS.md):
- board_calibration.csv prose cells are cleaned by fixed rules: a row with more than 8 fields has an unquoted comma in its
  last (prose) cell and the extra fields are joined back into it; a numeric cell is the FIRST number in the cell; the band is
  the first two numbers of the band cell; sign and in_band are the FIRST word of the cell (lower case), the rest of the
  cell is kept as `note`. in_band is also recomputed from the numbers (band_lo <= public <= band_hi).
- Interval labels: "forecast band" = the band written in the manifest BEFORE the upload (instrument CI or judgmental);
  "95% CI (date bootstrap, 500 draws)" = the joint date bootstrap of the I-S-INEL / INEL-FOLD / SEED6 scorers.
- F6, F7, F8b use only released cells. Predictions are the INEL-FOLD chain (C77 class, FV-INEL W1.0 members trained on the
  other fold of INEL cells) at held-out INEL cells (released but ineligible, never scored). Scored target cells hold no
  truth in the cache (asserted) and are drawn as blank.
"""
from __future__ import annotations

import argparse
import os
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TFB_DEV_ROOT", HERE.parents[2]))  # the development repository (default: two levels above final_solution)
OUT = HERE / "out"
sys.path.insert(0, str(ROOT / "src"))

# ----------------------------------------------------------------------------------------------- style
OK = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7", grey="#999999", lgrey="#DDDDDD")
COL_W = 3.5      # single column, inches
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "lines.linewidth": 1.0,
    "xtick.major.width": .6, "ytick.major.width": .6, "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.hashsalt": "tfb", "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 100,
})
NUM = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
CAPTIONS: dict[str, dict] = {}
MADE: list[tuple[str, str]] = []
C77_FC = .90081  # C77 = C76 .90079 + W1.0 read on C75 (+.00002); unread (ADOPTED.md row 34)


def f5(x, d=5):
    """+.00041 style (no leading zero)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    s = f"{x:+.{d}f}"
    return s.replace("+0.", "+.").replace("-0.", "-.")


def p0(x, d=5):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("0.") else s


def nums(text):
    return [float(x) for x in re.findall(NUM, str(text))] if isinstance(text, str) else []


def first_num(text):
    v = nums(text)
    return v[0] if v else np.nan


def first_word(text):
    m = re.match(r"\s*([A-Za-z]+)", str(text)) if isinstance(text, str) else None
    return m.group(1).lower() if m else ""


def save(fig, name, data: pd.DataFrame | dict[str, pd.DataFrame]):
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / f"{name}.png", dpi=200, metadata={"Software": None}, **kw)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None}, **kw)
    plt.close(fig)
    if isinstance(data, dict):
        for k, df in data.items():
            df.to_csv(OUT / f"{name}_{k}.csv", index=False, lineterminator="\n")
    else:
        data.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
    MADE.append((name, "figure"))


def save_table(name, df: pd.DataFrame, title: str, notes: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
    lines = [f"# {title}", ""] + [f"- {n}" for n in notes] + [""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    MADE.append((name, "table"))


def md_tables(path: Path):
    """All pipe tables of a Markdown file: list of (header cells, rows of cells)."""
    out, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = [cells, []]
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            else:
                cur[1].append(cells)
        elif cur is not None:
            out.append(tuple(cur)); cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def find_table(path, first_header):
    for h, rows in md_tables(path):
        if h and h[0].lower().startswith(first_header.lower()):
            return h, rows
    raise KeyError(f"{first_header} not in {path}")


def row_by(rows, start):
    for r in rows:
        if r[0].lower().startswith(start.lower()):
            return r
    raise KeyError(start)


# ----------------------------------------------------------------------------------------------- sources
P = dict(
    adopted=ROOT / "docs/report/ADOPTED.md", calib=Path(__file__).resolve().parents[1] / "data/board_calibration.csv",
    manifest=ROOT / "artifacts/claude_ref/probes/manifest.csv", registry=ROOT / "docs/research/registry/REGISTRY.tsv",
    review=ROOT / "docs/report/REVIEW_2026-10-11.md", r4=ROOT / "docs/research/ideas_R4/R4.md",
    r5=ROOT / "docs/research/ideas_R5/R5.md", r6a=ROOT / "docs/research/ideas_R6/R6A.md",
    isinel_gate=ROOT / "artifacts/claude_ref/l_isinel/GATE.json", inelfold_gate=ROOT / "artifacts/claude_ref/l_inelfold/GATE.json",
    s7cf_log=ROOT / "exp/L_isinel/logs/s7cf_run.log", seed6=ROOT / "artifacts/claude_ref/l_seed6/SEED6_POWER.json",
    seed6_mem_v=ROOT / "artifacts/claude_ref/l_seed6/member_validation.json",
    seed6_mem_p=ROOT / "artifacts/claude_ref/l_seed6/member_private.json",
    fvi_mar_w03=ROOT / "exp/L_fvinel/logs/fvi_verify_inst.log", fvi_mar_w10=ROOT / "exp/L_fvinel/logs/w10/fvi_verify_inst.log",
    fvi_mar_lw=ROOT / "exp/L_fvinel/logs/lw/fvi_verify_inst.log", fvi_apr_w03=ROOT / "exp/L_fvinel/logs/apr/fvi_apr_verify.log",
    fvi_apr_w10=ROOT / "exp/L_fvinel/logs/apr_w10/fvi_aprw10_verify.log",
    inel_count_v=ROOT / "artifacts/claude_ref/l_fvinel/inel_count_validation.json",
    inel_count_p=ROOT / "artifacts/claude_ref/l_fvinel/inel_count_private.json",
    fold_chain=ROOT / "artifacts/claude_ref/l_inelfold/isi/chain", rules=ROOT / "RULES.md",
    digest=ROOT / "DECISIONS_DIGEST.md", final_plan=ROOT / "docs/report/FINAL_PLAN_2026-10-21.md",
)
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S",
          "D12_I405_N", "D12_I405_S"]

KIND_RULES = [  # ordered; re.fullmatch on the calibration file name. Every row must match exactly one rule.
    ("knockout or isolation (diagnostic)", r"0133_.*|0134_.*|c21_.*|c22_.*|c_b1ko_.*|c_bundle18_s2b"),
    ("state: target-month labels", r"c_fvft_.*|c_fv2_.*|c_fvinel_.*|c_isinel_.*"),
    ("state and physics (target split)", r"c0[1789]_.*|c1[0-5]_.*|c17_.*|c20_.*|c26_.*|c30_.*|c34_.*|c36_s08.*|c37_.*"
                                         r"|c_b6.*|c_gap.*|c_w2gncfdq.*|c_lmicro2.*|c_bundle18_s2b_s7b"),
    ("ongoing: per-date adaptation", r"c03_.*|c29_.*|c35_.*|c36_coldxsd.*|c_xsd5s.*|c_poolhi.*|c_llabp1.*|c_a2spd.*"
                                     r"|c_a3xsd.*|c_tinyad.*|c_n47Afix.*|c_n2swa.*|c_coldaux2.*|c_warmsd.*|c_qaccum.*"
                                     r"|c_onltrust.*|c_f33_.*"),
    ("ongoing: decode or selection-conditioned input", r"c19_.*|c23_.*|c24_.*|b0708_.*"),
    ("ongoing: members trained on train months", r"c02_.*|c25_.*|c27_.*|c31_.*|c32_.*|c33_.*|c_w35.*|c_tinye24.*"
                                                 r"|c_tinyvm.*|c_linfF.*"),
    ("onset", r"0145_.*|c05_.*|c16_.*|c18_.*|c_onsclk.*"),
    ("early whole-file and ODME probes", r"0120_.*|0146_.*|0159_.*|0164_.*|0193_.*|0204_.*|0214_.*|0229_.*|0309_.*"),
]
KIND_COL = {
    "state and physics (target split)": OK["blue"], "state: target-month labels": OK["sky"],
    "ongoing: per-date adaptation": OK["green"], "ongoing: decode or selection-conditioned input": OK["orange"],
    "ongoing: members trained on train months": OK["verm"], "onset": OK["purple"],
    "early whole-file and ODME probes": OK["grey"], "knockout or isolation (diagnostic)": OK["black"],
}


def kind_of(file):
    hits = [k for k, rx in KIND_RULES if re.fullmatch(rx, file)]
    assert len(hits) >= 1, f"calibration row without a kind rule: {file}"
    return hits[0]


def read_calibration_raw():
    """Rule: the file has 8 columns; a row with more fields has an unquoted comma in its last (prose) cell, so the
    extra fields are joined back into that last cell with ', '."""
    import csv
    with open(P["calib"], encoding="utf-8", newline="") as fh:
        rd = list(csv.reader(fh))
    head, body = rd[0], rd[1:]
    n = len(head)
    fixed = [r[:n - 1] + [", ".join(r[n - 1:])] if len(r) > n else r + [""] * (n - len(r)) for r in body if r]
    return pd.DataFrame(fixed, columns=head)


def load_calibration():
    raw = read_calibration_raw()
    n_raw = len(raw)
    rows = []
    for _, r in raw.iterrows():
        band = nums(r.band)
        rows.append(dict(
            file=r.file, kind=kind_of(r.file), forecast=first_num(r.forecast),
            band_lo=band[0] if len(band) >= 2 else np.nan, band_hi=band[1] if len(band) >= 2 else np.nan,
            public=first_num(r.public), d_pred=first_num(r.d_pred), d_board=first_num(r.d_board),
            sign_recorded=first_word(r.sign), in_band_recorded=first_word(r.in_band),
            note=" ".join(x for x in [r.d_pred if len(nums(r.d_pred)) and not re.fullmatch(NUM, r.d_pred.strip()) else "",
                                      r.sign if first_word(r.sign) != r.sign.strip().lower() else "",
                                      r.in_band if first_word(r.in_band) != r.in_band.strip().lower() else ""] if x)))
    df = pd.DataFrame(rows)
    df["in_band_calc"] = (df.public >= df.band_lo - 1e-12) & (df.public <= df.band_hi + 1e-12)
    tol = 1e-5
    s_p = np.sign(np.where(df.d_pred.abs() < tol, 0, df.d_pred)); s_b = np.sign(np.where(df.d_board.abs() < tol, 0, df.d_board))
    df["sign_calc"] = np.where(df.d_pred.isna() | df.d_board.isna(), "", np.where(s_p == s_b, "agree", "disagree"))
    return df, n_raw


def load_manifest():
    m = pd.read_csv(P["manifest"], dtype=str, keep_default_na=False)
    m["stem"] = m.file.str.replace(r"\.csv$", "", regex=True)
    m["public_num"] = pd.to_numeric(m.public, errors="coerce")
    m["read_day"] = m.read_utc.str.slice(0, 10)
    return m


def adopted_rows():
    rows = {}
    for line in P["adopted"].read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[int(m.group(1))] = dict(piece=cells[1], evidence=cells[2], board=cells[3], text=line)
    return rows


# ----------------------------------------------------------------------------------------------- the adopted chain
# (label, read file, ADOPTED rows, component, mode). mode: read = public read of this file is the new level;
# private = private rows only (public unchanged by construction); sibling = level is the forecast, the delta is a
# read of the same rows on a sibling parent.
CHAIN = [
    ("C10 state v/q models + phys3", "c10_state_spd_hedge_flow_phys3_on_c03", [1], "state", "read"),
    ("C11 stage-2 residual stack", "c11_state_stack_s2stage_on_c03", [1], "state", "read"),
    ("C12 B6 U-Net imputer blend", "c12_state_c11_b6blp_on_c03", [1], "state", "read"),
    ("C13 polish chain", "c13_state_c12_polish_on_c03", [1], "state", "read"),
    ("C14 LSA + big stage 3", "c14_state_c13_lsa_s3big", [1], "state", "read"),
    ("C15 S08 input, phys4, gap rows v3", "c15_k6_s08_phys4_gap_selcond_on_c03", [1], "state", "read"),
    ("C16 onset Q03 + Q0436", "c16_k6_q0436_onset_on_c15", [2], "onset", "read"),
    ("C17 occupancy + demand factor", "c17_k6_s0447_state_on_c15", [1], "state", "read"),
    ("C19 ongoing ong2 (sccnn A, band)", "c19_ong2_sccnnA_on_c17", [4], "ongoing", "read"),
    ("C26 phys5 stage 4 (build A)", "c26_final_A_build", [1], "state", "read"),
    ("C28 ConvNeXt-Tiny member", "c27_bb_tinye8_drop20_on_c26", [6], "ongoing", "read"),
    ("C29 adapted/unadapted mix .75", "c29_noad75_ong_on_c28", [7], "ongoing", "read"),
    ("C30 stage 2 l63 x r600", "c30_s2l63r600_on_c29", [8], "state", "read"),
    ("C31 e16 ongoing members", "c31_e16s01_on_c30", [9], "ongoing", "read"),
    ("C33 5 seeds per member", "c33_e16s5_on_c31", [10], "ongoing", "read"),
    ("C34 stage 2 on 3M rows", "c34_s2n3r600_on_c33", [11], "state", "read"),
    ("C35 adaptation recipe mh5", "c35_mh5_on_c34", [12], "ongoing", "read"),
    ("C36 cold-start pools XSD", "c36_coldxsd_on_c35", [13], "ongoing", "read"),
    ("C43 S08 refit l127 x r4800", "c36_s08l127r4800_on_c36", [15], "state", "read"),
    ("C44 B6 imputer 12k it", "c_b6l12k_on_c43", [16], "state", "read"),
    ("C47 B6 imputer 24k it", "c_b6l24k_on_c44", [16], "state", "read"),
    ("C48 gap2 ramp-cond. gap U-Net", "c_gap2_on_c47", [19], "state", "read"),
    ("C54 L_micro knobs + B6 blend", "c_lmicro2_on_c50", [20], "state", "read"),
    ("C55 Tiny member, saved weights", "c_tinyvm_on_c54", [21], "ongoing", "read"),
    ("C57 A3 speed term, cold fits", "c_a3xsd_on_c55", [22], "ongoing", "read"),
    ("C59 Tiny per-date adaptation", "c_tinyad_on_c57", [23], "ongoing", "read"),
    ("C64 B6 S2 ramp + occupancy", "c_b6s2_on_c59", [24], "state", "read"),
    ("C65 GAPW gap rows", "c_gapw_on_c64", [25], "state", "read"),
    ("C66 BUNDLE18 S2b + S7b", "c_bundle18_s2b_s7b", [26], "state", "read"),
    ("C69 L_gap2 gap rows", "c_gap_on_c66", [27], "state", "read"),
    ("C70 FV-FT March labels", "c_fvft_mar_on_c69", [28], "state", "read"),
    ("C71 gap3 + April FV-FT", "c_gap3_aprfv_on_c70", [29], "state", "read"),
    ("C72 FV2 dose +6k", "c_fv2_on_b19", [30], "state", "read"),
    ("C73 FV-INEL W.3 labels", "c_fvinel_on_b19", [31], "state", "read"),
    ("C74P FV-INEL April rows", None, [31], "state", "private"),
    ("C75 gap rows on FV chain", "c_gapfv_on_c74p", [32], "state", "read"),
    ("C76 Q-ACCUM ongoing", "c_qaccum_on_c75", [33], "ongoing", "read"),
    ("C77 FV-INEL W1.0 (unread)", "c_fvinel_w10_on_c75", [34], "state", "sibling"),
    ("C79 S7c-INEL stage learner", "c_isinel_s7c_mar_on_c77", [35], "state", "read"),
    ("C80 April W1.0 + S7c April", None, [34, 35], "state", "private"),
]
COMP_COL = {"state": OK["blue"], "onset": OK["purple"], "ongoing": OK["green"]}


def build_chain():
    cal, _ = load_calibration()
    cal = cal.set_index("file")
    man = load_manifest().set_index("stem")
    level = float(man.loc["c03_queue_cnn_adapt_ongoing", "public_num"])
    start = level
    rows = []
    for label, f, arows, comp, mode in CHAIN:
        prev = level
        fc = lo = hi = np.nan
        if f is not None:
            c = cal.loc[f]
            fc, lo, hi = c.forecast - prev, c.band_lo - prev, c.band_hi - prev
        if mode == "read":
            new = float(man.loc[f, "public_num"])
            delta = new - prev
            how = f"public read of {f}"
        elif mode == "private":
            new, delta, how = prev, 0.0, "private rows only; public rows unchanged by construction"
        else:  # sibling: same rows read on the sibling parent C75; C77 level = forecast
            c = cal.loc[f]
            parent_pub = c.public - c.d_board
            delta = c.public - parent_pub
            fc, lo, hi = c.forecast - parent_pub, c.band_lo - parent_pub, c.band_hi - parent_pub
            new = prev + delta
            how = f"not read on its parent; delta = read of {f} on C75"
        rows.append(dict(step=label, file=f or "", adopted_rows=";".join(map(str, arows)), component=comp, mode=mode,
                         level_before=round(prev, 6), level_after=round(new, 6), board_delta=round(delta, 6),
                         forecast_delta=round(fc, 6) if np.isfinite(fc) else np.nan,
                         band_lo_delta=round(lo, 6) if np.isfinite(lo) else np.nan,
                         band_hi_delta=round(hi, 6) if np.isfinite(hi) else np.nan, how=how))
        level = new
    df = pd.DataFrame(rows)
    return df, start


# ----------------------------------------------------------------------------------------------- F2
def fig_F2():
    df, start = build_chain()
    final = df.level_after.iloc[-1]
    assert abs(final - .90088) < 1e-9, final
    split_at = df.index[df.step.str.startswith("C47")][0] + 1
    parts = [df.iloc[:split_at], df.iloc[split_at:]]
    heights = [len(parts[0]), len(parts[1])]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 7.6), gridspec_kw=dict(height_ratios=heights, hspace=.16))
    for k, (ax, part) in enumerate(zip(axes, parts)):
        y = np.arange(len(part))[::-1]
        for yi, (_, r) in zip(y, part.iterrows()):
            col = COMP_COL[r.component]
            lo, hi = sorted([r.level_before, r.level_after])
            if r["mode"] == "private":
                ax.plot([r.level_before], [yi], marker="|", ms=7, color=col, mew=1.2)
            else:
                ax.barh(yi, hi - lo if hi > lo else 0, left=lo, height=.62, color=col,
                        hatch="////" if r["mode"] == "sibling" else None, edgecolor="white" if r["mode"] != "sibling" else col,
                        linewidth=.3, alpha=.55 if r["mode"] == "sibling" else 1)
            if np.isfinite(r.forecast_delta):
                ax.plot([r.level_before + r.band_lo_delta, r.level_before + r.band_hi_delta], [yi - .36, yi - .36],
                        color="k", lw=.6)
                ax.plot([r.level_before + r.forecast_delta], [yi - .36], marker="o", ms=2.2, color="k")
            txt = "private only" if r["mode"] == "private" else f5(r.board_delta)
            ax.text(1.0, yi, txt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.6,
                    color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(part.step, fontsize=5.6)
        ax.tick_params(axis="y", length=0)
        xs = np.r_[part.level_before, part.level_after, (part.level_before + part.band_lo_delta).dropna(),
                   (part.level_before + part.band_hi_delta).dropna()]
        lo_x, hi_x = np.nanmin(xs), np.nanmax(xs)
        pad = (hi_x - lo_x) * .02
        ax.set_xlim(lo_x - pad, hi_x + (hi_x - lo_x) * .27)
        ax.set_ylim(-.9, len(part) - .4)
        ax.grid(axis="x", color=OK["lgrey"], lw=.4)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: p0(v, 4)))
        ax.set_title(["a  C03 .88225 to C47 .89776", "b  C47 to C80 .90088 (zoomed)"][k], fontsize=7)
    axes[1].set_xlabel("public score (March)")
    handles = [Patch(color=COMP_COL[c], label=f"Task {'1 state' if c == 'state' else '2 ' + c}") for c in COMP_COL]
    handles += [Patch(facecolor="white", edgecolor=OK["blue"], hatch="////", label="unread level (forecast)"),
                Line2D([], [], color="k", lw=.6, marker="o", ms=2.2, label="forecast and band written before upload")]
    fig.subplots_adjust(left=.43, right=.985, top=.915, bottom=.055)
    fig.legend(handles=handles, loc="upper center", fontsize=5.4, handlelength=1.4, ncol=2, bbox_to_anchor=(.55, .998),
               columnspacing=.8)
    total = final - start
    by_comp = df.groupby("component").board_delta.sum()
    save(fig, "F2_waterfall", df)
    top = df.sort_values("board_delta", ascending=False).head(3)
    CAPTIONS["F2_waterfall"] = dict(
        title="Where the score came from: adopted pieces C03 to C80",
        claim=(f"The public score rose {f5(total)} from C03 ({p0(start)}) to C80 ({p0(final)}) in {len(df)} adopted steps; "
               f"Task 1 state pieces gave {f5(by_comp.get('state', 0))}, Task 2 ongoing {f5(by_comp.get('ongoing', 0))}, "
               "onset 0 on the public board (its Q0436 cells change private windows). The three largest steps: "
               + "; ".join(f"{r.step} {f5(r.board_delta)}" for _, r in top.iterrows()) + "."),
        number=f"total {f5(total)}; state {f5(by_comp.get('state', 0))}; ongoing {f5(by_comp.get('ongoing', 0))}",
        intervals="Black segment = forecast band written in the manifest before the upload (instrument CI or judgmental), "
                  "dot = forecast. C77 is unread: its bar is the read of the same March rows on the sibling parent C75.",
        source="docs/report/ADOPTED.md (rows), artifacts/claude_ref/probes/manifest.csv (public reads), "
               "docs/report/board_calibration.csv (forecasts).",
        notes="Private-only steps (C74P, C80) change April rows only and are drawn as ticks.")


# ----------------------------------------------------------------------------------------------- F3
def fig_F3():
    cal, n_raw = load_calibration()
    d = cal.dropna(subset=["d_pred", "d_board"]).copy()
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    lt = 1e-4
    for kind, col in KIND_COL.items():
        s = d[d.kind == kind]
        if len(s):
            ax.scatter(s.d_pred, s.d_board, s=11, color=col, edgecolor="white", linewidth=.3, label=f"{kind} ({len(s)})",
                       zorder=3)
    lim = [-.5, .05]
    ax.plot(lim, lim, color="k", lw=.6, ls="--", zorder=1)
    ax.axhline(0, color=OK["lgrey"], lw=.5, zorder=0); ax.axvline(0, color=OK["lgrey"], lw=.5, zorder=0)
    ax.set_xscale("symlog", linthresh=lt); ax.set_yscale("symlog", linthresh=lt)
    ticks = [-.1, -.01, -.001, -.0001, 0, .0001, .001, .01]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    labs = ["-.1", "-.01", "-.001", "-.0001", "0", ".0001", ".001", ".01"]
    ax.set_xticklabels(labs, rotation=45); ax.set_yticklabels(labs)
    ax.set_xlim(-.5, .05); ax.set_ylim(-.5, .05)
    ax.set_xlabel("forecast change (written before upload)")
    ax.set_ylabel("board change (public read)")
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.2), fontsize=5.3, handletextpad=.2, ncol=2, columnspacing=.6)
    st = d[d.kind.isin(["state and physics (target split)"])]
    queue = d[d.kind.str.startswith("ongoing") | (d.kind == "onset")]
    agree = (d.sign_calc == "agree").mean()
    rho = d[["d_pred", "d_board"]].rank().corr().iloc[0, 1]
    st_inb = cal[cal.kind == "state and physics (target split)"].in_band_calc.mean()
    st_miss = (st.d_board - st.d_pred).abs().median()
    q_agree = (queue.sign_calc == "agree").mean()
    save(fig, "F3_forecast_vs_board", cal)
    CAPTIONS["F3_forecast_vs_board"] = dict(
        title="Offline forecasts predict the board for state pieces, not for queue pieces",
        claim=(f"Across {len(d)} reads with a written forecast, forecast and board change agree in sign "
               f"{agree:.0%} of the time (Spearman {rho:.2f}). Task 1 state/physics pieces (n={len(st)}) land in their band "
               f"{st_inb:.0%} of the time with a median miss of {p0(st_miss, 5)}; Task 2 queue pieces (n={len(queue)}) agree "
               f"in sign only {q_agree:.0%} of the time."),
        number=f"state in band {st_inb:.0%}, median |miss| {p0(st_miss, 5)}; queue sign agreement {q_agree:.0%}",
        intervals="No intervals drawn; each point is one public read. Axes are symmetric-log (linear inside +-.0001).",
        source="docs/report/board_calibration.csv (cleaned by the rules in the Data rules section).",
        notes=(f"{n_raw} rows in the file; "
               f"{len(cal) - len(d)} row(s) without a forecast change are kept in the CSV only. Kind of change is set by "
               "explicit file-name rules (KIND_RULES in the script)."))


# ----------------------------------------------------------------------------------------------- F4
REVIEW_C = {"C17": "c17_k6_s0447_state_on_c15", "C20": "c20_phys5_stepF_on_c17", "C26": "c26_final_A_build",
            "C30": "c30_s2l63r600_on_c29", "C34": "c34_s2n3r600_on_c33", "C38": "c36_s08l63r2400_on_c36",
            "C03": "c03_queue_cnn_adapt_ongoing", "C35": "c35_mh5_on_c34", "C19": "c19_ong2_sccnnA_on_c17",
            "C36": "c36_coldxsd_on_c35", "C27": "c27_bb_tinye8_drop20_on_c26", "C33": "c33_e16s5_on_c31",
            "C39": "c_w35_on_c38", "C40": "c_tinye24_on_c38", "C16": "c16_k6_q0436_onset_on_c15",
            "C18": "c18_onset4_p1_on_c12"}
REVIEW_GROUP_OF_KIND = {"state and physics (target split)": 0, "ongoing: per-date adaptation": 1,
                        "ongoing: decode or selection-conditioned input": 2, "ongoing: members trained on train months": 4,
                        "onset": 5, "state: target-month labels": 6}
COLD = {"c36_coldxsd_on_c35", "c_xsd5s_on_c43", "c_a3xsd_on_c55", "c_coldaux2_on_c59"}


def fig_F4():
    h, rows = find_table(P["review"], "Kind of change")
    cal, _ = load_calibration()
    cal = cal.set_index("file")
    man = load_manifest().set_index("stem")
    groups = [r[0] for r in rows] + ["Target-month labels (after the review)"]
    stated = [r[3] for r in rows] + ["(not in the review)"]
    recs = []
    for gi, r in enumerate(rows):
        for c in re.findall(r"C\d\d", r[4]):
            f = REVIEW_C[c]
            x = cal.loc[f]
            recs.append(dict(group=groups[gi], group_i=gi, cnum=c, file=f, d_pred=x.d_pred, d_board=x.d_board,
                             named_in_review=True))
    named = {x["file"] for x in recs}
    for f, x in cal.iterrows():
        if f in named or x.kind not in REVIEW_GROUP_OF_KIND:
            continue
        ru = man.read_utc.get(f, "") if f in man.index else ""
        if not ru or ru[:16] <= "2026-10-11T07:00":
            continue
        gi = 3 if f in COLD else REVIEW_GROUP_OF_KIND[x.kind]
        recs.append(dict(group=groups[gi], group_i=gi, cnum="", file=f, d_pred=x.d_pred, d_board=x.d_board,
                         named_in_review=False))
    d = pd.DataFrame(recs)
    d["ratio"] = np.where(d.d_pred.abs() >= 1e-5, d.d_board / d.d_pred.where(d.d_pred.abs() >= 1e-5), np.nan)
    d["excluded_reason"] = np.where(d.ratio.isna(), "forecast change below .00001 (ratio undefined)", "")
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    rng = np.random.default_rng(11)
    clip = 30
    for gi in range(len(groups)):
        s = d[(d.group_i == gi) & d.ratio.notna()]
        xj = gi + rng.uniform(-.18, .18, len(s))
        yv = s.ratio.clip(-clip, clip)
        filled = s.named_in_review.to_numpy()
        ax.scatter(xj[filled], yv[filled], s=14, color=OK["blue"], zorder=3, edgecolor="white", lw=.3)
        ax.scatter(xj[~filled], yv[~filled], s=12, facecolor="none", edgecolor=OK["verm"], lw=.7, zorder=3)
        for xx, (_, rr) in zip(xj[filled], s[filled].iterrows()):
            if gi == 0:
                continue
            ax.annotate(rr.cnum, (xx, np.clip(rr.ratio, -clip, clip)), fontsize=4.8, xytext=(2, 1),
                        textcoords="offset points", color="#333333")
        if len(s):
            ax.plot([gi - .3, gi + .3], [np.median(s.ratio)] * 2, color="k", lw=1.0)
    ax.axhline(1, color=OK["green"], lw=.7, ls="--"); ax.axhline(0, color=OK["grey"], lw=.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([-30, -10, -3, -1, 0, 1, 3, 10, 30]); ax.set_yticklabels(["<=-30", "-10", "-3", "-1", "0", "1", "3", "10", ">=30"])
    short = ["Task 1\nstate,\nphysics", "ongoing\nadapt-\nation", "select.-\ncond.\ninput", "cold-\nstart\npools",
             "ongoing\nmembers\n(train\nmonths)", "onset\nsite\nmodels", "target-\nmonth\nlabels"]
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(short, fontsize=5.2)
    ax.set_ylabel("board change / forecast change")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=OK["blue"], ms=3.5, label="named in the 10-11 review"),
                       Line2D([], [], ls="", marker="o", mfc="none", mec=OK["verm"], ms=3.5, label="later read, same kind"),
                       Line2D([], [], color="k", label="median (all points)"),
                       Line2D([], [], color=OK["green"], ls="--", label="full transfer (1x)")],
              loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=5.2, ncol=2)
    ok_ = d[d.ratio.notna()]
    med = ok_.groupby("group_i").ratio.median()
    med_named = ok_[ok_.named_in_review].groupby("group_i").ratio.median()
    med_later = ok_[~ok_.named_in_review].groupby("group_i").ratio.median()
    n_g = ok_.groupby("group_i").size()
    save(fig, "F4_transfer_rule", d.assign(review_stated_ratio=d.group_i.map(dict(enumerate(stated)))))
    CAPTIONS["F4_transfer_rule"] = dict(
        title="Task 1 state gains learned on the target month carry at about 1x; queue forecasts do not",
        claim=(f"Task 1 state and physics pieces read at a median {med.get(0, np.nan):.2f}x their forecast (n={n_g.get(0, 0)}). "
               f"Ongoing members trained on train months read at {med.get(4, np.nan):.2f}x (n={n_g.get(4, 0)}; two of the "
               f"named member reads are negative). Ongoing adaptation looked like 1x to 4x in the 10-11 review (median "
               f"{med_named.get(1, np.nan):.1f}x of the named reads), but the later adaptation reads have a median of "
               f"{med_later.get(1, np.nan):.1f}x: queue forecasts are not reliable in either direction. Target-month label "
               f"pieces read above their forecast (median {med.get(6, np.nan):.1f}x) because no offline test could see "
               "their full value."),
        number=f"state {med.get(0, np.nan):.2f}x vs members {med.get(4, np.nan):.2f}x; later adaptation {med_later.get(1, np.nan):.1f}x",
        intervals="No intervals; each point is one board read divided by its pre-upload forecast change. Ratios outside "
                  "+-30 are drawn at the edge; reads with a forecast change below .00001 are left out (listed in the CSV).",
        source="docs/report/REVIEW_2026-10-11.md section 2 table (groups and named examples); docs/report/board_calibration.csv "
               "(changes); artifacts/claude_ref/probes/manifest.csv (read dates for the later reads).",
        notes=("Named state examples: C17, C20, C26, C30, C34, C38. The cold-start group is C36 plus later cold-date XSD "
               "reads. C18 (forecast 0, board -.0063) has no ratio. This record led to the 10-17 ruling to focus on state "
               "pieces (DECISIONS_DIGEST.md)."))


# ----------------------------------------------------------------------------------------------- F5
CLASS_ORDER = ["ADOPTED", "passed offline, not shipped", "UNDETECTED", "REFUTED", "BAD-INSTRUMENT", "REASONING-ONLY"]
CLASS_COL = {"ADOPTED": OK["green"], "passed offline, not shipped": OK["yellow"], "UNDETECTED": OK["sky"],
             "REFUTED": OK["verm"], "BAD-INSTRUMENT": OK["grey"], "REASONING-ONLY": OK["lgrey"]}
TASK_ORDER = ["state", "physics", "onset", "ongoing", "queue", "odme", "cross", "instrument"]


def load_registry():
    r = pd.read_csv(P["registry"], sep="\t", dtype=str, keep_default_na=False)
    r["vclass"] = r.verdict_class.replace({"PASS-OFFLINE": "passed offline, not shipped",
                                           "ADOPTED-candidate": "passed offline, not shipped"})
    r["tm"] = r.learns_from_target_month.str.extract(r"^(yes|partly|no)")[0]
    return r


def fig_F5():
    r = load_registry()
    assert set(r.vclass) <= set(CLASS_ORDER), set(r.vclass)
    ct = pd.crosstab(r.task, r.vclass).reindex(index=TASK_ORDER, columns=CLASS_ORDER, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.6), gridspec_kw=dict(width_ratios=[2.1, 1], wspace=.45))
    ax = axes[0]
    left = np.zeros(len(ct))
    y = np.arange(len(ct))[::-1]
    for c in CLASS_ORDER:
        ax.barh(y, ct[c], left=left, color=CLASS_COL[c], edgecolor="white", lw=.3, height=.7, label=c)
        left += ct[c].to_numpy()
    for yi, tot in zip(y, left):
        ax.text(tot + 1, yi, str(int(tot)), va="center", fontsize=5.6)
    ax.set_yticks(y); ax.set_yticklabels(["state", "physics", "onset", "ongoing", "queue (both)", "ODME", "cross-task",
                                          "instrument"])
    ax.set_xlabel("ideas tested (registry rows)")
    ax.set_title(f"a  {len(r)} ideas by task and verdict")
    ax.legend(fontsize=5.0, loc="lower right", handlelength=1)
    ax2 = axes[1]
    tm = pd.crosstab(r.tm, r.vclass).reindex(index=["yes", "partly", "no"], columns=CLASS_ORDER, fill_value=0)
    share = tm.div(tm.sum(axis=1), axis=0)
    bottom = np.zeros(3)
    for c in CLASS_ORDER:
        ax2.bar(range(3), share[c], bottom=bottom, color=CLASS_COL[c], edgecolor="white", lw=.3, width=.7)
        bottom += share[c].to_numpy()
    ax2.set_xticks(range(3)); ax2.set_xticklabels([f"yes\n{tm.loc['yes'].sum()}", f"partly\n{tm.loc['partly'].sum()}",
                                                   f"no\n{tm.loc['no'].sum()}"], fontsize=5.6)
    ax2.set_ylabel("share of ideas"); ax2.set_ylim(0, 1)
    ax2.set_title("b  learns from the\ntarget month?", fontsize=6.2)
    adopt = share["ADOPTED"]
    data = ct.reset_index().rename(columns={"index": "task"})
    data2 = tm.reset_index().rename(columns={"tm": "learns_from_target_month"})
    save(fig, "F5_registry", {"by_task": data, "by_target_month": data2})
    counts = r.vclass.value_counts()
    CAPTIONS["F5_registry"] = dict(
        title="Breadth and honesty of the search: every idea has a verdict class",
        claim=(f"{len(r)} ideas were tested and classified: ADOPTED {counts.get('ADOPTED', 0)}, REFUTED "
               f"{counts.get('REFUTED', 0)}, UNDETECTED {counts.get('UNDETECTED', 0)}, BAD-INSTRUMENT "
               f"{counts.get('BAD-INSTRUMENT', 0)}, REASONING-ONLY {counts.get('REASONING-ONLY', 0)}, passed offline but "
               f"not shipped {counts.get('passed offline, not shipped', 0)}. Over a third of the ideas were refuted with a "
               f"stated test power; the adoption rate is about the same whether an idea learns from the target month "
               f"({adopt['yes']:.0%}) or not ({adopt['no']:.0%}), because train-month ideas were screened by board reads "
               "early (F4) and stopped."),
        number=f"adoption rate yes {adopt['yes']:.0%} / partly {adopt['partly']:.0%} / no {adopt['no']:.0%}",
        intervals="Counts, no intervals.",
        source="docs/research/registry/REGISTRY.tsv (columns task, verdict_class, learns_from_target_month).",
        notes="PASS-OFFLINE (R241) and ADOPTED-candidate (R248) are shown as 'passed offline, not shipped'. "
              "learns_from_target_month variants such as 'yes (March fit)' are mapped to their first word.")


# ----------------------------------------------------------------------------------------------- F6 / F7 / F8b data
def inel_view(split="validation"):
    """Per panel: cache, FD parameters, fold-chain predictions at INEL cells. Released cells only."""
    from tfb.data import load_cache
    from tfb import metric
    out = {}
    for pn in PANELS:
        c = load_cache(pn, split)
        a = c.arrays
        sp, fl = np.asarray(a["speed"], float), np.asarray(a["flow"], float)
        tg, el, gp = np.asarray(a["target"], bool), np.asarray(a["elig"], bool), np.asarray(a["gap"], bool)
        assert int((np.isfinite(sp) & tg).sum()) == 0, "cache must not hold values at scored target cells"
        hs = np.asarray(a["hist"], bool)
        fin = np.isfinite(sp) & np.isfinite(fl)
        inel = fin & ~el & ~gp & ~hs   # the I-S-INEL view's INEL set (exp/L_isinel/isi_lib.py inel_mask)
        z = np.load(P["fold_chain"] / split / "fold" / f"{pn}.npz")
        m = z["inel"]
        d_, t_, s_ = z["d"][m].astype(int), z["t"][m].astype(int), z["s"][m].astype(int)
        assert inel[d_, t_, s_].all() and m.sum() == inel.sum(), f"{pn}: fold chain INEL cells differ from the cache"
        pv, pq = np.full(sp.shape, np.nan), np.full(sp.shape, np.nan)
        pv[d_, t_, s_] = z["v"][m]; pq[d_, t_, s_] = z["q"][m]
        par, _, _, _ = metric._parameters(c, ROOT)
        out[pn] = dict(c=c, sp=sp, fl=fl, tg=tg, el=el, gp=gp, inel=inel, pv=pv, pq=pq, par=par,
                       regimes=np.asarray(c.regimes))
    return out


_VIEW = {}


def view():
    if "v" not in _VIEW:
        _VIEW["v"] = inel_view("validation")
    return _VIEW["v"]


def fig_F6():
    V = view()
    pn = "D12_I405_S"
    x = V[pn]
    vcut = x["par"].v_cut.to_numpy(float)
    cong = x["inel"] & (x["sp"] < vcut[None, None, :])
    day = int(np.argmax(cong.sum(axis=(1, 2))))
    order = np.argsort(x["par"].order_index.to_numpy())
    obs = x["sp"][day][:, order].T
    ours = np.where(x["inel"][day], x["pv"][day], x["sp"][day])[:, order].T
    inel = x["inel"][day][:, order].T
    tg = x["tg"][day][:, order].T
    err = np.where(inel, ours - obs, np.nan)
    # published: the cell type and our error at held-out INEL cells; released speed values are not drawn or written
    ctype = np.where(tg, 2, np.where(inel, 1, 0)).astype(float)
    ctype[~np.isfinite(obs) & ~tg] = np.nan
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.1), sharex=True, gridspec_kw=dict(hspace=.3))
    from matplotlib.colors import ListedColormap
    cm1 = ListedColormap(["#C9D1CB", "#0B6644", "#C98A1B"]); cm1.set_bad("#FFFFFF")
    axes[0].imshow(np.ma.masked_invalid(ctype), aspect="auto", cmap=cm1, vmin=-.5, vmax=2.5, origin="lower",
                   interpolation="nearest", extent=[0, 24, -.5, ctype.shape[0] - .5])
    axes[0].set_title("a  cell types: released (grey), held-out INEL (green), scored target (amber)", fontsize=6.6)
    axes[0].set_ylabel("station\n(corridor order)", fontsize=6)
    lim = np.nanpercentile(np.abs(err), 99)
    cm2 = plt.get_cmap("PuOr_r").copy(); cm2.set_bad("#F2F2F2")
    im = axes[1].imshow(np.ma.masked_invalid(err), aspect="auto", cmap=cm2, vmin=-lim, vmax=lim, origin="lower",
                        interpolation="nearest", extent=[0, 24, -.5, err.shape[0] - .5])
    axes[1].set_title("b  error at held-out INEL cells (ours - released)", fontsize=6.6)
    axes[1].set_ylabel("station", fontsize=6); axes[1].set_xlabel("hour of day")
    cb = fig.colorbar(im, ax=axes, pad=.01, fraction=.03); cb.set_label("km/h", fontsize=5.8); cb.ax.tick_params(labelsize=5.5)
    axes[1].set_xticks(range(0, 25, 3))
    e = err[np.isfinite(err)]
    o = obs[inel]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sd = float(np.std(o))
    congm = cong[day][:, order].T[inel]
    rmse_c = float(np.sqrt(np.mean(e[congm[np.isfinite(err[inel])]] ** 2))) if congm.any() else np.nan
    date = x["c"].dates[day]
    rows = []
    for si in range(obs.shape[0]):
        for ti in range(obs.shape[1]):
            rows.append(dict(station_rank=si, step=ti, inel=bool(inel[si, ti]), scored_target=bool(tg[si, ti]),
                             err_kmh=round(float(err[si, ti]), 1) if inel[si, ti] else None))
    save(fig, "F6_spacetime", pd.DataFrame(rows))
    CAPTIONS["F6_spacetime"] = dict(
        title="Held-out speed error on one March day is small",
        claim=(f"On {pn}, {date} (March; the day with the most released congested INEL cells, chosen by that rule), "
               f"the held-out prediction at {int(inel.sum()):,} INEL cells ({inel.mean():.1%} of the grid) has speed RMSE "
               f"{rmse:.1f} km/h against a released-speed SD of {sd:.1f} km/h at the same cells; at congested INEL cells "
               f"the RMSE is {rmse_c:.1f} km/h. Released speed values are not drawn."),
        number=f"RMSE {rmse:.1f} km/h vs SD {sd:.1f} km/h (INEL cells of the day)",
        intervals="No intervals; one day, one panel, all INEL cells of the day.",
        source="artifacts/cache (released cells, official view), artifacts/claude_ref/l_inelfold/isi/chain/validation/fold "
               "(INEL-FOLD chain, exp/L_inelfold/LINE.md).",
        notes=("Predictions are the INEL-FOLD chain: C77-class members (FV-INEL W1.0) that took INEL labels only from the "
               "other fold, so every INEL cell shown is held out. INEL cells are released and never scored; scored target "
               "cells carry no value in the cache (asserted) and are blank. Not C80 itself (S7c is not in this chain)."))


def fd_curve(k_n, r):
    """Normalised published triangle: x = k/kc, y = q/cap; congested branch with r = kj/kc."""
    return np.where(k_n <= 1, k_n, np.maximum(r - k_n, 0) / (r - 1))


def fig_F7():
    V = view()
    recs, pts = [], {}
    for pn, x in V.items():
        par = x["par"]
        lanes = par.lanes.to_numpy(float); cap = par.capacity_vph.to_numpy(float); kc = par.critical_density.to_numpy(float)
        kj = par.k_jam.to_numpy(float)
        r = (kj / kc)[None, None, :] * np.ones_like(x["sp"])
        for src, v, q in (("released", x["sp"], x["fl"]), ("ours", x["pv"], x["pq"])):
            k = q / np.maximum(v, 1.0)
            kn = k / kc[None, None, :]; qn = q / cap[None, None, :]
            m = x["inel"] & np.isfinite(kn) & np.isfinite(qn)
            for reg in ("R1", "R2", "R3"):
                mm = m & (x["regimes"] == reg)[:, None, None]
                pts.setdefault((src, reg), []).append(np.c_[kn[mm], qn[mm], r[mm]])
                q_l = q / lanes[None, None, :]; k_l = k / lanes[None, None, :]
                qfd = np.where(k_l <= (kc / lanes)[None, None, :], par.free_speed_kmh.to_numpy(float)[None, None, :] * k_l,
                               (cap / lanes / np.maximum(kj / lanes - kc / lanes, 1e-9))[None, None, :]
                               * np.maximum((kj / lanes)[None, None, :] - k_l, 0))
                recs.append(dict(panel=pn, source=src, regime=reg, cells=int(mm.sum()),
                                 fd_abs_err=float(np.abs(q_l - qfd)[mm].sum()), q_abs=float(np.abs(q_l)[mm].sum())))
    t = pd.DataFrame(recs)
    g = t.groupby(["source", "regime"])[["cells", "fd_abs_err", "q_abs"]].sum()
    g["fd_loss"] = g.fd_abs_err / g.q_abs
    fig, axes = plt.subplots(3, 2, figsize=(COL_W, 4.2), sharex=True, sharey=True, gridspec_kw=dict(hspace=.25, wspace=.08))
    rmed = np.median(np.concatenate([np.concatenate(pts[("released", reg)])[:, 2] for reg in ("R1", "R2", "R3")]))
    kk = np.linspace(0, 4, 300)
    for i, reg in enumerate(("R1", "R2", "R3")):
        for j, src in enumerate(("released", "ours")):
            a = np.concatenate(pts[(src, reg)])
            ax = axes[i, j]
            hb = ax.hexbin(a[:, 0], a[:, 1], gridsize=45, extent=(0, 4, 0, 1.3), bins="log", mincnt=1,
                           cmap="Greys" if src == "released" else "Blues", linewidths=0)
            ax.plot(kk, fd_curve(kk, rmed), color=OK["verm"], lw=.8)
            ax.set_xlim(0, 4); ax.set_ylim(0, 1.3)
            ax.text(.97, .93, f"{reg}, {src}\nFD loss {g.loc[(src, reg), 'fd_loss']:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=5.4)
            if i == 2:
                ax.set_xlabel("density / critical density")
            if j == 0:
                ax.set_ylabel("flow / capacity")
    fig.legend(handles=[Line2D([], [], color=OK["verm"], lw=.8, label=f"published triangle (median kjam/kc = {rmed:.1f})")],
               loc="upper center", fontsize=5.6, ncol=1, bbox_to_anchor=(.5, .995))
    save(fig, "F7_fundamental_diagram", g.reset_index())
    fl_r = {reg: g.loc[("released", reg), "fd_loss"] for reg in ("R1", "R2", "R3")}
    fl_o = {reg: g.loc[("ours", reg), "fd_loss"] for reg in ("R1", "R2", "R3")}
    n = int(g.loc["ours"].cells.sum())
    CAPTIONS["F7_fundamental_diagram"] = dict(
        title="Held-out predictions are physically plausible",
        claim=(f"At {n:,} held-out INEL cells of March (10 panels), our predictions sit on the published fundamental "
               f"diagram at least as closely as the released observations: FD loss (sum |q - q_FD(k)| / sum q, per lane) "
               + ", ".join(f"{reg} {fl_o[reg]:.3f} vs {fl_r[reg]:.3f}" for reg in fl_o) + " (ours vs released)."),
        number="; ".join(f"{reg} {fl_o[reg]:.3f}/{fl_r[reg]:.3f}" for reg in fl_o),
        intervals="Hexbin counts on a log colour scale; no intervals.",
        source="artifacts/cache (released cells), artifacts/claude_ref/l_inelfold/isi/chain/validation/fold, network "
               "fd_parameters.csv through src/tfb/metric.py _parameters (the scorer replica's FD form).",
        notes="Axes are normalised per station (kc = capacity / free speed, as in the scorer), so all stations share one "
              "free-flow branch; the congested branch differs by station and the drawn one uses the median kjam/kc. "
              "Lower FD loss for ours partly reflects that predictions are smoother than observations (hypothesis).")


# ----------------------------------------------------------------------------------------------- F8
def fig_F8():
    h, rows = find_table(P["r5"], "class (March")
    terc = []
    for name, key in (("short", "short links"), ("middle", "middle tercile"), ("long", "long links")):
        r = row_by(rows, key)
        terc.append(dict(tercile=name, cell_share=first_num(r[1]), dState=first_num(r[2]), dPhys=first_num(r[3]),
                         fc=first_num(r[4])))
    terc = pd.DataFrame(terc)
    allr = row_by(rows, "all target cells")
    tot_s, tot_p = first_num(allr[2]), first_num(allr[3])
    h1, rows1 = find_table(P["r5"], "item")
    run = row_by(rows1, "LWR loss by run length")
    mar, apr = nums(run[1]), nums(run[2])
    runs = pd.DataFrame(dict(run=["lone", "2-3", "4-12"], march=mar, april=apr))
    # INEL-cell proxy: share of the density x length transition residual by length tercile (all March panels)
    V = view()
    acc = []
    for pn, x in V.items():
        L = x["par"].length_km.to_numpy(float)
        No = x["fl"] / np.maximum(x["sp"], 1.0) * L[None, None, :]
        Nh = np.where(x["inel"], x["pq"] / np.maximum(x["pv"], 1.0) * L[None, None, :], No)
        e = Nh - No
        ok = np.isfinite(No[:, :-1]) & np.isfinite(No[:, 1:]) & (x["inel"][:, :-1] | x["inel"][:, 1:])
        res = np.abs(e[:, 1:] - e[:, :-1])
        for s in range(L.size):
            mm = ok[:, :, s]
            if mm.any():
                acc.append(dict(panel=pn, length_km=L[s], transitions=int(mm.sum()), residual=float(res[:, :, s][mm].sum())))
    a = pd.DataFrame(acc)
    q = np.quantile(np.repeat(a.length_km.to_numpy(), a.transitions.to_numpy()), [1 / 3, 2 / 3])
    a["tercile"] = np.where(a.length_km <= q[0], "short", np.where(a.length_km <= q[1], "middle", "long"))
    prox = a.groupby("tercile")[["transitions", "residual"]].sum().reindex(["short", "middle", "long"])
    prox = prox / prox.sum()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2), gridspec_kw=dict(wspace=.45, width_ratios=[1, 1.35]))
    ax = axes[0]
    w = .36
    ax.bar(np.arange(3) - w / 2, runs.march, w, color=OK["blue"], label="March")
    ax.bar(np.arange(3) + w / 2, runs.april, w, color=OK["orange"], label="April")
    ax.set_xticks(range(3)); ax.set_xticklabels(["lone", "2-3", "4-12"]); ax.set_xlabel("hidden run length (cells)")
    ax.set_ylabel("LWR loss (own view)"); ax.legend(fontsize=5.4); ax.set_title("a  conservation loss\nby run length")
    ax = axes[1]
    xs = np.arange(3); w = .26
    ax.bar(xs - w, terc.dState / tot_s, w, color=OK["sky"], label="state sensitivity (own view)")
    ax.bar(xs, terc.dPhys / tot_p, w, color=OK["verm"], label="physics sensitivity (own view)")
    ax.bar(xs + w, prox.residual.to_numpy(), w, color=OK["verm"], alpha=.45, hatch="///", edgecolor=OK["verm"], lw=.3,
           label="mass-balance residual\n(held-out INEL cells, proxy)")
    ax.axhline(1 / 3, color=OK["grey"], lw=.5, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(["short", "middle", "long"]); ax.set_xlabel("link length tercile")
    ax.set_ylabel("share"); ax.set_ylim(0, 1); ax.legend(fontsize=4.9, loc="upper left")
    ax.set_title("b  where physics error sits")
    save(fig, "F8_physics_anatomy", {"run_length": runs, "terciles": terc.assign(
        state_share=terc.dState / tot_s, phys_share=terc.dPhys / tot_p,
        inel_proxy_residual_share=prox.residual.to_numpy(), inel_proxy_transition_share=prox.transitions.to_numpy())})
    lone_share = mar[0] / sum(mar)
    CAPTIONS["F8_physics_anatomy"] = dict(
        title="Mass balance breaks at lone hidden cells on long links",
        claim=(f"Lone hidden cells carry {lone_share:.0%} of the LWR conservation loss (March own view). The long-link "
               f"tercile holds {terc.dPhys.iloc[2] / tot_p:.0%} of the physics sensitivity but only "
               f"{terc.dState.iloc[2] / tot_s:.0%} of the state sensitivity; on held-out INEL cells it holds "
               f"{prox.residual.iloc[2]:.0%} of the density x length transition residual (proxy)."),
        number=f"long links: physics {terc.dPhys.iloc[2] / tot_p:.0%}, state {terc.dState.iloc[2] / tot_s:.0%}",
        intervals="Published point values (R5, 10% error cut per class); no intervals were computed in R5.",
        source="docs/research/ideas_R5/R5.md tables 1b and 1c (own view, C66/C69 chain); proxy bars computed here from "
               "artifacts/cache and the INEL-FOLD chain.",
        notes=("The space x time map of the conservation residual named in FINAL_PLAN 4.3 needs the own-view chain grids "
               "of R5 (scratch, not in the repo); see TODO_F6_F7.md section F8. The proxy uses INEL cells, which the "
               "physics scorer does not score (they are ineligible), so it is a cross-check of the length effect only."))


# ----------------------------------------------------------------------------------------------- F9
def fig_F9():
    h, rows = find_table(P["r6a"], "slice")
    keep = [("queue FRONT", "queue front (2 steps / 2 stations)"), ("congested, not front", "congested, not front"),
            ("free flow", "free flow"), ("4 panels", "4 panels (D12 I5 S, I405 N/S; D7 I210 W)"), ("hours 15-19", "hours 15-19"),
            ("regime R3", "regime R3 days"), ("hidden run 4-12", "hidden run 4-12 cells"),
            ("nearest visible station 3-5", "nearest visible station 3-5 away"), ("long links", "long links (top tercile)"),
            ("lanes 6", "6-lane stations")]
    recs = []
    for key, lab in keep:
        r = row_by(rows, key)
        cs = [x / 100 for x in nums(r[1])]
        imp = [x / 100 for x in nums(r[3])]
        cs = cs * 2 if len(cs) == 1 else cs
        recs.append(dict(slice=lab, cell_share_mar=cs[0], cell_share_apr=cs[1], improvable_mar=imp[0], improvable_apr=imp[1],
                         loss_share_mar=nums(r[2])[0] / 100, loss_share_apr=nums(r[2])[-1] / 100))
    d = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    y = np.arange(len(d))[::-1]
    hgt = .26
    ax.barh(y + hgt, d.cell_share_mar, hgt, color=OK["grey"], label="share of cells (March)")
    ax.barh(y, d.improvable_mar, hgt, color=OK["blue"], label="share of improvable state error, March")
    ax.barh(y - hgt, d.improvable_apr, hgt, color=OK["orange"], label="share of improvable state error, April")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(max(r.improvable_mar, r.improvable_apr, r.cell_share_mar) + .01, yi,
                f"{r.improvable_mar:.0%}/{r.improvable_apr:.0%} in {r.cell_share_mar:.1%}",
                va="center", fontsize=4.9)
    ax.set_yticks(y); ax.set_yticklabels(d.slice, fontsize=5.8)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.legend(fontsize=5.2, loc="lower center", bbox_to_anchor=(.4, 1.0), ncol=1)
    ax.set_xlabel("share (slices overlap; each bar is its own slice)")
    save(fig, "F9_error_anatomy", d)
    fr = d.iloc[0]
    four = d.iloc[3]
    CAPTIONS["F9_error_anatomy"] = dict(
        title="The remaining state error lives at queue fronts on four panels",
        claim=(f"Queue-front cells are {fr.cell_share_mar:.1%} / {fr.cell_share_apr:.1%} of cells but hold "
               f"{fr.improvable_mar:.0%} / {fr.improvable_apr:.0%} of the improvable state error (March / April); four "
               f"panels hold {four.improvable_mar:.0%} / {four.improvable_apr:.0%} of it in {four.cell_share_mar:.0%} of "
               "cells. A tail of about 300 single-step hedges a month (speed error > 15 km/h, .031% of cells) carries "
               "19% / 17% of the whole linear state loss."),
        number=f"fronts {fr.cell_share_mar:.1%} of cells, {fr.improvable_mar:.0%}/{fr.improvable_apr:.0%} of improvable error",
        intervals="Published point shares; R6A reports no intervals for the slice shares.",
        source="docs/research/ideas_R6/R6A.md section 1 table (FV2 chain at INEL cells, official geometry). The scratch "
               "outputs of R6A (r6a_sum.json) are not in the repository, so the published numbers are plotted and cited.",
        notes="Improvable = error above a heteroscedastic noise floor (R6A section 0). Slices overlap.")


# ----------------------------------------------------------------------------------------------- F10
def fvi(path):
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("FVI_INFO"):
            return json.loads(line[len("FVI_INFO"):].strip())
    raise KeyError(path)


def fig_F10():
    cal, _ = load_calibration()
    c = cal.set_index("file")
    man = load_manifest().set_index("stem")
    pub = lambda f: float(man.loc[f, "public_num"])  # noqa: E731
    steps = [  # cumulative label-source gain on the board (March reads)
        ("S2b chain\n(own-view labels)", 0.0, "C69 level"),
        ("FV1 +6k\nofficial view", pub("c_fvft_mar_on_c69") - pub("c_gap_on_c66"), "C70 - C69"),
        ("FV2 +12k\nsame labels", pub("c_fv2_on_b19") - pub("c_gap3_aprfv_on_c70"), "C72 - C71"),
        ("FV-INEL W.3\nINEL labels", pub("c_fvinel_on_b19") - pub("c_fv2_on_b19"), "C73 - C72"),
        ("W1.0\nINEL weight 1", c.loc["c_fvinel_w10_on_c75", "d_board"], "read on C75"),
        ("S7c-INEL\nstage learner", pub("c_isinel_s7c_mar_on_c77") - c.loc["c_isinel_s7c_mar_on_c77", "forecast"]
         + (c.loc["c_isinel_s7c_mar_on_c77", "forecast"] - .90081), "C79 - C77 forecast"),
    ]
    lab = [s[0] for s in steps]; inc = np.array([s[1] for s in steps]); cum = np.cumsum(inc)
    fc = [np.nan, c.loc["c_fvft_mar_on_c69", "d_pred"], c.loc["c_fv2_on_b19", "d_pred"],
          c.loc["c_fvinel_on_b19", "forecast"] - pub("c_fv2_on_b19"), c.loc["c_fvinel_w10_on_c75", "d_pred"],
          c.loc["c_isinel_s7c_mar_on_c77", "forecast"] - C77_FC]
    G = json.loads(P["inelfold_gate"].read_text())
    k = G["k"]
    fold_m, fold_a = G["march"], G["april"]
    S7 = {}
    for line in P["s7cf_log"].read_text(encoding="utf-8").splitlines():
        m = re.search(r"S7CF_READ (\S+) march fc (\S+) \[(\S+), (\S+)\].*board (\S+) april fc (\S+) \[(\S+), (\S+)\].*board (\S+)$", line)
        if m:
            S7[m.group(1)] = [float(x) for x in m.groups()[1:]]
    honest = S7["fold->s7c_fv2x"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=.5))
    ax = axes[0]
    x = np.arange(len(steps))
    ax.plot(x, cum * 1e4, color=OK["blue"], marker="o", ms=3, label="board, cumulative (March reads)")
    ax.plot(x[1:], (cum[:-1] + np.array(fc[1:])) * 1e4, ls="", marker="_", ms=7, color="k", mew=1, label="forecast of the step")
    fold_lo, fold_hi = k * fold_m["ci_fc"][0], k * fold_m["ci_fc"][1]
    ax.errorbar([4.25], [(cum[2] + k * fold_m["fc"]) * 1e4], yerr=[[(k * fold_m["fc"] - fold_lo) * 1e4], [(fold_hi - k * fold_m["fc"]) * 1e4]],
                fmt="s", ms=3, color=OK["verm"], capsize=1.5, lw=.7, label="INEL-FOLD held-out (FV2 + W1.0 class)")
    ax.errorbar([5.25], [(cum[4] + honest[3]) * 1e4], yerr=[[(honest[3] - k * honest[1]) * 1e4], [(k * honest[2] - honest[3]) * 1e4]],
                fmt="D", ms=2.6, color=OK["purple"], capsize=1.5, lw=.7, label="S7C-FOLD held-out")
    ax.set_xticks(x); ax.set_xticklabels(["S2b\n(own\nview)", "FV1\n+6k", "FV2\n+12k", "INEL\nW.3", "INEL\nW1.0", "S7c-\nINEL"],
                                         fontsize=5.2)
    ax.set_ylabel("label-source gain (x 1e-4 public)")
    ax.legend(fontsize=4.8, loc="lower right")
    ax.set_title("a  new labels jump, more steps on\nthe same labels stay flat", fontsize=6.4)
    ax2 = axes[1]
    mw03, mw10, aw03, aw10 = fvi(P["fvi_mar_w03"]), fvi(P["fvi_mar_w10"]), fvi(P["fvi_apr_w03"]), fvi(P["fvi_apr_w10"])
    rows = []
    for mon, a3, a10 in (("March", mw03, mw10), ("April", aw03, aw10)):
        g = a3["groups"]["new_cells"]
        rows += [dict(month=mon, group="new official-view cells", model=m_, loss=v_) for m_, v_ in
                 (("S2b parent", g["s2b_parent"]), ("FV", g["fv"]), ("FV2", a3["fv2_same_keys"]["groups"]["new_cells"]),
                  ("FV-INEL W.3", g["fvinel"]), ("W1.0", a10["groups"]["new_cells"]["fvinel"]))]
        rows += [dict(month=mon, group="INEL cells", model=m_, loss=v_) for m_, v_ in
                 (("FV", a3["inel_err"]["fv"]), ("FV-INEL W.3", a3["inel_err"]["fvinel"]), ("W1.0", a10["inel_err"]["fvinel"]))]
    mem = pd.DataFrame(rows)
    models = ["S2b parent", "FV", "FV2", "FV-INEL W.3", "W1.0"]
    for mon, col, off in (("March", OK["blue"], -.1), ("April", OK["orange"], .1)):
        s = mem[(mem.month == mon) & (mem.group == "new official-view cells")].set_index("model").reindex(models)
        ax2.plot(np.arange(5) + off, s.loss * 1e3, marker="o", ms=2.5, color=col, label=f"{mon}, new cells")
        s2 = mem[(mem.month == mon) & (mem.group == "INEL cells")].set_index("model").reindex(models)
        ax2.plot(np.arange(5) + off, s2.loss * 1e3, marker="^", ms=2.5, color=col, ls="--", label=f"{mon}, INEL cells")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(["S2b", "FV", "FV2", "INEL\nW.3", "INEL\nW1.0"], fontsize=5.2)
    ax2.set_ylabel("member loss (x 1e-3, metric form)"); ax2.legend(fontsize=4.6, loc="upper right")
    ax2.set_title("b  member error", fontsize=6.4)
    data = pd.DataFrame(dict(step=[s.replace("\n", " ") for s in lab], board_increment=inc, board_cumulative=cum,
                             forecast_increment=fc, how=[s[2] for s in steps]))
    extra = pd.DataFrame([dict(step="INEL-FOLD March (board units k x fc)", board_increment=k * fold_m["fc"],
                               board_cumulative=np.nan, forecast_increment=np.nan, how=f"k={k}, CI {k * fold_m['ci_fc'][0]:.6f}..{k * fold_m['ci_fc'][1]:.6f}"),
                          dict(step="INEL-FOLD April (board units k x fc)", board_increment=k * fold_a["fc"], board_cumulative=np.nan,
                               forecast_increment=np.nan, how=f"k={k}, CI {k * fold_a['ci_fc'][0]:.6f}..{k * fold_a['ci_fc'][1]:.6f}"),
                          dict(step="S7C-FOLD held-out March / April (board)", board_increment=honest[3], board_cumulative=np.nan,
                               forecast_increment=honest[7], how="exp/L_isinel/logs/s7cf_run.log fold->s7c_fv2x")])
    save(fig, "F10_label_dose", {"board": pd.concat([data, extra], ignore_index=True), "member_loss": mem})
    CAPTIONS["F10_label_dose"] = dict(
        title="New label sources beat more training on the same labels",
        claim=(f"Official-view labels of the target month gave {f5(inc[1])} (FV1); 6k more steps on the same labels gave "
               f"{f5(inc[2])} (FV2); released-but-ineligible (INEL) cells as a new label source gave {f5(inc[3])}, weight 1.0 "
               f"{f5(inc[4])}, and INEL labels in a stage learner {f5(inc[5])}: {f5(cum[-1])} in total from label sources. "
               f"Held out, INEL-FOLD reads the INEL class at {f5(k * fold_m['fc'])} March and {f5(k * fold_a['fc'])} April "
               "(board units), with 95% CIs above zero in both months."),
        number=f"label sources {f5(cum[-1])} public; INEL-FOLD Apr {f5(k * fold_a['fc'])}",
        intervals=("Squares and diamonds: 95% CI, joint date bootstrap (500 draws) of I-S-INEL, scaled to board units by "
                   f"k = {k}. Ticks: forecast written before each upload. Panel b: mean over 30 members, no interval."),
        source="artifacts/claude_ref/probes/manifest.csv, docs/report/board_calibration.csv, artifacts/claude_ref/l_inelfold/GATE.json, "
               "exp/L_isinel/logs/s7cf_run.log (S7CF_READ), exp/L_fvinel/logs/**/FVI_INFO lines (member loss).",
        notes=("Member loss at INEL cells is IN SAMPLE for the INEL-label members (they were trained on those cells); "
               "only the held-out markers in panel a are honest. Loss = .54 (dv/25)^2 + .46 (dq per lane/600)^2, the "
               "metric's per-cell form (exp/L_fvinel/fvi_step.py inel_err). The S7c step is read against the unread C77 "
               "forecast .90081."))


# ----------------------------------------------------------------------------------------------- F11
def fig_F11():
    I = json.loads(P["isinel_gate"].read_text())
    G = json.loads(P["inelfold_gate"].read_text())
    cal, _ = load_calibration(); c = cal.set_index("file")
    man = load_manifest().set_index("stem")
    k = I["instrument"]["k_fc"]
    fv1_board = float(man.loc["c_fvft_mar_on_c69", "public_num"] - man.loc["c_gap_on_c66", "public_num"])
    pts = [
        dict(what="FV1 (defines k)", forecast=k * I["instrument"]["march"]["s2b_fv"]["fc"], board=fv1_board, kind="calibration"),
        dict(what="FV2 step", forecast=I["instrument"]["forecast_fv2_board"], board=c.loc["c_fv2_on_b19", "d_board"], kind="test"),
        dict(what="S7c-INEL", forecast=I["s7c"]["board_forecast_public"],
             board=float(man.loc["c_isinel_s7c_mar_on_c77", "public_num"]) - C77_FC, kind="test"),
        dict(what="INEL class (fold)", forecast=G["march"]["board_units"],
             board=float(man.loc["c_fvinel_on_b19", "public_num"] - man.loc["c_fv2_on_b19", "public_num"])
             + c.loc["c_fvinel_w10_on_c75", "d_board"], kind="test"),
    ]
    pts = pd.DataFrame(pts)
    S7 = []
    for line in P["s7cf_log"].read_text(encoding="utf-8").splitlines():
        m = re.search(r"S7CF_READ (\S+) march fc (\S+) \[(\S+), (\S+)\] dState \S+ board (\S+) april fc (\S+) \[(\S+), (\S+)\] dState \S+ board (\S+)", line)
        if m:
            g = m.groups()
            S7.append(dict(pair=g[0], mar_fc=float(g[1]), mar_lo=float(g[2]), mar_hi=float(g[3]), mar_board=float(g[4]),
                           apr_fc=float(g[5]), apr_lo=float(g[6]), apr_hi=float(g[7]), apr_board=float(g[8])))
    S7 = pd.DataFrame(S7)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.4), gridspec_kw=dict(wspace=.55, width_ratios=[1, 1.25]))
    ax = axes[0]
    for _, r in pts.iterrows():
        mk = "s" if r.kind == "calibration" else "o"
        ax.scatter(r.forecast * 1e4, r.board * 1e4, marker=mk, s=16, color=OK["blue"] if r.kind == "test" else OK["grey"], zorder=3)
        off_ = {"INEL class (fold)": (-34, -9), "FV1 (defines k)": (-40, 4)}.get(r.what, (3, -6))
        ax.annotate(r.what, (r.forecast * 1e4, r.board * 1e4), fontsize=4.9, xytext=off_, textcoords="offset points")
    lo_ci, hi_ci = k * G["march"]["ci_fc"][0], k * G["march"]["ci_fc"][1]
    r = pts.iloc[3]
    ax.plot([lo_ci * 1e4, hi_ci * 1e4], [r.board * 1e4] * 2, color=OK["blue"], lw=.7)
    lim = [.1, 12]
    ax.plot(lim, lim, color="k", ls="--", lw=.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("I-S-INEL forecast (x 1e-4)"); ax.set_ylabel("board read (x 1e-4)")
    ax.set_title("a  forecast vs board", fontsize=6.4)
    ax2 = axes[1]
    labels = {"fold->s7c_fv2fit": "S7c on its own\ntraining cells", "fold->s7c_fv2x": "S7c held out\n(shipped fit)",
              "fold->s7c_fold_x": "S7c held out\n(fold refit)", "s7c_fv2x->s7c_fold_x": "fold refit vs\nshipped fit"}
    sub = S7[S7.pair.isin(labels)].set_index("pair").reindex(list(labels))
    y = np.arange(len(sub))[::-1]
    for off, mon, col in ((.13, "mar", OK["blue"]), (-.13, "apr", OK["orange"])):
        b = k * sub[f"{mon}_fc"].to_numpy()  # centre = k x fc (the logged board column is rounded to 1e-6)
        lo = k * sub[f"{mon}_lo"].to_numpy(); hi = k * sub[f"{mon}_hi"].to_numpy()
        ax2.errorbar(b * 1e4, y + off, xerr=[(b - lo) * 1e4, (hi - b) * 1e4], fmt="o", ms=2.6, color=col, capsize=1.3, lw=.7,
                     label="March" if mon == "mar" else "April")
    s7_board = float(man.loc["c_isinel_s7c_mar_on_c77", "public_num"]) - C77_FC
    ax2.axvline(s7_board * 1e4, color=OK["verm"], lw=.7, ls="--", label="board read of S7c (March)")
    ax2.axvline(0, color=OK["grey"], lw=.5)
    ax2.set_yticks(y); ax2.set_yticklabels(sub.index.map(labels), fontsize=5)
    ax2.set_xlabel("board units (x 1e-4)")
    ax2.legend(fontsize=4.8, loc="upper center", bbox_to_anchor=(.35, -.25), ncol=2)
    ax2.set_title("b  S7C-FOLD: held-out vs in-sample", fontsize=6.4)
    save(fig, "F11_instrument_validation", {"forecast_vs_board": pts, "s7cf": S7})
    ratio = I["instrument"]["ratio_board_over_forecast_fv2"]
    CAPTIONS["F11_instrument_validation"] = dict(
        title="Instruments were validated before use, and their misses are reported",
        claim=(f"I-S-INEL, calibrated once on FV1 (k = {k:.3f}), forecast the next step FV2 at "
               f"{f5(I['instrument']['forecast_fv2_board'], 6)} and the board read {f5(pts.iloc[1].board)} "
               f"(ratio {ratio:.2f}). INEL-FOLD put the INEL class at {f5(G['march']['board_units'])} "
               f"[{f5(lo_ci)}, {f5(hi_ci)}], with the board at {f5(r.board)} inside that interval. S7c read "
               f"{f5(s7_board)} on the board against {f5(I['s7c']['board_forecast_public'])} forecast; held out on "
               f"the shipped chain it is worth {f5(sub.loc['fold->s7c_fv2x', 'mar_board'], 6)} (March) / "
               f"{f5(sub.loc['fold->s7c_fv2x', 'apr_board'], 6)} (April), while its in-sample value "
               f"{f5(sub.loc['fold->s7c_fv2fit', 'mar_board'], 6)} matches the board read."),
        number=f"FV2 board/forecast {ratio:.2f}; S7c held-out {f5(sub.loc['fold->s7c_fv2x', 'mar_board'], 6)} vs board {f5(s7_board)}",
        intervals="Horizontal bars: 95% CI, joint date bootstrap (500 draws), converted to board units by k.",
        source="artifacts/claude_ref/l_isinel/GATE.json, artifacts/claude_ref/l_inelfold/GATE.json, "
               "exp/L_isinel/logs/s7cf_run.log (S7CF_READ lines), manifest and calibration for board reads.",
        notes=("FV1 is the point that defines k, so it lies on y = x by construction (grey square). The S7c miss has "
               "several possible explanations (one read's variance about .00001, front-concentrated gains, stage-learner "
               "effects the instrument under-reads); none is claimed (FINAL_PLAN section 8)."))


# ----------------------------------------------------------------------------------------------- F12
def fig_F12():
    S = json.loads(P["seed6"].read_text())
    mv, mp = json.loads(P["seed6_mem_v"].read_text()), json.loads(P["seed6_mem_p"].read_text())
    rows = []
    for mon, M in (("March", mv), ("April", mp)):
        for grp in ("small5_pooled", "all10_pooled"):
            e = M[grp]["e_rel"]; lo, hi = M[grp]["ci"]
            r2 = 1 / (1 + e)  # e(2)/e(3)
            rows += [dict(month=mon, group=grp, seeds=2, rel_err=r2, lo=1 / (1 + hi), hi=1 / (1 + lo), kind="measured"),
                     dict(month=mon, group=grp, seeds=3, rel_err=1.0, lo=1.0, hi=1.0, kind="reference"),
                     dict(month=mon, group=grp, seeds=6, rel_err=2 - r2, lo=2 - 1 / (1 + lo), hi=2 - 1 / (1 + hi),
                          kind="projected e(n) = a + b/n")]
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2), gridspec_kw=dict(wspace=.55))
    ax = axes[0]
    for mon, col in (("March", OK["blue"]), ("April", OK["orange"])):
        for grp, ls in (("small5_pooled", "-"), ("all10_pooled", ":")):
            s = d[(d.month == mon) & (d.group == grp)].sort_values("seeds")
            ax.plot(s.seeds[:2], s.rel_err[:2], color=col, ls=ls, marker="o", ms=2.6)
            ax.plot(s.seeds[1:], s.rel_err[1:], color=col, ls="--" if ls == "-" else ":", marker="o", ms=2.6, mfc="white")
            ax.fill_between(s.seeds, s.lo, s.hi, color=col, alpha=.15, lw=0)
    ax.set_xticks([2, 3, 6]); ax.set_xlabel("B6 seeds per panel"); ax.set_ylabel("member error at INEL cells\n(relative to 3 seeds)")
    ax.legend(handles=[Line2D([], [], color=OK["blue"], label="March"), Line2D([], [], color=OK["orange"], label="April"),
                       Line2D([], [], color="k", label="5 small panels"), Line2D([], [], color="k", ls=":", label="all 10 panels"),
                       Line2D([], [], color="k", ls="--", marker="o", mfc="white", ms=2.6, label="6 seeds: projected")],
              fontsize=4.8, loc="upper right")
    ax.set_title("a  member error vs seeds", fontsize=6.4)
    ax2 = axes[1]
    for i, (mon, key, col) in enumerate((("March", "march", OK["blue"]), ("April", "april", OK["orange"]))):
        be, (lo, hi) = S[key]["board_equiv"], S[key]["board_ci"]
        ax2.errorbar([i - .12], [be * 1e5], yerr=[[(be - lo) * 1e5], [(hi - be) * 1e5]], fmt="o", color=col, ms=3, capsize=1.5, lw=.7)
        ax2.errorbar([i + .12], [be * 1e5], yerr=[[(be - lo) * 1e5], [(hi - be) * 1e5]], fmt="o", mfc="white", color=col, ms=3,
                     capsize=1.5, lw=.7, ls="")
    ax2.axhline(0, color=OK["grey"], lw=.5)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["March", "April"])
    ax2.set_ylabel("chain gain, board units (x 1e-5)")
    ax2.legend(handles=[Line2D([], [], ls="", marker="o", color="k", ms=3, label="2 -> 3 seeds (measured)"),
                        Line2D([], [], ls="", marker="o", mfc="white", color="k", ms=3, label="3 -> 6 seeds (projected)")],
               fontsize=4.8, loc="upper left")
    ax2.set_title("b  chain gain", fontsize=6.4)
    ch = pd.DataFrame([dict(month=m, pair="2->3 measured (LOO_AVG)", board_equiv=S[k]["board_equiv"], lo=S[k]["board_ci"][0],
                            hi=S[k]["board_ci"][1], fc=S[k]["fc"], p_fc_gt0=S[k]["p_fc_gt0"]) for m, k in (("March", "march"), ("April", "april"))])
    save(fig, "F12_seed_curve", {"member": d, "chain": ch})
    CAPTIONS["F12_seed_curve"] = dict(
        title="More seeds cut variance; the gain shows up in April, not March",
        claim=(f"Going from 2 to 3 B6 seeds lowers member error at INEL cells by {-mv['small5_pooled']['e_rel']:.1%} (March) "
               f"and {-mp['small5_pooled']['e_rel']:.1%} (April) on the five small panels. At the chain this is "
               f"{f5(S['april']['board_equiv'], 6)} [{f5(S['april']['board_ci'][0], 6)}, {f5(S['april']['board_ci'][1], 6)}] "
               f"board units in April but only {f5(S['march']['board_equiv'], 6)} [{f5(S['march']['board_ci'][0], 6)}, "
               f"{f5(S['march']['board_ci'][1], 6)}] in March."),
        number=f"April {f5(S['april']['board_equiv'], 6)} vs March {f5(S['march']['board_equiv'], 6)}",
        intervals="95% CI, joint date bootstrap (500 draws); member CIs in panel a from the same bootstrap.",
        source="artifacts/claude_ref/l_seed6/SEED6_POWER.json, member_validation.json, member_private.json; method "
               "exp/L_seed6/LINE.md (leave-one-seed-out arms, exchangeable seeds).",
        notes=("6 seeds were NOT run to the end (SEED6 step 1 paused at the end of development, 10-21); the 3 -> 6 points "
               "are the projection e(2) - e(3) = e(3) - e(6) = b/6 under exchangeable seeds, not measurements."))


# ----------------------------------------------------------------------------------------------- F13
def fig_F13():
    tabs = md_tables(P["r4"])
    t1b = [t for t in tabs if any(r and r[0].startswith("distance of the nearest cell") for r in t[1])][0][1]
    dist = row_by(t1b, "distance of the nearest cell")
    q_m, q_a = nums(dist[1]), nums(dist[2])
    dw = row_by(t1b, "DW-ADAPT: largest move among")
    todr = row_by(t1b, "TODR: the same")
    wc = row_by(t1b, "windows changed: DW")
    sc = row_by(t1b, "windows changed with the DW delta scaled")
    near = row_by(t1b, "windows with any cell within")
    scale = [1, 2, 5, 10, 20]
    wm = [nums(wc[1])[0]] + nums(sc[1]); wa = [nums(wc[2])[0]] + nums(sc[2])
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.3), gridspec_kw=dict(wspace=.62))
    ax = axes[0]
    qs = ["p10", "p25", "median", "p75"]
    ax.plot(q_m, range(4), marker="o", ms=3, color=OK["blue"], label="March")
    ax.plot(q_a, range(4), marker="s", ms=3, color=OK["orange"], label="April")
    for v, ls, lab in ((nums(dw[1])[0], "--", "DW-ADAPT move (Mar)"), (nums(todr[1])[0], ":", "TODR move (Mar)")):
        ax.axvline(v, color=OK["verm"], ls=ls, lw=.8, label=lab)
    ax.set_xscale("log"); ax.set_yticks(range(4)); ax.set_yticklabels(qs)
    ax.set_xlabel("nearest cell's distance\nto the decode threshold")
    ax.legend(fontsize=4.6, loc="upper left"); ax.set_title("a  decisive cells are far", fontsize=6.4)
    ax2 = axes[1]
    ax2.plot(scale, wm, marker="o", ms=3, color=OK["blue"], label="March (40 windows)")
    ax2.plot(scale, wa, marker="s", ms=3, color=OK["orange"], label="April (40 windows)")
    ax2.axhline(3, color=OK["grey"], ls="--", lw=.7, label="3-window upload rule")
    ax2.set_xscale("log"); ax2.set_xticks(scale); ax2.set_xticklabels([f"x{s}" for s in scale])
    ax2.set_xlabel("DW-ADAPT move, scaled"); ax2.set_ylabel("official windows changed")
    ax2.legend(fontsize=4.8, loc="upper left"); ax2.set_title("b  windows flipped vs dose", fontsize=6.4)
    d1 = pd.DataFrame(dict(quantile=qs, march=q_m, april=q_a))
    d2 = pd.DataFrame(dict(dw_scale=scale, windows_march=wm, windows_april=wa))
    d3 = pd.DataFrame(dict(item=["DW-ADAPT largest move near threshold", "TODR largest move near threshold",
                                 "windows with a cell within .01/.02/.05"],
                           march=[dw[1], todr[1], near[1]], april=[dw[2], todr[2], near[2]]))
    save(fig, "F13_decode_bottleneck", {"distance": d1, "dose": d2, "moves": d3})
    CAPTIONS["F13_decode_bottleneck"] = dict(
        title="Small queue gains do not show because the decode is sharp",
        claim=(f"The median official window has no cell within {q_m[2]:.3f} (March) / {q_a[2]:.3f} (April) of the eIoU "
               f"threshold, while in-model target-month pieces move the cells near it by {nums(dw[1])[0]} (DW-ADAPT) and "
               f"{nums(todr[1])[0]} (TODR). At the real dose DW changes {wm[0]:.0f} March / {wa[0]:.0f} April "
               f"window(s); the move must be scaled x{next(s_ for s_, w_ in zip(scale, wm) if w_ >= 3)} (March) and "
               f"x{next(s_ for s_, w_ in zip(scale, wa) if w_ >= 3)} (April) to change the 3 windows a board read needs."),
        number=f"median distance {q_m[2]:.3f} vs move {nums(dw[1])[0]}; windows changed at x1 = {wm[0]:.0f} (Mar)",
        intervals="Quantiles over the 40 official windows per month; no intervals.",
        source="docs/research/ideas_R4/R4.md table 1b (official windows, A3 chain + C59 Tiny decode). Counts are of "
               "changed decoded windows only; no window truth is used.",
        notes="Decode threshold = r/(1+r) with r the window's expected IoU (about .47 on official windows).")


# ----------------------------------------------------------------------------------------------- tables
def table_T2():
    df, start = build_chain()
    A = adopted_rows()
    man = load_manifest().set_index("stem")
    recs = []
    for _, r in df.iterrows():
        rows_ = [int(x) for x in r.adopted_rows.split(";")]
        text = " ".join(A[i]["text"] for i in rows_)
        ev = " / ".join(A[i]["evidence"] for i in rows_)
        if len(ev) < 25 and r.file and r.file in man.index:
            ev = f"{ev}: {man.loc[r.file, 'hypothesis']}"
        ev = re.sub(r"\s+", " ", ev)
        ev = ev if len(ev) <= 200 else ev[:197] + "..."
        m = re.search(r"private (?:fc|expected) ([^;|]*)", text)
        priv = m.group(1).strip().rstrip(".") if m else ("0 (private rows only change April)" if False else "not recorded")
        if r["mode"] == "private":
            priv = "April rows only; see ADOPTED rows " + r.adopted_rows
        fc = "" if not np.isfinite(r.forecast_delta) else f"{f5(r.forecast_delta)} [{f5(r.band_lo_delta)}, {f5(r.band_hi_delta)}]"
        recs.append(dict(step=r.step, adopted_row=r.adopted_rows, component=r.component, offline_evidence=ev,
                         forecast_delta_band=fc, board_delta=("private only" if r["mode"] == "private" else f5(r.board_delta)),
                         board_note=r.how, private_expectation=priv))
    t = pd.DataFrame(recs)
    save_table("T2_ablation", t, "T2. Ablation: every adopted step from C03 (.88225) to C80 (.90088)", [
        "Board delta = public read of the step minus the previous level (exact, one read; one read's noise for state pieces "
        "is about .00001, for queue pieces about one window = .00375 x share).",
        "Forecast [band] = written in the manifest before the upload, relative to the previous level (instrument CI or judgmental).",
        "Private expectation = the value written in ADOPTED.md when the piece was adopted; 'not recorded' where none was written.",
        "Sources: docs/report/ADOPTED.md, artifacts/claude_ref/probes/manifest.csv, docs/report/board_calibration.csv."])


EFFORT_W = {"S": 1, "M": 2, "L": 3}


def effort_weight(s):
    s = str(s)
    w = EFFORT_W.get(s.split(" ")[0], 0)
    m = re.search(r"(\d+)k tokens", s)
    if m:
        w = max(w, 3 if int(m.group(1)) >= 500 else 2)
    hrs = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*h\b", s)]
    if hrs:
        w = max(w, 3 if max(hrs) >= 10 else 2)
    if "GPU" in s or "Kaggle" in s:
        w += .5
    return w


def table_T3():
    r = load_registry()
    x = r[r.verdict_class == "REFUTED"].copy()
    x["power"] = x.min_effect_test_could_see.map(lambda s: first_num(s))
    x = x[x.power.notna()].copy()
    x["w"] = x.effort.map(effort_weight)
    x = x.sort_values(["w", "power", "id"], ascending=[False, True, True])
    pick, per = [], {}
    for _, row in x.iterrows():
        if per.get(row.task, 0) >= 4:
            continue
        pick.append(row); per[row.task] = per.get(row.task, 0) + 1
        if len(pick) == 15:
            break
    t = pd.DataFrame(pick)
    cut = lambda s, n: (s if len(s) <= n else s[:n - 3] + "...")  # noqa: E731
    out = pd.DataFrame(dict(id=t.id, task=t.task, idea=t.mechanism.map(lambda s: cut(s, 150)),
                            best_version=t.best_version_tried.map(lambda s: cut(s, 90)), instrument=t.instrument,
                            result=t.result.map(lambda s: cut(s, 150)), smallest_effect_the_test_could_see=t.min_effect_test_could_see,
                            effort=t.effort))
    save_table("T3_negative_results", out, "T3. What did not work: 15 refuted ideas with the power of their test", [
        "Selection rule (mechanical): REFUTED rows of REGISTRY.tsv with a numeric 'min_effect_test_could_see'; sorted by "
        "effort (L > M > S; >= 500k tokens or >= 10 h counts as L, GPU/Kaggle +.5), then by the smallest detectable "
        "effect (most powerful test first), then id; at most 4 rows per task.",
        f"Pool: {len(r[r.verdict_class == 'REFUTED'])} REFUTED rows, {len(x)} with a numeric power.",
        "Power = the smallest effect (S_total units unless stated) the test could have detected; a REFUTED verdict is only "
        "as strong as this number. Source: docs/research/registry/REGISTRY.tsv."])


def table_T4():
    man = load_manifest()
    cal, n_raw = load_calibration()
    reads = man[man.public_num.notna()]
    per_day = reads.groupby("read_day").size()
    not_up = man[man.public.str.contains("NOT_UPLOADED", na=False)]
    by_kind = cal.groupby("kind").agg(reads=("file", "size"), in_band=("in_band_calc", "mean"),
                                      sign_agree=("sign_calc", lambda s: (s == "agree").mean()))
    rows = [
        dict(item="board reads with a written forecast (board_calibration.csv)", value=n_raw, note=""),
        dict(item="  in the pre-written band (recomputed)", value=f"{cal.in_band_calc.mean():.0%}", note=""),
        dict(item="  forecast and board agree in sign (abs change >= .00001)", value=f"{(cal.sign_calc == 'agree').mean():.0%}",
             note="rows without a forecast change counted as not agreeing"),
        dict(item="probe manifest rows", value=len(man), note="artifacts/claude_ref/probes/manifest.csv"),
        dict(item="  read on the board", value=len(reads), note=f"{len(not_up)} prepared but not uploaded"),
        dict(item="  UTC days with at least one read", value=per_day.size, note=f"{per_day.index.min()} to {per_day.index.max()}"),
        dict(item="  most reads on one UTC day", value=int(per_day.max()), note="cap 5 per UTC day (RULES.md 7)"),
        dict(item="  median reads per active day", value=float(per_day.median()), note=""),
    ]
    for k_, r_ in by_kind.iterrows():
        rows.append(dict(item=f"reads of kind: {k_}", value=int(r_.reads),
                         note=f"in band {r_.in_band:.0%}, sign agree {r_.sign_agree:.0%}"))
    # compute records that exist as receipts or logs
    comp = []
    for name, key in (("FV-INEL W.3 March, 30 members", "fvi_mar_w03"), ("FV-INEL W1.0 March, 30 members", "fvi_mar_w10"),
                      ("FV-INEL LW March, 30 members", "fvi_mar_lw"), ("FV-INEL W.3 April, 30 members", "fvi_apr_w03"),
                      ("FV-INEL W1.0 April, 30 members", "fvi_apr_w10")):
        info = fvi(P[key])
        comp.append(dict(item=f"GPU training: {name}", value=f"{info['minutes'] / 60:.1f} h",
                         note=f"{', '.join(info['devices'])}; FVI_INFO in {P[key].relative_to(ROOT).as_posix()}"))
    comp.append(dict(item="CPU rebuild of C47 from code (base of the final)", value="about 9 h",
                     note="docs/report/FINAL_PLAN_2026-10-21.md section 1; docs/report/REPRO_README.md"))
    comp.append(dict(item="CPU build entry from shipped weights (file B qualification)", value="2570 s",
                     note="docs/report/REPRO_README.md (peak 4.4 GiB)"))
    reg = load_registry()
    gpu_h = []
    for _, r_ in reg.iterrows():
        for m in re.finditer(r"GPU ~?(\d+(?:\.\d+)?)\s*h", r_.effort):
            gpu_h.append(float(m.group(1)))
    comp.append(dict(item="registry rows with an explicit GPU-hour effort", value=f"{len(gpu_h)} rows, {sum(gpu_h):.0f} h",
                     note="lower bound; most rows give effort as S/M/L or tokens"))
    comp.append(dict(item="total GPU and CPU hours of the project", value="not tallied",
                     note="not logged in one place: VM time (Vast, 2 x RTX 5070, until 10-17), Kaggle weekly quota "
                          "(used up 10-20), laptop RTX 3050 Ti. Recipe: sum the job logs under exp/*/logs and the "
                          "Kaggle kernel run times (exp/C_backbone/KAGGLE.md)."))
    t = pd.DataFrame(rows + comp)
    t["value"] = t["value"].astype(str)
    save_table("T4_board_reads_compute", t, "T4. Board use and compute", [
        "Every upload had a manifest line with hypothesis, prediction and band written first (D-0069, D-0068).",
        "Sources: docs/report/board_calibration.csv, artifacts/claude_ref/probes/manifest.csv, FVI_INFO log lines, "
        "REPRO_README.md, REGISTRY.tsv effort column.",
        "The early uploads before C01 appear only where they have a calibration row."])
    pd.DataFrame(dict(read_day=per_day.index, reads=per_day.values)).to_csv(OUT / "T4_reads_per_day.csv", index=False,
                                                                            lineterminator="\n")


def inel_counts():
    """Recompute INEL x scored-target intersections from the local cache (label-free), and compare with the stored JSONs."""
    sys.path.insert(0, str(ROOT / "exp/L_fvinel"))
    import inel_count as ic  # noqa: E402  (exp/L_fvinel/inel_count.py, reads masks only)
    res = {}
    for split, key in (("validation", "inel_count_v"), ("private", "inel_count_p")):
        tot = dict(inel=0, target=0, gap=0, inel_and_target=0, inel_and_gap=0, target_not_elig=0)
        for pn in PANELS:
            r = ic.count(pn, split)
            tot["inel"] += r["inel"]; tot["target"] += r["target"]; tot["gap"] += r["gap"]
            tot["inel_and_target"] += r["check_inel_and_target"]; tot["inel_and_gap"] += r["check_inel_and_gap"]
            tot["target_not_elig"] += r["check_target_not_elig"]
        stored = json.loads(P[key].read_text())["total"]
        tot["matches_stored_json"] = (stored["inel"] == tot["inel"] and stored["check_inel_and_target"] == tot["inel_and_target"])
        res[split] = tot
    return res


def table_T5():
    ic = inel_counts()
    man = load_manifest()
    per_day = man[man.public_num.notna()].groupby("read_day").size()
    v, p = ic["validation"], ic["private"]
    inel_txt = (f"COMPUTED: INEL cells x scored target cells = {v['inel_and_target']} (March, {v['inel']:,} INEL vs "
                f"{v['target']:,} target cells) and {p['inel_and_target']} (April, {p['inel']:,} vs {p['target']:,}); INEL x gap "
                f"= {v['inel_and_gap']} / {p['inel_and_gap']}; target cells outside eligibility = {v['target_not_elig']} / "
                f"{p['target_not_elig']}. Recomputed from the cache by exp/L_fvinel/inel_count.py; equal to the stored JSONs: "
                f"{v['matches_stored_json'] and p['matches_stored_json']}.")
    items = [
        ("Task 2 online rule: a window with origin T uses only data stamped <= T on its date, earlier dates of the split, and train",
         "RULES.md 2; forum 742068", "docs/report/TASK2_ONLINE_AUDIT_2026-10-11.md: no violation found; poisoned-future test in asm2_run_all.sh (--poison)",
         "PASS (audit 10-11); TODO before the final build: extend poison to ramp_states, window_history, window_index and every origin (ADOPTED.md follow-ups)"),
        ("April may use released March data (Task 2); March never uses April data", "RULES.md 2 (D-0062); DECISIONS_DIGEST Data use",
         "audit 10-11 section 1: no piece loads two evaluation splits together for a March forecast", "PASS; TODO: mechanical check that no March Task 2 step opens an April file (final_solution/checks)"),
        ("Task 1 and 3 offline: any released observation of the split; INEL cells (released, ineligible) used as labels", "RULES.md 2 (Tasks 1 and 3)",
         inel_txt, "PASS (computed)"),
        ("No IDs, seeds, row order, window numbering or file timestamps as features", "RULES.md 3 and 5", "by construction of the feature code; registry has no row using them",
         "TODO: grep check in final_solution/checks over the build's feature code"),
        ("No matching of a hidden day to a train day to copy values", "RULES.md 3; CLAUDE.md hard rules", "no method in ADOPTED.md matches days; all target-month learning uses the split's own released cells",
         "DECLARED"),
        ("Upload cap 5 per UTC day; each upload with a reason and a pre-written forecast", "RULES.md 7; D-0069; D-0068",
         f"manifest: most reads on one UTC day = {int(per_day.max())}", "PASS (computed from the manifest)"),
        ("Organiser-only files (queue truth, boundary flux, path-flow truth) never used", "RULES.md 3", "no code path reads them", "DECLARED"),
        ("Declared April methods: a March board verdict of a METHOD chooses it for April rows; April gap and FV methods fitted on April's own view",
         "DECISIONS_DIGEST rulings 2026-10-15 (a), 10-17 (precedence), 10-18 (April gap methods)", "ADOPTED.md rows 22, 23, 27-35 state 'declared method'", "DECLARED in the report"),
        ("U1 selection-conditioned inputs (sccnn flags, type A, persistence cap, band raise; about +.0045) read nothing after T",
         "FINAL_PLAN decision 3; ADOPTED.md online follow-ups", "disclosed plainly", "DECLARED"),
        ("External data: PeMS screen (failed gate, not in any file); Chronos-Bolt zero-shot screen (not used)", "RULES.md 3",
         "RULES.md disclosures 2026-10-13", "DECLARED; TODO(C): Chronos-Bolt repo id and revision"),
        ("Pretrained weights: timm ConvNeXt-Tiny (ImageNet-1k) in the ongoing member; TabPFN-v2 only in file B (S0631 off by default)",
         "RULES.md 3 (D-0048)", "pins and SHA256 in RULES.md and artifacts/claude_ref/c_bb/weights_manifest.json", "DECLARED"),
        ("No code or predictions shared outside the team; no public notebook copied", "RULES.md 3", "", "DECLARED"),
        ("Reproducibility: rebuilt file within 1% of the selected file's private score", "RULES.md 6; D-0051",
         "C47 rebuilt from code, state rows max diff 4e-13 (REPRO_README.md); C80 rebuild from saved weights in final_solution/",
         "PASS for C47 (C80 - C47 = +.00312 public, inside the gate); TODO: C80 rebuild parity"),
    ]
    t = pd.DataFrame(items, columns=["item", "rule_source", "evidence", "status"])
    save_table("T5_compliance", t, "T5. Compliance checklist (skeleton)", [
        "Status: PASS = checked with the evidence named; COMPUTED numbers are produced by make_figures.py; DECLARED = "
        "stated in the report, no mechanical check; TODO = what is still to run and how.",
        "Sources: RULES.md, DECISIONS_DIGEST.md, docs/report/ADOPTED.md, docs/report/TASK2_ONLINE_AUDIT_2026-10-11.md."])
    pd.DataFrame([dict(split=s, **{k: v for k, v in d.items()}) for s, d in ic.items()]).to_csv(
        OUT / "T5_inel_target_intersection.csv", index=False, lineterminator="\n")
    return ic


# ----------------------------------------------------------------------------------------------- captions
def write_captions(ic):
    order = ["F1_pipeline", "F2_waterfall", "F3_forecast_vs_board", "F4_transfer_rule", "F5_registry", "F6_spacetime",
             "F7_fundamental_diagram", "F8_physics_anatomy", "F9_error_anatomy", "F10_label_dose",
             "F11_instrument_validation", "F12_seed_curve", "F13_decode_bottleneck", "F14_bootstrap"]
    L = ["# Figure captions (generated by make_figures.py; do not edit by hand, rerun the script)", "",
         "Final file C80, public .90088 (sha 3e27d66c). One claim per figure; the number is computed by the script from repo files.",
         "Files: figures/out/<name>.png (200 dpi), .pdf, and the CSV of the plotted numbers (<name>.csv or <name>_<part>.csv).", ""]
    for k in order:
        if k not in CAPTIONS:
            continue
        c = CAPTIONS[k]
        L += [f"## {k}: {c['title']}", "", f"Caption. {c['claim']}", "", f"- Number it proves: {c['number']}",
              f"- Intervals: {c['intervals']}", f"- Data: {c['source']}"]
        if c.get("notes"):
            L.append(f"- Notes: {c['notes']}")
        L.append("")
    L += ["## Data rules", "",
          "- board_calibration.csv: a row with more than 8 fields (unquoted comma in the last prose cell) gets its extra "
          "fields joined back into that cell; a numeric cell is its FIRST number; band = first two numbers; sign and in_band = first word "
          "(lower case); the remaining prose is kept as `note` in the CSVs. in_band is recomputed (band_lo <= public <= "
          "band_hi) and sign agreement is recomputed with a .00001 zero tolerance; the recomputed values are used.",
          "- Kind of change: fixed regular expressions on the file name (KIND_RULES in make_figures.py), first match wins.",
          "- The adopted chain (F2, T2) is the CHAIN list in make_figures.py: one entry per ADOPTED.md row in the board "
          "file, in adoption order; the level after each step is its public read, except C77 (unread; level = C76 + the "
          "read of the same rows on C75) and the private-only steps C74P and C80.",
          "- F6, F7, F8 proxy: released cells only; predictions = INEL-FOLD chain at held-out INEL cells; the script asserts "
          "that the cache holds no value at scored target cells.",
          ""]
    if ic:
        v, p = ic["validation"], ic["private"]
        L += ["## Computed compliance count (T5)", "",
              f"Scored target cells that are INEL cells: March {v['inel_and_target']}, April {p['inel_and_target']} "
              f"(INEL cells {v['inel']:,} / {p['inel']:,}; scored target cells {v['target']:,} / {p['target']:,}).", ""]
    (HERE / "CAPTIONS.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------- main
def _f1_f14(name):
    """F1 and F14 live in make_f1_f14.py (worker F2); they share this module's style, save() and CAPTIONS."""
    def job():
        sys.path.insert(0, str(HERE))
        import make_f1_f14 as X
        return getattr(X, name)(sys.modules[__name__])
    return job


JOBS = {"F1": _f1_f14("fig_F1"), "F14": _f1_f14("fig_F14"), "F2": fig_F2, "F3": fig_F3, "F5": fig_F5, "F4": fig_F4, "F10": fig_F10, "F11": fig_F11, "F12": fig_F12,
        "F13": fig_F13, "F9": fig_F9, "F8": fig_F8, "F6": fig_F6, "F7": fig_F7,
        "T2": table_T2, "T3": table_T3, "T4": table_T4, "T5": table_T5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = a.only or list(JOBS)
    ic = None
    for j in jobs:
        res = JOBS[j]()
        if j == "T5":
            ic = res
        print("done", j, flush=True)
    if a.only is None:
        write_captions(ic)
    print("MADE", MADE)


if __name__ == "__main__":
    main()
