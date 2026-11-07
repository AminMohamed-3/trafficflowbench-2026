"""Private board (April) after the close, 2026-11-07: how much of each public (March) gain carried.

  python report/figures/private_board.py            (from final_solution/)

Inputs (report/data/):
  submissions_2026-11-07.csv   our Kaggle submissions with public and private S_total (kaggle CLI export)
  leaderboard_2026-11-07.csv   final private leaderboard, top 16 teams
and figures/out/F2_waterfall.csv, figures/out/F3_forecast_vs_board.csv.
Outputs (figures/out/): P1_public_private.csv (every read file), P2_waterfall_private.csv (adopted steps with the private
delta of the same step), P3_by_kind.csv (carry ratio by kind of change), P4_leaderboard.csv, P_summary.json, P1/P2 figures.
"""
import csv, json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent
REP = FIG.parent
OUT = FIG / "out"
DATA = REP / "data"


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wr(p, rows, cols):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


subs = rd(DATA / "submissions_2026-11-07.csv")
priv = {r["fileName"].removesuffix(".csv"): fnum(r["privateScore"]) for r in subs}
pub = {r["fileName"].removesuffix(".csv"): fnum(r["publicScore"]) for r in subs}
calib = {r["file"]: r for r in rd(OUT / "F3_forecast_vs_board.csv")}

# P1: every file we read on the board
p1 = []
for r in sorted(subs, key=lambda r: r["date"]):
    f = r["fileName"].removesuffix(".csv")
    c = calib.get(f, {})
    p1.append(dict(file=f, date=r["date"][:16], kind=c.get("kind", ""), forecast=c.get("forecast", ""),
                   public=pub[f], private=priv[f], gap=round(pub[f] - priv[f], 5) if None not in (pub[f], priv[f]) else ""))
wr(OUT / "P1_public_private.csv", p1, list(p1[0]))

# P2: adopted steps of F2; parent = latest earlier step whose public level equals this step's level_before
wf = rd(OUT / "F2_waterfall.csv")
p2, seen = [], []
for r in wf:
    f = r["file"]
    lb = fnum(r["level_before"])
    par = next((s for s in reversed(seen) if s["file"] and abs(fnum(s["level_after"]) - lb) < 5e-6), None)
    pa, pb = priv.get(f), priv.get(par["file"]) if par else None
    d_priv = round(pa - pb, 5) if None not in (pa, pb) else None
    d_pub = fnum(r["board_delta"])
    p2.append(dict(step=r["step"], file=f, parent=par["file"] if par else "", component=r["component"],
                   public_before=lb, public_after=fnum(r["level_after"]), d_public=d_pub, d_forecast=fnum(r["forecast_delta"]),
                   private_before=pb, private_after=pa, d_private=d_priv,
                   carry=round(d_priv / d_pub, 2) if d_priv is not None and d_pub else None))
    seen.append(r)

# Corrections where the step's file is not a child of the previous step's file (critic review 2026-11-07, B2):
#  C17 c17_k6_s0447_state_on_c15 is built on C15 (no Q0436): its pair is C15. C26 c26_final_A_build re-adds Q0436 on top
#  of C19: its private change = Q0436 (+.00714, booked once at C16, where C16 - C15 measures it) + its own state change.
#  C77 (W1.0, March rows) was never read in its real form; C76, the C77 class and C79 have the same April rows (private
#  .89874), so C77 and C79 change April by 0.
FIX = {"C10": ("c03_queue_cnn_adapt_ongoing", None, ""),
       "C17": ("c15_k6_s08_phys4_gap_selcond_on_c03", None, "pair C15 (built on C15, without Q0436)"),
       "C26": ("c19_ong2_sccnnA_on_c17", -(priv["c16_k6_q0436_onset_on_c15"] - priv["c15_k6_s08_phys4_gap_selcond_on_c03"]),
               "C26 - C19 minus Q0436 (re-added here, booked at C16)"),
       "C77": (None, 0.0, "unread; public level is the forecast; April rows unchanged"),
       "C79": (None, 0.0, "April rows equal C76 (private .89874)")}
for r in p2:
    k = r["step"].split()[0]; r["note"] = ""
    if k in FIX:
        par, adj, note = FIX[k]; r["note"] = note
        if par:
            r["parent"] = par; r["private_before"] = priv[par]
            r["d_private"] = round(priv[r["file"]] - priv[par] + (adj or 0), 5)
        else:
            r["d_private"] = adj
        r["carry"] = round(r["d_private"] / r["d_public"], 2) if r["d_public"] else None
# private level along the chain = C03 private + cumulative corrected private changes (ends at the scored file's private)
lvl_p = priv["c03_queue_cnn_adapt_ongoing"]
for r in p2:
    if r["d_private"] is not None:
        lvl_p += r["d_private"]
    r["private_cum"] = round(lvl_p, 5)
wr(OUT / "P2_waterfall_private.csv", p2, list(p2[0]))

# P3: carry by component (sum of private over sum of public on steps with both)
p3 = []
for comp in sorted({r["component"] for r in p2}):
    rs = [r for r in p2 if r["component"] == comp and r["d_private"] is not None and r["d_public"] is not None]
    if not rs:
        continue
    sp, sv = sum(r["d_public"] for r in rs), sum(r["d_private"] for r in rs)
    p3.append(dict(component=comp, steps=len(rs), sum_public=round(sp, 5), sum_private=round(sv, 5),
                   carry=round(sv / sp, 2) if sp else None,
                   same_sign=sum((r["d_private"] > 0) == (r["d_public"] > 0) for r in rs)))
wr(OUT / "P3_by_kind.csv", p3, list(p3[0]))

lb = rd(DATA / "leaderboard_2026-11-07.csv")
p4 = [dict(rank=i + 1, team=r["teamName"], private=fnum(r["score"]), last_submission=r["submissionDate"][:10]) for i, r in enumerate(lb)]
wr(OUT / "P4_leaderboard.csv", p4, list(p4[0]))

def lvl(f):
    return pub[f], priv[f]
summ = dict(
    final_file="c_isinel_s7c_mar_on_c77 (C79; C80 differs only in April rows and was not uploaded)",
    rank=next(r["rank"] for r in p4 if r["team"] == "Amin Mohamed"), teams_listed=len(p4),
    final=dict(zip(("public", "private"), lvl("c_isinel_s7c_mar_on_c77"))),
    c47=dict(zip(("public", "private"), lvl("c_b6l24k_on_c44"))),
    c03=dict(zip(("public", "private"), lvl("c03_queue_cnn_adapt_ongoing"))) if "c03_queue_cnn_adapt_ongoing" in pub else None,
    baseline=dict(zip(("public", "private"), lvl("0120_u1_canonical"))),
    best_private_read=max(((f, priv[f]) for f in priv if priv[f] is not None), key=lambda x: x[1]),
    winner=p4[0]["private"],
)
for a, b in (("c47", "final"), ("baseline", "final")):
    dp = summ[b]["public"] - summ[a]["public"]; dv = summ[b]["private"] - summ[a]["private"]
    summ[f"gain_{a}_to_final"] = dict(public=round(dp, 5), private=round(dv, 5), carry=round(dv / dp, 3))
ratios = [r["carry"] for r in p2 if r["carry"] is not None]
summ["step_carry_median"] = statistics.median(ratios) if ratios else None
# after C47 by component (rows after the C47 step), and the two large early April movers
i47 = next(i for i, r in enumerate(p2) if r["step"].startswith("C47"))
after = {}
for r in p2[i47 + 1:]:
    if r["d_private"] is None or r["d_public"] is None:
        continue
    a = after.setdefault(r["component"], dict(steps=0, public=0.0, private=0.0))
    a["steps"] += 1; a["public"] += r["d_public"]; a["private"] += r["d_private"]
for a in after.values():
    a["public"], a["private"] = round(a["public"], 5), round(a["private"], 5)
    a["carry"] = round(a["private"] / a["public"], 2) if a["public"] else None
summ["after_c47_by_component"] = after
summ["big_april_steps"] = [dict(step=r["step"], d_public=r["d_public"], d_private=r["d_private"]) for r in p2
                           if r["d_private"] is not None and abs(r["d_private"]) > .005]
# rank agreement public vs private (Spearman, average ranks)
def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rk(xs), rk(ys); ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return round(cov / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5, 2)
both = [r for r in p1 if r["public"] is not None and r["private"] is not None]
late = [r for r in both if r["date"] >= "2026-10-12"]
summ["spearman"] = dict(all=dict(n=len(both), rho=spearman([r["public"] for r in both], [r["private"] for r in both])),
                        since_10_12=dict(n=len(late), rho=spearman([r["public"] for r in late], [r["private"] for r in late])))
summ["uploads"] = dict(total=len(subs))
(OUT / "P_summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")

# P1 figure: public vs private for every read file; P2: cumulative public and private along the adopted chain
fig, ax = plt.subplots(figsize=(6.2, 5.2))
xs = [r for r in p1 if r["public"] and r["private"] and r["public"] > .85]
ax.scatter([r["public"] for r in xs], [r["private"] for r in xs], s=14, c="#4c78a8", alpha=.75)
lo, hi = .875, .905
ax.plot([lo, hi], [lo, hi], color="#999", lw=.8, ls="--")
fin = next(r for r in p1 if r["file"] == "c_isinel_s7c_mar_on_c77")
ax.scatter([fin["public"]], [fin["private"]], s=60, c="#e45756", zorder=3, label="final (C79)")
ax.set_xlabel("public S_total (March)"); ax.set_ylabel("private S_total (April)")
ax.set_title("Every board read: public vs private"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "P1_public_private.png", dpi=160); plt.close(fig)

ch = [r for r in p2 if r["d_private"] is not None]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(range(len(ch)), [r["public_after"] for r in ch], marker="o", ms=3, label="public (March)", color="#4c78a8")
ax.plot(range(len(ch)), [r["private_cum"] for r in ch], marker="o", ms=3, label="private (April)", color="#e45756")
ax.set_xticks(range(len(ch))); ax.set_xticklabels([r["step"].split()[0] for r in ch], rotation=70, fontsize=7)
ax.set_ylabel("S_total"); ax.set_title("Adopted chain: public and private level after each step"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "P2_chain_public_private.png", dpi=160); plt.close(fig)
print(json.dumps(summ, indent=1))
