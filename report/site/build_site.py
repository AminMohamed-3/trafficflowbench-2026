"""Build the solution website: report/site/template.html + data -> report/site/index.html (one self-contained file).

  python report/site/build_site.py          (from final_solution/; run report/figures/private_board.py first)

Data: report/figures/out/*.csv|json (the report's figure tables, P* = private board), report/data/registry.tsv (the idea
registry). The space-time map gets only the cell types and our error at held-out cells: no released values are published.
"""
import csv, json
from pathlib import Path

SITE = Path(__file__).resolve().parent
REP = SITE.parent
OUT = REP / "figures" / "out"


def rd(name, base=OUT, delim=","):
    with open(base / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def num(x, nd=6):
    try:
        v = float(x)
        return round(v, nd)
    except (TypeError, ValueError):
        return None


D = {}
D["summary"] = json.loads((OUT / "P_summary.json").read_text(encoding="utf-8"))
D["leaderboard"] = [dict(rank=int(r["rank"]), team=r["team"], private=num(r["private"], 5)) for r in rd("P4_leaderboard.csv")]
D["reads"] = [dict(file=r["file"], date=r["date"], kind=r["kind"], fc=num(r["forecast"]), pub=num(r["public"], 5),
                   priv=num(r["private"], 5)) for r in rd("P1_public_private.csv")]
D["chain"] = [dict(step=r["step"], file=r["file"], comp=r["component"], pub=num(r["public_after"], 5), dpub=num(r["d_public"], 5),
                   dfc=num(r["d_forecast"], 5), priv=num(r["private_cum"], 5) if r["d_private"] else None, dpriv=num(r["d_private"], 5),
                   note=r["note"]) for r in rd("P2_waterfall_private.csv")]
# the four steps before C03 (REPORT Appendix A), with their private change from the same files
P1 = {r["file"]: r for r in rd("P1_public_private.csv")}
EARLY = [("U4 Q03 onset classifier", "onset", "0120_u1_canonical", "0145_u4_q03_only", .0127),
         ("Q04 tree model for ongoing", "ongoing", "0145_u4_q03_only", "0164_q04_ongoing_only", .0112),
         ("F4 state base", "state", "0164_q04_ongoing_only", "0193_f4_state_only", .0076),
         ("C03 adapted CNN for ongoing", "ongoing", "0193_f4_state_only", "c03_queue_cnn_adapt_ongoing", .0039)]
D["early"] = [dict(step=a, comp=c, dpub=round(float(P1[f]["public"]) - float(P1[par]["public"]), 5),
                   dpriv=round(float(P1[f]["private"]) - float(P1[par]["private"]), 5), dfc=fc) for a, c, par, f, fc in EARLY]
wf = {r["step"]: r for r in rd("F2_waterfall.csv")}
for c in D["chain"]:
    w = wf.get(c["step"], {})
    c["lo"], c["hi"], c["rows"], c["how"] = num(w.get("band_lo_delta")), num(w.get("band_hi_delta")), w.get("adopted_rows", ""), w.get("how", "")
D["carry"] = [dict(comp=r["component"], steps=int(r["steps"]), pub=num(r["sum_public"], 5), priv=num(r["sum_private"], 5),
                   carry=num(r["carry"], 2), same=int(r["same_sign"])) for r in rd("P3_by_kind.csv")]
D["calib"] = [dict(file=r["file"], kind=r["kind"], dfc=num(r["d_pred"]), dboard=num(r["d_board"]), lo=num(r["band_lo"]),
                   hi=num(r["band_hi"]), fc=num(r["forecast"]), inband=r["in_band_calc"] == "True")
              for r in rd("F3_forecast_vs_board.csv") if r["d_pred"] and r["d_board"]]
D["pipeline"] = [dict(id=r["block"], lane=r["lane"], title=r["title"], body=r["body"], train=r["reads_train"] == "1",
                      month=r["reads_month"] == "1", online=r["reads_online"] == "1", mar4apr=r["reads_mar4apr"] == "1")
                 for r in rd("F1_pipeline.csv")]
D["anatomy"] = [dict(slice=r["slice"], cell_mar=num(r["cell_share_mar"], 3), cell_apr=num(r["cell_share_apr"], 3),
                     imp_mar=num(r["improvable_mar"], 3), imp_apr=num(r["improvable_apr"], 3)) for r in rd("F9_error_anatomy.csv")]
D["dose"] = [dict(step=r["step"], inc=num(r["board_increment"]), fc=num(r["forecast_increment"]), how=r["how"]) for r in rd("F10_label_dose_board.csv")]
D["decode"] = dict(distance=[dict(q=r["quantile"], mar=num(r["march"], 3), apr=num(r["april"], 3)) for r in rd("F13_decode_bottleneck_distance.csv")],
                   dose=[dict(scale=num(r["dw_scale"]), mar=num(r["windows_march"]), apr=num(r["windows_april"])) for r in rd("F13_decode_bottleneck_dose.csv")])
D["fd"] = [dict(src=r["source"], regime=r["regime"], cells=int(r["cells"]), loss=num(r["fd_loss"], 4)) for r in rd("F7_fundamental_diagram.csv")]
D["boot"] = [dict(piece=r["piece"], inst=r["instrument"], row=r["adopted_row"], month=r["month"], point=num(r["point"], 7),
                  lo=num(r["lo95"], 7), hi=num(r["hi95"], 7), p=num(r["p_gt0"], 3)) for r in rd("F14_bootstrap_summary.csv")]
D["registry_by_task"] = rd("F5_registry_by_task.csv")
D["negatives"] = rd("T3_negative_results.csv")
D["compute"] = [r for r in rd("T4_board_reads_compute.csv") if r["value"] not in ("TODO", "not tallied")]
D["registry"] = [{k: r[k] for k in ("id", "task", "mechanism", "info_source", "learns_from_target_month", "instrument", "result",
                                    "min_effect_test_could_see", "verdict_class", "why_class")}
                 for r in rd("registry.tsv", REP / "data", "\t")]
# F6: 44 stations x 288 five-minute steps of one March day. Published: the cell type (0 released, 1 held-out INEL, 2 scored
# target) and our error (ours - released, 0.1 km/h) at held-out INEL cells only (F6_spacetime.csv holds nothing else).
st = rd("F6_spacetime.csv")
ns, nt = 1 + max(int(r["station_rank"]) for r in st), 1 + max(int(r["step"]) for r in st)
err, mask = [[None] * nt for _ in range(ns)], [[0] * nt for _ in range(ns)]
for r in st:
    i, t = int(r["station_rank"]), int(r["step"])
    mask[i][t] = 2 if r["scored_target"] == "True" else (1 if r["inel"] == "True" else 0)
    if mask[i][t] == 1 and r["err_kmh"]:
        err[i][t] = num(r["err_kmh"], 1)
D["spacetime"] = dict(panel="D12_I405_S", date="2031-03-18", stations=ns, steps=nt, mask=mask, err=err)

html = (SITE / "template.html").read_text(encoding="utf-8")
blob = json.dumps(D, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
import os
repo = os.environ.get("REPO_URL", "https://github.com/AminMohamed-3/trafficflowbench-2026").rstrip("/")
status = os.environ.get("CHAIN_STATUS", 'passed on 2026-11-09. Rows 19 to 35 ran in 11.6 h on the laptop from a fresh folder made only from this repository, the release assets and the competition data, with every Python file open audited: nothing outside that folder was read except the two stored submission files used for the final comparison. Every rebuilt member set matched development by sha256, and C79 and C80 came out byte-equal to the submitted files (VERIFY_PASS). The base C47 was rebuilt separately in WSL (VERIFY_PASS, state rows within 4e-13).')
html = html.replace("__CHAIN_STATUS__", status)
html = html.replace("__REPO_URL__", repo).replace("__REPO_NAME__", repo.rsplit("/", 1)[-1])
(SITE / "index.html").write_text(html.replace("/*__DATA__*/null", blob), encoding="utf-8", newline="\n")
print(f"index.html {len((SITE / 'index.html').read_bytes()) / 1e6:.2f} MB; registry {len(D['registry'])} rows; reads {len(D['reads'])}")
