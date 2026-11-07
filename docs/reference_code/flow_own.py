"""Claude helper 2026-10-04: I-S-OWN March (validation) state check of flow_krige vs retained F4 OWN predictions.
Band weights: pooled OPT weights (train held dates, all panels) from artifacts/claude_ref/flow/opt/opt_summary.json; no March fit.
Kriging input = the blinded OWN view (hidden + official targets are blank), so held-out labels never enter.
Usage: python docs/reference_code/flow_own.py"""
import sys, json, importlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_krige as FK
import tfb
tfb.__path__.insert(0, str(Path(P.P["D_worktree"]) / "src/tfb"))
own = importlib.import_module("tfb.lane_s.own_instrument")
binding = importlib.import_module("tfb.lane_d.f4_own_metadata_binding")
OUT = P.ROOT / "artifacts/claude_ref/flow/own"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = P.P
    w = np.asarray(json.loads((P.ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    ref_rows, new_rows, per = [], [], {}
    for item in items:
        view, tv, tq, score_target, hidden = own.load_item(P.ROOT, binding.bound_own_item(p, item))
        pv, pq = (np.load(P.D / "F4_predictions/OWN" / f"{item['panel']}_validation" / f"{n}.npy").astype(float) for n in ("speed", "flow"))
        N, den = FK.krige_log_flow(np.asarray(view.flow, float), FK.positions(view.stations))
        nq = FK.blend(pv, pq, N, np.asarray(score_target, bool), view.stations, w, den=den)
        a, _ = own.state_rows(view, tv, tq, pv, pq, score_target); b, _ = own.state_rows(view, tv, tq, pv, nq, score_target)
        ref_rows += a; new_rows += b
        la = view.stations.lanes.to_numpy(float); m = np.asarray(score_target, bool)
        gap_rows = m & np.asarray(view.gap, bool)
        per[item["panel"]] = dict(rmse_q_F4=float(np.sqrt(np.mean(((pq - tq) / la)[m] ** 2))), rmse_q_new=float(np.sqrt(np.mean(((nq - tq) / la)[m] ** 2))),
                                  share_changed=float((np.abs(nq - pq)[m] > 1e-6).mean()), share_block_rows=float(gap_rows.sum() / m.sum()),
                                  rmse_q_F4_nonblock=float(np.sqrt(np.mean(((pq - tq) / la)[m & ~gap_rows] ** 2))), rmse_q_new_nonblock=float(np.sqrt(np.mean(((nq - tq) / la)[m & ~gap_rows] ** 2))))
        print(item["panel"], per[item["panel"]], flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in ref_rows})
    res = own.paired_state(ref_rows, new_rows, panels, dates, groups)
    print(json.dumps(res, indent=1))
    (OUT / "own_march.json").write_text(json.dumps(dict(paired=res, per_panel=per, weights=w.tolist()), indent=1))


if __name__ == "__main__":
    main()
