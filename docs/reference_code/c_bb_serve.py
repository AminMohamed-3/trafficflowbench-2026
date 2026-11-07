"""C-bb 2026-10-08: official ongoing probabilities from a backbone serving model (board read: does a pretrained backbone
member transfer). Same inputs as the shipped sc members (ong2_sccnn_official.predict): qcnn_predict_adapt.official_x
(masked cache cube through T, raw window_history t-12..t-1, prior/pos/tod dropped, ongoing flag 1) plus selection flags
by official type (artifacts/claude_ref/ong2/official_types.csv): A (1,1,0), B (1,0,1), C (1,0,0). No per-date adaptation
(the model is a plain serving member fit through 2031-02-28, the shipped members' base cutoff).
Future poison: for the middle date of each split, every masked cell on later dates and on day d after its last origin
is set to 7; that date's predictions must not change.
Output: artifacts/claude_ref/c_bb/serve/<run>.parquet (window_id, timestamp, link_id, prob) + .json receipt.
Usage (VM): python c_bb_serve.py <run_name>"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import c_bb_arch
import qcnn_predict_adapt as qpa
import ong2_sccnn as S
FL = {"A": (1., 1., 0.), "B": (1., 0., 1.), "C": (1., 0., 0.)}
OUT = ROOT / "artifacts/claude_ref/c_bb"


def predict(model, data, d_str, drop, dev, types, amp):
    rows = []
    for p, v in data.items():
        wi = v["wi"][[str(pd.Timestamp(o).date()) == d_str for o in v["wi"].forecast_origin]]
        for w, t, x in qpa.official_x(v["c3"], v["rc"], v["stat"], v["st"], v["links"], wi, v["hist"], drop):
            x = S.fill(x, [FL[types[w.window_id]]])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pr = torch.sigmoid(model(x.to(dev)).float())[0].cpu().numpy()
            T = pd.Timestamp(w.forecast_origin)
            ts = [(T + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(6)]
            rows.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(v["links"])),
                                      "link_id": np.tile(v["links"], 6), "prob": pr.reshape(-1)}))
    return pd.concat(rows, ignore_index=True) if rows else None


def main(run, make=None, amp=None):
    """make / amp: optional overrides for non-c_bb_arch members (c_bb_tfm.py serve: transformer, fp32)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((OUT / "runs" / run / "config.json").read_text())
    assert cfg["fit_end"] == "2031-02-28" and cfg.get("sccnn"), cfg.get("fit_end")
    drop = cfg["drop"]; assert "prior" in drop
    model = (make or c_bb_arch.make)(cfg["spec"]).to(dev)
    model.load_state_dict(torch.load(OUT / "runs" / run / "model.pt", map_location=dev)); model.eval()
    amp = (cfg["spec"]["bb"] != "cnn" if amp is None else amp) and dev == "cuda"
    S.enable()
    types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv").set_index("window_id").typ.to_dict()
    allrows, pois = [], []
    for split in ("validation", "private"):
        data = qpa.load_split(split, False)
        dates = sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin})
        for k, d_str in enumerate(dates):
            got = predict(model, data, d_str, drop, dev, types, amp); allrows.append(got)
            print(split, d_str, "windows", got.window_id.nunique(), flush=True)
            if k == len(dates) // 2:
                saved = {p: v["c3"].copy() for p, v in data.items()}
                for p, v in data.items():
                    v["c3"][v["dates"] > d_str] = np.float16(7.)
                    di = [i for i, x in enumerate(v["dates"]) if x == d_str]
                    ts = [pd.Timestamp(o).hour * 12 + pd.Timestamp(o).minute // 5 for o in v["wi"].forecast_origin if str(pd.Timestamp(o).date()) == d_str]
                    if di and ts:
                        v["c3"][di[0], max(ts) + 1:] = np.float16(7.)
                again = predict(model, data, d_str, drop, dev, types, amp)
                dmax = float(np.abs(again.prob.to_numpy() - got.prob.to_numpy()).max())
                pois.append(dict(split=split, date=d_str, max_abs_dprob=dmax)); print("POISON", pois[-1], flush=True)
                assert dmax < 1e-6, "future poison changed predictions"
                for p, v in data.items():
                    v["c3"] = saved[p]
    df = pd.concat(allrows, ignore_index=True)
    types_seen = pd.Series([types[w] for w in df.window_id.unique()]).value_counts().to_dict()
    (OUT / "serve").mkdir(parents=True, exist_ok=True); f = OUT / "serve" / f"{run}.parquet"; df.to_parquet(f)
    rec = dict(run=run, model_sha256=cfg.get("sha256"), rows=len(df), windows=int(df.window_id.nunique()), types=types_seen,
               poison=pois, flags=FL, inputs="qcnn_predict_adapt.official_x", sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    f.with_suffix(".json").write_text(json.dumps(rec, indent=1)); print("saved", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
