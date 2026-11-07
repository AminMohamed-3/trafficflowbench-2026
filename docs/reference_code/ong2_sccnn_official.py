"""ong2 2026-10-05: official ongoing probabilities from the selection-conditioned CNN (ong2_sccnn.py), C03 adaptation
(qcnn_predict_adapt.finetune: same-split dates < d of the same day type, no cap as shipped, seed = date index; fine-tune
flags from pseudo labels with P_UNK dropout), official inputs (qcnn_predict_adapt.official_x) plus selection flags by
official type (ong2_official_types.csv): A (1, 1, 0), B (1, 0, 1), C (1, 0, 0). Seeds averaged.
Optional --poison: for the middle date of each split, garbage on later dates and on day d after its last T; predictions
for that date must not change.
Output: <out>.parquet (window_id, timestamp, link_id, prob).
Usage: python ong2_sccnn_official.py <run_dir[,run_dir]> <out.parquet> [--poison] [--cap=N]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from qcnn_train import Net, CTX
import qcnn_predict_adapt as qpa
import ong2_sccnn as S

FL = {"A": (1., 1., 0.), "B": (1., 0., 1.), "C": (1., 0., 0.)}


def predict(models, data, d_str, drop, dev, types):
    rows = []
    for p, v in data.items():
        wi = v["wi"][[str(pd.Timestamp(o).date()) == d_str for o in v["wi"].forecast_origin]]
        for w, t, x in qpa.official_x(v["c3"], v["rc"], v["stat"], v["st"], v["links"], wi, v["hist"], drop):
            x = S.fill(x, [FL[types[w.window_id]]])
            with torch.no_grad():
                pr = torch.stack([torch.sigmoid(m(x.to(dev)))[0] for m in models]).mean(0).cpu().numpy()
            T = pd.Timestamp(w.forecast_origin)
            ts = [(T + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(6)]
            rows.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(v["links"])),
                                      "link_id": np.tile(v["links"], 6), "prob": pr.reshape(-1)}))
    return pd.concat(rows, ignore_index=True)


def main(runs, out, *flags):
    qpa.configure_determinism()
    poison = "--poison" in flags; cap = next((int(x[6:]) for x in flags if x.startswith("--cap=")), None)
    S.enable()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv").set_index("window_id").typ.to_dict()
    bases = []
    for r in runs.split(","):
        cfg = json.loads((Path(r) / "config.json").read_text()); assert cfg.get("sccnn"); drop = cfg["drop"]
        m = Net(cin=CTX * 5 + 16).to(dev); m.load_state_dict(torch.load(Path(r) / "model.pt", map_location=dev)); m.eval(); bases.append(m)
    allrows = []
    for split in ("validation", "private"):
        data = qpa.load_split(split, False)
        dates = sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin})
        for k, d_str in enumerate(dates):
            S.MODE = "train"; S._rng = np.random.default_rng(1000 + k)
            models = [qpa.checkpointed_adaptation(b, r, split, d_str, k,
                      lambda b=b: qpa.finetune(b, data, d_str, drop, dev, seed=k, cap=cap),
                      settings=dict(daytype=True, cap=cap, drop=list(drop), selection_dropout=S.P_UNK,
                                    selection_rng_seed=1000+k, ensemble_position=j,
                                    selection_source_sha256=qpa.hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest()))
                      for j, (b, r) in enumerate(zip(bases, runs.split(",")))]
            got = predict(models, data, d_str, drop, dev, types); allrows.append(got)
            print(split, d_str, "windows", got.window_id.nunique(), flush=True)
            if poison and k == len(dates) // 2:
                saved = {p: v["c3"].copy() for p, v in data.items()}
                for p, v in data.items():
                    v["c3"][v["dates"] > d_str] = np.float16(7.)
                    di = [i for i, x in enumerate(v["dates"]) if x == d_str]
                    ts = [pd.Timestamp(o).hour * 12 + pd.Timestamp(o).minute // 5 for o in v["wi"].forecast_origin if str(pd.Timestamp(o).date()) == d_str]
                    if di and ts:
                        v["c3"][di[0], max(ts) + 1:] = np.float16(7.)
                S.MODE = "train"; S._rng = np.random.default_rng(1000 + k)
                again = predict([qpa.finetune(b, data, d_str, drop, dev, seed=k, cap=cap) for b in bases], data, d_str, drop, dev, types)
                dmax = float(np.abs(again.prob.to_numpy() - got.prob.to_numpy()).max())
                print("POISON", split, d_str, "max|dprob|", dmax, flush=True)
                assert dmax < 1e-6, "future poison changed predictions"
                for p, v in data.items():
                    v["c3"] = saved[p]
    pd.concat(allrows, ignore_index=True).to_parquet(out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
