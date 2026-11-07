"""Claude 2026-10-01: online same-split adapted CNN forecasts for official Task 2 ongoing windows.

For every distinct window date d of split s (validation, private): copy the base model, fine-tune it on
I-Q-SPLIT pseudo ongoing windows of split s on dates < d (RULES 2 (b); masked released layer only), then
predict the official ongoing windows of date d with the official inputs of qcnn_predict.py (masked layer
<= T of day d, raw history T-60..T-5 from window_history.parquet). Onset windows are not predicted here
(the probe keeps the parent's onset rows).
Poison check (--poison): for one date per split, overwrite every masked value on dates > d and on day d
after T of each window with garbage; predictions for date d must stay identical.
Usage: python docs/reference_code/qcnn_predict_adapt.py <run_dir> <thr_or_eiou> <out.csv> [--poison]"""
import sys, json, copy, os, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube, ramp_cube
from qcnn_train import Net, PANELS, CTX, apply_drop, RAMP_C
from qcnn_shift import windows, eiou_mask
from qcnn_adapt import make_x, statics, weekend

REL = ROOT / "data/raw/kaggle_public/task2"
EPOCHS, LR, BS = 3, 3e-4, 32
POOL_MAX_PERS = None   # 2026-10-03: optional selector match, keep pool windows with persistence IoU <= this (official max .878)


def configure_determinism():
    if os.environ.get("TFB_ASM2_DET") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        torch.set_num_interop_threads(1)


def checkpointed_adaptation(base, run, split, date, seed, fit, settings=None):
    """Optional saved per-date weights for exact serving; default behavior is unchanged."""
    directory = os.environ.get("TFB_ASM2_ADAPT_ROOT")
    if not directory:
        return fit()
    out = Path(directory) / Path(run).name / split / date
    model_file = out / "model.pt"; receipt_file = out / "inputs.json"
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sources = ["qcnn_predict_adapt.py", "qcnn_train.py", "qcnn_adapt.py", "qcnn_shift.py", "qcnn_data.py"]
    expected = dict(schema=2, base_sha256=digest(Path(run) / "model.pt"),
                    config_sha256=hashlib.sha256((Path(run) / "config.json").read_bytes()).hexdigest(),
                    split=split, before=date, seed=seed, epochs=EPOCHS, learning_rate=LR, batch_size=BS,
                    weight_decay=1e-4, pool_max_persistence=POOL_MAX_PERS,
                    settings=settings or {}, torch_version=str(torch.__version__),
                    device=str(next(base.parameters()).device), threads=torch.get_num_threads(),
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    source_sha256={name:digest(ROOT / "docs/reference_code" / name) for name in sources})
    if os.environ.get("TFB_ASM2_USE_ADAPTED") == "1":
        receipt = json.loads(receipt_file.read_text()) if receipt_file.is_file() else {}
        assert receipt.get("inputs") == expected, f"adaptation input mismatch: {out}"
        assert receipt.get("model_sha256") == digest(model_file), f"adaptation model bytes changed: {out}"
        model = copy.deepcopy(base)
        model.load_state_dict(torch.load(model_file, map_location=next(model.parameters()).device)); model.eval()
        return model
    model = fit(); out.mkdir(parents=True, exist_ok=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, model_file)
    receipt_file.write_text(json.dumps(dict(inputs=expected, model_sha256=digest(model_file)), indent=2) + "\n")
    return model


def pers_iou(c3, d, t):
    """Persistence IoU of pseudo window (d, t): last history step's queue repeated over T+5..T+30, observed cells only."""
    q, obs = c3[d, t + 1:t + 7, 4] > .5, c3[d, t + 1:t + 7, 3] > .5
    p = np.broadcast_to(c3[d, t - 1, 4] > .5, q.shape) & obs; y = q & obs
    u = (p | y).sum()
    return 1. if u == 0 else (p & y).sum() / u


def official_x(c3m, rc, stat, st, links, wi, hist, drop):
    """Official ongoing window inputs, same construction as qcnn_predict.py (prior channels zero: dropped)."""
    out = []
    col = {l: i for i, l in enumerate(links)}
    for w in wi.itertuples(index=False):
        T = pd.Timestamp(w.forecast_origin); d = int(w.day); t = T.hour * 12 + T.minute // 5
        L = len(links); lo = t - CTX + 1
        ctx = np.zeros((CTX, 5, L), np.float32)
        for s in range(max(lo, 0), t + 1):
            if not (t - 12 <= s <= t - 1):
                ctx[s - lo] = c3m[d, s]
        h = hist[hist.window_id == w.window_id]
        hs = ((pd.to_datetime(h.timestamp) - T.normalize()).dt.total_seconds() // 300).astype(int).to_numpy()
        assert hs.min() == t - 12 and hs.max() == t - 1
        sp = np.full((12, L), np.nan, np.float32); fl = sp.copy(); oc = sp.copy()
        j = h.link_id.map(col).to_numpy(); k = hs - (t - 12)
        sp[k, j] = h.speed_kmh; fl[k, j] = h.flow_vph; oc[k, j] = h.occupancy
        ctx[11:23] = cube(sp[None], fl[None], oc[None], st)[0]
        tod = 2 * np.pi * t / 288
        x = torch.cat([torch.tensor(ctx.reshape(1, CTX * 5, L)), torch.tensor(stat)[None], torch.zeros(1, 6, L),
                       torch.tensor([np.sin(tod), np.cos(tod), 1.], dtype=torch.float32)[None, :, None].expand(1, 3, L)], 1)
        if rc is not None:
            r = np.zeros((CTX, RAMP_C, L), np.float32); r[max(lo, 0) - lo:] = rc[d, max(lo, 0):t + 1]
            x = torch.cat([x, torch.tensor(r.reshape(1, CTX * RAMP_C, L))], 1)
        out.append((w, t, apply_drop(x, drop)))
    return out


def finetune(base, data, before, drop, dev, seed, daytype=True, cap=None):
    """Fine-tune on pseudo windows of dates < before (same day type as `before` when daytype).
    cap: optional max windows per epoch (random subset; C03 official predictions used no cap)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    kind = weekend(before)
    pool = {p: [(d, t) for d, t in v["W"] if v["dates"][d] < before and (not daytype or weekend(v["dates"][d]) == kind)]
            for p, v in data.items()}
    if POOL_MAX_PERS is not None:
        pool = {p: [(d, t) for d, t in w if pers_iou(data[p]["c3"], d, t) <= POOL_MAX_PERS] for p, w in pool.items()}
    if sum(len(w) for w in pool.values()) == 0:
        return base
    model = copy.deepcopy(base); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS):
        sub = pool
        if cap:
            n = sum(len(w) for w in pool.values()); keep = rng.random(n) < min(1., cap / n); it = iter(keep)
            sub = {p: [w for w in pool[p] if next(it)] for p in PANELS}
        order = [(p, i) for p in PANELS for i in range(0, len(sub[p]), BS)]
        rng.shuffle(order)
        for p, i in order:
            v = data[p]; W = sub[p]; B = [W[j] for j in rng.permutation(len(W))[:BS]]
            x = make_x(v["c3"], v["rc"], v["stat"], B, drop).to(dev)
            obs = np.stack([v["c3"][d, t + 1:t + 7, 3] > .5 for d, t in B]); q = np.stack([v["c3"][d, t + 1:t + 7, 4] > .5 for d, t in B])
            y = torch.tensor(q & obs, dtype=torch.float32, device=dev); m = torch.tensor(obs, dtype=torch.float32, device=dev)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def load_split(split, ramps):
    data = {}
    for panel in PANELS:
        c = load_cache(panel, split); st = c.stations.sort_values("idx")
        c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
        wi = pd.read_csv(REL / panel / split / "window_index.csv")
        wi = wi[wi.condition == "queue_ongoing"].copy(); wi["day"] = [c.day_index(str(pd.Timestamp(o).date())) for o in wi.forecast_origin]
        data[panel] = dict(c=c, st=st, c3=c3, rc=ramp_cube(c, st) if ramps else None, stat=statics(st),
                           dates=np.array(c.dates), W=windows(c3, step=1), links=st.link_id.to_numpy().astype(str), wi=wi,
                           hist=pd.read_parquet(REL / panel / split / "window_history.parquet"),
                           sample=pd.read_csv(REL / panel / split / "sample_submission_queue.csv"))
    return data


def predict_date(model, data, d_str, decide, drop, dev):
    rows = []
    for p, v in data.items():
        wi = v["wi"][[str(pd.Timestamp(o).date()) == d_str for o in v["wi"].forecast_origin]]
        for w, t, x in official_x(v["c3"], v["rc"], v["stat"], v["st"], v["links"], wi, v["hist"], drop):
            with torch.no_grad():
                pr = torch.sigmoid(model(x.to(dev)))[0].cpu().numpy()
            pred = decide(pr); T = pd.Timestamp(w.forecast_origin)
            ts = [(T + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(6)]
            rows.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(ts, len(v["links"])),
                                      "link_id": np.tile(v["links"], 6), "queue_pred": pred.reshape(-1).astype(int),
                                      "prob": pr.reshape(-1)}))
    return pd.concat(rows, ignore_index=True) if rows else None


def main(run, rule, out, poison=False):
    configure_determinism()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    cfg = json.loads((Path(run) / "config.json").read_text()); drop, ramps = cfg["drop"], cfg.get("ramps", False)
    assert "prior" in drop, "adaptation path assumes the prior channels are dropped"
    base = Net(cin=CTX * 5 + 16 + (CTX * RAMP_C if ramps else 0)).to(dev)
    base.load_state_dict(torch.load(Path(run) / "model.pt", map_location=dev)); base.eval()
    decide = eiou_mask if rule == "eiou" else (lambda pr: pr > float(rule))
    allrows = []
    for split in ("validation", "private"):
        data = load_split(split, ramps)
        dates = sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin})
        for k, d_str in enumerate(dates):
            model = checkpointed_adaptation(base, run, split, d_str, k, lambda: finetune(base, data, d_str, drop, dev, seed=k),
                                            settings=dict(daytype=True, cap=None, drop=list(drop)))
            got = predict_date(model, data, d_str, decide, drop, dev)
            n_fit = sum(sum(1 for d, t in v["W"] if v["dates"][d] < d_str and weekend(v["dates"][d]) == weekend(d_str)) for v in data.values())
            print(split, d_str, "fit windows", n_fit, "official windows", got.window_id.nunique(), "queued share", round(float(got.queue_pred.mean()), 4), flush=True)
            allrows.append(got)
            if poison and k == len(dates) // 2:
                # garbage on every date >= d (later days fully; day d after each window's T is not used by construction,
                # so poisoning dates > d and day d after the latest T of that date is a strict superset test)
                saved = {p: v["c3"].copy() for p, v in data.items()}
                for p, v in data.items():
                    later = v["dates"] > d_str
                    v["c3"][later] = np.float16(7.)
                    di = [i for i, x in enumerate(v["dates"]) if x == d_str]
                    ts = [pd.Timestamp(o).hour * 12 + pd.Timestamp(o).minute // 5 for o in v["wi"].forecast_origin if str(pd.Timestamp(o).date()) == d_str]
                    if di and ts:
                        v["c3"][di[0], max(ts) + 1:] = np.float16(7.)
                again = predict_date(finetune(base, data, d_str, drop, dev, seed=k), data, d_str, decide, drop, dev)
                same = (again.queue_pred.to_numpy() == got.queue_pred.to_numpy()).all()
                print("POISON", split, d_str, "binary identical", bool(same), "max|dprob|", float(np.abs(again.prob - got.prob).max()), flush=True)
                assert same, "future poison changed predictions"
                for p, v in data.items():
                    v["c3"] = saved[p]
    sub = pd.concat(allrows, ignore_index=True)
    sub.drop(columns="prob").to_csv(out, index=False); sub.to_parquet(Path(out).with_suffix(".parquet"))
    print("rows", len(sub), "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--poison" in sys.argv)
