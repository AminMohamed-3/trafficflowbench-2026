"""Claude 2026-10-01: does fine-tuning on the split's own earlier days help (legal under RULES 2 (b))?

Each split is its own scenario (SPLITSHIFT). RULES 2 allows, for a window on date d of split s, any released
data of split s on dates before d. Test: fine-tune a split-robust CNN on I-Q-SPLIT pseudo ongoing windows of
the first days of a split, score pseudo windows of later days against the same model without fine-tuning.
  validation: adapt on Mar 1-7, score Mar 8-12   (official validation windows are Mar 1-12)
  private:    adapt on Apr 1-7, score Apr 8-17   (official private windows are Apr 1-17)
Inputs and labels follow qcnn_shift.py (masked layer for context; labels on released cells).
Usage: python docs/reference_code/qcnn_adapt.py <run_dir> [epochs] [lr]"""
import sys, json, math, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube, ramp_cube
from qcnn_train import Net, PANELS, CTX, apply_drop, RAMP_C
from qcnn_shift import windows, eiou_mask

SETUPS = {"validation": (("2031-03-01", "2031-03-07"), ("2031-03-08", "2031-03-12")),
          "private": (("2031-04-01", "2031-04-07"), ("2031-04-08", "2031-04-17"))}


def statics(st):
    L = len(st)
    return np.stack([st.v_cut / st.free_speed_kmh, st.lanes / 5, np.log(st.length_km.clip(lower=.01)),
                     st.capacity_vph / 8000, st.has_on_ramp, st.has_off_ramp, np.arange(L) / L]).astype(np.float32)


def make_x(c3, rc, stat, B, drop):
    L = c3.shape[-1]
    ctx = np.stack([c3[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * 5, L) for d, t in B])
    tod = np.array([2 * math.pi * t / 288 for d, t in B])
    glob = np.stack([np.sin(tod), np.cos(tod), np.ones(len(B))], 1).astype(np.float32)
    x = torch.cat([torch.tensor(ctx), torch.tensor(stat)[None].expand(len(B), -1, -1), torch.zeros(len(B), 6, L),
                   torch.tensor(glob)[:, :, None].expand(len(B), 3, L)], 1)
    if rc is not None:
        x = torch.cat([x, torch.tensor(np.stack([rc[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * RAMP_C, L) for d, t in B]))], 1)
    return apply_drop(x, drop)


def iou(pred, lab, val):
    pv = pred & val; u = (pv | lab).sum()
    return (pv & lab).sum() / u if u else 1.


def weekend(date_str):
    return pd.Timestamp(str(date_str)).dayofweek >= 5


def main(run, epochs=3, lr=3e-4, daytype=False):
    """daytype: fine-tune one model per day type (weekday/weekend) on earlier days of that type only;
    a type with no earlier days uses the base model."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((Path(run) / "config.json").read_text()); drop, ramps = cfg["drop"], cfg.get("ramps", False)
    assert "prior" in drop
    base = Net(cin=CTX * 5 + 16 + (CTX * RAMP_C if ramps else 0)).to(dev)
    base.load_state_dict(torch.load(Path(run) / "model.pt", map_location=dev)); base.eval()
    rows = []
    for split, (fit_rng, test_rng) in SETUPS.items():
        data = {}; data_dates = {}
        for panel in PANELS:
            c = load_cache(panel, split); st = c.stations.sort_values("idx")
            c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
            rc = ramp_cube(c, st) if ramps else None
            W = windows(c3, step=1)
            dates = np.array(c.dates)
            fitW = [(d, t) for d, t in W if fit_rng[0] <= dates[d] <= fit_rng[1]]
            testW = [(d, t) for d, t in W if test_rng[0] <= dates[d] <= test_rng[1]][::3]
            data[panel] = (c3, rc, statics(st), fitW, testW); data_dates[panel] = dates
        models = {}
        for kind in ((False, True) if daytype else (None,)):
          sub = {p: (v[0], v[1], v[2], [(d, t) for d, t in v[3] if kind is None or weekend(data_dates[p][d]) == kind], v[4])
                 for p, v in data.items()}
          if sum(len(v[3]) for v in sub.values()) == 0:
              models[kind] = base; continue
          model = copy.deepcopy(base); model.train()
          opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
          rng = np.random.default_rng(0)
          for ep in range(epochs):
            order = [(p, i) for p in PANELS for i in range(0, len(sub[p][3]), 32)]
            rng.shuffle(order); tot = 0.
            for p, i in order:
                c3, rc, stat, fitW, _ = sub[p]
                B = [fitW[j] for j in rng.permutation(len(fitW))[:32]]
                x = make_x(c3, rc, stat, B, drop).to(dev)
                obs = np.stack([c3[d, t + 1:t + 7, 3] > .5 for d, t in B]); q = np.stack([c3[d, t + 1:t + 7, 4] > .5 for d, t in B])
                y = torch.tensor(q & obs, dtype=torch.float32, device=dev); v = torch.tensor(obs, dtype=torch.float32, device=dev)
                loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * v).sum() / (v.sum() + 1)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            print(split, kind, "epoch", ep, "loss", round(tot / max(len(order), 1), 4), "fit windows", sum(len(sub[p][3]) for p in PANELS), flush=True)
          model.eval(); models[kind] = model
        for p in PANELS:
            c3, rc, stat, _, testW = data[p]
            for i in range(0, len(testW), 256):
                B = testW[i:i + 256]; x = make_x(c3, rc, stat, B, drop).to(dev)
                with torch.no_grad():
                    pb = torch.sigmoid(base(x)).cpu().numpy()
                    pa = np.stack([torch.sigmoid(models[weekend(data_dates[p][d]) if daytype else None](x[j:j + 1]))[0].cpu().numpy()
                                   for j, (d, t) in enumerate(B)])
                for (d, t), b, a in zip(B, pb, pa):
                    obs, qq = c3[d, :, 3] > .5, c3[d, :, 4] > .5
                    val, lab = obs[t + 1:t + 7], qq[t + 1:t + 7] & obs[t + 1:t + 7]
                    last = np.zeros(c3.shape[-1], bool)
                    for s in range(t - 12, t + 1):
                        last = np.where(obs[s], qq[s], last)
                    rows.append(dict(split=split, panel=p, base=iou(b > .5, lab, val), adapted=iou(a > .5, lab, val),
                                     base_eiou=iou(eiou_mask(b), lab, val), adapted_eiou=iou(eiou_mask(a), lab, val),
                                     pers=iou(np.broadcast_to(last, val.shape), lab, val)))
    df = pd.DataFrame(rows); df.to_parquet(Path(run) / "adapt_windows.parquet")
    for split, g in df.groupby("split"):
        pn = g.groupby("panel")[["base", "adapted", "base_eiou", "adapted_eiou", "pers"]].mean()
        fam = pn.groupby(pn.index.str.rsplit("_", n=1).str[0]).mean().mean()
        print(split, "test windows", len(g), fam.round(4).to_dict(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) if i == 0 else float(a) for i, a in enumerate(sys.argv[2:])))
