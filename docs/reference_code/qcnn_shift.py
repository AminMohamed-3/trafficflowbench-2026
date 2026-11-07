"""Claude 2026-10-01: does the CNN's ongoing edge over persistence survive in Mar-Apr 2031 (validation/private)?

C02 lost ~.066 public ongoing although I-Q-ADV (Jan-Feb) said +.056 over Q04. Test for a regime shift with
pseudo-windows built the same way on both sides from a masked layer only:
  Jan-Feb 2031: train-split masked layer (synthetic Task 1 mask), labels = masked observed cells;
  Mar/Apr 2031: released validation/private layer (official Task 1 targets and Task 2 horizons are NaN).
Window rule (ongoing only): history t-12..t-1 coverage >= .5, some link with >= 2 queued steps there, a queued
observed cell in t+1..t+6.  Inputs: the whole context from the masked layer (no raw history).
Scores: per-window IoU on observed horizon cells; CNN (ADV fit, thr .5) and last-observed persistence
(per link, last observed state in t-12..t). Macro: panel, then family. A gap that holds in Jan-Feb but
collapses in Mar-Apr means drift; a gap that holds in both points at labels/scoring instead.
Usage: python docs/reference_code/qcnn_shift.py [run_dir]"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube, ramp_cube
from qcnn_train import Net, PANELS, CTX, DATA, apply_drop, RAMP_C
import json


def windows(c3, step=3):
    q, obs = c3[:, :, 4] > .5, c3[:, :, 3] > .5
    out = []
    for d in range(c3.shape[0]):
        for t in range(CTX, 282, step):
            if obs[d, t - 12:t].mean() < .5:
                continue
            if not ((q[d, t - 12:t].sum(0) >= 2).any() and (q[d, t + 1:t + 7] & obs[d, t + 1:t + 7]).any()):
                continue
            out.append((d, t))
    return out


def eiou_mask(pr):
    """Top-m cells with m maximising expected IoU ~ sum_top p / (m + sum_all p - sum_top p) (Q04's rule)."""
    f = pr.ravel(); o = np.argsort(-f); c = np.cumsum(f[o]); m = np.arange(1, len(f) + 1)
    e = c / (m + f.sum() - c); k = int(np.argmax(e)) + 1
    out = np.zeros(len(f), bool); out[o[:k]] = True
    return out.reshape(pr.shape)


def main(runs):
    """runs: one run dir, or several comma-separated (same config): probabilities are averaged (seed ensemble)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = runs.split(","); run = runs[0]
    cfgs = [json.loads((Path(r) / "config.json").read_text()) if (Path(r) / "config.json").exists() else {} for r in runs]
    drop, ramps = cfgs[0].get("drop", []), cfgs[0].get("ramps", False)
    assert all((c.get("drop", []), c.get("ramps", False)) == (drop, ramps) for c in cfgs), "ensemble members differ in inputs"
    models = []
    for r in runs:
        m = Net(cin=CTX * 5 + 16 + (CTX * RAMP_C if ramps else 0)).to(dev)
        m.load_state_dict(torch.load(Path(r) / "model.pt", map_location=dev)); m.eval(); models.append(m)
    fit_end = "2031-02-28" if "run_full" in str(run) else "2030-12-31"
    rows = []
    for panel in PANELS:
        tr = load_cache(panel, "train"); fit = np.array([x <= fit_end for x in tr.dates])
        if (DATA / f"{panel}.npz").exists():
            z = np.load(DATA / f"{panel}.npz"); static = torch.tensor(z["static"]); L = len(z["link_id"])
            S, N = z["queued_raw"][fit].sum(0).astype(np.float32), z["observed_raw"][fit].sum(0).astype(np.float32)
        else:                      # laptop fallback: statics from station metadata; prior only valid if dropped
            assert "prior" in drop, "prior needs the panel npz"
            st0 = tr.stations.sort_values("idx"); L = len(st0)
            static = torch.tensor(np.stack([st0.v_cut / st0.free_speed_kmh, st0.lanes / 5, np.log(st0.length_km.clip(lower=.01)),
                                            st0.capacity_vph / 8000, st0.has_on_ramp, st0.has_off_ramp, np.arange(L) / L]).astype(np.float32))
            S = N = np.zeros((288, L), np.float32)
        for split in ("train", "validation", "private"):
            c = tr if split == "train" else load_cache(panel, split)
            st = c.stations.sort_values("idx")
            c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
            days = [i for i, x in enumerate(c.dates) if split != "train" or x >= "2031-01-01"]
            rc = ramp_cube(c, st)[days] if ramps else None
            c3 = c3[days]; dates = [c.dates[i] for i in days]
            W = windows(c3)
            for i in range(0, len(W), 512):
                B = W[i:i + 512]
                ctx = np.stack([c3[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * 5, L) for d, t in B])
                prior = np.stack([S[t + 1:t + 7] / np.maximum(N[t + 1:t + 7], 1) for d, t in B])
                tod = np.array([2 * math.pi * t / 288 for d, t in B])
                glob = np.stack([np.sin(tod), np.cos(tod), np.ones(len(B))], 1).astype(np.float32)
                x = torch.cat([torch.tensor(ctx), static[None].expand(len(B), -1, -1), torch.tensor(prior),
                               torch.tensor(glob)[:, :, None].expand(len(B), 3, L)], 1)
                if ramps:
                    x = torch.cat([x, torch.tensor(np.stack([rc[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * RAMP_C, L) for d, t in B]))], 1)
                x = apply_drop(x, drop).to(dev)
                with torch.no_grad():
                    p = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0).cpu().numpy()
                for (d, t), pr in zip(B, p):
                    obs, q = c3[d, :, 3] > .5, c3[d, :, 4] > .5
                    val, lab = obs[t + 1:t + 7], q[t + 1:t + 7] & obs[t + 1:t + 7]
                    last = np.zeros(L, bool)
                    for s in range(t - 12, t + 1):
                        last = np.where(obs[s], q[s], last)
                    res = dict(panel=panel, split=split, date=dates[d], t=t)
                    for name, pred in (*((f"cnn{th}", pr > th) for th in (.3, .4, .5, .6)), ("cnn_eiou", eiou_mask(pr)), ("pers", np.broadcast_to(last, val.shape))):
                        pv = pred & val; u = (pv | lab).sum()
                        res[name] = (pv & lab).sum() / u if u else 1.
                    rows.append(res)
            print(panel, split, len(W), flush=True)
    df = pd.DataFrame(rows); df["period"] = np.where(df.split == "train", "JanFeb", np.where(df.split == "validation", "Mar", "Apr"))
    df.to_parquet(Path(run) / ("shift_windows.parquet" if len(runs) == 1 else f"shift_windows_ens{len(runs)}.parquet"))
    for per, g in df.groupby("period"):
        cols = [c for c in g.columns if c.startswith("cnn")] + ["pers"]
        pn = g.groupby("panel")[cols].mean()
        fam = pn.groupby(pn.index.str.rsplit("_", n=1).str[0]).mean().mean()
        print(per, "windows", len(g), "macro", fam.round(4).to_dict(), flush=True)
    print(df.groupby(["period", "panel"])[["cnn0.5", "pers"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA / "run_adv_s0"))
