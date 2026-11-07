"""Claude 2026-10-01: corridor CNN queue forecaster (Task 2), trained on train-split candidate windows.

Input per window, along the L links of a panel (upstream -> downstream):
  24 context steps x 5 channels (masked t-23..t-13, raw t-12..t-1, masked t), 7 link statics,
  6 historical queued-frequency channels for steps t+1..t+6 (fit days only, own day left out),
  time of day (sin, cos) and the ongoing flag.  Output: 6 x L logits (T+5..T+30).
Model: 1x1 conv to 128, residual dilated 1D convs (kernel 3) along the corridor, 1x1 head.
Loss: masked BCE (valid = raw observed), onset windows weighted x ONSET_W.
Scoring: per-window IoU on valid cells, mean per panel/condition, then panel, then family (I-Q-ADV macro).
Modes: inner = fit June-Nov, score December (choose epochs and thresholds);
       adv   = fit June-Dec, score Q's frozen I-Q-ADV broad windows (Jan-Feb 2031), one read.
Usage: python docs/reference_code/qcnn_train.py inner|adv [--epochs N] [--seed S] [--thr-on X --thr-onset Y]"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/claude_ref/qcnn"
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
CTX, ONSET_W = 24, 20.
THRS = np.round(np.arange(.2, .81, .05), 2)
# input channel groups (see inputs): context 0..119, statics 120..126 (126 = corridor position),
# prior 127..132, sin/cos time of day 133..134, ongoing flag 135
GROUPS = {"prior": list(range(127, 133)), "pos": [126], "tod": [133, 134], "statics": list(range(120, 127))}


def apply_drop(x, drop):
    """Zero channel groups that carry train-scenario structure (split-robust variants, C02 finding)."""
    for g in drop:
        x[:, GROUPS[g]] = 0
    return x


RAMP_C = 3


class Panel:
    def __init__(self, name, fit_mask, ramps=False):
        z = np.load(DATA / f"{name}.npz")
        self.ramp = np.load(DATA / f"{name}_ramp.npz")["ramp"] if ramps else None
        self.name, self.raw, self.masked = name, z["raw"], z["masked"]
        self.labels, self.valid, self.static = z["labels"], z["valid"], torch.tensor(z["static"])
        q, o = z["queued_raw"].astype(np.float32), z["observed_raw"].astype(np.float32)
        self.q, self.o = q, o
        self.S, self.N = q[fit_mask].sum(0), o[fit_mask].sum(0)
        self.fit_mask = fit_mask
        self.L = self.raw.shape[-1]

    def batch(self, rows):
        X, P, R = [], [], []
        for d, t in rows:
            lo = t - CTX + 1
            ctx = np.zeros((CTX, 5, self.L), np.float32)
            for s in range(max(lo, 0), t + 1):
                src = self.raw if t - 12 <= s <= t - 1 else self.masked
                ctx[s - lo] = src[d, s]
            X.append(ctx.reshape(CTX * 5, self.L))
            if self.ramp is not None:
                r = np.zeros((CTX, RAMP_C, self.L), np.float32); s0 = max(lo, 0)
                r[s0 - lo:] = self.ramp[d, s0:t + 1]
                R.append(r.reshape(CTX * RAMP_C, self.L))
            S, N = self.S[t + 1:t + 7], self.N[t + 1:t + 7]
            if self.fit_mask[d]:
                S, N = S - self.q[d, t + 1:t + 7], N - self.o[d, t + 1:t + 7]
            P.append(S / np.maximum(N, 1))
        X = torch.tensor(np.stack(X)); P = torch.tensor(np.stack(P), dtype=torch.float32)
        return (X, P, torch.tensor(np.stack(R))) if R else (X, P, None)


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv1d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        return x + self.c2(F.gelu(self.n2(h)))


class Net(nn.Module):
    def __init__(self, cin=CTX * 5 + 7 + 6 + 3, c=128, dils=(1, 2, 4, 8, 16, 32, 1, 4, 16)):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in dils])
        self.out = nn.Conv1d(c, 6, 1)

    def forward(self, x):
        return self.out(F.gelu(self.blocks(self.inp(x))))


def inputs(panel, W, dev, drop=()):
    ctx, prior, ramp = panel.batch(list(zip(W.day, W.t)))
    B, L = len(W), panel.L
    tod = 2 * math.pi * W.t.to_numpy() / 288
    glob = np.stack([np.sin(tod), np.cos(tod), (W.condition == "queue_ongoing").to_numpy()], 1).astype(np.float32)
    x = torch.cat([ctx, panel.static[None].expand(B, -1, -1), prior,
                   torch.tensor(glob)[:, :, None].expand(B, 3, L)] + ([ramp] if ramp is not None else []), 1)
    return apply_drop(x, drop).to(dev)


def predict(model, panels, W, dev, bs=256, drop=()):
    model.eval(); out = {}
    with torch.no_grad():
        for name, g in W.groupby("panel"):
            p = panels[name]
            for i in range(0, len(g), bs):
                c = g.iloc[i:i + bs]
                out.update(zip(c.index, torch.sigmoid(model(inputs(p, c, dev, drop))).cpu().numpy()))
    return out


def window_iou(prob, lab, val, thr):
    pv, tv = (prob > thr) & val, lab & val
    u = (pv | tv).sum()
    return (pv & tv).sum() / u if u else (1. if val.any() else 0.)


def macro(df, col):
    pc = df.groupby(["panel", "condition"])[col].mean()
    pn = pc.groupby("panel").mean()
    fam = pn.groupby(pn.index.str.rsplit("_", n=1).str[0]).mean()
    return float(fam.mean()), pc


def score(probs, panels, W, thr_on, thr_onset):
    rows = []
    for i, r in W.iterrows():
        p = panels[r.panel]; thr = thr_on if r.condition == "queue_ongoing" else thr_onset
        pr = probs[i]
        if r.condition == "queue_onset":       # selector: no queue at T+5..T+25 in onset windows
            pr = pr.copy(); pr[:5] = 0
        rows.append(window_iou(pr, p.labels[r.row], p.valid[r.row], thr))
    return np.array(rows)


def main():
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["inner", "adv", "full"])
    a.add_argument("--epochs", type=int, default=8); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--thr-on", type=float); a.add_argument("--thr-onset", type=float)
    a.add_argument("--drop", default="", help="comma list of channel groups to zero: prior,pos,tod,statics")
    a.add_argument("--tag", default="")
    a.add_argument("--ramps", action="store_true", help="append released ramp-flow context (CTX x 3 channels)")
    args = a.parse_args()
    drop = [g for g in args.drop.split(",") if g]
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = pd.read_parquet(DATA / "windows.parquet")
    fit_end = {"inner": "2030-11-30", "adv": "2030-12-31", "full": "2031-02-28"}[args.mode]
    train = W[W.date <= fit_end]
    if args.mode == "inner":
        test = W[(W.date >= "2030-12-01") & (W.date <= "2030-12-31")]
    elif args.mode == "full":
        test = W.iloc[:0]
    else:
        # all Jan-Feb candidates; the frozen I-Q-ADV subset is taken on the laptop (one read)
        test = W[(W.date >= "2031-01-01") & (W.date <= "2031-02-28")]
    sys.path.insert(0, str(ROOT / "src"))
    from tfb.data import load_cache
    panels = {}
    for name in PANELS:
        fit_mask = np.array([x <= fit_end for x in load_cache(name, "train").dates])
        panels[name] = Panel(name, fit_mask, args.ramps)
    print(dev, "train", len(train), "test", len(test), flush=True)
    model = Net(cin=CTX * 5 + 16 + (CTX * RAMP_C if args.ramps else 0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    groups = {n: g for n, g in train.groupby("panel")}
    steps_per_epoch = len(train) // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * steps_per_epoch)
    hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.
        order = rng.permutation(np.concatenate([np.full(len(g) // 64, i) for i, g in enumerate(groups.values())]))
        names = list(groups)
        for k, pi in enumerate(order[:steps_per_epoch]):
            g = groups[names[pi]]; p = panels[names[pi]]
            on = g[g.condition == "queue_onset"]
            c = pd.concat([g.sample(56, random_state=int(rng.integers(1 << 31))),
                           on.sample(8, random_state=int(rng.integers(1 << 31)), replace=len(on) < 8)])
            x = inputs(p, c, dev, drop)
            y = torch.tensor(p.labels[c.row.to_numpy()], dtype=torch.float32, device=dev)
            v = torch.tensor(p.valid[c.row.to_numpy()], dtype=torch.float32, device=dev)
            w = torch.tensor(np.where(c.condition == "queue_onset", ONSET_W, 1.), dtype=torch.float32, device=dev)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * v).sum((1, 2))
            loss = (loss * w).sum() / (v.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if args.mode == "full" or (args.mode == "adv" and ep < args.epochs - 1):
            print(json.dumps(dict(epoch=ep, loss=round(tot / steps_per_epoch, 4), minutes=round((time.time() - t0) / 60, 2))), flush=True)
            continue
        probs = predict(model, panels, test, dev, drop=drop)
        res = {}
        for cond, col in (("queue_ongoing", "on"), ("queue_onset", "onset")):
            sub = test[test.condition == cond]
            res[col] = {float(th): float(score(probs, panels, sub, th, th).mean()) for th in THRS}
        bon = max(res["on"], key=res["on"].get); bonset = max(res["onset"], key=res["onset"].get)
        thr_on = args.thr_on if args.thr_on is not None else bon
        thr_onset = args.thr_onset if args.thr_onset is not None else bonset
        test = test.assign(iou=score(probs, panels, test, thr_on, thr_onset))
        S, pc = macro(test, "iou")
        pers, _ = macro(test, "persistence_iou")
        hist.append(dict(epoch=ep, loss=tot / steps_per_epoch, S_queue=S, persistence=pers, thr_on=thr_on,
                         thr_onset=thr_onset, best_on=bon, best_onset=bonset,
                         on_mean=float(test[test.condition == "queue_ongoing"].iou.mean()),
                         onset_mean=float(test[test.condition == "queue_onset"].iou.mean()), minutes=(time.time() - t0) / 60))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hist[-1].items()}), flush=True)
    out = DATA / f"run_{args.mode}_s{args.seed}{args.tag}"
    out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(drop=drop, ramps=args.ramps, epochs=args.epochs, seed=args.seed, mode=args.mode)))
    if args.mode == "full":
        torch.save(model.state_dict(), out / "model.pt"); print("saved", out); return
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    test.drop(columns=[]).to_parquet(out / "window_scores.parquet")
    torch.save(model.state_dict(), out / "model.pt")
    print("per panel/condition:\n", pc.round(4).to_string())


if __name__ == "__main__":
    main()
