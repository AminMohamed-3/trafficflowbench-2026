"""ong2 2026-10-05: train the base corridor CNN (qcnn_train.py full mode, npt recipe: --drop prior,pos,tod, 8 epochs)
with selection flags (ong2_sccnn.py) or without (--ctl, the C03 recipe: reproduces run_full_s<seed>_npt bytes when
the host is deterministic). Batch sampling uses the same RNG sequence as qcnn_train (flag dropout has its own RNG).
Output: artifacts/claude_ref/ong2/run_sc_s<seed>[_ctl]/ (model.pt, config.json).
Usage: python ong2_sccnn_train.py --seed 0 [--ctl]
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import qcnn_train
from qcnn_train import Panel, Net, PANELS, CTX, ONSET_W, DATA
from tfb.data import load_cache

OUT = ROOT / "artifacts/claude_ref/ong2"


def main():
    a = argparse.ArgumentParser(); a.add_argument("--seed", type=int, default=0); a.add_argument("--ctl", action="store_true")
    a.add_argument("--epochs", type=int, default=8)
    args = a.parse_args(); drop = ["prior", "pos", "tod"]
    if not args.ctl:
        import ong2_sccnn; ong2_sccnn.enable(); ong2_sccnn.MODE = "train"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = pd.read_parquet(DATA / "windows.parquet")
    fit_end = "2031-02-28"; train = W[W.date <= fit_end]
    panels = {name: Panel(name, np.array([x <= fit_end for x in load_cache(name, "train").dates])) for name in PANELS}
    print(dev, "train", len(train), flush=True)
    model = Net(cin=CTX * 5 + 16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    groups = {n: g for n, g in train.groupby("panel")}
    steps_per_epoch = len(train) // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * steps_per_epoch)
    nf = np.zeros(3)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.
        order = rng.permutation(np.concatenate([np.full(len(g) // 64, i) for i, g in enumerate(groups.values())]))
        names = list(groups)
        for k, pi in enumerate(order[:steps_per_epoch]):
            g = groups[names[pi]]; p = panels[names[pi]]
            on = g[g.condition == "queue_onset"]
            c = pd.concat([g.sample(56, random_state=int(rng.integers(1 << 31))),
                           on.sample(8, random_state=int(rng.integers(1 << 31)), replace=len(on) < 8)])
            x = qcnn_train.inputs(p, c, dev, drop)
            if not args.ctl and ep == 0:
                nf += x[:, 127:130, 0].sum(0).cpu().numpy()
            y = torch.tensor(p.labels[c.row.to_numpy()], dtype=torch.float32, device=dev)
            v = torch.tensor(p.valid[c.row.to_numpy()], dtype=torch.float32, device=dev)
            w = torch.tensor(np.where(c.condition == "queue_onset", ONSET_W, 1.), dtype=torch.float32, device=dev)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * v).sum((1, 2))
            loss = (loss * w).sum() / (v.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(json.dumps(dict(epoch=ep, loss=round(tot / steps_per_epoch, 4), minutes=round((time.time() - t0) / 60, 2),
                              flag_sums_ep0=nf.round(0).tolist())), flush=True)
    out = OUT / (f"run_sc_s{args.seed}" + ("_ctl" if args.ctl else "")); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(drop=drop, ramps=False, epochs=args.epochs, seed=args.seed, mode="full", sccnn=not args.ctl)))
    torch.save(model.state_dict(), out / "model.pt")
    print("saved", out, "sha256", hashlib.sha256((out / "model.pt").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
