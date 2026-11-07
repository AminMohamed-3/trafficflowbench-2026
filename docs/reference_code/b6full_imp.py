"""Claude helper 2026-10-04 (ledger 0399, idea B6 full): per-split masked conv U-Net imputer.
Trained only on the visible cells of ONE layer (one split, or one OPT held train month); never sees target/hidden cells.
Inputs per (day, time, station): speed/vf and flow/cap at visible cells (0 elsewhere), the visibility mask, time-of-day
sin/cos, station position (milepost scaled to [0,1]) and lanes/6 (published constants of the layer's own panel).
Training: batch of 4 days; a further random 20-50% of visible cells hidden, half the days also one 90-min full-corridor
block; loss on hidden visible cells weighted like S_state (.54 (dv/25)^2 + .46 (dq_lane/600)^2); AdamW 1e-3 one-cycle,
grad clip 1, fp32. Prediction: full visible input, outputs clipped to [3, 1.3 vf] km/h and [0, 1.3 cap] vph.
Modes (layer): own = I-S-OWN March blinded view; opt_jun / opt_jul = OPT held dates of that train month (masked layer);
val / priv = official validation / private view (for the probe only). Mode own also saves imp_valown_*: the
OWN-trained model applied to the official validation view (validation-only data; used for the probe's validation rows).
Usage: python docs/reference_code/b6full_imp.py <mode> <panel|all> <seed> [iters]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P

torch.set_num_threads(1)
OUT = P.ROOT / "artifacts/claude_ref/b6full"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PANELS = P.P["panels"]


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    def __init__(self, cin=7, w=32):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w)
        self.d2, self.d1 = Block(6 * w, 2 * w), Block(3 * w, w)
        self.out = nn.Conv2d(w, 2, 1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(F.avg_pool2d(a, 2)); c = self.e3(F.avg_pool2d(b, 2))
        b2 = self.d2(torch.cat([b, F.interpolate(c, size=b.shape[2:])], 1))
        a2 = self.d1(torch.cat([a, F.interpolate(b2, size=a.shape[2:])], 1))
        return self.out(a2)


def layer(mode, panel):
    """Returns speed, flow (D,T,S) float arrays with only released visible values used, vis mask, stations, dates."""
    if mode == "own":
        import flow_own as FO
        items = [i for i in json.loads(Path(P.P["OWN_metadata_binding"]["manifest"]).read_text())["items"]
                 if i["period"] == "validation" and i["panel"] == panel]
        view, tv, tq, st_, hidden = FO.own.load_item(P.ROOT, FO.binding.bound_own_item(P.P, items[0]))
        hide = np.asarray(view.target, bool) | np.asarray(st_, bool) | np.asarray(view.gap, bool)
        sp, fl, el, stations, dates = view.speed, view.flow, view.elig, view.stations, list(view.dates)
    elif mode.startswith("opt_"):
        held, tv, tq, pv, pq = P.held_panel(panel)
        mon = {"opt_jun": "2030-06", "opt_jul": "2030-07"}[mode]
        dd = [i for i, d in enumerate(P.P["OPT_held_dates"]) if d.startswith(mon)]
        hide = (np.asarray(held.target, bool) | np.asarray(held.gap, bool))[dd]
        sp, fl, el = (np.asarray(x)[dd] for x in (held.speed, held.flow, held.elig))
        stations, dates = held.stations, [P.P["OPT_held_dates"][i] for i in dd]
    else:
        sys.path.insert(0, str(P.ROOT / "src"))
        from tfb.data import load_cache
        c = load_cache(panel, {"val": "validation", "priv": "private"}[mode])
        hide = np.asarray(c.target, bool) | np.asarray(c.gap, bool)
        sp, fl, el, stations, dates = c.speed, c.flow, c.elig, c.stations, list(c.dates)
    sp = np.asarray(sp, float); fl = np.asarray(fl, float)
    vis = np.asarray(el, bool) & ~hide & np.isfinite(sp) & np.isfinite(fl)
    return np.where(vis, sp, np.nan), np.where(vis, fl, np.nan), vis, stations, dates


def inputs(sp, fl, vis, stations):
    st = stations.sort_values("idx")
    vf = st.free_speed_kmh.to_numpy(np.float32); cap = st.capacity_vph.to_numpy(np.float32)
    lanes = np.maximum(st.lanes.to_numpy(np.float32), 1)
    mp = st.milepost.to_numpy(np.float32); pos = (mp - mp.min()) / max(float(mp.max() - mp.min()), 1e-6)
    D, T, S = sp.shape
    sn = np.where(vis, sp / vf, 0).astype(np.float32); fn = np.where(vis, fl / cap, 0).astype(np.float32)
    tod = np.arange(T) / T * 2 * np.pi
    b = lambda a: np.broadcast_to(a, (D, T, S)).astype(np.float32)
    return np.stack([sn, fn, vis.astype(np.float32), b(np.sin(tod)[None, :, None]), b(np.cos(tod)[None, :, None]),
                     b(pos[None, None, :]), b(lanes[None, None, :] / 6)], 1), vf, cap, lanes


def train_predict(sp, fl, vis, stations, iters=6000, seed=0, log=print, extra=()):
    """extra: further (sp, fl, vis) layers of the same panel to predict with the trained model (no training on them)."""
    Xn, vf, cap, lanes = inputs(sp, fl, vis, stations)
    D, T, S = sp.shape
    X = torch.tensor(Xn, device=DEV)
    VIS = torch.tensor(vis, device=DEV)
    VF = torch.tensor(vf, device=DEV); CAP = torch.tensor(cap, device=DEV); LN = torch.tensor(lanes, device=DEV)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = UNet(cin=X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters)
    tt = torch.arange(T, device=DEV)[None, :, None]
    t0 = time.time(); B = min(4, D); run = 0.
    for it in range(iters):
        idx = torch.tensor(rng.choice(D, B, replace=False), device=DEV)
        x = X[idx].clone(); vm = VIS[idx]
        rate = torch.tensor(rng.uniform(.2, .5, B), device=DEV, dtype=torch.float32)[:, None, None]
        drop = (torch.rand(vm.shape, device=DEV) < rate) & vm
        blk = torch.tensor(rng.random(B) < .5, device=DEV)[:, None, None]
        t1 = torch.tensor(rng.integers(0, T - 18, B), device=DEV)[:, None, None]
        drop |= blk & (tt >= t1) & (tt < t1 + 18) & vm
        keep = (vm & ~drop).float()
        x[:, 0] *= keep; x[:, 1] *= keep; x[:, 2] = keep
        y = model(x)
        dv = (y[:, 0] - X[idx, 0]) * VF / 25; dq = (y[:, 1] - X[idx, 1]) * CAP / LN / 600
        loss = ((.54 * dv ** 2 + .46 * dq ** 2) * drop).sum() / drop.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = .99 * run + .01 * float(loss) if it else float(loss)
        if it % 1000 == 0 or it == iters - 1:
            log(f"  it {it} loss {run:.5f} {time.time() - t0:.0f}s")
    model.eval()

    def pred(Xa):
        with torch.no_grad():
            Y = torch.cat([model(torch.tensor(Xa[i:i + 4], device=DEV)) for i in range(0, len(Xa), 4)]).cpu().numpy().astype(float)
        return (np.clip(Y[:, 0] * vf, 3., 1.3 * vf).astype(np.float32), np.clip(Y[:, 1] * cap, 0., 1.3 * cap).astype(np.float32))
    del X
    iv, iq = pred(Xn)
    ex = [pred(inputs(a, b_, c, stations)[0]) for a, b_, c in extra]
    del model
    torch.cuda.empty_cache()
    return iv, iq, float(run), ex


def main(mode, panel, seed, iters=6000):
    OUT.mkdir(parents=True, exist_ok=True)
    for pn in (PANELS if panel == "all" else panel.split(",")):
        f = OUT / f"imp_{mode}_{pn}_s{seed}.npz"
        if f.exists():
            print("skip", f.name, flush=True); continue
        sp, fl, vis, stations, dates = layer(mode, pn)
        t0 = time.time()
        extra = []
        if mode == "own":  # same model also predicts the official validation view (more visible cells) for the probe
            ev = layer("val", pn); assert list(ev[4]) == list(dates); extra = [ev[:3]]
        iv, iq, loss, ex = train_predict(sp, fl, vis, stations, int(iters), int(seed), log=lambda s: print(pn, s, flush=True), extra=extra)
        np.savez_compressed(f, speed=iv, flow=iq, dates=np.array(dates))
        if ex:
            np.savez_compressed(OUT / f"imp_valown_{pn}_s{seed}.npz", speed=ex[0][0], flow=ex[0][1], dates=np.array(dates))
        print(mode, pn, "seed", seed, "shape", sp.shape, "vis", round(float(vis.mean()), 3), "loss", round(loss, 5),
              "min", round((time.time() - t0) / 60, 1), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
