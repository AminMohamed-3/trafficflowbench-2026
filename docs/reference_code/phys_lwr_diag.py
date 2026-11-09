"""Claude 2026-10-04: where does the S_LWR loss sit, and does a density-only smoother help J_S? Tier 0 screen.
OPT setup of D's 0354 screen (train held dates, frozen F4 OPT predictions retained by D, local truth-projected replica).
Transition classes: oo (both cells observed: error 0), ot/to (one target), tt (both target). Variants change v only
(q kept, so flow state is unchanged): density n = q/v smoothed in time over target cells.
Usage: python docs/reference_code/phys_lwr_diag.py [panel ...]"""
import sys, json, importlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "exp/0354_d_f4_density_score"
P = json.loads((D / "inputs.json").read_text())
sys.path.insert(0, str(ROOT / "src"))
import tfb
tfb.__path__.insert(0, str(ROOT / "src/tfb"))   # 2026-11-09: this root, not inputs.json S_source_root (old lane-s worktree, same code up to line endings)
from tfb import metric
base = importlib.import_module("tfb.lane_s.evaluate_baselines")
data = importlib.import_module("tfb.data")


def held_panel(panel):
    cache = data.load_cache(panel, "train", root=ROOT)
    held = base.subset_cache(cache, [cache.dates.index(d) for d in P["OPT_held_dates"]])
    tv, tq = base.raw_truth(held, ROOT)
    pv, pq = (np.load(D / "F4_predictions/OPT" / panel / f"{n}.npy") for n in ("speed", "flow"))
    return held, np.asarray(tv, float), np.asarray(tq, float), pv.astype(float), pq.astype(float)


def smooth_density(pv, pq, target, w):
    """Centered moving average of density over consecutive target cells (window 2w+1, observed cells untouched)."""
    k = pq / np.maximum(pv, 1.)
    out = k.copy()
    D_, T, L = k.shape
    num = np.zeros_like(k); den = np.zeros_like(k)
    for s in range(-w, w + 1):
        sh = np.roll(k, s, axis=1); ok = np.roll(target, s, axis=1)
        if s > 0: ok[:, :s] = False
        if s < 0: ok[:, s:] = False
        num += np.where(ok, sh, 0); den += ok
    out = np.where(target & (den > 0), num / np.maximum(den, 1), k)
    return np.where(target, pq / np.maximum(out, 1e-6), pv)


def lwr_parts(held, tv, tq, pv, pq):
    """E_LWR numerator split by transition class (approximation: all links observable, no ramp filter)."""
    p = held.stations.sort_values("idx")
    length = np.asarray(metric._parameters(held, ROOT)[0].length_km, float)
    tgt = np.asarray(held.target, bool); el = np.asarray(held.elig, bool)
    v = np.where(tgt, pv, tv); q = np.where(tgt, pq, tq)
    n = q / np.maximum(v, 1) * length; nt = tq / np.maximum(tv, 1) * length
    ok = el[:, :-1] & el[:, 1:] & np.isfinite(n[:, :-1]) & np.isfinite(n[:, 1:]) & np.isfinite(nt[:, :-1]) & np.isfinite(nt[:, 1:])
    err = np.abs(np.diff(n, axis=1) - np.diff(nt, axis=1)); rhs = np.abs(np.diff(nt, axis=1))
    a, b = tgt[:, :-1], tgt[:, 1:]
    out = {"scale": float(rhs[ok].sum())}
    for name, m in (("ot+to", a ^ b), ("tt", a & b)):
        out[name] = round(float(err[ok & m].sum()) / out["scale"], 4)
        out[name + "_share"] = round(float((ok & m).sum()) / ok.sum(), 3)
    return out


def score(held, tv, tq, pv, pq):
    ph = metric.physics_score(held, pv, pq, truth_speed=tv, truth_flow=tq, mode="A")
    st = metric.state_score(tv, tq, pv, pq, target=held.target, lanes=held.stations.sort_values("idx").lanes.to_numpy(), regimes=held.regimes)
    r = ph["regimes"]; lw = np.mean([r[x]["S_LWR"] for x in r]); fd = np.mean([r[x]["S_FD"] for x in r])
    return dict(S_phys=round(ph["S_physics"], 5), S_FD=round(fd, 5), S_LWR=round(lw, 5), S_state=round(st["S_state"], 5),
                J=round(.35 * st["S_state"] + .15 * ph["S_physics"], 5))


def main(*panels):
    for panel in panels or ("D12_I5_N",):
        held, tv, tq, pv, pq = held_panel(panel)
        tgt = np.asarray(held.target, bool)
        print(panel, "target share", round(tgt.mean(), 3), "LWR parts", lwr_parts(held, tv, tq, pv, pq), flush=True)
        ref = score(held, tv, tq, pv, pq); print(" F4", ref, flush=True)
        tq_or = np.where(np.isfinite(tq), tq, pq); tv_or = np.where(np.isfinite(tv), tv, pv)
        print(" oracle v (true speed on target)", score(held, tv, tq, np.where(tgt, tv_or, pv), pq), flush=True)
        for w in (1, 2, 4):
            nv = smooth_density(pv, pq, tgt, w)
            print(f" smooth density w{w}", score(held, tv, tq, nv, pq), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
