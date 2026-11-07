"""phys5 (C helper, 2026-10-05): where does C14's S_LWR loss sit on OPT, and how far above the iid-noise floor is it?
Floor: target density replaced by truth x exp(eps), eps ~ N(0, s_k) with s_k = panel free-flow cell noise of log(q/v)
(double difference: time high-pass x adjacent-station, / sqrt 3). Classes by truth speed at the target cell:
F free (v > .85 vf), C congested (v < v_cut), M middle. Excess = actual - floor per class.
Usage: python docs/reference_code/phys5_diag.py [panel ...]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import phys4_common as C
import phys_lwr_diag as P

OUT = P.ROOT / "artifacts/claude_ref/phys5"


def noise_sd(tv, tq, el, vf):
    ok = el & np.isfinite(tv) & np.isfinite(tq) & (tq > 0) & (tv > .9 * vf[None, None, :])
    res = {}
    for name, x in (("v", np.log(np.where(ok, tv, np.nan))), ("q", np.log(np.where(ok, tq, np.nan)))):
        hp = x[:, 1:-1] - .5 * (x[:, :-2] + x[:, 2:]); dd = hp[:, :, :-1] - hp[:, :, 1:]
        dd = dd[np.isfinite(dd)]; res[name] = float(1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(3))
    return res


def load(pn, tag=None):
    """C14 OPT arrays; with tag (phys5_s3 pred tag, e.g. 'AL+BL') the stage predictions are applied on their rows."""
    held, tv, tq, _, _ = P.held_panel(pn)
    z = np.load(OUT / f"c14_opt_{pn}.npz")
    v, q = z["v"].astype(float), z["q"].astype(float)
    if tag:
        import phys5_s3 as S
        v, q = S.apply(pn, tag, v, q, np.load(OUT / f"feat_{pn}.npz")["ix"])
    return held, np.asarray(tv, float), np.asarray(tq, float), v, q, z["p"].astype(float)


def main(*panels):
    import os
    tag = os.environ.get("PHYS5_TAG") or None
    res = {}
    rng = np.random.default_rng(0)
    for pn in panels or C.PANELS:
        held, tv, tq, v, q, p = load(pn, tag)
        tr, c = C.lwr_mask(held, tv, tq, v, q)
        T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); ln = c["length"]; vf = c["vf"]
        sd = noise_sd(tv, tq, el, vf); sk = np.hypot(sd["v"], sd["q"])
        nt = tq / np.maximum(tv, 1) * ln
        n = np.where(T, q / np.maximum(v, 1) * ln, nt)
        nf = np.where(T, nt * np.exp(rng.normal(0, sk, nt.shape)), nt)
        dt = np.diff(nt, axis=1); scale = np.abs(dt)[tr].sum()
        e = np.abs(np.diff(n, axis=1) - dt); ef = np.abs(np.diff(nf, axis=1) - dt)
        vc = held.stations.sort_values("idx").v_cut.to_numpy(float)
        cls = np.where(tv > .85 * vf[None, None, :], 0, np.where(tv < vc[None, None, :], 2, 1))
        # attribute each transition to the target cell in it (if both target, to the later one's class = max class)
        a, b = T[:, :-1], T[:, 1:]
        tc = np.where(b, cls[:, 1:], cls[:, :-1]); tc = np.where(a & b, np.maximum(cls[:, :-1], cls[:, 1:]), tc)
        r = dict(sd_v=round(sd["v"], 4), sd_q=round(sd["q"], 4), E=e[tr].sum() / scale, E_floor=ef[tr].sum() / scale)
        for k, nm in enumerate("FMC"):
            m = tr & (a | b) & (tc == k)
            r[nm + "_E"] = e[m].sum() / scale; r[nm + "_floor"] = ef[m].sum() / scale; r[nm + "_ncell"] = float((T & (cls == k)).sum() / T.sum())
        # misclassified (pred class != truth class) share
        pc = np.where(v > .85 * vf[None, None, :], 0, np.where(v < vc[None, None, :], 2, 1))
        mis = T & (pc != cls) & ((pc == 2) | (cls == 2))
        mt = np.where(b, mis[:, 1:], mis[:, :-1]) | (a & b & (mis[:, :-1] | mis[:, 1:]))
        r["mis_E"] = e[tr & mt].sum() / scale; r["mis_ncell"] = float(mis.sum() / T.sum())
        # top 1% cells by error
        ee = e[tr & (a | b)]; ee = np.sort(ee)[::-1]; r["top1pct_share"] = ee[:max(1, len(ee) // 100)].sum() / ee.sum()
        res[pn] = {k: round(float(x), 4) for k, x in r.items()}
        print(pn, res[pn], flush=True)
    keys = list(next(iter(res.values())).keys())
    res["mean"] = {k: round(float(np.mean([res[x][k] for x in res])), 4) for k in keys}
    print("MEAN", res["mean"])
    (OUT / f"diag{('_' + tag.replace('+', '_')) if tag else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
