"""gap2: per-ramp calibrated congestion sensors for on-ramps (OR, as state6 v6) AND off-ramps (FR, new). Off-ramp flow drops
vs mainline when the attached mainline is congested (log FR - log main shifts -.31..-.54, gap2_fr.py), so FR ramps are a second
in-gap congestion sensor. Copy of state6_gaptrain.ramp_calib_fields with a ramp-type argument.
calib_fields(pn, split, kind, types=("OR",)|("FR",)|("OR","FR")) -> {ZC1, ZW1, ZC2, ZW2, ZC4, ZW4} (D, T, S); also the per-ramp
SNR list via calib_fields(..., info=True)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_lib as G
import warnings; warnings.simplefilter("ignore")
S6 = G.ROOT / "artifacts/claude_ref/state6"


def calib_fields(pn, split, kind, types=("OR",), info=False, min_shift=.05):
    sys.path.insert(0, str(G.ROOT / "src"))
    from tfb.data import load_cache
    c = load_cache(pn, split); A = c.arrays
    rf = np.asarray(A["ramp_flow"], float); re = np.asarray(A["ramp_elig"], bool)
    rf = np.where(re, rf, np.nan)
    lr = np.log(rf + 5.); an = lr - np.nanmedian(lr, 0, keepdims=True)
    D, T, _ = rf.shape
    an3 = np.nanmean(np.stack([np.roll(an, k, 1) for k in (-1, 0, 1)], 0), 0)
    if kind == "own":
        z = np.load(S6 / f"base_{'mar' if split == 'validation' else 'apr'}_{pn}.npz")
        vis = np.where(z["target"], np.nan, z["vobs"].astype(float))
    else:
        vis = np.where(np.asarray(A["target"], bool) | np.asarray(A["gap"], bool), np.nan, np.asarray(A["speed"], float))
    st = c.stations.sort_values("idx"); vc = st.v_cut.to_numpy(float); S = len(vc)
    Cg = np.where(np.isfinite(vis), vis < vc[None, None, :], False); Vi = np.isfinite(vis)
    zs, ws, ss, inf = [], [], [], []
    for r in c.ramps.itertuples():
        if r.ramp_type not in types or not np.isfinite(r.station_idx):
            continue
        s0 = int(r.station_idx); sl = slice(max(0, s0 - 1), min(S, s0 + 2))
        anyc = Cg[:, :, sl].any(2); anyv = Vi[:, :, sl].any(2)
        a = an3[:, :, int(r.idx)]; ok = np.isfinite(a)
        mc_ = ok & anyc; mf_ = ok & anyv & ~anyc
        if mc_.sum() < 20 or mf_.sum() < 100:
            continue
        mc, mf = a[mc_].mean(), a[mf_].mean()
        var = .5 * (a[mc_].var() + a[mf_].var()) + 1e-3
        if mc - mf > -min_shift:
            continue
        zs.append(np.clip((a - mf) / (mc - mf), -1, 2)); ws.append((mc - mf) ** 2 / var); ss.append(s0)
        inf.append((r.ramp_type, s0, mc - mf, (mc - mf) ** 2 / var))
    out = {}
    for nb in (1, 2, 4):
        num = np.zeros((D, T, S)); den = np.zeros((D, T, S))
        for zr, wr, s0 in zip(zs, ws, ss):
            ok = np.isfinite(zr)
            for s in range(max(0, s0 - nb), min(S, s0 + nb + 1)):
                num[:, :, s] += np.where(ok, wr * zr, 0); den[:, :, s] += np.where(ok, wr, 0)
        out[f"ZC{nb}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[f"ZW{nb}"] = den
    return (out, inf) if info else out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    for pn in G.PANELS:
        _, inf = calib_fields(pn, split, "off", ("OR", "FR"), info=True)
        for t in ("OR", "FR"):
            x = [i for i in inf if i[0] == t]
            if x:
                print(split, pn, t, "n", len(x), "median shift %.2f median snr %.2f sum snr %.1f" % (np.median([i[2] for i in x]), np.median([i[3] for i in x]), sum(i[3] for i in x)))
