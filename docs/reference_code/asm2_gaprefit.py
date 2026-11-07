"""asm2 (ledger 0475): refit the gap specialist v3 (ledger 0412) from the raw release into artifacts/claude_ref/asm2/gaprefit/:
gap_build.build('off', split, panel) (simulated selector-like blocks on the split's own official visible data), then the
LightGBM fit/predict of gap_probe.py (255 leaves, 1500 rounds, base an, seeds 0/1, 3 threads), saved as offpred_v3.npz in the
same key layout as artifacts/claude_ref/gap/offpred_v3.npz (unsmoothed; asm applies smooth 1 and the shipped blend weights).
Usage: python docs/reference_code/asm2_gaprefit.py"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, time, types
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asm2_phys_lwr_diag as _P
sys.modules["phys_lwr_diag"] = _P
import gap_lib as G
OUT = G.ROOT / os.environ.get("TFB_ASM2_GAPOUT", "artifacts/claude_ref/asm2/gaprefit")   # second dir = same-platform rerun test
G.OUT = OUT
import gap_build as GB
# gap_train imports flow_own at module level, which needs the lane-d worktree (OWN instrument binding). The specialist
# fit/predict used here never touches it (only the I-S-OWN blend-weight fit does), so a stub keeps this refit portable.
sys.modules.setdefault("flow_own", types.ModuleType("flow_own"))
import gap_train as GT


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    a = types.SimpleNamespace(leaves=255, rounds=1500, base="an")
    PRED = {}
    for split in ("validation", "private"):
        for pn in G.PANELS:
            if not (OUT / f"rows_off_{split}_{pn}.npz").exists():
                GB.build("off", split, pn)
        R, X = GT.load("off", split, G.PANELS)
        names = [str(n) for n in R[G.PANELS[0]]["names"]]
        models, keep = GT.fit(R, X, names, a); del X
        for pn in G.PANELS:
            r = R[pn]; M = r["Mev"]
            if len(M) == 0:
                continue
            sv, sq = GT.predict(models, keep, names, r["Xev"], a)
            s = M[:, 6]
            PRED[(split, pn)] = (M, np.clip(sv * r["vf"][s], 3, 1.3 * r["vf"][s]), np.clip(sq * r["cap"][s], 0, 1.3 * r["cap"][s]))
        del R, models
        print(split, "specialist trained and predicted", round(time.time() - t0), "s", flush=True)
    np.savez_compressed(OUT / "offpred_v3.npz", **{f"{sp}|{pn}|{k}": v for (sp, pn), x in PRED.items() for k, v in zip(("M", "v", "q"), x)})
    rp = G.ROOT / os.environ.get("TFB_ASM2_GAPREF", "artifacts/claude_ref/gap/offpred_v3.npz")
    ref = np.load(rp) if rp.exists() else None
    if ref is not None:
        for (sp, pn), (M, v, q) in PRED.items():
            k = f"{sp}|{pn}|"
            same_M = np.array_equal(ref[k + "M"], M)
            print("cmp", sp, pn, "M equal", same_M, "max|dv|", float(np.abs(ref[k + "v"] - v).max()) if same_M else None,
                  "max|dq|", float(np.abs(ref[k + "q"] - q).max()) if same_M else None, flush=True)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
