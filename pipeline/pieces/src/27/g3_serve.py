"""Row 27 G3 serve: v6r gap specialist predictions on the official gap rows from the SAVED LightGBM models (worker R2 2026-10-21).
The fit script exp/L_gap2/g3_v6.py (sha256 349338609be2dd327030a0e14b7346825ba65bbbed8ab3b341ec9f80f386b7e9) ran
docs/reference_code/state6_gaptrain.py main (sha256 17ae50a6255aa012f96796907c677ec5353dce1fe69b51d8b4ed9ae046950dbf) "v6r off
SPLIT --calib": load() -> fit() -> predict on R[pn]["Xev"] -> pred_v6r_off_SPLIT.npz + model_v6r_off_SPLIT_{v,q}.txt. Here fit() is
replaced by loading those two model files; the eval features are built by the same functions as load() (rows_off_SPLIT_PANEL.npz
eval part + ramp_fields + ramp_calib_fields(kind off) via ramp_feats on Mev; the training matrix and its subsample are not needed
for prediction); the prediction lines are copied from state6_gaptrain.main unchanged. Feature names asserted equal to the stored
names_v6r.json. Usage: python g3_serve.py SPLIT MODEL_V MODEL_Q NAMES.json OUT_DIR [--panels P1,P2]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
sys.path.insert(0, str(ROOT / "docs/reference_code")); sys.path.insert(0, str(ROOT / "src"))
import state6_gaptrain as T
import lightgbm as lgb
G = T.G
a = sys.argv[1:]
panels = None
if "--panels" in a:
    i = a.index("--panels"); panels = a[i + 1].split(","); del a[i:i + 2]
period, mv, mq, namesf, outd = a
t0 = time.time()
names_saved = json.loads(Path(namesf).read_text())
models = [lgb.Booster(model_file=mv), lgb.Booster(model_file=mq)]
ia, iq = names_saved.index("an_v"), names_saved.index("an_q")
out = {}
for pn in G.PANELS:
    if panels and pn not in panels:
        continue
    with np.load(G.OUT / f"rows_off_{period}_{pn}.npz") as z:
        r = {k: z[k] for k in z.files if k not in ("Xtr",)}
    if len(r["Mev"]) == 0:
        out[f"{pn}_v"] = np.zeros(0, np.float32); out[f"{pn}_q"] = np.zeros(0, np.float32); continue
    Fd = T.ramp_fields(pn, period); Fd.update(T.ramp_calib_fields(pn, period, "off"))
    rn, xe = T.ramp_feats(Fd, r["Mev"])
    assert [str(x) for x in r["names"]] + list(rn) == names_saved, (pn, "feature names differ from names_v6r.json")
    Xe = np.concatenate([r["Xev"], xe], 1)
    # --- copied from state6_gaptrain.main (prediction part) ---
    s = r["Mev"][:, 6]
    sv = models[0].predict(Xe, num_threads=3) + Xe[:, ia]; sq = models[1].predict(Xe, num_threads=3) + Xe[:, iq]
    vs = sv * r["vf"][s]; qs = sq * r["cap"][s]
    out[f"{pn}_v"] = np.clip(vs, 3, 1.3 * r["vf"][s]).astype(np.float32); out[f"{pn}_q"] = np.clip(qs, 0, 1.3 * r["cap"][s]).astype(np.float32)
    print("g3", period, pn, len(Xe), "rows", round(time.time() - t0), "s", flush=True)
Path(outd).mkdir(parents=True, exist_ok=True)
f = Path(outd) / (f"pred_v6r_off_{period}.npz" if not panels else f"pred_v6r_off_{period}_{'_'.join(panels)}.npz")
np.savez_compressed(f, **out)
print("G3_SERVE_DONE", f, round(time.time() - t0), "s", flush=True)
