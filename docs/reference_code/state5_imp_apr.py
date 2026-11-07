"""Claude helper 2026-10-04 (state5): run the frozen B6 per-split imputer (b6full_imp.train_predict, unchanged recipe) on the
I-S-OWN April (private) blinded view, so the C13 stack can be rebuilt on OWN April. Same hiding rule as b6full_imp mode own
(official target | OWN score target | gap hidden). Output artifacts/claude_ref/state5/imp_ownapr_<panel>_s<seed>.npz.
Usage: python docs/reference_code/state5_imp_apr.py <panel|all> <seed> [iters]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import b6full_imp as BI

OUT = P.ROOT / "artifacts/claude_ref/state5"


def layer_apr(panel):
    import flow_own as FO
    items = [i for i in json.loads(Path(P.P["OWN_metadata_binding"]["manifest"]).read_text())["items"]
             if i["period"] == "private" and i["panel"] == panel]
    view, tv, tq, st_, hidden = FO.own.load_item(P.ROOT, FO.binding.bound_own_item(P.P, items[0]))
    hide = np.asarray(view.target, bool) | np.asarray(st_, bool) | np.asarray(view.gap, bool)
    sp = np.asarray(view.speed, float); fl = np.asarray(view.flow, float)
    vis = np.asarray(view.elig, bool) & ~hide & np.isfinite(sp) & np.isfinite(fl)
    return np.where(vis, sp, np.nan), np.where(vis, fl, np.nan), vis, view.stations, list(view.dates)


def main(panel, seed, iters=6000):
    for pn in (BI.PANELS if panel == "all" else panel.split(",")):
        f = OUT / f"imp_ownapr_{pn}_s{seed}.npz"
        if f.exists():
            print("skip", f.name, flush=True); continue
        sp, fl, vis, stations, dates = layer_apr(pn)
        t0 = time.time()
        iv, iq, loss, _ = BI.train_predict(sp, fl, vis, stations, int(iters), int(seed), log=lambda s: print(pn, s, flush=True))
        np.savez_compressed(f, speed=iv, flow=iq, dates=np.array(dates))
        print("ownapr", pn, "seed", seed, "shape", sp.shape, "vis", round(float(vis.mean()), 3), "loss", round(loss, 5),
              "min", round((time.time() - t0) / 60, 1), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
