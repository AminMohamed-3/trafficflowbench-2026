# R3 (rows 28-35) copy of exp/L_gapfv/gfv73_build.py (sha256 c73ee7717129190d6c2462c9fb381bd435bdfd73d824022fe95b6b1df5aef110).
# Edits: ROOT = $TFB_ROOT; RSS file next to OUT; no output folder check. Logic unchanged. Usage: python gfv73_build.py OUT.csv TUNE.json --panels P
"""L_gapfv c73p copy of gfv_build.py (only the RSS file gfv73_build_rss.txt and the output folder check l_gapfv/c73p/build/ differ; tune exp/L_gapfv/tune_gapfv_c73p.json = FV-INEL March arm + FV2 April arm). gfv_build.py text: L_gapfv step 1: the FV state build with the gap-input hook.
Body = exp/L_fvft/mar_build.py (per-split stage-3 models "s3_split", S7B_GAPCFG default s2 = shipped gap config) + the passive
capture of exp/L_gap2/g2_build.py (gap stage INPUTS at the official gap&target rows: chain = stage before "gap", spec_v3 smoothed,
class, 1.15 vf clip, and the GAPW arm). Tune = exp/L_gapfv/tune_gapfv.json (FV arms in both splits: mar/serve valown_, apr/serve
privown_; + S7b stage-3 models), so the parquet must equal l_fvft/mar/build (validation rows) and l_fvft/apr/build (private rows).
Usage (repo root): python exp/L_gapfv/gfv_build.py artifacts/claude_ref/l_gapfv/build/t1.csv exp/L_gapfv/tune_gapfv.json [--panels P]
(one panel per process keeps RSS under 3.5 GiB; with --panels asm2_final writes no parquet, only OUT.<P>.gaprows.npz)"""
import os, sys, json, threading, time
from pathlib import Path
import psutil
ROOT = Path(os.environ["TFB_ROOT"]); os.chdir(ROOT)
NT = os.environ.get("LM_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = NT
os.environ["CUDA_VISIBLE_DEVICES"] = ""
p = psutil.Process()


def _watch():
    peak = 0
    while True:
        r = p.memory_info().rss; peak = max(peak, r)
        Path(str(sys.argv[1]) + ".rss.txt").write_text("peak_rss_mb %.0f now_mb %.0f\n" % (peak / 2 ** 20, r / 2 ** 20))
        if r > float(os.environ.get("LM_RSS_GB", "3.5")) * 2 ** 30:
            print("STOP: RSS above LM_RSS_GB", flush=True); os._exit(3)
        time.sleep(15)


threading.Thread(target=_watch, daemon=True).start()
S08 = "artifacts/claude_ref/c_budget/s08/official/l127_r4800"
ART = {f"artifacts/lane_s/0398_s08/official_{s}.parquet": f"{S08}/official_{s}.parquet" for s in ("validation", "private")}
assert os.environ.get("S7B_GAPCFG", "s2") == "s2"   # the shipped gap config, as the FV builds and the g2_build hook build
os.environ["TFB_ASM2_ART"] = json.dumps(ART)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "docs/reference_code"))
out, tune = sys.argv[1], sys.argv[2]
T = json.loads(Path(tune).read_text()); assert not T.get("lm_i511") and T.get("s3") and not T.get("s4")
import numpy as np
import pandas as pd
import asm2_build as AB
_rp = AB.Stages.run_panel
WNEW = {int(k): v for k, v in json.loads((ROOT / "artifacts/claude_ref/l_b6/gapw/gap_cfg_gapw.json").read_text())["weights"].items()}
CAP = {}


def run_panel_split(self, split, *a, **k):
    """exp/L_fvft/mar_build.py run_panel_split (lm_split knobs + s3_split stage-3 models), unchanged."""
    o = self.tune.get("lm_split", {}).get(split)
    if o:
        self.chain = o["polish_chain"]; self.s3scale = list(o["s3_scale"])
        assert getattr(self, "s3bag", None), "s3 bag path needed for the per-split scale"
    sm = self.tune.get("s3_split", {}).get(split)
    if sm:
        import lightgbm as lgb
        self.s3bag = [tuple(lgb.Booster(model_file=str(self._in(ROOT / f))) for f in pair) for pair in sm]
        print("S7B_S3_SPLIT", split, sm, flush=True)
    return _rp(self, split, *a, **k)


def run_panel_cap(self, split, panel, c, d, t, sidx, *a, **k):
    """exp/L_gap2/g2_build.py run_panel_cap, unchanged except the inner call (run_panel_split above)."""
    out_ = run_panel_split(self, split, panel, c, d, t, sidx, *a, **k)
    import gap_lib as G
    comps = self.comps; assert "gap" in comps and comps[comps.index("gap") + 1] == "clip"
    prev = comps[comps.index("gap") - 1]
    tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool); I = (d, t, sidx)
    if f"{split}|{panel}|M" not in self.gap_pred:
        assert not (gap & tgt).any(); return out_

    def grid(a_):
        g = np.full(tgt.shape, np.nan); g[d, t, sidx] = a_; return g

    st = c.stations.sort_values("idx")
    topo = pd.read_csv(ROOT / f"data/raw/kaggle_public/corridors/{panel}/network/lwr_mainline_topology.csv")
    cl_ = 1.15 * topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
    cs = cl_[sidx]
    M = self.gap_pred[f"{split}|{panel}|M"]; gs = M[:, 6]
    vs, qs = G.smooth_rows(M, self.gap_pred[f"{split}|{panel}|v"], self.gap_sm), G.smooth_rows(M, self.gap_pred[f"{split}|{panel}|q"], self.gap_sm)
    dd, tt = M[:, 1], M[:, 2] + M[:, 5]
    cl = np.clip(M[:, 3], 0, 1)
    res = {}
    for arm, W in (("old", self.gap_w), ("new", WNEW)):
        v, q = grid(out_[prev][0]), grid(out_[prev][1])
        memv = [v[dd, tt, gs].copy(), vs]; memq = [q[dd, tt, gs].copy(), qs]
        v[dd, tt, gs] = sum(np.array([W[0][0][i], W[1][0][i]])[cl] * memv[i] for i in range(2))
        q[dd, tt, gs] = sum(np.array([W[0][1][i], W[1][1][i]])[cl] * memq[i] for i in range(2))
        vv, qq = v[I], q[I]
        vv = np.where(vv > cs + 1e-9, cs, vv)
        res[arm] = (vv, qq)
    ok = np.array_equal(res["old"][0], out_["clip"][0]) and np.array_equal(res["old"][1], out_["clip"][1])
    assert ok, (split, panel, "old-weight recompute differs from the build clip output")
    g = gap[d, t, sidx]
    assert int(g.sum()) == len(M)
    v0, q0 = grid(out_[prev][0]), grid(out_[prev][1])
    key = np.asarray(d, np.int64) * 1000000 + np.asarray(t, np.int64) * 1000 + np.asarray(sidx, np.int64)
    gi = np.flatnonzero(g); kg = key[gi]; o = np.argsort(kg)
    kk = dd.astype(np.int64) * 1000000 + tt.astype(np.int64) * 1000 + gs.astype(np.int64)
    j = o[np.searchsorted(kg[o], kk)]; assert (kg[j] == kk).all() and len(np.unique(j)) == len(j) == len(gi)
    ch_v = np.full(len(gi), np.nan); ch_q = np.full(len(gi), np.nan); sp_v = np.full(len(gi), np.nan); sp_q = np.full(len(gi), np.nan); cls_ = np.full(len(gi), -1)
    ch_v[j] = v0[dd, tt, gs]; ch_q[j] = q0[dd, tt, gs]; sp_v[j] = vs; sp_q[j] = qs; cls_[j] = cl
    assert np.isfinite(ch_v).all() and np.isfinite(ch_q).all() and np.isfinite(sp_v).all() and (cls_ >= 0).all()
    CAP[f"{split}|{panel}"] = dict(d=np.asarray(d)[g], t=np.asarray(t)[g], s=np.asarray(sidx)[g], v_old=res["old"][0][g], q_old=res["old"][1][g],
                                   v_new=res["new"][0][g], q_new=res["new"][1][g], chain_v=ch_v, chain_q=ch_q, spec_v=sp_v, spec_q=sp_q,
                                   cls=cls_, cs=np.asarray(cs)[g])
    print("GFV_HOOK", split, panel, "gap rows", int(g.sum()), "old == build clip EXACT", flush=True)
    return out_


AB.Stages.run_panel = run_panel_cap
import s2stage as S2
S2.OUT = S2.P.ROOT / "artifacts/claude_ref/c_budget/s2" / "all_l63_m2000_n3000000_r600"
assert (S2.OUT / "m_v.txt").exists() and (S2.OUT / "m_q.txt").exists()
import asm2_final
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only", "--tune", tune] + sys.argv[3:]
asm2_final.main()
assert CAP
keys = sorted(CAP)
pa = [x for x in sys.argv[3:] if x.startswith("--panels")]
tag = (sys.argv[sys.argv.index("--panels") + 1].replace(",", "+") + ".") if "--panels" in sys.argv else ""
np.savez_compressed(str(out) + f".{tag}gaprows.npz", keys=np.array(keys), **{f"{k}|{f}": CAP[k][f] for k in keys for f in CAP[k]})
print("GFV_BUILD_DONE", len(keys), "panels,", sum(len(CAP[k]["d"]) for k in keys), "gap rows ->", str(out) + f".{tag}gaprows.npz", flush=True)
