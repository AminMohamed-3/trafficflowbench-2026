"""Shared hooked Task 1 state build for rows 25, 26 and 27 (worker R2 piece 25-27, 2026-10-21).
COPY of exp/L_gap2/g2_build.py (sha256 e9d04a5f201a5e39f45bec55992cafb8cb3415e1524b8ce3d4ab6cbd0d49a77d), whose hook is a superset
of exp/L_b6/gapw_build.py (3dcabf0d, row 25) and exp/L_b6/bundle18_build.py (5254b042, row 26): same keys v_old/q_old/v_new/q_new
computed by the same code, plus chain/spec/cls/cs (row 27). Logic unchanged; only these are parameterised:
  OUT anywhere (original asserted the l_gap2/build folder), RSS file next to OUT, the W3 file (--w3, default the registered
  gap_cfg_gapw.json; asserted equal to docs/research/ideas_L10/a/gapw_march_fit.json W3 when that file exists, as the original),
  --arm-dir DIR: the B6 member folder of the tune's b6 block (rewritten into OUT.tune.json as a path relative to
  artifacts/claude_ref, the form asm2_build.parse_arm expects), --s08-dir DIR: the S08 official parquets (base build
  intermediate; default the folder the stored builds used, artifacts/claude_ref/c_budget/s08/official/l127_r4800).
Libraries are imported from the development repo ($TFB_OLD_REPO: docs/reference_code, src), the build reads its fitted inputs
from there (asm2_build ROOT). Writes OUT.state.parquet, OUT.state.json (asm2_final) and OUT.gaprows.npz (hook).
Usage: python gap_hook_build.py OUT.csv TUNE.json [--w3 FILE] [--arm-dir DIR] [--s08-dir DIR] [--panels P1,P2]
  (--panels: asm2_final debug mode, no state parquet; the hook npz is still written = the cheap parity variant)"""
import os, sys, json, threading, time
from pathlib import Path
import psutil
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve(); os.chdir(ROOT)
NT = os.environ.get("LM_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = NT
os.environ["CUDA_VISIBLE_DEVICES"] = ""
p = psutil.Process()
argv = sys.argv[1:]


def opt(f, d=None):
    if f in argv:
        i = argv.index(f); v = argv[i + 1]; del argv[i:i + 2]; return v
    return d


W3F = opt("--w3"); ARMD = opt("--arm-dir"); S08D = opt("--s08-dir")
out, tune = argv[0], argv[1]; rest = argv[2:]
Path(out).parent.mkdir(parents=True, exist_ok=True)
RSS = Path(str(out) + ".rss.txt")


def _watch():
    peak = 0
    while True:
        r = p.memory_info().rss; peak = max(peak, r)
        RSS.write_text("peak_rss_mb %.0f now_mb %.0f\n" % (peak / 2 ** 20, r / 2 ** 20))
        if r > float(os.environ.get("LM_RSS_GB", "4")) * 2 ** 30:
            print("STOP: RSS above LM_RSS_GB", flush=True); os._exit(3)
        time.sleep(15)


threading.Thread(target=_watch, daemon=True).start()


def rel(x):
    return os.path.relpath(Path(x).resolve(), ROOT).replace("\\", "/")


S08 = rel(S08D) if S08D else "artifacts/claude_ref/c_budget/s08/official/l127_r4800"
os.environ["TFB_ASM2_ART"] = json.dumps({f"artifacts/lane_s/0398_s08/official_{s}.parquet": f"{S08}/official_{s}.parquet" for s in ("validation", "private")})
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "docs/reference_code"))
T = json.loads(Path(tune).read_text()); assert not T.get("lm_i511")
if ARMD:   # same tune, B6 member folder moved (arm string "dir:pre:seeds:suf", dir relative to artifacts/claude_ref)
    d = os.path.relpath(Path(ARMD).resolve(), ROOT / "artifacts/claude_ref").replace("\\", "/"); assert ":" not in d
    for sp, x in T["b6"].items():
        a = x["arm"].split(":"); a[0] = d; x["arm"] = ":".join(a)
    tune = str(out) + ".tune.json"; Path(tune).write_text(json.dumps(T, indent=1))
    print("ARM_DIR", d, "->", tune, flush=True)
import numpy as np
import pandas as pd
import asm2_build as AB
_rp = AB.Stages.run_panel
W3F = W3F or str(ROOT / "artifacts/claude_ref/l_b6/gapw/gap_cfg_gapw.json")
WNEW = {int(k): v for k, v in json.loads(Path(W3F).read_text())["weights"].items()}
_fit = ROOT / "docs/research/ideas_L10/a/gapw_march_fit.json"
if _fit.exists():
    assert WNEW == {int(k): v for k, v in json.loads(_fit.read_text())["W3"].items()}
CAP = {}


def run_panel_split(self, split, *a, **k):
    """Per-split knob settings (tune "lm_split": {split: {polish_chain, s3_scale}}), as lm_build.py (copy of s2_build.py)."""
    o = self.tune.get("lm_split", {}).get(split)
    if o:
        self.chain = o["polish_chain"]; self.s3scale = list(o["s3_scale"])
        assert getattr(self, "s3bag", None), "s3 bag path needed for the per-split scale"
    return _rp(self, split, *a, **k)


def run_panel_cap(self, split, panel, c, d, t, sidx, *a, **k):
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
    # gap stage inputs in M (gap_pred) order, mapped to the key order of the panel's gap rows
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
    print("HOOK", split, panel, "gap rows", int(g.sum()), "old == build clip EXACT", "new mean |dv| %.4f |dq| %.3f" % (
        np.abs(res["new"][0] - res["old"][0])[g].mean(), np.abs(res["new"][1] - res["old"][1])[g].mean()), flush=True)
    return out_


AB.Stages.run_panel = run_panel_cap
import s2stage as S2
S2.OUT = S2.P.ROOT / "artifacts/claude_ref/c_budget/s2" / "all_l63_m2000_n3000000_r600"
assert (S2.OUT / "m_v.txt").exists() and (S2.OUT / "m_q.txt").exists()
import asm2_final
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only", "--tune", tune] + rest
asm2_final.main()
if CAP:
    keys = sorted(CAP)
    np.savez_compressed(str(out) + ".gaprows.npz", keys=np.array(keys), **{f"{k}|{f}": CAP[k][f] for k in keys for f in CAP[k]})
    print("HOOK_BUILD_DONE", len(keys), "panels,", sum(len(CAP[k]["d"]) for k in keys), "gap rows ->", str(out) + ".gaprows.npz", flush=True)
