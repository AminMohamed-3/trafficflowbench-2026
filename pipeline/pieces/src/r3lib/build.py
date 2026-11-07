# R3 (rows 28-35) parameterised copy of exp/L_isinel/isi_build.py (sha256 5d28a4bb369f5f5e9d80a07e5280d24246de659cd3e1167b821938cdd238d1d5; = mar_build.py / fv2_build.py / fvi_build.py body,
# which differ only in the output folder check, RSS file and done line). Edits: ROOT = $TFB_ROOT (old-layout root) instead of
# parents[2]; RSS file next to OUT; no output folder check; done line R3_BUILD_DONE. Logic unchanged. Usage: python build.py OUT.csv TUNE.json
"""L_isinel copy of exp/L_fvft/fv2_build.py (only the output folder check l_isinel/, the RSS file exp/L_isinel/logs/isi_build_rss.txt and the done line ISI_BUILD_DONE differ; run with an exp/L_isinel/tunes/cand_*.json tune). fv2_build.py text: L_fvft FV2 copy of exp/L_fvft/mar_build.py (only the output folder check l_fvft/fv2/, the RSS file fv2_logs/fv2_build_rss.txt and the done line FV2_BUILD_DONE differ; run with fv2_tune_mar.json, fv2_tune_apr.json, fv2_tune_apr_par.json or mar_tune.json for the March parity build). mar_build.py text: L_fvft attempt 2 copy of exp/L_b6/bundle18_s7bbuild.py (only the output folder check, the RSS file and the done line differ; run with exp/L_fvft/mar_tune.json = bundle18_tune_s7b.json with ONLY the validation b6 arm -> the FV serve dir, or with bundle18_tune_s7b.json itself for the parity build). bundle18_s7bbuild.py text: BUNDLE18 copy of exp/L_b6/s7b_build.py (only the output folder check, RSS file and done line differ; run with bundle18_tune_s7b.json = tune_s2b.json b6 block + S2b-chain C models). Original:
L_b6 S7b, official side (copy of exp/L_b6/s7_build.py; per-split stage-3 models via tune "s3_split": {split: [[yq, yv]]}): the S2 state build path (copy of exp/L_b6/s2_build.py: C54..C64 lm_build path, per-split polish chain and
stage-3 scale) with the GAPW gap config (TFB_ASM2_ART override of the shipped gap config by artifacts/claude_ref/l_b6/gapw/gap_cfg_gapw.json,
= the C65 gap stage) and the stage models given by the tune file (tune "s3" bag = stage 3, tune "s4" bag = stage 4, asm2_final.py):
  exp/L_b6/tune_s7_parity.json  shipped s3_yqL/s3_yvL and s4_AL+FL_yq through the same tune keys -> parquet must equal
                                artifacts/claude_ref/l_b6/gapw/build/t1_gapw.state.parquet bit for bit (the C65 state source)
  exp/L_b6/tune_s7.json         the refit all-March models (artifacts/claude_ref/l_b6/s7/models)
Writes OUT.state.parquet / OUT.state.json (OUT under artifacts/claude_ref/l_b6/s7b/).
Usage (repo root): python exp/L_b6/s7_build.py OUT.csv TUNE.json"""
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
        if r > float(os.environ.get("LM_RSS_GB", "4")) * 2 ** 30:
            print("STOP: RSS above LM_RSS_GB", flush=True); os._exit(3)
        time.sleep(15)


threading.Thread(target=_watch, daemon=True).start()
S08 = "artifacts/claude_ref/c_budget/s08/official/l127_r4800"
ART = {f"artifacts/lane_s/0398_s08/official_{s}.parquet": f"{S08}/official_{s}.parquet" for s in ("validation", "private")}
if os.environ.get("S7B_GAPCFG", "s2") == "gapw":   # default s2: the shipped (C64) gap config, so non-gap rows = C65 non-gap rows
    ART["artifacts/claude_ref/gap/probes/c16_gap_v3sm1_on_c13_delta.json"] = "artifacts/claude_ref/l_b6/gapw/gap_cfg_gapw.json"
print("S7B_GAPCFG", os.environ.get("S7B_GAPCFG", "s2"), flush=True)
os.environ["TFB_ASM2_ART"] = json.dumps(ART)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "docs/reference_code"))
out, tune = sys.argv[1], sys.argv[2]
T = json.loads(Path(tune).read_text()); assert not T.get("lm_i511") and T.get("s3") and not T.get("s4")  # stage 4: default path as tune_s2 (an s4 bag changes 8,565 flow rows)
import asm2_build as AB
_rp = AB.Stages.run_panel


def run_panel_split(self, split, *a, **k):
    """Per-split knob settings (tune "lm_split": {split: {polish_chain, s3_scale}}), as lm_build.py / s2_build.py."""
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


AB.Stages.run_panel = run_panel_split
import s2stage as S2
S2.OUT = S2.P.ROOT / "artifacts/claude_ref/c_budget/s2" / "all_l63_m2000_n3000000_r600"
assert (S2.OUT / "m_v.txt").exists() and (S2.OUT / "m_q.txt").exists()
import asm2_final
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only", "--tune", tune] + sys.argv[3:]
asm2_final.main()
print("R3_BUILD_DONE", out, flush=True)
