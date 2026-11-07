"""Row 26 step 4: S7b state build (worker R2 2026-10-21). COPY of exp/L_b6/bundle18_s7bbuild.py (sha256
603311b63d8a61d2fc0229f488ef86572e77de7784dec5b4898911c3f7763979) = s7b_build.py on the S2b chain: the S2 state build path (C54
lm_build per-split polish chain and stage-3 scale) with per-split stage-3 models (tune "s3_split": shipped s3L + 60 own-month trees,
C_*_mar_all for validation rows, C_*_apr_all for private rows), shipped gap config (S7B_GAPCFG default s2). Logic unchanged;
parameterised: OUT anywhere, RSS file next to OUT, --s08-dir (base-build S08 official parquets, default the stored folder).
Usage: python s7b_build.py OUT.csv TUNE.json [--s08-dir DIR] [--panels P]"""
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
S08D = None
if "--s08-dir" in argv:
    i = argv.index("--s08-dir"); S08D = argv[i + 1]; del argv[i:i + 2]
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
S08 = os.path.relpath(Path(S08D).resolve(), ROOT).replace("\\", "/") if S08D else "artifacts/claude_ref/c_budget/s08/official/l127_r4800"
ART = {f"artifacts/lane_s/0398_s08/official_{s}.parquet": f"{S08}/official_{s}.parquet" for s in ("validation", "private")}
if os.environ.get("S7B_GAPCFG", "s2") == "gapw":   # default s2: the shipped (C64) gap config, so non-gap rows = C65 non-gap rows
    ART["artifacts/claude_ref/gap/probes/c16_gap_v3sm1_on_c13_delta.json"] = "artifacts/claude_ref/l_b6/gapw/gap_cfg_gapw.json"
print("S7B_GAPCFG", os.environ.get("S7B_GAPCFG", "s2"), flush=True)
os.environ["TFB_ASM2_ART"] = json.dumps(ART)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "docs/reference_code"))
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
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only", "--tune", tune] + rest
asm2_final.main()
print("B18_S7B_BUILD_DONE", out, flush=True)
