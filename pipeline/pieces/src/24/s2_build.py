"""Row 24 (C64) state build: official Task 1 state rows of the C54..C59 state build path with the B6R (S2) members (final_solution
copy, R2 worker). Parameterised copy of exp/L_b6/s2_build.py (sha256 4165f31f7231c994a727c6be8a5e92d275903949f45f6f428f36e6c2eaecbc0e),
itself a copy of exp/L_micro/lm_build.py without the lm_i511 branch; the script that wrote artifacts/claude_ref/l_b6/s2/build/
t1.csv.state.parquet (exp/L_b6/s2_finish.sh: LM_THREADS=4 LM_RSS_GB=5). Tune src/24/tune_s2.json = exp/L_b6/tune_s2.json =
tune_lm2.json (row 20's knobs: lm_split stage 3 scale .8/.7, polish a 1.0, s3 bag) with the b6 block -> arm l_b6/s2/arm36k
(the served B6R members) and weights artifacts/claude_ref/l_b6/s2/b6w_s2.json. Same text as src/20/lm_build.py (whose docstring
lists the differences to the original; here: B6_DIR = the row 24 serve output, original artifacts/claude_ref/l_b6/s2/arm36k;
RSS file original exp/L_b6/logs/s2_build_rss.txt; the "l_b6/s2/ in OUT" assert dropped); defaults LM_THREADS 4, LM_RSS_GB 5.
Writes OUT.state.parquet / OUT.state.json.  Usage: python s2_build.py OUT.csv TUNE.json [--panels P]"""
import os, sys, json, threading, time
from pathlib import Path
import numpy as np
import psutil
ROOT = Path(os.environ["TFB_CODE_ROOT"]).resolve(); os.chdir(ROOT)
NT = os.environ.get("LM_THREADS", "4")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = NT
os.environ["CUDA_VISIBLE_DEVICES"] = ""
p = psutil.Process()
RSS_FILE = Path(os.environ["LM_RSS_FILE"])


def _watch():
    peak = 0
    while True:
        r = p.memory_info().rss; peak = max(peak, r)
        RSS_FILE.write_text("peak_rss_mb %.0f now_mb %.0f\n" % (peak / 2 ** 20, r / 2 ** 20))
        if r > float(os.environ.get("LM_RSS_GB", "5")) * 2 ** 30:
            print("STOP: RSS above LM_RSS_GB", flush=True); os._exit(3)
        time.sleep(15)


threading.Thread(target=_watch, daemon=True).start()
out, tune = sys.argv[1], sys.argv[2]
T = json.loads(Path(tune).read_text()); assert not T.get("lm_i511")


def rel(x):
    x = Path(x).resolve()
    return str(x.relative_to(ROOT)).replace("\\", "/")   # ValueError = mapped file outside ROOT (see docstring)


S08 = Path(os.environ["S08_DIR"]); B6 = Path(os.environ["B6_DIR"]); B6W = Path(os.environ["B6W"])
ART = {f"artifacts/lane_s/0398_s08/official_{s}.parquet": rel(S08 / f"official_{s}.parquet") for s in ("validation", "private")}
for sp, x in T["b6"].items():
    d, pre, seeds, suf = x["arm"].split(":")
    for pn in ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']:
        for k_ in seeds.split(","):
            src = f"artifacts/claude_ref/{d}/imp_{pre}{pn}_s{k_}.npz"; dst = rel(B6 / f"imp_{pre}{pn}_s{k_}.npz")
            if dst != src:
                ART[src] = dst
    if rel(B6W) != x["weights"]:
        ART[x["weights"]] = rel(B6W)
os.environ["TFB_ASM2_ART"] = json.dumps(ART)
print("TFB_ASM2_ART", len(ART), "entries", flush=True)
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "docs/reference_code"))
import asm2_build as AB
_rp = AB.Stages.run_panel


def run_panel_split(self, split, *a, **k):
    """Per-split knob settings (tune "lm_split": {split: {polish_chain, s3_scale}}): March rows and April rows take the
    setting chosen on the other month."""
    o = self.tune.get("lm_split", {}).get(split)
    if o:
        self.chain = o["polish_chain"]; self.s3scale = list(o["s3_scale"])
        assert getattr(self, "s3bag", None), "s3 bag path needed for the per-split scale"
    return _rp(self, split, *a, **k)


AB.Stages.run_panel = run_panel_split
import s2stage as S2
S2.OUT = S2.P.ROOT / "artifacts/claude_ref/c_budget/s2" / "all_l63_m2000_n3000000_r600"
assert (S2.OUT / "m_v.txt").exists() and (S2.OUT / "m_q.txt").exists()
if "--panels" in sys.argv[3:]:   # cheap parity: write the selected panels' rows (in-memory patch of one line)
    import types
    fn = ROOT / "docs/reference_code/asm2_final.py"; txt = fn.read_text(encoding="utf-8")
    old = '            print("debug panels: no file written"); return'
    new = ('            _m = st_rows & key.panel.astype(str).isin(panels).to_numpy()\n'
           '            pd.DataFrame({"submission_id": ids[_m], "speed_kmh": speed[_m], "flow_vph": flow[_m]}).to_parquet(str(a.out) + ".panels.parquet", index=False)\n'
           '            print("debug panels: wrote", str(a.out) + ".panels.parquet", int(_m.sum()), "rows"); return')
    assert txt.count(old) == 1, "asm2_final.py: panels line not found once"
    mod = types.ModuleType("asm2_final"); mod.__file__ = str(fn); sys.modules["asm2_final"] = mod
    exec(compile(txt.replace(old, new), str(fn), "exec"), mod.__dict__)
import asm2_final
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only", "--tune", tune] + sys.argv[3:]
asm2_final.main()
