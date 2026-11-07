# R3 (rows 28-35) copy of exp/L_qaccum/qa_build.py (sha256 76d19121415a2689db8fa3538eef9c5f778cf32088a6af6c1c903add13369445).
# Edits: ROOT = $TFB_ROOT; output folders under $QA_OUT (was artifacts/claude_ref/l_qaccum); the three Kaggle serve outputs may be pointed to rerun pulls by $QA_DW, $QA_TODR, $QA_MARGIN (default: the stored pulls, sha-checked by 33_qaccum.sh). Logic unchanged.
"""L_qaccum build (LINE.md): bundle of DW-ADAPT + TODR + MARGIN-ADAPT official k1 - k0 deltas, then the MARGIN build path
(exp/L_margin/mg_build.py, unchanged, imported; folders redirected to artifacts/claude_ref/l_qaccum/build).
Cold: p_xall, p_xsd = clip(shipped + dDW + dTODR + dMG, 0, 1) (mg_build.members with its 'official' arm k1 replaced by
MG k1 + dDW + dTODR in float64; base = MG k0). Warm: MG only (mg_build unchanged).
  members | chain | rows | delta PARENT.csv OUT_NAME   (name fixed 'qa')"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ["TFB_ROOT"])
sys.path.insert(0, str(ROOT / "exp/L_margin"))
import mg_build as MG
MG.A = Path(os.environ["QA_OUT"]); MG.OUT = MG.A / "build"   # QA_OUT must lie inside TFB_ROOT (mg_build uses relative_to)
DW = Path(os.environ.get("QA_DW") or ROOT / "artifacts/claude_ref/l_dwadapt/kg_serve/artifacts/claude_ref/l_dwadapt/official")
TD = Path(os.environ.get("QA_TODR") or ROOT / "artifacts/claude_ref/l_todr/kg_serve/artifacts/claude_ref/l_todr/official")
MGR = Path(os.environ.get("QA_MARGIN") or ROOT / "artifacts/claude_ref/l_margin/kg_inst/artifacts/claude_ref/l_margin")
K = MG.K; _arm = MG.arm; STATS = {}


def piece_delta(d, t, keys):
    a1 = pd.read_parquet(d / "k1" / f"{t}.parquet").set_index(K).reindex(keys); a0 = pd.read_parquet(d / "k0" / f"{t}.parquet").set_index(K).reindex(keys)
    assert a1.p_xsd.notna().all() and a0.p_xsd.notna().all(), f"{d}: {t} keys differ from MARGIN's"
    return {c: a1[c].to_numpy(np.float64) - a0[c].to_numpy(np.float64) for c in ("p_xall", "p_xsd")}


def arm(root, sub, a, t):
    f = _arm(root, sub, a, t)
    if sub != "official" or a != "k1":
        return f
    keys = pd.MultiIndex.from_frame(f[K]); f = f.copy(); st = {}
    dd = {n: piece_delta(d, t, keys) for n, d in (("dw", DW), ("todr", TD))}
    for c in ("p_xall", "p_xsd"):
        f[c] = f[c].to_numpy(np.float64) + dd["dw"][c] + dd["todr"][c]
    for n in dd:
        st[n] = dict(cells_moved=int((dd[n]["p_xsd"] != 0).sum()), max_move=float(np.abs(dd[n]["p_xsd"]).max()))
    STATS[t] = st
    return f


MG.arm = arm

if __name__ == "__main__":
    c = sys.argv[1]; a = [x for x in sys.argv[2:] if not x.startswith("--")]
    if c == "members":
        MG.members(MGR, "k1", "k0", "qa")
        (MG.OUT / "members_qa" / "pieces.json").write_text(json.dumps(STATS, indent=1)); print("QA_PIECES", json.dumps({t: {n: v["cells_moved"] for n, v in s.items()} for t, s in STATS.items()}), flush=True)
    elif c == "chain":
        MG.chain("qa")
    elif c == "rows":
        MG.rows("qa", parity="--no-parity" not in sys.argv)
    elif c == "delta":
        MG.delta(a[0], "qa", a[1])
    sys.stdout.flush(); __import__("os")._exit(0)
