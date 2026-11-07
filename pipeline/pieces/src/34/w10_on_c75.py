# R3 (rows 28-35) copy of exp/L_fvinel/fvi_lw_on_c75.py (sha256 dbb540a62138980dd292280fe5ec9816ec32cddc72ba547d38e0f408b11bf1c0).
# Edit: ROOT = $TFB_ROOT. Logic unchanged. Row 34 uses it with LW.csv = the W1.0 March file (c_fvinel_w10_mar_on_c70) -> c_fvinel_w10_on_c75 (sha 67d02edf).
"""FV-INEL-LW March rows onto C75 (LINE.md "Variant LW"): OUT = C75 with every VALIDATION non-gap state row taken from the LW finish file
c_fvinel_lw_mar_on_c70 (LW replaces FV-INEL W .3 on those rows); gap rows, private rows and every other task stay C75.
Asserts: C75's validation non-gap state rows == the W .3 file c_fvinel_mar_on_c70 (01bff667) EXACTLY (C75 March non-gap = FV-INEL W .3);
LW vs W .3 differ only inside those rows; OUT vs C75 differs only inside them. Then tools/probe.py make (delta vs C75) and
tools/window_guard.py --against C75 --no-record (C's upload records the guard line itself).
Usage (repo root): python exp/L_fvinel/fvi_lw_on_c75.py C75.csv W3.csv LW.csv OUT.csv ; last line FVI_LW_C75_DONE <sha> | traceback
Written 2026-10-19 by the L_fvinel worker; dry-tested with LW.csv = W3.csv and OUT = --dry (asserts and counts only; OUT == C75, 0 rows changed)."""
import os, sys, subprocess, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ["TFB_ROOT"])
spec = importlib.util.spec_from_file_location("probe", ROOT / "tools/probe.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
C75_SHA, W3_SHA = "15280925", "01bff667"
c75p, w3p, lwp, outp = sys.argv[1:5]
assert P.sha256(c75p)[:8] == C75_SHA, "C75 sha"
assert P.sha256(w3p)[:8] == W3_SHA, "W .3 file sha (artifacts/claude_ref/l_fvinel/c_fvinel_mar_on_c70.csv)"
c75 = P.read_full(c75p); w3 = P.read_full(w3p); lw = P.read_full(lwp)
for x in (w3, lw):
    assert (x.submission_id.to_numpy() == c75.submission_id.to_numpy()).all(), "row order differs"
km = np.load(ROOT / "artifacts/claude_ref/xfer/keymap.npz")
i = pd.Index(km["sid"].astype(np.int64)).get_indexer(c75.submission_id.to_numpy(np.int64)); assert (i >= 0).all()
st, gap, spl = km["state"][i], km["gap"][i] & km["state"][i], km["split"][i]
R = st & ~gap & (spl == 0)
cols = ["speed_kmh", "flow_vph"]


def ch(a, b):
    r = np.zeros(len(a), bool)
    for c in P.VALUE_COLS:
        if c in a.columns:
            r |= P.changed_mask(a, b, [c])[c]
    return r


d75_w3, dlw_w3 = ch(c75, w3), ch(lw, w3)
print("R (validation non-gap state rows):", int(R.sum()), "| C75 vs W.3 inside R:", int((d75_w3 & R).sum()), "(must be 0) | outside R:", int((d75_w3 & ~R).sum()),
      "| LW vs W.3 inside R:", int((dlw_w3 & R).sum()), "| outside R:", int((dlw_w3 & ~R).sum()), "(must be 0)", flush=True)
assert int((d75_w3 & R).sum()) == 0, "C75's March non-gap state rows are not FV-INEL W .3: stop (ask C)"
assert int((dlw_w3 & ~R).sum()) == 0, "LW differs from W .3 outside the validation non-gap state rows"
out = c75.copy()
for c in cols:
    v = out[c].to_numpy().copy(); v[R] = lw[c].to_numpy()[R]; out[c] = v
d = ch(out, c75)
assert not (d & ~R).any(), "changes outside validation non-gap state rows"
print("OUT vs C75 rows changed:", int(d.sum()), "(all validation non-gap state)", flush=True)
del w3, lw, c75
if outp == "--dry":   # test: assertions and counts only, nothing written
    print("FVI_LW_C75_DRY_OK", flush=True); sys.exit(0)
out.to_csv(outp, index=False, lineterminator=P.line_ending(c75p)); del out
subprocess.run([sys.executable, "-B", str(ROOT / "tools/probe.py"), "make", c75p, outp, outp.replace(".csv", ".delta.csv.gz")], check=True)
g = subprocess.run([sys.executable, "-B", str(ROOT / "tools/window_guard.py"), outp, "--against", c75p, "--no-record"], capture_output=True, text=True)
print("\n".join(l for l in g.stdout.splitlines() if "->" in l or "PASS" in l or "REFUSE" in l), flush=True)
assert g.returncode == 0, f"window_guard exit {g.returncode}"
print("FVI_LW_C75_DONE", P.sha256(outp), outp, flush=True)
