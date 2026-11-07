# R3 (rows 28-35) copy of exp/L_fvft/fv2_apr_on_c72.py (sha256 66cc86e376025b04cf9c47a3907ef19931c325de6466bf078c92fb1a0493970d). Edit: tools/probe.py and the keymap read from $TFB_ROOT instead of the working directory. Logic unchanged. Used for C72P, C73P, C74P and C78 (arguments C72 BOTH C70 OUT).
"""FV2 April rows onto C72 (C 2026-10-19): out = C72 with every private non-gap state row taken from c_fv2_both_on_c70
(FV2 April replaces FV1 April; declared method, precedence ruling for April rows). Everything else stays C72.
Usage: python exp/L_fvft/fv2_apr_on_c72.py C72.csv FV2_BOTH.csv C70.csv OUT.csv ; last line FV2APR72_DONE sha"""
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("probe", os.path.join(os.environ["TFB_ROOT"], "tools/probe.py")); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
c72p, bothp, c70p, outp = sys.argv[1:5]
c72, both, c70 = P.read_full(c72p), P.read_full(bothp), P.read_full(c70p)
for x in (both, c70):
    assert (x.submission_id.to_numpy() == c72.submission_id.to_numpy()).all(), "row order differs"
km = np.load(os.path.join(os.environ["TFB_ROOT"], "artifacts/claude_ref/xfer/keymap.npz"))
i = pd.Index(km["sid"].astype(np.int64)).get_indexer(c72.submission_id.to_numpy(np.int64)); assert (i >= 0).all()
st, gap, spl = km["state"][i], km["gap"][i] & km["state"][i], km["split"][i]
R = st & ~gap & (spl == 1)
cols = ["speed_kmh", "flow_vph"]
def ch(a, b, m=None):
    r = np.zeros(len(a), bool)
    for c in P.VALUE_COLS:
        if c in a.columns: r |= P.changed_mask(a, b, [c])[c]
    return r
d72_70, dboth_70 = ch(c72, c70), ch(both, c70)
print("C72 vs C70 rows in R (FV1 April):", int((d72_70 & R).sum()), "| outside R:", int((d72_70 & ~R).sum()))
print("FV2_BOTH vs C70 rows in R:", int((dboth_70 & R).sum()), "| private outside R:", int((dboth_70 & ~R & (spl == 1)).sum()))
assert int((dboth_70 & ~R & (spl == 1)).sum()) == 0
out = c72.copy()
for c in cols:
    v = out[c].to_numpy().copy(); v[R] = both[c].to_numpy()[R]; out[c] = v
d = ch(out, c72)
assert not (d & ~R).any(), "changes outside private non-gap state rows"
print("OUT vs C72 rows changed:", int(d.sum()), "(all private non-gap state)")
out.to_csv(outp, index=False, lineterminator=P.line_ending(c72p))
print("FV2APR72_DONE", P.sha256(outp))
