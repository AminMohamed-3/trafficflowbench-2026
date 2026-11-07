"""Exact comparison of a (partial) state parquet with the stored one: every row of BUILT (submission_id, speed_kmh, flow_vph)
must equal the stored row of the same id bit for bit. Prints one JSON line; exit 0 = EQUAL.
Usage: python cmp_state.py BUILT.parquet STORED.parquet"""
import json, sys
import numpy as np
import pandas as pd
b, s = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
i = pd.Index(s.submission_id).get_indexer(b.submission_id)
out = dict(built=sys.argv[1], stored=sys.argv[2], rows=int(len(b)), missing=int((i < 0).sum()))
if out["missing"] == 0 and len(b):
    for c in ("speed_kmh", "flow_vph"):
        x, y = b[c].to_numpy(float), s[c].to_numpy(float)[i]
        out[c] = dict(differ=int((x != y).sum()), max_abs=float(np.abs(x - y).max()))
ok = out["missing"] == 0 and len(b) > 0 and all(out[c]["differ"] == 0 for c in ("speed_kmh", "flow_vph"))
out["result"] = "EQUAL" if ok else "DIFFER"
print(json.dumps(out)); sys.exit(0 if ok else 1)
