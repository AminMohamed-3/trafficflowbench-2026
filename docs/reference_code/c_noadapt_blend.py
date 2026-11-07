"""C 2026-10-09 (SIGNAL probe helper): partial de-adaptation of the shipped ongoing CNN members.

Per member probability file p = a * adapted + (1 - a) * unadapted (cnn_s0, cnn_s1: base CNN seeds; sccnn_official: the
seed-mean of the selection-conditioned CNN, which is linear so equal to blending each seed). The blended files then go
through the unchanged chain (qcnn_avg_probs -> ong2_probe build --noT -> asm2_bbmember decode). a = 1 returns the
adapted bytes exactly (asserted).

  python c_noadapt_blend.py ON_DIR OFF_DIR A OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

K = ["window_id", "timestamp", "link_id"]


def main(on_dir, off_dir, a, out_dir):
    a = float(a); assert 0. <= a <= 1.; out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(x.name for x in Path(on_dir).glob("cnn_s*.parquet")) + ["sccnn_official.parquet"]   # any seed count
    assert files[:2] == ["cnn_s0.parquet", "cnn_s1.parquet"], files
    for f in files:
        on = pd.read_parquet(Path(on_dir) / f, use_threads=False); off = pd.read_parquet(Path(off_dir) / f, use_threads=False)
        assert list(on.columns) == list(off.columns) and len(on) == len(off)
        assert (on[K].to_numpy() == off[K].to_numpy()).all(), f"row order differs {f}"
        p = a * on.prob.to_numpy(np.float64) + (1. - a) * off.prob.to_numpy(np.float64)
        new = on.copy(); new["prob"] = p.astype(on.prob.dtype)
        if a == 1.: assert np.array_equal(new.prob.to_numpy(), on.prob.to_numpy())
        new.to_parquet(out / f)
        d = np.abs(new.prob.to_numpy(np.float64) - on.prob.to_numpy(np.float64))
        print("blend", f, "a", a, "rows", len(new), "mean|dp| vs adapted", round(float(d.mean()), 5), "max", round(float(d.max()), 4), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
    import os; sys.stdout.flush(); os._exit(0)
