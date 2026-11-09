"""C_gap2c47: build c_gap2_on_c47 = C47 with the official gap&target state rows blended with the gap2 CNN member.
  new gap row = clip((1 - w) * C47 gap row + w * mean(cnr0, cnr1, cnr2 official predictions, sm1)), speed clip 1.15 vf,
  w = frozen March weights (artifacts/claude_ref/c_gap2c47/weights_mar.json key m3), same for both splits.
Everything is streamed line by line (the 6.98M-row files are never loaded whole). Outputs:
  artifacts/claude_ref/probes/c_gap2_on_c47.csv                      full file
  artifacts/claude_ref/c_gap2c47/c_gap2_on_c47.delta.csv.gz + .json  delta in the tools/probe.py make format
Steps: rows (pass 1: read the parent's gap rows) -> write (pass 2: full file + delta) -> check (pass 3: independent
re-application of the delta to the parent = streamed 'expand', sha compared with the written file and the json).
Usage (repo root): python exp/C_gap2c47/g2_build.py"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, gzip, hashlib, csv, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "docs/reference_code"):
    sys.path.insert(0, str(ROOT / p))
import gap_lib as G

MY = ROOT / "artifacts/claude_ref/c_gap2c47"
PARENT = ROOT / "artifacts/claude_ref/probes/c_b6l24k_on_c44.csv"
NAME = "c_gap2_on_c47"
FULL = ROOT / "artifacts/claude_ref/probes" / f"{NAME}.csv"
DELTA = MY / f"{NAME}.delta.csv.gz"
GAP2 = ROOT / "artifacts/claude_ref/gap2"
TAGS = ("cnr0", "cnr1", "cnr2")
SPLITS = ["validation", "private"]
PANELS = ['D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S', 'D12_I5_N', 'D12_I5_S', 'D12_I405_N', 'D12_I405_S']
W3 = {0: [[0.0, 1.0], [0.05, 0.95]], 1: [[0.5, 0.5], [0.2, 0.8]]}
HEAD = b"submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clipvec(panel):
    from tfb.data import load_cache
    c = load_cache(panel, "validation"); st = c.stations.sort_values("idx")
    topo = pd.read_csv(ROOT / f"data/raw/kaggle_public/corridors/{panel}/network/lwr_mainline_topology.csv")
    return 1.15 * topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)


def main():
    t0 = time.time()
    km = np.load(ROOT / "artifacts/claude_ref/xfer/keymap.npz"); sid = km["sid"]; st = km["state"]; gap = km["gap"] & st
    N = len(sid); assert N == 6980503
    gi = np.flatnonzero(gap); pos = np.full(N, -1, np.int64); pos[gi] = np.arange(len(gi))
    # ---- pass 1: the parent's gap rows
    pv = np.zeros(len(gi)); pq = np.zeros(len(gi)); gapl = gap.tolist(); sidl = None
    with open(PARENT, "rb") as fin:
        head = fin.readline(); assert head.strip() == HEAD
        j = 0
        for i, line in enumerate(fin):
            if gapl[i]:
                a = line.rstrip(b"\r\n").split(b",")
                assert int(a[0]) == sid[i] and a[1] == b"state"
                pv[j] = float(a[2]); pq[j] = float(a[3]); j += 1
        assert i == N - 1 and j == len(gi)
    print("pass 1 done: parent gap rows", len(gi), round(time.time() - t0), "s", flush=True)
    W = {tuple(int(x) for x in k.split(",")): v for k, v in json.loads((MY / "weights_mar.json").read_text())["m3"].items()}
    print("weights (class, channel):", W, flush=True)
    z3 = np.load(G.OUT / "offpred_v3.npz"); Z = {sp: {t: np.load(GAP2 / f"pred_{t}_off_{sp}.npz") for t in TAGS} for sp in SPLITS}
    nv = pv.copy(); nq = pq.copy(); info = dict(per_panel={}, by_split={}); done = np.zeros(len(gi), bool)
    for si, sp in enumerate(SPLITS):
        tot = dict(rows=0, dv=0., dq=0.)
        for pi, pn in enumerate(PANELS):
            with np.load(G.OUT / f"rows_off_{sp}_{pn}.npz") as r:
                M = r["Mev"]
            sel = np.flatnonzero(st & (km["split"] == si) & (km["panel"] == pi))
            if not len(M):
                assert not gap[sel].any(), (sp, pn); continue
            D = int(km["d"][sel].max()) + 1
            grid = np.full((D, 288, int(km["s"][sel].max()) + 1), -1, np.int64)
            grid[km["d"][sel], km["t"][sel], km["s"][sel]] = sel
            ri = grid[M[:, 1], M[:, 2] + M[:, 5], M[:, 6]]
            assert (ri >= 0).all() and gap[ri].all() and len(np.unique(ri)) == len(ri)
            assert len(ri) == int((gap & (km["split"] == si) & (km["panel"] == pi)).sum())
            k = pos[ri]; assert (k >= 0).all() and not done[k].any(); done[k] = True
            cl = np.clip(M[:, 3], 0, 1); cv = clipvec(pn)[M[:, 6]]
            # what produces the rows today: onset speed = clip(v3 sm1) exactly (W3 onset speed weight on the chain is 0)
            M3 = z3[f"{sp}|{pn}|M"]; assert np.array_equal(M3, M)
            v3 = G.smooth_rows(M, z3[f"{sp}|{pn}|v"].astype(float), 1); q3 = G.smooth_rows(M, z3[f"{sp}|{pn}|q"].astype(float), 1)
            on = cl == 0; e_on = float(np.abs(pv[k][on] - np.where(v3[on] > cv[on] + 1e-9, cv[on], v3[on])).max()) if on.any() else 0.
            # implied chain value on ongoing rows: (row - .5 spec) / .5 (speed), (row - .8 spec) / .2 (flow)
            og = ~on; clipped = int((pv[k] >= cv - 1e-9).sum())
            ch_v = (pv[k][og] - W3[1][0][1] * v3[og]) / W3[1][0][0] if og.any() else np.zeros(0)
            mv = np.mean([G.smooth_rows(M, Z[sp][t][f"{pn}_v"].astype(float), 1) for t in TAGS], 0)
            mq = np.mean([G.smooth_rows(M, Z[sp][t][f"{pn}_q"].astype(float), 1) for t in TAGS], 0)
            assert np.isfinite(mv).all() and np.isfinite(mq).all()
            wv = np.array([W[(0, 0)], W[(1, 0)]])[cl]; wq = np.array([W[(0, 1)], W[(1, 1)]])[cl]
            bv = (1 - wv) * pv[k] + wv * mv; bq = (1 - wq) * pq[k] + wq * mq
            bv = np.where(bv > cv + 1e-9, cv, bv)
            dv = bv - pv[k]; dq = bq - pq[k]; nv[k] = bv; nq[k] = bq
            info["per_panel"][f"{sp}/{pn}"] = dict(rows=int(len(ri)), onset_rows=int(on.sum()), ongoing_rows=int(og.sum()), rms_dv=float(np.sqrt(np.mean(dv ** 2))), rms_dq=float(np.sqrt(np.mean(dq ** 2))),
                                                  onset_mean_dv=float(dv[on].mean()) if on.any() else 0., ongoing_mean_dv=float(dv[og].mean()) if og.any() else 0.,
                                                  parent_onset_speed_vs_clip_v3sm1_maxabs=e_on, parent_rows_at_clip=clipped,
                                                  implied_chain_v_ongoing_min_max=[float(ch_v.min()), float(ch_v.max())] if og.any() else None)
            print(sp, pn, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in info["per_panel"][f"{sp}/{pn}"].items()}, flush=True)
            tot["rows"] += len(ri); tot["dv"] += float((dv ** 2).sum()); tot["dq"] += float((dq ** 2).sum())
        info["by_split"][sp] = dict(rows=tot["rows"], rms_dv=float(np.sqrt(tot["dv"] / max(tot["rows"], 1))), rms_dq=float(np.sqrt(tot["dq"] / max(tot["rows"], 1))))
        print(sp, info["by_split"][sp], flush=True)
    assert done.all() and np.isfinite(nv).all() and np.isfinite(nq).all() and (nv > 0).all() and (nq >= 0).all(), (float(nv.min()), float(nq.min()))
    ch = (nv != pv) | (nq != pq)
    # ---- pass 2: full file + delta
    h = hashlib.sha256(); wrote = 0; split_of = km["split"]; nchg = {0: 0, 1: 0}
    tmp = FULL.with_suffix(".csv.tmp")
    with open(PARENT, "rb") as fin, open(tmp, "wb") as fo, gzip.open(DELTA, "wt", compresslevel=6, newline="") as fd:
        head = fin.readline(); h.update(head); fo.write(head); eol = b"\r\n" if head.endswith(b"\r\n") else b"\n"
        fd.write("submission_id,speed_kmh,flow_vph\n")
        for i, line in enumerate(fin):
            if gapl[i]:
                k = pos[i]
                if ch[k]:
                    a = line.rstrip(b"\r\n").split(b",")
                    sv, sq = repr(float(nv[k])), repr(float(nq[k]))
                    a[2] = sv.encode(); a[3] = sq.encode(); line = b",".join(a) + eol
                    fd.write(f"{int(a[0])},{sv},{sq}\n"); wrote += 1; nchg[int(split_of[i])] += 1
            h.update(line); fo.write(line)
    assert wrote == int(ch.sum())
    os.replace(tmp, FULL)
    full_sha = h.hexdigest(); assert sha(FULL) == full_sha
    print("pass 2 done: wrote", FULL.name, full_sha, "changed rows", wrote, round(time.time() - t0), "s", flush=True)
    # ---- pass 3: independent expand check (delta + parent -> sha), and line-level diff of the written file vs the parent
    dl = {}
    with gzip.open(DELTA, "rt", newline="") as fd:
        rd = csv.reader(fd); assert next(rd) == ["submission_id", "speed_kmh", "flow_vph"]
        for a, b, c in rd:
            dl[int(a)] = (b, c)
    h2 = hashlib.sha256(); used = 0; ndiff = 0; bad = 0
    with open(PARENT, "rb") as fin, open(FULL, "rb") as ff:
        head = fin.readline(); h2.update(head); assert ff.readline() == head
        for i, (line, lf) in enumerate(zip(fin, ff)):
            if line != lf:
                ndiff += 1
                if not gapl[i]:
                    bad += 1
            if gapl[i]:
                a = line.rstrip(b"\r\n").split(b","); x = dl.get(int(a[0]))
                if x is not None:
                    assert float(x[0]) == nv[pos[i]] and float(x[1]) == nq[pos[i]]
                    a[2] = x[0].encode(); a[3] = x[1].encode(); line = b",".join(a) + eol; used += 1
            h2.update(line)
        assert i == N - 1 and ff.readline() == b""
    ok = h2.hexdigest() == full_sha and used == len(dl) == wrote and bad == 0 and ndiff == wrote
    print("pass 3 (expand check): sha from parent + delta", h2.hexdigest(), "MATCH" if h2.hexdigest() == full_sha else "MISMATCH", "| delta rows used", used,
          "| lines that differ from the parent", ndiff, "of which outside gap&target state rows", bad, flush=True)
    meta = dict(parent=PARENT.name, parent_sha256=sha(PARENT), full=FULL.name, full_sha256=full_sha, delta=DELTA.name, delta_sha256=sha(DELTA), rows_changed=wrote,
                columns=["speed_kmh", "flow_vph"], rows_by_task={"state": wrote}, changed_validation=nchg[0], changed_private=nchg[1], gap_rows=int(len(gi)),
                expand_check="MATCH" if ok else "FAILED", weights={f"{a},{b}": v for (a, b), v in W.items()}, members=list(TAGS),
                note="C47 with official gap&target state rows = clip((1-w) parent row + w mean(cnr0,cnr1,cnr2 official predictions, sm1)); w (class,channel) fitted on March I-S-OWN selector-like blocks, all dates",
                info=info)
    (MY / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "info"}, indent=1), flush=True)
    print("BUILD", "DONE" if ok else "FAILED", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
