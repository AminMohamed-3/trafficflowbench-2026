# R3 (rows 28-35) copy of exp/L_gap3/g3_official.py (sha256 31912f37b823c7934c5d6f5b1b081babc2e2f553eb4fa5904647f020d5e88847).
# Edits: ROOT = $TFB_ROOT; output folder D = $G3_OUT; parent C69 = $G3_PARENT (sha still asserted); knobs file = $G3_KNOBS (default the stored l_gap3/knobs_march.json). Logic unchanged.
"""L_gap3 official gap rows: C69 recipe (exp/L_gap2/g2_official.py step 7 of artifacts/claude_ref/l_gap2/RECEIPT.json) with the
member smoothing h as a parameter and the EDGE-TAPER F1 stage after the v6r step, streamed over the parent C69
(artifacts/claude_ref/l_gap2/c_gap_on_c66.csv, sha d1ada3d6): ONLY the 128,465 gap rows may change.
  W3 (hook chain/spec, GAPW March / G2 April-own all-dates) -> gap2 m: clip((1-m) row + m mean_s sm_h(off8k_s)) ->
  v6r a: clip((1-a) row + a sm_h(v6r)) -> [new arm: guard speed >= 3, flow >= 0] -> F1: p + b r(j) (chain - p), speed clip.
Arms: par = h 1, b 0 (must equal C69's gap rows bit for bit, else exit 2); new = h 0, b = March all-dates knobs fitted on U0
(artifacts/claude_ref/l_gap3/knobs_march.json "U0"), the same knobs for March and April rows.
Also re-checks hook parities p1 (GAPW arm == recomputed) and p2 (== C66 t1_gapwf gap rows).
Usage (repo root): python exp/L_gap3/g3_official.py NAME [--parity-only]
Out: artifacts/claude_ref/l_gap3/NAME.csv, NAME.delta.csv.gz (tools/probe.py make), NAME.info.json"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(os.environ["TFB_ROOT"])
for p in ("src", "docs/reference_code", "exp/C_gap2c47", "exp/L_b6"):
    sys.path.insert(0, str(ROOT / p))
CR = ROOT / "artifacts/claude_ref"; D = Path(os.environ["G3_OUT"]); G2 = CR / "l_gap2"
C69 = Path(os.environ.get("G3_PARENT") or G2 / "c_gap_on_c66.csv"); C69SHA = "" if os.environ.get("SKIP_PARENT_SHA") == "1" else "d1ada3d6"   # chain: parent is the rebuilt C69 (value-equal); the par-arm parity below still checks its gap rows bit for bit
GAPWF = CR / "l_b6/bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet"
HOOK = G2 / "build/t1.csv.gaprows.npz"
WJF = G2 / "g2/weights_g2.json"; MEMD = G2 / "g1/pred_cpu"; MTAGS = ("off8k_s0", "off8k_s1", "off8k_s2"); V6D = G2 / "g3"
A6F = {"validation": G2 / "g3/bench_validation.json", "private": G2 / "g3/bench_private_aprfit.json"}
KNOBS = Path(os.environ.get("G3_KNOBS") or CR / "l_gap3/knobs_march.json")
N = 6980503; N_GAP2 = 128465
SIDES = [(1, "start"), (1, "end"), (0, "end")]
import c0_official as C0   # sha, state_rows, rel


def w3_apply(W, cls, chain, spec, cs, ch):
    v = sum(np.array([W[0][ch][i], W[1][ch][i]])[cls] * (chain, spec)[i] for i in range(2))
    return np.where(v > cs + 1e-9, cs, v) if ch == 0 else v


def ramp(j, side):
    j = j.astype(float)
    return np.clip(1 - (j - 1) / 2, 0, 1) if side == "start" else np.clip(1 - (18 - j) / 2, 0, 1)


def main(name, parity_only=False):
    t0 = time.time()
    import g2_build as B
    import asm2_build_gap2w as AB
    import gap_lib as GLb
    sha69 = C0.sha(C69); assert sha69.startswith(C69SHA), "C69 sha"
    km = np.load(CR / "xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(np.int64), km["t"].astype(np.int64), km["s"].astype(np.int64)
    del km
    gap = gapk & st; assert int(gap.sum()) == N_GAP2
    spos = np.cumsum(st) - 1
    pv, pq = C0.state_rows(C69, st, sid)
    P = pd.read_parquet(GAPWF); assert np.array_equal(P.submission_id.to_numpy(np.int64), sid[st].astype(np.int64))
    fv, fq = P.speed_kmh.to_numpy(float), P.flow_vph.to_numpy(float); del P
    Wg = {int(k): v for k, v in json.loads((CR / "l_b6/gapw/gap_cfg_gapw.json").read_text())["weights"].items()}
    m3 = {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads((CR / "c_gap2c47/weights_mar.json").read_text())["m3"].items()}
    WJ = json.loads(WJF.read_text())
    A6 = {sp: {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads(f.read_text())["fits"]["all"].items()} for sp, f in A6F.items()}
    KN = {(int(a), int(b), s): float(v) for k, v in json.loads(KNOBS.read_text())["U0"]["all"].items() for a, b, s in [k.split(",")]}
    MEM = {sp: [np.load(MEMD / f"pred_{tg}_off_{sp}.npz") for tg in MTAGS] for sp in ("validation", "private")}
    V6 = {sp: np.load(V6D / f"pred_v6r_off_{sp}.npz") for sp in ("validation", "private")}
    z = np.load(HOOK)
    S = AB.Stages(["gap", "clip", "gap2"], None)   # only for gap_pred (M per split|panel)
    arms = {"par": (pv.copy(), pq.copy()), "new": (pv.copy(), pq.copy())}
    hit = np.zeros(len(sid), int); bad = {"p1": 0, "p2": 0}; per = {}; guard = [0, 0]; nramp = 0
    for key in z["keys"]:
        sp, pn = str(key).split("|"); si, pi = B.SPLITS.index(sp), B.PANELS.index(pn)
        sel = np.flatnonzero(st & (ksplit == si) & (kpanel == pi)); g = gap[sel]
        kk = kd[sel] * 1000000 + kt[sel] * 1000 + ks[sel]; o = np.argsort(kk)
        qk = z[f"{key}|d"].astype(np.int64) * 1000000 + z[f"{key}|t"].astype(np.int64) * 1000 + z[f"{key}|s"].astype(np.int64)
        j = o[np.searchsorted(kk[o], qk)]; assert (kk[j] == qk).all() and len(np.unique(j)) == len(j) == int(g.sum()) and g[j].all(), key
        rows = sel[j]; hit[rows] += 1
        cls, cs = z[f"{key}|cls"].astype(int), z[f"{key}|cs"].astype(float)
        chv, chq, spv, spq = (z[f"{key}|{f}"].astype(float) for f in ("chain_v", "chain_q", "spec_v", "spec_q"))
        gv, gq = w3_apply(Wg, cls, chv, spv, cs, 0), w3_apply(Wg, cls, chq, spq, cs, 1)
        bad["p1"] += int(((gv != z[f"{key}|v_new"]) | (gq != z[f"{key}|q_new"])).sum())
        bad["p2"] += int(((z[f"{key}|v_new"] != fv[spos[rows]]) | (z[f"{key}|q_new"] != fq[spos[rows]])).sum())
        cs_full = B.clipvec(pn)[ks[sel]]
        if sp in WJ:
            W3 = {int(k): v for k, v in WJ[sp]["W3"].items()}; Gm = {tuple(int(x) for x in k.split(",")): float(v) for k, v in WJ[sp]["gap2"].items()}
        else:
            W3, Gm = Wg, m3
        M = S.gap_pred[f"{sp}|{pn}|M"]
        kk2 = M[:, 1].astype(np.int64) * 1000000 + (M[:, 2] + M[:, 5]).astype(np.int64) * 1000 + M[:, 6].astype(np.int64)
        k2 = o[np.searchsorted(kk[o], kk2)]; assert (kk[k2] == kk2).all() and len(np.unique(k2)) == len(k2)
        assert set(k2.tolist()) == set(j.tolist()), (key, "M rows != hook gap rows")
        c2 = np.clip(M[:, 3], 0, 1); jj = M[:, 5]
        assert ((jj >= 1) & (jj <= 18)).all() and np.isin(M[:, 3], (0, 1)).all(), key
        full = lambda x: (lambda a: (a.__setitem__(j, x), a)[1])(np.full(len(sel), np.nan))
        assert np.array_equal(full(cls.astype(float))[k2], c2.astype(float)), (key, "hook class != M class")
        chM = (full(chv)[k2], full(chq)[k2]); assert np.isfinite(chM[0]).all() and np.isfinite(chM[1]).all()
        for arm, h in (("par", 1), ("new", 0)):
            nv, nq = w3_apply(W3, cls, chv, spv, cs, 0), w3_apply(W3, cls, chq, spq, cs, 1)
            ov = pv[spos[sel]].copy(); oq = pq[spos[sel]].copy(); ov[j] = nv; oq[j] = nq
            # gap2 step (asm2_build_gap2w.Stages.gap2_rows with smoothing h)
            mv = np.mean([GLb.smooth_rows(M, x[f"{pn}_v"].astype(float), h) for x in MEM[sp]], 0)
            mq = np.mean([GLb.smooth_rows(M, x[f"{pn}_q"].astype(float), h) for x in MEM[sp]], 0)
            assert len(mv) == len(M) and np.isfinite(mv).all() and np.isfinite(mq).all()
            wv = np.array([Gm[(0, 0)], Gm[(1, 0)]])[c2]; wq = np.array([Gm[(0, 1)], Gm[(1, 1)]])[c2]
            bv = (1 - wv) * ov[k2] + wv * mv; oq[k2] = (1 - wq) * oq[k2] + wq * mq
            ov[k2] = np.where(bv > cs_full[k2] + 1e-9, cs_full[k2], bv)
            # v6r step (g2_official --v6 with smoothing h)
            x6v = GLb.smooth_rows(M, V6[sp][f"{pn}_v"].astype(float), h); x6q = GLb.smooth_rows(M, V6[sp][f"{pn}_q"].astype(float), h)
            av = np.array([A6[sp][(0, 0)], A6[sp][(1, 0)]])[c2]; aq = np.array([A6[sp][(0, 1)], A6[sp][(1, 1)]])[c2]
            bv = (1 - av) * ov[k2] + av * x6v; oq[k2] = (1 - aq) * oq[k2] + aq * x6q
            ov[k2] = np.where(bv > cs_full[k2] + 1e-9, cs_full[k2], bv)
            if arm == "new":
                guard[0] += int((ov[k2] < 3.).sum()); guard[1] += int((oq[k2] < 0.).sum())
                ov[k2] = np.maximum(ov[k2], 3.); oq[k2] = np.maximum(oq[k2], 0.)
                pM = (ov[k2].copy(), oq[k2].copy()); out = []
                for ch in (0, 1):
                    p = pM[ch].copy()
                    for cc, side in SIDES:
                        p = p + KN[(cc, ch, side)] * ramp(jj, side) * (M[:, 3] == cc) * (chM[ch] - pM[ch])
                    out.append(p)
                nramp += int(((ramp(jj, "start") > 0) & (M[:, 3] == 1) | (ramp(jj, "end") > 0)).sum())
                ov[k2] = np.where(out[0] > cs_full[k2] + 1e-9, cs_full[k2], out[0]); oq[k2] = out[1]
            arms[arm][0][sel[g]] = ov[g]; arms[arm][1][sel[g]] = oq[g]
        dv = np.abs(arms["new"][0][sel[g]] - pv[sel[g]]); dq = np.abs(arms["new"][1][sel[g]] - pq[sel[g]])
        per[str(key)] = dict(gap_rows=int(g.sum()), changed=int(((dv > 0) | (dq > 0)).sum()), mean_abs_dv=float(dv.mean()), mean_abs_dq=float(dq.mean()))
    assert (hit[gap] == 1).all() and not hit[~gap].any(), "gap row coverage"
    pc = int(((arms["par"][0] != pv) | (arms["par"][1] != pq))[gap].sum())
    print("GAP3OFF_PARITY p1 (hook GAPW arm == recomputed)", "EXACT" if bad["p1"] == 0 else f"FAIL {bad['p1']}",
          "| p2 (hook GAPW arm == t1_gapwf gap rows)", "EXACT" if bad["p2"] == 0 else f"FAIL {bad['p2']}",
          "| pC69 (h 1, b 0 here == C69 gap rows)", "EXACT" if pc == 0 else f"FAIL {pc}", "| guard rows", guard, "| taper rows", nramp, flush=True)
    if bad["p1"] or bad["p2"] or pc:
        sys.exit(2)
    if parity_only:
        print("GAP3OFF parity only %.0f s" % (time.time() - t0)); return
    nv, nq = arms["new"]
    assert np.isfinite(nv[gap]).all() and np.isfinite(nq[gap]).all() and (nv[gap] > 0).all() and (nq[gap] >= 0).all()
    chg = ((nv != pv) | (nq != pq)) & st; assert not (chg & ~gap).any()
    # disk floor 13.5 GiB: the full file is written gzipped (content sha computed while writing); guard and probe.py read .csv.gz
    import gzip, hashlib
    full = D / f"{name}.csv.gz"; assert not full.exists(), full
    tmp = D / f"{name}.csv.gz.tmp"; gl = gap.tolist(); nvl, nql = nv.tolist(), nq.tolist(); cl = chg.tolist(); hsh = hashlib.sha256()
    with open(C69, "rb") as f, gzip.open(tmp, "wb", compresslevel=6) as o:
        x = f.readline(); o.write(x); hsh.update(x)
        for i, line in enumerate(f):
            if gl[i] and cl[i]:
                a = line.split(b",", 4); eol = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                rest = a[4]; assert rest.endswith(eol)
                line = b",".join([a[0], a[1], repr(nvl[i]).encode(), repr(nql[i]).encode(), rest])
            o.write(line); hsh.update(line)
    os.replace(tmp, full); content_sha = hsh.hexdigest()
    sl = sid.tolist(); wv = np.full(len(st), np.nan); wq = np.full(len(st), np.nan); stl = st.tolist(); n = 0
    with open(C69, "rb") as f, gzip.open(full, "rb") as g_:
        f.readline(); g_.readline()
        for i, (a, b) in enumerate(zip(f, g_)):
            n += 1
            if not gl[i]:
                if a != b:
                    raise SystemExit(f"line {i} changed outside the gap rows")
                continue
            x = b.split(b",", 4); assert int(x[0]) == sl[i]
            wv[i] = float(x[2]); wq[i] = float(x[3])
    assert n == N, n
    assert np.array_equal(wv[gap], nv[gap]) and np.array_equal(wq[gap], nq[gap]), "written gap rows differ"
    dv = np.abs(nv - pv)[gap]; dq = np.abs(nq - pq)[gap]
    info = dict(probe=C0.rel(full), sha256=content_sha, sha256_note="sha256 of the CSV content (gunzipped), as uploaded", gz_sha256=C0.sha(full), parent=C0.rel(C69), parent_sha256=sha69, rows=N, gap2_rows=N_GAP2, rows_changed=int(chg.sum()),
                rows_changed_by_split={sp: int((chg & (ksplit == si)).sum()) for si, sp in enumerate(B.SPLITS)},
                change_size=dict(mean_abs_dv=float(dv.mean()), max_dv=float(dv.max()), mean_abs_dq=float(dq.mean()), max_dq=float(dq.max())),
                per_panel=per, knobs=C0.rel(KNOBS), knobs_sha256=C0.sha(KNOBS), knobs_used={f"{a},{b},{s}": v for (a, b, s), v in KN.items()},
                members=C0.rel(MEMD), member_tags=list(MTAGS), member_smoothing_h=0, v6=dict(dir=C0.rel(V6D), smoothing_h=0, a={sp: {f"{k[0]},{k[1]}": v for k, v in A6[sp].items()} for sp in A6}),
                weights=C0.rel(WJF), hook=C0.rel(HOOK), hook_sha256=C0.sha(HOOK), guard_rows=dict(speed_lt3=guard[0], flow_lt0=guard[1]), taper_rows=nramp,
                parity=dict(p1="EXACT", p2="EXACT", pC69="EXACT"), seconds=round(time.time() - t0))
    (D / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    print("GAP3OFF_PROBE", json.dumps({k: info[k] for k in ("probe", "sha256", "rows_changed", "rows_changed_by_split", "change_size", "guard_rows")}), flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/probe.py"), "make", str(C69), str(full), str(D / f"{name}.delta.csv.gz")], check=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], "--parity-only" in sys.argv)
