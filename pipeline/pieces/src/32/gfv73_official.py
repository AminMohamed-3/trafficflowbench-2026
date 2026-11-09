# R3 (rows 28-35) copy of exp/L_gapfv/gfv73_official.py (sha256 b4273202df00451e3f47b369dbd7a7e7eebfd4ddeca65a13938380a51a732e0d).
# Edits: ROOT = $TFB_ROOT; output folder D = $GFV_OUT; parent C73P = $GFV_PARENT (sha still asserted); RSS file in D; new hook = $GFV73_NEW_HOOK (original env). Logic unchanged.
"""L_gapfv c73p copy of gfv_official.py (parent, folders, RSS file and the plain-CSV output differ; the gap form is unchanged).
gfv_official.py text, with the c73p names: L_gapfv official gap rows: the L_gap3 shipped form (exp/L_gap3/g3_official.py arm "new":
h 0, F1 knobs knobs_march.json U0 all) with the chain member (W3 and F1) taken from a hook build. Streamed over the parent P = C73P
(artifacts/claude_ref/l_fvinel/c73p/c_fvinel_fv2apr.csv, sha 5827d508; .csv or .csv.gz): ONLY the 128,465 gap rows may change.
Arms: par = S2b hook (artifacts/claude_ref/l_gap2/build/t1.csv.gaprows.npz) -> must equal P's gap rows bit for bit (else exit 2);
      new = FV hook (artifacts/claude_ref/l_gapfv/c73p/build/t1.csv.gaprows.npz: FV-INEL March, FV2 April), every weight, knob and other member unchanged.
Also re-checks hook parities p1 (GAPW arm == recomputed; both hooks) and p2 (S2b hook GAPW arm == t1_gapwf gap rows), and that the
FV hook equals the S2b hook on keys, d, t, s, class, clip and spec_v3 (only the chain may differ).
Usage (repo root): python exp/L_gapfv/gfv73_official.py NAME [--parity-only]
Out: artifacts/claude_ref/l_gapfv/c73p/NAME.csv (plain), NAME.delta.csv.gz (tools/probe.py make), NAME.info.json, change_rows.npz"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, subprocess, time, gzip, hashlib, threading
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
ROOT = Path(os.environ["TFB_ROOT"])
for p in ("src", "docs/reference_code", "exp/C_gap2c47", "exp/L_b6"):
    sys.path.insert(0, str(ROOT / p))
CR = ROOT / "artifacts/claude_ref"; D = Path(os.environ["GFV_OUT"]); G2 = CR / "l_gap2"; G3 = CR / "l_gap3"
PAR = Path(os.environ["GFV_PARENT"]); PARSHA = "" if os.environ.get("SKIP_PARENT_SHA") == "1" else "5827d508"   # chain: rebuilt parent (value-equal)
GAPWF = CR / "l_b6/bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet"
HOOK = {"par": G2 / "build/t1.csv.gaprows.npz", "new": Path(os.environ.get("GFV73_NEW_HOOK", str(D / "build/t1.csv.gaprows.npz")))}  # override only for the code-parity dry run
WJF = G2 / "g2/weights_g2.json"; MEMD = G2 / "g1/pred_cpu"; MTAGS = ("off8k_s0", "off8k_s1", "off8k_s2"); V6D = G2 / "g3"
A6F = {"validation": G2 / "g3/bench_validation.json", "private": G2 / "g3/bench_private_aprfit.json"}
KNOBS = G3 / "knobs_march.json"
N = 6980503; N_GAP2 = 128465; H = 0
SIDES = [(1, "start"), (1, "end"), (0, "end")]
SAME = ("d", "t", "s", "cls", "cs", "spec_v", "spec_q")
import c0_official as C0   # sha, rel
proc = psutil.Process()


def _watch():
    peak = 0
    while True:
        r = proc.memory_info().rss; peak = max(peak, r)
        (D / "gfv73_official_rss.txt").write_text("peak_rss_mb %.0f now_mb %.0f\n" % (peak / 2 ** 20, r / 2 ** 20))
        if r > 3.5 * 2 ** 30:
            print("STOP: RSS above 3.5 GiB", flush=True); os._exit(3)
        time.sleep(10)


def parent_path():
    for p in (PAR, Path(str(PAR) + ".gz")):
        if p.exists():
            return p
    raise SystemExit("parent missing")


def opn(p):
    return gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")


def content_sha(p):
    h = hashlib.sha256()
    with opn(p) as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def state_rows(path, st, sid):
    v = np.full(len(st), np.nan); q = np.full(len(st), np.nan); stl = st.tolist(); sl = sid.tolist()
    with opn(path) as f:
        f.readline()
        for i, line in enumerate(f):
            a = line.split(b",", 4)
            if int(a[0]) != sl[i]:
                raise SystemExit(f"{path}: row {i} id {a[0]} differs from the keymap")
            if stl[i]:
                v[i] = float(a[2]); q[i] = float(a[3])
    assert i == len(st) - 1
    return v, q


def w3_apply(W, cls, chain, spec, cs, ch):
    v = sum(np.array([W[0][ch][i], W[1][ch][i]])[cls] * (chain, spec)[i] for i in range(2))
    return np.where(v > cs + 1e-9, cs, v) if ch == 0 else v


def ramp(j, side):
    j = j.astype(float)
    return np.clip(1 - (j - 1) / 2, 0, 1) if side == "start" else np.clip(1 - (18 - j) / 2, 0, 1)


def main(name, parity_only=False):
    t0 = time.time(); threading.Thread(target=_watch, daemon=True).start()
    import g2_build as B
    import asm2_build_gap2w as AB
    import gap_lib as GLb
    PP = parent_path(); shaP = content_sha(PP); assert shaP.startswith(PARSHA), ("parent sha", shaP)
    km = np.load(CR / "xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(np.int64), km["t"].astype(np.int64), km["s"].astype(np.int64)
    del km
    gap = gapk & st; assert int(gap.sum()) == N_GAP2
    spos = np.cumsum(st) - 1
    pv, pq = state_rows(PP, st, sid)
    P = pd.read_parquet(GAPWF); assert np.array_equal(P.submission_id.to_numpy(np.int64), sid[st].astype(np.int64))
    fv, fq = P.speed_kmh.to_numpy(float), P.flow_vph.to_numpy(float); del P
    Wg = {int(k): v for k, v in json.loads((CR / "l_b6/gapw/gap_cfg_gapw.json").read_text())["weights"].items()}
    m3 = {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads((CR / "c_gap2c47/weights_mar.json").read_text())["m3"].items()}
    WJ = json.loads(WJF.read_text())
    A6 = {sp: {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads(f.read_text())["fits"]["all"].items()} for sp, f in A6F.items()}
    KN = {(int(a), int(b), s): float(v) for k, v in json.loads(KNOBS.read_text())["U0"]["all"].items() for a, b, s in [k.split(",")]}
    MEM = {sp: [np.load(MEMD / f"pred_{tg}_off_{sp}.npz") for tg in MTAGS] for sp in ("validation", "private")}
    V6 = {sp: np.load(V6D / f"pred_v6r_off_{sp}.npz") for sp in ("validation", "private")}
    Z = {a: np.load(f) for a, f in HOOK.items()}
    assert list(Z["par"]["keys"]) == list(Z["new"]["keys"]), "hook keys differ"
    S = AB.Stages(["gap", "clip", "gap2"], None)   # only for gap_pred (M per split|panel)
    arms = {"par": (pv.copy(), pq.copy()), "new": (pv.copy(), pq.copy())}
    hit = np.zeros(len(sid), int); bad = {"p1_par": 0, "p1_new": 0, "p2": 0, "same": 0}; per = {}; guard = {"par": [0, 0], "new": [0, 0]}; nramp = 0
    CH = {f: [] for f in ("row", "jj", "cls", "dchv", "dchq")}
    for key in Z["par"]["keys"]:
        sp, pn = str(key).split("|"); si, pi = B.SPLITS.index(sp), B.PANELS.index(pn)
        zs, zf = Z["par"], Z["new"]
        for f in SAME:
            bad["same"] += int(not np.array_equal(zs[f"{key}|{f}"], zf[f"{key}|{f}"]))
        sel = np.flatnonzero(st & (ksplit == si) & (kpanel == pi)); g = gap[sel]
        kk = kd[sel] * 1000000 + kt[sel] * 1000 + ks[sel]; o = np.argsort(kk)
        qk = zs[f"{key}|d"].astype(np.int64) * 1000000 + zs[f"{key}|t"].astype(np.int64) * 1000 + zs[f"{key}|s"].astype(np.int64)
        j = o[np.searchsorted(kk[o], qk)]; assert (kk[j] == qk).all() and len(np.unique(j)) == len(j) == int(g.sum()) and g[j].all(), key
        rows = sel[j]; hit[rows] += 1
        cls, cs = zs[f"{key}|cls"].astype(int), zs[f"{key}|cs"].astype(float)
        spv, spq = zs[f"{key}|spec_v"].astype(float), zs[f"{key}|spec_q"].astype(float)
        bad["p2"] += int(((zs[f"{key}|v_new"] != fv[spos[rows]]) | (zs[f"{key}|q_new"] != fq[spos[rows]])).sum())
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
        # gap2 / v6r member values do not depend on the chain: computed once (h = 0)
        mv = np.mean([GLb.smooth_rows(M, x[f"{pn}_v"].astype(float), H) for x in MEM[sp]], 0)
        mq = np.mean([GLb.smooth_rows(M, x[f"{pn}_q"].astype(float), H) for x in MEM[sp]], 0)
        assert len(mv) == len(M) and np.isfinite(mv).all() and np.isfinite(mq).all()
        x6v = GLb.smooth_rows(M, V6[sp][f"{pn}_v"].astype(float), H); x6q = GLb.smooth_rows(M, V6[sp][f"{pn}_q"].astype(float), H)
        chs = {}
        for arm, z in (("par", zs), ("new", zf)):
            chv, chq = z[f"{key}|chain_v"].astype(float), z[f"{key}|chain_q"].astype(float)
            chs[arm] = (chv, chq)
            gv, gq = w3_apply(Wg, cls, chv, spv, cs, 0), w3_apply(Wg, cls, chq, spq, cs, 1)
            bad["p1_" + arm] += int(((gv != z[f"{key}|v_new"]) | (gq != z[f"{key}|q_new"])).sum())
            chM = (full(chv)[k2], full(chq)[k2]); assert np.isfinite(chM[0]).all() and np.isfinite(chM[1]).all()
            nv, nq = w3_apply(W3, cls, chv, spv, cs, 0), w3_apply(W3, cls, chq, spq, cs, 1)
            ov = pv[spos[sel]].copy(); oq = pq[spos[sel]].copy(); ov[j] = nv; oq[j] = nq
            wv = np.array([Gm[(0, 0)], Gm[(1, 0)]])[c2]; wq = np.array([Gm[(0, 1)], Gm[(1, 1)]])[c2]
            bv = (1 - wv) * ov[k2] + wv * mv; oq[k2] = (1 - wq) * oq[k2] + wq * mq
            ov[k2] = np.where(bv > cs_full[k2] + 1e-9, cs_full[k2], bv)
            av = np.array([A6[sp][(0, 0)], A6[sp][(1, 0)]])[c2]; aq = np.array([A6[sp][(0, 1)], A6[sp][(1, 1)]])[c2]
            bv = (1 - av) * ov[k2] + av * x6v; oq[k2] = (1 - aq) * oq[k2] + aq * x6q
            ov[k2] = np.where(bv > cs_full[k2] + 1e-9, cs_full[k2], bv)
            guard[arm][0] += int((ov[k2] < 3.).sum()); guard[arm][1] += int((oq[k2] < 0.).sum())
            ov[k2] = np.maximum(ov[k2], 3.); oq[k2] = np.maximum(oq[k2], 0.)
            pM = (ov[k2].copy(), oq[k2].copy()); out = []
            for ch in (0, 1):
                p = pM[ch].copy()
                for cc, side in SIDES:
                    p = p + KN[(cc, ch, side)] * ramp(jj, side) * (M[:, 3] == cc) * (chM[ch] - pM[ch])
                out.append(p)
            if arm == "par":
                nramp += int(((ramp(jj, "start") > 0) & (M[:, 3] == 1) | (ramp(jj, "end") > 0)).sum())
            ov[k2] = np.where(out[0] > cs_full[k2] + 1e-9, cs_full[k2], out[0]); oq[k2] = out[1]
            arms[arm][0][sel[g]] = ov[g]; arms[arm][1][sel[g]] = oq[g]
        # change-map rows (M order): full-file row, j, class, chain change FV - S2b
        CH["row"].append(sel[k2]); CH["jj"].append(jj.astype(np.int8)); CH["cls"].append(c2.astype(np.int8))
        CH["dchv"].append(full(chs["new"][0] - chs["par"][0])[k2]); CH["dchq"].append(full(chs["new"][1] - chs["par"][1])[k2])
        dv = np.abs(arms["new"][0][sel[g]] - pv[sel[g]]); dq = np.abs(arms["new"][1][sel[g]] - pq[sel[g]])
        dcv = np.abs(chs["new"][0] - chs["par"][0])
        per[str(key)] = dict(gap_rows=int(g.sum()), changed=int(((dv > 0) | (dq > 0)).sum()), mean_abs_dv=float(dv.mean()),
                             mean_abs_dq=float(dq.mean()), chain_changed=int((dcv > 0).sum()), chain_mean_abs_dv=float(dcv.mean()))
    assert (hit[gap] == 1).all() and not hit[~gap].any(), "gap row coverage"
    pc = int(((arms["par"][0] != pv) | (arms["par"][1] != pq))[gap].sum())
    print("GFVOFF_PARITY p1 S2b hook", "EXACT" if bad["p1_par"] == 0 else f"FAIL {bad['p1_par']}",
          "| p1 FV hook", "EXACT" if bad["p1_new"] == 0 else f"FAIL {bad['p1_new']}",
          "| p2 (S2b hook GAPW arm == t1_gapwf)", "EXACT" if bad["p2"] == 0 else f"FAIL {bad['p2']}",
          "| FV hook == S2b hook on keys/d/t/s/cls/cs/spec", "EXACT" if bad["same"] == 0 else f"FAIL {bad['same']}",
          "| pP (S2b chain, gap3 form == C73P gap rows)", "EXACT" if pc == 0 else f"FAIL {pc}",
          "| guard rows", guard, "| taper rows", nramp, flush=True)
    if any(bad.values()) or pc:
        sys.exit(2)
    if parity_only:
        print("GFVOFF parity only %.0f s" % (time.time() - t0)); return
    nv, nq = arms["new"]
    assert np.isfinite(nv[gap]).all() and np.isfinite(nq[gap]).all() and (nv[gap] > 0).all() and (nq[gap] >= 0).all()
    chg = ((nv != pv) | (nq != pq)) & st; assert not (chg & ~gap).any()
    rr = np.concatenate(CH["row"])
    np.savez_compressed(D / "change_rows.npz", row=rr, sid=sid[rr], split=ksplit[rr], panel=kpanel[rr], d=kd[rr], t=kt[rr], s=ks[rr],
                        jj=np.concatenate(CH["jj"]), cls=np.concatenate(CH["cls"]), p_v=pv[rr], p_q=pq[rr], dv=nv[rr] - pv[rr], dq=nq[rr] - pq[rr],
                        dchv=np.concatenate(CH["dchv"]), dchq=np.concatenate(CH["dchq"]))
    if os.environ.get("GFV_ROWS_ONLY"):
        print("GFVOFF rows only (dry run)"); return
    del arms
    full = D / f"{name}.csv"; assert not full.exists(), full
    tmp = D / f"{name}.csv.tmp"; gl = gap.tolist(); cl = chg.tolist(); hsh = hashlib.sha256()
    idx = np.flatnonzero(chg); vals = {int(i): (repr(float(nv[i])).encode(), repr(float(nq[i])).encode()) for i in idx}
    with opn(PP) as f, open(tmp, "wb") as o:
        x = f.readline(); o.write(x); hsh.update(x)
        for i, line in enumerate(f):
            if cl[i]:
                a = line.split(b",", 4); eol = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                rest = a[4]; assert rest.endswith(eol)
                line = b",".join([a[0], a[1], vals[i][0], vals[i][1], rest])
            o.write(line); hsh.update(line)
    os.replace(tmp, full); csha = hsh.hexdigest(); del vals
    sl = sid.tolist(); wv = np.full(len(st), np.nan); wq = np.full(len(st), np.nan); n = 0
    with opn(PP) as f, open(full, "rb") as g_:
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
    assert C0.sha(full) == csha, "file sha != streamed sha"
    dv = np.abs(nv - pv)[gap]; dq = np.abs(nq - pq)[gap]
    info = dict(probe=C0.rel(full), sha256=csha, sha256_note="sha256 of the CSV file, as uploaded", file_sha256=C0.sha(full),
                parent=C0.rel(PAR), parent_sha256=shaP, rows=N, gap2_rows=N_GAP2, rows_changed=int(chg.sum()),
                rows_changed_by_split={sp: int((chg & (ksplit == si)).sum()) for si, sp in enumerate(B.SPLITS)},
                change_size=dict(mean_abs_dv=float(dv.mean()), max_dv=float(dv.max()), mean_abs_dq=float(dq.mean()), max_dq=float(dq.max())),
                per_panel=per, knobs=C0.rel(KNOBS), knobs_sha256=C0.sha(KNOBS), knobs_used={f"{a},{b},{s}": v for (a, b, s), v in KN.items()},
                members=C0.rel(MEMD), member_tags=list(MTAGS), member_smoothing_h=H, v6=dict(dir=C0.rel(V6D), smoothing_h=H),
                weights=C0.rel(WJF), hooks={a: dict(path=C0.rel(f), sha256=C0.sha(f)) for a, f in HOOK.items()},
                guard_rows=guard, taper_rows=nramp,
                parity=dict(p1_s2b="EXACT", p1_fv="EXACT", p2="EXACT", fv_hook_same_fields="EXACT", pP_parent_gap_rows="EXACT"),
                seconds=round(time.time() - t0))
    (D / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    print("GFVOFF_PROBE", json.dumps({k: info[k] for k in ("probe", "sha256", "rows_changed", "rows_changed_by_split", "change_size", "guard_rows")}), flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/probe.py"), "make", str(PP), str(full), str(D / f"{name}.delta.csv.gz")], check=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], "--parity-only" in sys.argv)
