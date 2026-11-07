"""Row 27 piece (worker R2 2026-10-21): COPY of exp/L_gap2/g2_official.py (sha256
ed84b37be695c96e483eb2561a2878c59afc469bb0da7aca18a853ec1b3b80a6), values logic unchanged. Parameterised (env, defaults = the
original stored paths): G2O_PARENT (C66 file; sha must start f1a5278f unless G2O_SKIP_PARENT_SHA=1), G2O_GAPWF (row 26
post/t1_gapwf.state.parquet, parity p2), G2O_HOOK (row 26 hooked S2b build npz; the row-26 hook is the same g2_build.py hook),
G2O_OUT (output csv; info json next to it; no delta file), G2O_GAPCFG (gap_cfg_gapw.json), G2O_M3 (weights_mar.json),
G2O_SHIP (shipped gap2 members cnr0-2 = row 19 pred_cpu, used for parity p3). Library paths from $TFB_OLD_REPO.
Original text:
L_gap2 official gap rows from the captured gap-stage inputs (exp/L_gap2/g2_build.py hook: chain, spec v3 sm, class, clip per gap row),
then the shipped gap2 step (exp/C_gap2c47/asm2_build_gap2w.py Stages.gap2_rows, members = c_gap2c47/pred_cpu unless --members DIR),
streamed over the parent C66 (artifacts/claude_ref/l_b6/bundle18/c_bundle18_s2b_s7b.csv, sha f1a5278f): ONLY the 128,465 gap rows
may change. Weights: a JSON {"validation": {"W3": {class: [[v chain, v spec], [q chain, q spec]]}, "gap2": {"c,ch": m}}, "private": ...}.
Parities (all must hold, else exit 2):
  p1  hook GAPW arm (v_new/q_new) == the same formula recomputed from chain/spec with gap_cfg_gapw.json W3 (bit for bit)
  p2  hook GAPW arm == bundle18 off_c_bundle18_s2b/t1_gapwf.state.parquet gap rows (bit for bit)  -> the captured inputs ARE C66's
  p3  GAPW W3 + weights_mar.json m3 through this script == C66's gap rows bit for bit (both splits)
Usage (repo root): python exp/L_gap2/g2_official.py WEIGHTS.json NAME [--members DIR] [--v6 DIR --v6a MARCH.json,APRIL.json] [--parity-only]
  --v6: G3, pred_v6r_off_<split>.npz in DIR blended after the gap2 step with fits.all a of each split's json
Out: artifacts/claude_ref/l_gap2/NAME.csv, NAME.delta.csv.gz (tools/probe.py make), NAME.info.json"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
for p in ("src", "docs/reference_code", "exp/C_gap2c47", "exp/L_b6"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, os.environ.get("GAP2W_SRC") or str(ROOT / "exp/C_gap2c47"))
CR = ROOT / "artifacts/claude_ref"
E = os.environ.get
C66 = Path(E("G2O_PARENT") or CR / "l_b6/bundle18/c_bundle18_s2b_s7b.csv"); C66SHA = "" if E("G2O_SKIP_PARENT_SHA") == "1" else "f1a5278f"
GAPWF = Path(E("G2O_GAPWF") or CR / "l_b6/bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet")
HOOK = Path(E("G2O_HOOK") or CR / "l_gap2/build/t1.csv.gaprows.npz")
OUTF = Path(E("G2O_OUT") or CR / "l_gap2/x.csv"); D = OUTF.parent
GAPCFG = Path(E("G2O_GAPCFG") or CR / "l_b6/gapw/gap_cfg_gapw.json")
M3F = Path(E("G2O_M3") or CR / "c_gap2c47/weights_mar.json")
SHIP = Path(E("G2O_SHIP") or CR / "c_gap2c47/pred_cpu")
N = 6980503; N_GAP2 = 128465
V6 = A6 = None
import c0_official as C0   # sha, state_rows, rel (functions only)


def w3_apply(W, cls, chain, spec, cs, ch):
    """asm2_build gap formula, same operation order as the build hook: sum over (chain, spec) of weight[cls] * member, speed clip."""
    v = sum(np.array([W[0][ch][i], W[1][ch][i]])[cls] * (chain, spec)[i] for i in range(2))
    return np.where(v > cs + 1e-9, cs, v) if ch == 0 else v


def main(wfile, name, members=None, parity_only=False, v6dir=None, v6a=None):
    t0 = time.time()
    global V6, A6
    MTAGS = os.environ.get("G2_MEMBER_TAGS", "").split(",") if os.environ.get("G2_MEMBER_TAGS") else None
    V6 = {sp: np.load(Path(v6dir) / f"pred_v6r_off_{sp}.npz") for sp in ("validation", "private")} if v6dir else None
    A6 = {sp: {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads(Path(f).read_text())["fits"]["all"].items()}
          for sp, f in zip(("validation", "private"), v6a.split(","))} if v6a else None   # --v6a MARCH_FITS.json,APRIL_FITS.json
    import g2_build as B
    import asm2_build_gap2w as AB
    AB.GAP2_PRED = SHIP.resolve(); AB.GAP2_W = M3F.resolve()
    _in0 = AB.Stages._in

    def _in(s, p):   # files outside the development repo (clean machine): no ROOT-relative name, same file read
        p = Path(p)
        try:
            p.resolve().relative_to(AB.ROOT.resolve()) if p.is_absolute() else None
        except ValueError:
            s.inputs[str(p)] = C0.sha(p); return p
        return _in0(s, p)
    AB.Stages._in = _in
    sha66 = C0.sha(C66); assert sha66.startswith(C66SHA), "C66 sha"
    km = np.load(CR / "xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(np.int64), km["t"].astype(np.int64), km["s"].astype(np.int64)
    gap = gapk & st; assert int(gap.sum()) == N_GAP2
    spos = np.cumsum(st) - 1
    pv, pq = C0.state_rows(C66, st, sid)
    P = pd.read_parquet(GAPWF); assert np.array_equal(P.submission_id.to_numpy(np.int64), sid[st].astype(np.int64))
    fv, fq = P.speed_kmh.to_numpy(float), P.flow_vph.to_numpy(float)
    Wg = {int(k): v for k, v in json.loads(GAPCFG.read_text())["weights"].items()}
    m3 = {tuple(int(x) for x in k.split(",")): float(v) for k, v in json.loads(M3F.read_text())["m3"].items()}
    WJ = json.loads(Path(wfile).read_text()) if wfile != "none" else None
    z = np.load(HOOK)
    S = AB.Stages(["gap", "clip", "gap2"], None)
    ship_pred = S.gap2_pred
    new_pred = {sp: [np.load(Path(members) / f"pred_{tg}_off_{sp}.npz") for tg in (MTAGS or AB.GAP2_TAGS)] for sp in ("validation", "private")} if members else ship_pred
    arms = {"par": (pv.copy(), pq.copy()), "new": (pv.copy(), pq.copy())}
    hit = np.zeros(len(sid), int); bad = {"p1": 0, "p2": 0}; per = {}
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
        for arm in ("par", "new"):
            if arm == "par" or WJ is None or sp not in WJ:
                W3, G2 = Wg, m3
            else:
                W3 = {int(k): v for k, v in WJ[sp]["W3"].items()}; G2 = {tuple(int(x) for x in k.split(",")): float(v) for k, v in WJ[sp]["gap2"].items()}
            S.gap2_pred = ship_pred if arm == "par" else new_pred
            nv, nq = w3_apply(W3, cls, chv, spv, cs, 0), w3_apply(W3, cls, chq, spq, cs, 1)
            v0 = pv[spos[sel]].copy(); q0 = pq[spos[sel]].copy(); v0[j] = nv; q0[j] = nq
            S.gap2_w = G2
            ov, oq = S.gap2_rows(sp, pn, kd[sel], kt[sel], ks[sel], v0, q0, cs_full)
            if arm == "new" and V6 is not None:   # G3: v6r blended after the gap2 step, a per class x channel, speed clip
                import gap_lib as GLb
                M = S.gap_pred[f"{sp}|{pn}|M"]
                kk2 = M[:, 1].astype(np.int64) * 1000000 + (M[:, 2] + M[:, 5]).astype(np.int64) * 1000 + M[:, 6].astype(np.int64)
                k2 = o[np.searchsorted(kk[o], kk2)]; assert (kk[k2] == kk2).all() and len(np.unique(k2)) == len(k2)
                zv = V6[sp]; c2 = np.clip(M[:, 3], 0, 1)
                x6v = GLb.smooth_rows(M, zv[f"{pn}_v"].astype(float), 1); x6q = GLb.smooth_rows(M, zv[f"{pn}_q"].astype(float), 1)
                av = np.array([A6[sp][(0, 0)], A6[sp][(1, 0)]])[c2]; aq = np.array([A6[sp][(0, 1)], A6[sp][(1, 1)]])[c2]
                bv = (1 - av) * ov[k2] + av * x6v; oq[k2] = (1 - aq) * oq[k2] + aq * x6q
                ov[k2] = np.where(bv > cs_full[k2] + 1e-9, cs_full[k2], bv)
            arms[arm][0][sel[g]] = ov[g]; arms[arm][1][sel[g]] = oq[g]
        dv = np.abs(arms["new"][0][sel[g]] - pv[sel[g]]); dq = np.abs(arms["new"][1][sel[g]] - pq[sel[g]])
        per[str(key)] = dict(gap_rows=int(g.sum()), changed=int(((dv > 0) | (dq > 0)).sum()), mean_abs_dv=float(dv.mean()), mean_abs_dq=float(dq.mean()))
    assert (hit[gap] == 1).all() and not hit[~gap].any(), "gap row coverage"
    p3 = int(((arms["par"][0] != pv) | (arms["par"][1] != pq))[gap].sum())
    print("G2OFF_PARITY p1 (hook GAPW arm == recomputed)", "EXACT" if bad["p1"] == 0 else f"FAIL {bad['p1']}",
          "| p2 (hook GAPW arm == t1_gapwf gap rows)", "EXACT" if bad["p2"] == 0 else f"FAIL {bad['p2']}",
          "| p3 (GAPW + m3 here == C66 gap rows)", "EXACT" if p3 == 0 else f"FAIL {p3}", flush=True)
    if bad["p1"] or bad["p2"] or p3:
        sys.exit(2)
    if parity_only:
        print("G2OFF parity only %.0f s" % (time.time() - t0)); return
    nv, nq = arms["new"]
    assert np.isfinite(nv[gap]).all() and np.isfinite(nq[gap]).all() and (nv[gap] > 0).all() and (nq[gap] >= 0).all()
    chg = ((nv != pv) | (nq != pq)) & st; assert not (chg & ~gap).any()
    full = OUTF; assert not full.exists(), full
    tmp = Path(str(OUTF) + ".tmp"); gl = gap.tolist(); nvl, nql = nv.tolist(), nq.tolist()
    with open(C66, "rb") as f, open(tmp, "wb") as o:
        o.write(f.readline())
        for i, line in enumerate(f):
            if gl[i] and chg[i]:
                a = line.split(b",", 4); eol = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                rest = a[4]; assert rest.endswith(eol)
                o.write(b",".join([a[0], a[1], repr(nvl[i]).encode(), repr(nql[i]).encode(), rest]))
            else:
                o.write(line)
    os.replace(tmp, full)
    wv, wq = C0.state_rows(full, st, sid)
    assert np.array_equal(wv[gap], nv[gap]) and np.array_equal(wq[gap], nq[gap]), "written gap rows differ"
    assert np.array_equal(wv[st & ~gap], pv[st & ~gap]) and np.array_equal(wq[st & ~gap], pq[st & ~gap]), "non-gap state rows changed"
    with open(C66, "rb") as f, open(full, "rb") as g_:
        f.readline(); g_.readline()
        for i, (a, b) in enumerate(zip(f, g_)):
            if not gl[i] and a != b:
                raise SystemExit(f"line {i} changed outside the gap rows")
    dv = np.abs(nv - pv)[gap]; dq = np.abs(nq - pq)[gap]
    info = dict(probe=C0.rel(full), sha256=C0.sha(full), parent=C0.rel(C66), parent_sha256=sha66, rows=N, gap2_rows=N_GAP2, rows_changed=int(chg.sum()),
                rows_changed_by_split={sp: int((chg & (ksplit == si)).sum()) for si, sp in enumerate(B.SPLITS)},
                change_size=dict(mean_abs_dv=float(dv.mean()), max_dv=float(dv.max()), mean_abs_dq=float(dq.mean()), max_dq=float(dq.max())),
                per_panel=per, weights=C0.rel(wfile) if wfile != "none" else None, weights_sha256=C0.sha(wfile) if wfile != "none" else None,
                members=C0.rel(members) if members else C0.rel(SHIP), member_tags=list(MTAGS or AB.GAP2_TAGS), hook=C0.rel(HOOK), hook_sha256=C0.sha(HOOK),
                v6=dict(dir=C0.rel(v6dir), a={sp: {f"{k[0]},{k[1]}": v for k, v in A6[sp].items()} for sp in A6},
                        files={sp: C0.sha(Path(v6dir) / f"pred_v6r_off_{sp}.npz") for sp in A6}) if v6dir else None,
                parity=dict(p1="EXACT", p2="EXACT", p3="EXACT"), seconds=round(time.time() - t0))
    Path(str(OUTF) + ".info.json").write_text(json.dumps(info, indent=1))
    print("G2OFF_PROBE", json.dumps({k: info[k] for k in ("probe", "sha256", "rows_changed", "rows_changed_by_split", "change_size")}), flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = lambda f: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else None
    mem, v6dir, v6a = opt("--members"), opt("--v6"), opt("--v6a")
    a = [x for x in a if x not in (mem, v6dir, v6a)]
    main(a[0], a[1], mem, "--parity-only" in sys.argv, v6dir, v6a)
