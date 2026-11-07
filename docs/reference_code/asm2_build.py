"""asm2 (ledger 0475) copy of asm_build.py (asm, ledger 0411) for the VM rebuild test. Only changes:
- the C03 parent path can be overridden with env TFB_ASM2_PARENT (csv or parquet; artifacts/claude_ref/asm2/c03_parent.parquet is
  C03 converted losslessly, float64 round trip), so the build runs on the VM, where submissions/ is not synced;
- env TFB_ASM2_ART overrides fitted-artifact files: JSON {relative path in the manifest: replacement path} (refit test);
- outputs default to artifacts/claude_ref/asm2/.
Everything else is asm_build.py verbatim (see that file's docstring).
"""
import os, json
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time, hashlib, argparse, platform
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import asm2_phys_lwr_diag as _P; sys.modules["phys_lwr_diag"] = _P  # portable shim (no sibling worktrees needed)
CR = ROOT / "artifacts/claude_ref"
OUTDIR = CR / "asm2"
PARENT = Path(os.environ.get("TFB_ASM2_PARENT") or ROOT / "submissions/c03_queue_cnn_adapt_ongoing.csv")
PARENT = PARENT if PARENT.is_absolute() else ROOT / PARENT
_ART = json.loads(os.environ.get("TFB_ASM2_ART") or "{}")


def read_parent(**kw):
    if str(PARENT).endswith(".parquet"):
        df = pd.read_parquet(PARENT, columns=kw.get("usecols"))
        return df.astype({"task": "category"}) if "task" in df else df
    return pd.read_csv(PARENT, float_precision="round_trip", **kw)

KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"
NROWS = 6980503

T1_ORDER = ["s08in", "c10", "s2", "b6", "polish", "s3l", "lsa", "s3p4", "gap", "clip"]
NEEDS = {"s08in": "c10", "s2": "c10", "b6": "s2", "polish": "b6", "s3l": "polish", "lsa": "polish", "s3p4": "lsa", "gap": "polish"}
GAP_CFG = CR / "gap/probes/c16_gap_v3sm1_on_c13_delta.json"  # shipped gap specialist config (weights, smooth)
GAP_PRED = CR / "gap/offpred_v3.npz"
S08_DIR = ROOT / "artifacts/lane_s/0398_s08"
B6W_S08 = "artifacts/claude_ref/asm/b6w_s08.json"
Q_COMP = {"ongoing_selcond": ("queue_ongoing", CR / "gapres/selcond_official_rows.csv"),
          "onset4_p1": ("queue_onset", CR / "onset4/c14_onset4_p1_on_c12.csv"),
          "onset5_s1": ("queue_onset", CR / "onset5/c15_onset5_s1_on_c13.csv"),
          "onset5_s1loc": ("queue_onset", CR / "onset5/c16_onset5_s1loc_on_c13.csv")}
# retained probe files keyed by the Task 1 chain prefix they equal (for --check-refs)
REFS = {("c10",): CR / "probes/c10_state_spd_hedge_flow_phys3_on_c03.csv",
        ("c10", "s2"): CR / "probes/c11_state_stack_s2stage_on_c03.csv",
        ("c10", "s2", "b6"): CR / "probes/c12_state_c11_b6blp_on_c03.csv",
        ("c10", "s2", "b6", "polish"): CR / "probes/c13_state_c12_polish_on_c03.csv",
        ("c10", "s2", "b6", "polish", "lsa"): CR / "phys4/c14_lsa_on_c13.csv",
        ("c10", "s2", "b6", "polish", "lsa", "s3p4"): CR / "phys4/c15_s3_on_c14.csv",
        ("c10", "s2", "b6", "polish", "clip"): CR / "gen/c15_genclip_on_c13.csv",
        ("c10", "s2", "b6", "polish", "lsa", "s3p4L"): (CR / "phys4/c14_lsa_on_c13.csv", CR / "phys4/c15L_s3big_on_c14.delta.csv.gz"),
        ("s08in", "c10", "s2", "b6", "polish", "s3l", "clip"): CR / "state5/c17_state5_s08s3l_on_c13.csv",
        ("c10", "s2", "b6", "polish", "gap"): CR / "gap/probes/c16_gap_v3sm1_on_c13.csv"}
B6_V1 = dict(weights="artifacts/claude_ref/b6full/own_eval_s012.json", val_arm="b6full:valown_:1,2:", priv_arm="b6full:priv_:0:", gate=None)
POLISH_JSON = CR / "polish/c13_polish_on_c03.json"
LSA_JSON = CR / "phys4/c14_lsa_on_c13.json"
VERSIONS = {"c10": "ledger 0358/0360/0368/0369 (P2 path, s2stage_probe internals)", "s2": "ledger 0389 VQP m_v/m_q",
            "b6": "ledger 0399 BLP (v1) unless --b6", "polish": "ledger 0402 chain from c13_polish_on_c03.json",
            "lsa": "ledger 0405 LSA prm from c14_lsa_on_c13.json", "s3p4": "ledger 0405 s3_yq/s3_yv (phys4)",
            "clip": "ledger 0409 1.15 x free_flow_speed_kmh", "s08in": "ledger 0400/0407 S08 official rows (lane_s 0398_s08) as the stack input",
            "gap": "ledger 0412 gap specialist v3 sm1 (LightGBM on the split's own simulated selector-like blocks), per-class blend on official gap cells (gap_probe)",
            "s3l": "ledger 0407 split-local stage 3 (state5_probe_base.fit_s3l on the split's I-S-OWN cells) + soft phys3 + TV", "ongoing_selcond": "gapres selcond c=.92",
            "onset4_p1": "ledger 0406 P1", "onset5_s1": "ledger 0408 S1", "onset5_s1loc": "ledger 0408 S1-loc"}


def sha(p):
    """sha256 of the file; for .gz files sha256 of the decompressed content (= the CSV that gunzip restores)."""
    import gzip
    h = hashlib.sha256()
    with (gzip.open(p, "rb") if str(p).endswith(".gz") else open(p, "rb")) as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class _HashWriter:
    def __init__(s, f):
        s.f = f; s.h = hashlib.sha256()

    def write(s, x):
        b = x.encode(); s.h.update(b); s.f.write(b); return len(x)


def write_csv(df, out):
    """Writes df as the probe CSV (CRLF like every retained probe). out *.csv.gz: gzip stream, same CSV bytes inside
    (laptop disk). Returns sha256 of the CSV bytes."""
    import gzip
    if str(out).endswith(".gz"):
        with gzip.open(out, "wb", compresslevel=6) as g:
            w = _HashWriter(g); df.to_csv(w, index=False, lineterminator="\r\n")
        return w.h.hexdigest()
    df.to_csv(out, index=False)
    return sha(out)


def csv_sha(df):
    """sha256 of the full CSV bytes df.to_csv would write (CRLF), without touching the disk."""
    class _Null:
        def write(s, b):
            return len(b)
    w = _HashWriter(_Null()); df.to_csv(w, index=False, lineterminator="\r\n")
    return w.h.hexdigest()


def read_delta(p):
    p = str(p)
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, float_precision="round_trip")


def write_out(df, out, speed0, flow0, q0, full_csv=None, base=None):
    """out *.delta.parquet: only rows that differ from C03 (submission_id + changed columns, float64 exact, zstd);
    sha256 returned is that of the full CSV the delta expands to. Otherwise a full CSV (or .csv.gz). Optional full CSV copy.
    base: path of the --t1-from delta; speed0/flow0 are then its Task 1 values and the delta holds only the rows on top of it."""
    extra = {}
    if str(out).endswith(".delta.parquet"):
        cols = {"speed_kmh": speed0, "flow_vph": flow0, "queue_pred": q0}
        diff = {c: df[c].to_numpy(float) != v for c, v in cols.items()}
        rows = np.flatnonzero(np.logical_or.reduce(list(diff.values())))
        keep = [c for c in cols if diff[c].any()]
        dl = pd.DataFrame({"submission_id": df.submission_id.to_numpy()[rows], **{c: df[c].to_numpy(float)[rows] for c in keep}})
        dl.to_parquet(out, index=False, compression="zstd")
        h = csv_sha(df)
        extra = dict(delta_rows=int(len(rows)), delta_columns=keep, delta_sha256=sha(out), delta_vs="C03 " + sha(PARENT)[:12])
        if base:
            extra.update(delta_base=str(base).replace("\\", "/"), delta_base_sha256=sha(base), delta_vs="C03 + delta_base")
    else:
        h = write_csv(df, out)
    if full_csv:
        h2 = write_csv(df, full_csv); assert h2 == h; extra["full_csv"] = str(full_csv)
    return h, extra


def expand(delta, out):
    """Full submission CSV from an asm delta file: C03 + delta rows; checks the sha256 recorded in the delta's manifest."""
    mf = json.loads(Path(str(delta) + ".manifest.json").read_text())
    assert sha(delta) == mf["delta_sha256"], "delta file changed since its build"
    sub = read_parent(dtype={"task": "category"})
    chain = [delta]
    if mf.get("delta_base"):
        bm = json.loads(Path(mf["delta_base"] + ".manifest.json").read_text())
        assert sha(mf["delta_base"]) == mf["delta_base_sha256"] == bm["delta_sha256"] and not bm.get("delta_base"), "delta base changed"
        chain = [mf["delta_base"], delta]
    for f in chain:
        dl = read_delta(f)
        rows = np.searchsorted(sub.submission_id.to_numpy(), dl.submission_id.to_numpy())
        assert (sub.submission_id.to_numpy()[rows] == dl.submission_id.to_numpy()).all()
        for c in dl.columns:
            if c != "submission_id":
                v = sub[c].to_numpy(float).copy(); v[rows] = dl[c].to_numpy(float); sub[c] = v
    h = write_csv(sub, out)
    assert h == mf["sha256"], f"expanded CSV sha {h} != manifest {mf['sha256']}"
    print("wrote", out, "sha256", h, "(matches manifest)")


def parse_arm(a):
    d, pre, seeds, suf = a.split(":")
    return dict(dir=d, pre=pre, seeds=[int(s) for s in seeds.split(",")], suf=suf)


class Stages:
    """Loads models / params once; each method maps the current pinned (v, q) to the next stage for one split/panel."""

    def __init__(s, comps, b6cfg, s3tag="", tune=None):
        s.comps = comps; s.inputs = {}; s.s3tag = s3tag
        s.tune = tune or {}   # tune (ledger 0479-0481) knob config; empty = the shipped F1 chain, unchanged
        if "s08in" in comps and "b6" in comps and not (b6cfg or {}).get("weights"):
            b6cfg = dict(b6cfg or {}, weights=B6W_S08)
        if "c10" in comps or "s2" in comps:
            import phys3_stack as ST, s2stage as S2
            s.w = np.asarray(json.loads(s._in(CR / "flow/opt/opt_summary.json").read_text())["pooled_weights"])
            cf = json.loads(s._in(CR / "state2/opt_coef.json").read_text()); s.cf = (cf["b_free"], cf["b_cong"])
            import lightgbm as lgb   # models load from the registered path, so TFB_ASM2_ART overrides apply (0768 fix)
            s.m368 = tuple(lgb.Booster(model_file=str(s._in(ST.OUT / f"m368_{n}.txt"))) for n in ("clf", "gc", "gf"))
            if "s2" in comps:
                s.s2m = tuple(lgb.Booster(model_file=str(s._in(S2.OUT / f"m_{k}.txt"))) for k in ("v", "q"))
        if "b6" in comps:
            s.b6 = dict(B6_V1, **(b6cfg or {}))
            s.b6W = json.loads(s._in(ROOT / s.b6["weights"]).read_text())
            s.b6arms = {"validation": parse_arm(s.b6["val_arm"]), "private": parse_arm(s.b6["priv_arm"])}
            if s.b6.get("gate"):
                g = json.loads(s._in(ROOT / s.b6["gate"]).read_text()); s.gateA = g["gate_all_days"]; s.b6W = {"per_panel": g["weights_all_days"], "_gate": True}
        if "b6" in comps and s.tune.get("b6"):   # per-split imputer arm + per-split blend weights
            s.b6arms = {sp: parse_arm(x["arm"]) for sp, x in s.tune["b6"].items()}
            s.b6Wsplit = {sp: json.loads(s._in(ROOT / x["weights"]).read_text()) for sp, x in s.tune["b6"].items()}
        if "polish" in comps:
            s.chain = json.loads(s._in(POLISH_JSON).read_text())["chain"]
            if s.tune.get("polish_chain"):
                s.chain = s.tune["polish_chain"]
        if "lsa" in comps:
            s.lsa_prm = json.loads(s._in(LSA_JSON).read_text())["prm"]
            if s.tune.get("lsa"):
                s.lsa_prm = s.tune["lsa"]
        if "s3p4" in comps:
            import phys4_s3 as S3
            import lightgbm as lgb
            if s.tune.get("s3"):   # bag: mean of the members' log-residual predictions, times scale [q, v]
                s.s3bag = [tuple(lgb.Booster(model_file=str(s._in(ROOT / f))) for f in pair) for pair in s.tune["s3"]["models"]]
                s.s3scale = s.tune["s3"].get("scale", [1., 1.])
            else:
                s.mq, s.mv = (lgb.Booster(model_file=str(s._in(S3.OUT / f"s3_{k}{s3tag}.txt"))) for k in ("yq", "yv"))
            s._in(S3.OUT / "s3_names.json")
        if "gap" in comps:
            cfg = json.loads(s._in(GAP_CFG).read_text()); s.gap_w = {int(k): v for k, v in cfg["weights"].items()}; s.gap_sm = int(cfg["smooth"])
            assert cfg["tag"] == "v3" and cfg["cnn"] == ""
            z = np.load(s._in(GAP_PRED)); s.gap_pred = {k: z[k] for k in z.files}
        if "s3l" in comps:
            import state5_lib as SL
            for tg in ("mar", "apr"):
                for pn in SL.PANELS:
                    s._in(SL.OUT / f"own_{tg}_s08_{pn}.npz")
            s.s3l = {}

    def _in(s, p):
        p = Path(p); p = p if p.is_absolute() else ROOT / p
        r = str(p.relative_to(ROOT)).replace("\\", "/")
        if r in _ART:
            p = ROOT / _ART[r]; print("override", r, "->", _ART[r], flush=True)
        s.inputs[str(p.relative_to(ROOT)).replace("\\", "/")] = sha(p); return p

    def run_panel(s, split, panel, c, d, t, sidx, v03, q03, log, vin=None, qin=None):
        """Returns {stage: (v, q) values on this panel's target rows in key order}."""
        import state2_cm as CM, flow_krige as FK, phys3_proj as PJ, phys3_stack as ST, phys3_build_probe as PB, s2stage as S2
        out = {}
        st = c.stations.sort_values("idx"); vcut = st.v_cut.to_numpy(float)
        tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
        vobs = np.asarray(c.speed, float); fobs = np.asarray(c.flow, float); elig = np.asarray(c.elig, bool)
        observed = elig & ~tgt & ~gap & np.isfinite(vobs)

        def grid(a):
            g = np.full(tgt.shape, np.nan); g[d, t, sidx] = a; return g

        pv, pq = (grid(v03), grid(q03)) if vin is None else (grid(vin), grid(qin))  # chain input (C03 = F4, or S08)
        if vin is not None:
            out["s08in"] = (vin, qin)
        cur = (pv, pq); I = (d, t, sidx)
        keep = {}
        if "c10" in s.comps:
            gr, ncm = CM.common_factor(vobs, observed, c.stations)
            rv = np.where(tgt, CM.apply(pv, gr, vcut, s.cf), np.nan)  # 0360
            full = PB.fill_time(np.where(tgt, rv, np.where(observed, vobs, np.nan)))
            pqf = PB.fill_time(np.where(tgt, pq, np.where(np.isfinite(fobs), fobs, np.nan)))
            hvf, hp = S2.hedge_368(s.m368, vobs, observed, c.stations, full, pqf, tgt)  # 0368
            hv = np.where(tgt, hvf, np.nan)
            N, den = FK.krige_log_flow(fobs, FK.positions(c.stations))
            hq = FK.blend(np.where(tgt, hv, 0.), pq, N, tgt, c.stations, s.w, den=den)  # 0358
            sv, sq, _ = PJ.project(hv, hq, tgt, c.stations, ST.R, ST.A, ST.G, gap=gap)  # phys3
            keep.update(gr=gr, ncm=ncm, rv=rv, hp=hp, hv=hv, hq=hq, N=N, den=den, sv=sv, sq=sq)
            s.hp = hp   # 0368 hedge probability grid of the last panel (read by the S 0631 query step in asm2_final)
            out["c10"] = (sv[I], sq[I]); cur = (grid(sv[I]), grid(sq[I]))
        if "s2" in s.comps:
            k = keep
            nv, nq = S2.apply_stage2(s.s2m, (vobs, fobs, elig, tgt, gap), c.stations, pv, pq, k["rv"], k["gr"], k["ncm"], k["N"],
                                     k["den"], k["hp"], k["hv"], k["hq"], k["sv"], k["sq"], tgt, "VQP")
            out["s2"] = (nv[I], nq[I]); cur = (grid(nv[I]), grid(nq[I]))
        bl = None
        if "b6" in s.comps:
            a = s.b6arms[split]
            zs = [np.load(s._in(CR / a["dir"] / f"imp_{a['pre']}{panel}_s{k_}.npz")) for k_ in a["seeds"]]
            assert all(list(z["dates"]) == list(c.dates) for z in zs)
            iv = np.mean([z["speed" + a["suf"]] for z in zs], 0).astype(float); iq = np.mean([z["flow" + a["suf"]] for z in zs], 0).astype(float)
            if getattr(s, "b6Wsplit", None):
                import b6full_eval as EV
                bv, bq = EV.blend(cur[0], cur[1], iv, iq, gap, s.b6Wsplit[split]["per_panel"][panel]["weights_all_days"])
            elif s.b6W.get("_gate"):
                import b6v2_gate as GT
                svd = np.std([z["speed" + a["suf"]] for z in zs], 0).astype(float); sqd = np.std([z["flow" + a["suf"]] for z in zs], 0).astype(float)
                la = np.broadcast_to(np.maximum(st.lanes.to_numpy(float), 1.), tgt.shape)
                bv, bq = GT.gated(cur[0], cur[1], iv, iq, svd, sqd, la, gap, s.b6W["per_panel"][panel], s.gateA, np.zeros(tgt.shape, int))
            else:
                import b6full_eval as EV
                bv, bq = EV.blend(cur[0], cur[1], iv, iq, gap, s.b6W["per_panel"][panel]["weights_all_days"])
            bl = (bv, bq)
            nv, nq, _ = PJ.project(bv, bq, tgt, c.stations, ST.R, ST.A, ST.G, gap=gap)
            out["b6"] = (nv[I], nq[I]); cur = (grid(nv[I]), grid(nq[I]))
        sp = vobs; fl = fobs
        if "polish" in s.comps:
            import polish_eval as E
            obs = elig & ~tgt & ~gap & np.isfinite(sp)
            assert "s5" not in s.chain
            ctx = E.Ctx(c.stations, tgt, np.where(obs, sp, np.nan), np.full(tgt.shape, np.nan), bl[0], bl[1], gap)
            nv, nq = E.apply_chain(s.chain, ctx, cur[0], cur[1])
            out["polish"] = (nv[I], nq[I]); cur = (grid(nv[I]), grid(nq[I]))
        if "s3l" in s.comps:
            import state5_probe_base as SPB
            if split not in s.s3l:
                s.s3l[split] = SPB.fit_s3l(split)
            obs = elig & ~tgt & ~gap & np.isfinite(sp)
            nv, nq = SPB.apply_s3l(s.s3l[split], c, tgt, gap, obs, cur[0], cur[1], grid(v03), grid(q03))
            out["s3l"] = (nv[I], nq[I]); cur = (grid(nv[I]), grid(nq[I]))
        F = None
        if "lsa" in s.comps or "s3p4" in s.comps:
            import phys4_lsa_eval as LE
            from tfb import metric
            obs2 = elig & ~tgt & ~gap & np.isfinite(sp) & np.isfinite(fl)
            prm = metric._parameters(c, ROOT)[0]; vf = prm.free_speed_kmh.to_numpy(float)
            so, fo = np.where(obs2, sp, np.nan), np.where(obs2, fl, np.nan)
            F = LE.lsa_fields(so, fo, vf)
        if "lsa" in s.comps:
            nv, nq = LE.apply(cur[0], cur[1], tgt, vf, F, s.lsa_prm, gap)
            out["lsa"] = (nv[I], nq[I]); cur = (grid(nv[I]), grid(nq[I]))
        if "s3p4" in s.comps:
            import phys4_feat as PF, phys4_s3 as S3
            v14, q14 = cur
            m = tgt & ~gap & np.isfinite(v14) & np.isfinite(q14) & (q14 > 0) & (v14 > 0)
            X, names, ix = PF.build(so, fo, tgt, v14, q14, keep["hp"], c.stations, prm.length_km.to_numpy(float), m)
            dd, tt, ss = ix.T.astype(int)
            lq = np.log(np.maximum(q14, 1.)); lv = np.log(np.maximum(v14, 1.))
            X = np.concatenate([X, np.stack([(F[0] - lq)[dd, tt, ss], F[1][dd, tt, ss], (F[2] - lv)[dd, tt, ss], F[3][dd, tt, ss]], 1).astype(np.float32)], 1)
            assert names + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"] == json.loads((S3.OUT / "s3_names.json").read_text())
            v15, q15 = v14.copy(), q14.copy()
            if getattr(s, "s3bag", None):
                q15[dd, tt, ss] = q14[dd, tt, ss] * np.exp(s.s3scale[0] * np.mean([m[0].predict(X, num_threads=3) for m in s.s3bag], 0))
                v15[dd, tt, ss] = v14[dd, tt, ss] * np.exp(s.s3scale[1] * np.mean([m[1].predict(X, num_threads=3) for m in s.s3bag], 0))
            else:
                q15[dd, tt, ss] = q14[dd, tt, ss] * np.exp(s.mq.predict(X, num_threads=3))
                v15[dd, tt, ss] = v14[dd, tt, ss] * np.exp(s.mv.predict(X, num_threads=3))
            del X
            out["s3p4"] = (v15[I], q15[I]); cur = (grid(v15[I]), grid(q15[I]))
        if "gap" in s.comps:  # gap specialist blend on official gap&target cells (as gap_probe; members c12 = current chain, spec)
            import gap_lib as G
            v, q = cur[0].copy(), cur[1].copy()
            if f"{split}|{panel}|M" in s.gap_pred:
                M = s.gap_pred[f"{split}|{panel}|M"]; gs = M[:, 6]
                vs, qs = G.smooth_rows(M, s.gap_pred[f"{split}|{panel}|v"], s.gap_sm), G.smooth_rows(M, s.gap_pred[f"{split}|{panel}|q"], s.gap_sm)
                dd, tt = M[:, 1], M[:, 2] + M[:, 5]
                assert gap[dd, tt, gs].all() and tgt[dd, tt, gs].all() and len(M) == int((gap & tgt).sum())
                cl = np.clip(M[:, 3], 0, 1); W = s.gap_w
                memv = [v[dd, tt, gs].copy(), vs]; memq = [q[dd, tt, gs].copy(), qs]
                v[dd, tt, gs] = sum(np.array([W[0][0][i], W[1][0][i]])[cl] * memv[i] for i in range(2))
                q[dd, tt, gs] = sum(np.array([W[0][1][i], W[1][1][i]])[cl] * memq[i] for i in range(2))
            else:
                assert not (gap & tgt).any(), (split, panel, "gap cells without specialist predictions")
            out["gap"] = (v[I], q[I]); cur = (grid(v[I]), grid(q[I]))
        if "clip" in s.comps:
            topo = pd.read_csv(s._in(ROOT / f"data/raw/kaggle_public/corridors/{panel}/network/lwr_mainline_topology.csv"))
            cl = 1.15 * topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
            assert np.isfinite(cl).all()
            v, q = cur[0][I], cur[1][I]; cs = cl[sidx]
            v = np.where(v > cs + 1e-9, cs, v)
            out["clip"] = (v, q)
        return out


def _ref_vals(rf, done):
    if isinstance(rf, tuple):  # (base full csv, delta csv.gz with submission_id, speed_kmh, flow_vph on changed rows)
        ref = pd.read_csv(rf[0], float_precision="round_trip", usecols=["submission_id", "speed_kmh", "flow_vph"])
        dl = pd.read_csv(rf[1], float_precision="round_trip")
        pos = pd.Index(ref.submission_id).get_indexer(dl.submission_id); assert (pos >= 0).all()
        v, q = ref.speed_kmh.to_numpy(float).copy(), ref.flow_vph.to_numpy(float).copy(); del ref
        v[pos] = dl.speed_kmh.to_numpy(float); q[pos] = dl.flow_vph.to_numpy(float)
        return v[done], q[done], "+".join(str(Path(x).relative_to(ROOT)).replace("\\", "/") for x in rf)
    ref = pd.read_csv(rf, float_precision="round_trip", usecols=["speed_kmh", "flow_vph"])
    return ref.speed_kmh.to_numpy(float)[done], ref.flow_vph.to_numpy(float)[done], str(rf.relative_to(ROOT)).replace("\\", "/")


def ref_checks(t1, stage_vals, done, s3tag=""):
    """Compare each Task 1 stage (on the computed rows) with the retained probe file of the same chain prefix."""
    out = {}
    if stage_vals is None or not done.any():
        return out
    for i in range(len(t1)):
        pre = tuple(t1[:i + 1]); rf = REFS.get(tuple(x + s3tag if x == "s3p4" else x for x in pre))
        if rf is None or not all(Path(x).exists() for x in (rf if isinstance(rf, tuple) else (rf,))) or pre[-1] not in stage_vals:
            continue
        rv_, rq_, rname = _ref_vals(rf, done)
        nv, nq = np.asarray(stage_vals[pre[-1]][0])[done], np.asarray(stage_vals[pre[-1]][1])[done]
        out["+".join(pre)] = dict(ref=rname, rows=int(done.sum()), max_abs_dv=float(np.abs(nv - rv_).max()),
                                  max_abs_dq=float(np.abs(nq - rq_).max()), exact_share=float(np.mean((nv == rv_) & (nq == rq_))))
        print("ref", out["+".join(pre)], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    if len(sys.argv) > 1 and sys.argv[1] == "--expand":  # python asm_build.py --expand X.delta.parquet out.csv
        expand(sys.argv[2], sys.argv[3]); return
    ap.add_argument("--components", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--full-csv", default=None, help="also write the full CSV here (only for the file to upload)")
    ap.add_argument("--delta-vs-t1", action="store_true", help="with --t1-from X.delta.parquet: write only the rows on top of X (queue rows); --expand chains X")
    ap.add_argument("--extra-deltas", default=None, help="comma list of external queue delta files (submission_id, queue_pred) vs C03")
    ap.add_argument("--b6", default=None, help="JSON overriding the b6 config (weights, val_arm, priv_arm, gate)")
    ap.add_argument("--t1-from", default=None, help="reuse Task 1 rows of an earlier asm build (same Task 1 components)")
    ap.add_argument("--check-stages", default=None, help="stages kept for --check-refs (default: last stage with a retained ref)")
    ap.add_argument("--s3tag", default="", help="phys4 stage-3 model tag: '' = small (s3_yq.txt), 'L' = big (s3_yqL.txt)")
    ap.add_argument("--check-refs", action="store_true"); ap.add_argument("--panels", default=None, help="debug: subset, no file written")
    a = ap.parse_args()
    t0 = time.time()
    comps = [x.strip() for x in a.components.split(",") if x.strip()]
    bad = [x for x in comps if x not in T1_ORDER and x not in Q_COMP]
    assert not bad, f"unknown components {bad}"
    t1 = [x for x in T1_ORDER if x in comps]; qc = [x for x in Q_COMP if x in comps]
    assert [x for x in comps if x in T1_ORDER] == t1, "Task 1 components must be listed in the fixed order " + ",".join(T1_ORDER)
    for x in t1:
        assert NEEDS.get(x) is None or NEEDS[x] in t1, f"{x} needs {NEEDS[x]}"
    assert sum(Q_COMP[x][0] == "queue_onset" for x in qc) <= 1, "at most one onset component"
    b6cfg = json.loads(a.b6) if a.b6 else None
    out = Path(a.out); OUTDIR.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(out.resolve().parent).free
    need = .35e9 if str(out).endswith((".gz", ".parquet")) else 1.0e9
    assert a.panels or free > need, f"only {free/1e9:.2f} GB free on the output disk (need {need/1e9:.2f} GB)"

    sub = read_parent(dtype={"task": "category"})
    kcols = ["task", "split", "panel", "timestamp", "link_id", "window_id"]
    key = pd.read_csv(KEY, usecols=["submission_id"], dtype=np.int64)
    for col in kcols:
        key[col] = pd.read_csv(KEY, usecols=[col], dtype=str)[col].astype("category")
    assert len(sub) == NROWS == len(key) and (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    st_rows = (key.task == "state").to_numpy()
    speed0, flow0 = sub.speed_kmh.to_numpy(float), sub.flow_vph.to_numpy(float)
    speed, flow = speed0.copy(), flow0.copy()
    S = Stages(t1, b6cfg, a.s3tag); S._in(PARENT); S._in(KEY)
    from tfb.data import load_cache
    stage_vals = None
    if a.check_refs and t1:  # in RAM, float64, only the stages named in --check-stages (default: last stage with a retained ref)
        cand = [x for i, x in enumerate(t1) if tuple(y + a.s3tag if y == "s3p4" else y for y in t1[:i + 1]) in REFS]
        keepst = a.check_stages.split(",") if a.check_stages else cand[-1:]
        stage_vals = {x: (np.full(NROWS, np.nan), np.full(NROWS, np.nan)) for x in keepst}
    done = np.zeros(NROWS, bool)
    stats = {}; panels = set(a.panels.split(",")) if a.panels else None
    if t1 and a.t1_from:  # reuse the Task 1 rows of an earlier asm build with the same Task 1 components (sha checked)
        mf = json.loads(Path(a.t1_from + ".manifest.json").read_text())
        assert [x for x in mf["components"] if x in T1_ORDER] == t1 and mf["b6_config"] == getattr(S, "b6", None) and mf.get("s3tag", "") == a.s3tag, "t1-from components differ"
        if a.t1_from.endswith(".delta.parquet"):
            assert sha(a.t1_from) == mf["delta_sha256"]; S.inputs[a.t1_from.replace("\\", "/")] = mf["delta_sha256"]
            tf = read_delta(a.t1_from)
            rows = np.searchsorted(key.submission_id.to_numpy(), tf.submission_id.to_numpy())
            assert (key.submission_id.to_numpy()[rows] == tf.submission_id.to_numpy()).all()
            m = st_rows[rows]; speed[rows[m]] = tf.speed_kmh.to_numpy(float)[m]; flow[rows[m]] = tf.flow_vph.to_numpy(float)[m]; del tf
        else:
            assert sha(a.t1_from) == mf["sha256"]; S.inputs[a.t1_from.replace("\\", "/")] = mf["sha256"]
            tf = pd.read_csv(a.t1_from, float_precision="round_trip", usecols=["submission_id", "speed_kmh", "flow_vph"], dtype={"submission_id": np.int64})
            assert (tf.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
            speed[st_rows] = tf.speed_kmh.to_numpy(float)[st_rows]; flow[st_rows] = tf.flow_vph.to_numpy(float)[st_rows]; del tf
        stats = mf["task1_stage_changes"]; stats["_t1_from"] = a.t1_from
    elif t1:
        s08 = {}
        for (split, panel), g in key[st_rows].groupby(["split", "panel"], observed=True):
            if panels and panel not in panels:
                continue
            vin = qin = None
            if "s08in" in t1:  # S08 official rows (S lane 0400 delivery) replace the C03 (F4) rows as the stack input
                if split not in s08:
                    s08[split] = pd.read_parquet(S._in(S08_DIR / f"official_{split}.parquet"), columns=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"])
                m8 = g[["timestamp", "link_id"]].astype(str).merge(s08[split][s08[split].panel == panel].astype({"timestamp": str, "link_id": str}),
                                                                   on=["timestamp", "link_id"], how="left", validate="one_to_one")
                assert len(m8) == len(g) and m8.speed_kmh.notna().all() and m8.flow_vph.notna().all(), ("S08 rows missing", split, panel)
                vin, qin = m8.speed_kmh.to_numpy(float), m8.flow_vph.to_numpy(float); del m8
            tp = time.time()
            c = load_cache(panel, split); st = c.stations.sort_values("idx")
            ts = pd.to_datetime(g.timestamp.astype(str))
            d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            s = g.link_id.astype(str).map(dict(zip(st.link_id.astype(str), st.idx))).to_numpy()
            assert not pd.isna(s).any(); s = s.astype(int)
            tgt = np.asarray(c.target, bool)
            assert tgt[d, t, s].all() and tgt.sum() == len(g), (split, panel, tgt.sum(), len(g))
            idx = g.index.to_numpy()
            res = S.run_panel(split, panel, c, d, t, s, speed0[idx], flow0[idx], stats, vin, qin)
            prev = (speed0[idx], flow0[idx])
            for x in t1:
                nv, nq = res[x]
                assert np.isfinite(nv).all() and np.isfinite(nq).all(), (x, split, panel)
                r = stats.setdefault(x, {}).setdefault(split, dict(rows=0, abs_dv=0., abs_dq=0., dq=0., dv=0., changed=0))
                r["rows"] += len(idx); r["abs_dv"] += float(np.abs(nv - prev[0]).sum()); r["abs_dq"] += float(np.abs(nq - prev[1]).sum())
                r["dv"] += float((nv - prev[0]).sum()); r["dq"] += float((nq - prev[1]).sum())
                r["changed"] += int(((np.abs(nv - prev[0]) > 1e-9) | (np.abs(nq - prev[1]) > 1e-9)).sum())
                if stage_vals is not None and x in stage_vals:
                    stage_vals[x][0][idx] = nv; stage_vals[x][1][idx] = nq
                prev = (nv, nq)
            speed[idx], flow[idx] = prev; done[idx] = True
            print(split, panel, "rows", len(idx), "s", round(time.time() - tp), flush=True)
    for x in ([] if a.t1_from else stats):
        for sp_, r in stats[x].items():
            n = r.pop("rows"); stats[x][sp_] = dict(rows=n, changed_share=r["changed"] / n, mean_abs_dv=r["abs_dv"] / n,
                                                    mean_abs_dq=r["abs_dq"] / n, mean_dv=r["dv"] / n, mean_dq=r["dq"] / n)
    if panels:
        print(json.dumps(stats, indent=1)); ref_checks(t1, stage_vals, done, a.s3tag); return

    # ---- queue rows
    qpred = sub.queue_pred.to_numpy(float).copy(); q0 = sub.queue_pred.to_numpy(float)
    qrows = (key.task == "queue").to_numpy()
    wi = pd.concat([pd.read_csv(p, dtype=str) for p in (ROOT / "data/raw/kaggle_public/task2").glob("*/*/window_index.csv")])
    wi = wi[wi.split.isin(["validation", "private"])]
    qlog = {}
    for x in qc:
        cond, f = Q_COMP[x]; S._in(f)
        wins = set(wi.window_id[wi.condition == cond])
        sel = qrows & key.window_id.isin(wins).to_numpy()
        if x == "ongoing_selcond":
            r = pd.read_csv(f, dtype={"window_id": str, "timestamp": str, "link_id": str})
            kq = key.loc[sel, ["window_id", "timestamp", "link_id"]].astype(str).reset_index()
            m = kq.merge(r, on=["window_id", "timestamp", "link_id"], how="left", validate="one_to_one")
            assert m.queue_pred_new.notna().all() and len(r) == len(m), "selcond rows do not cover the ongoing windows"
            new = m.queue_pred_new.to_numpy(float); rows = m["index"].to_numpy()
            wsum = pd.read_csv(f.parent / "selcond_official_windows.csv")
            same = set(wsum.window_id[wsum.tag == "none"])
            un = m.window_id.isin(same).to_numpy()
            assert (new[un] == q0[rows][un]).all(), "selcond base differs from C03 on untouched windows"
        else:
            cf = pd.read_csv(f, usecols=["submission_id", "queue_pred"], dtype={"submission_id": np.int64})
            assert (cf.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
            qa = cf.queue_pred.to_numpy(float)
            outside = qrows & ~sel
            assert np.array_equal(qa[outside], q0[outside]), f"{x}: queue rows outside onset windows differ from C03"
            rows = np.flatnonzero(sel); new = qa[rows]
        ch = new != qpred[rows]
        qlog[x] = dict(condition=cond, rows=int(len(rows)), cells_changed=int(ch.sum()),
                       windows_changed=sorted(set(key.window_id.to_numpy()[rows][ch].astype(str))))
        qpred[rows] = new
        print(x, {k: (v if k != "windows_changed" else len(v)) for k, v in qlog[x].items()}, flush=True)

    new = sub; new["speed_kmh"] = speed; new["flow_vph"] = flow; new["queue_pred"] = qpred
    # ---- validation (build_state_probe pattern)
    assert len(new) == NROWS and not new.isna().any().any()
    assert np.isfinite(speed[st_rows]).all() and np.isfinite(flow[st_rows]).all()
    other = ~st_rows
    assert np.array_equal(speed[other], speed0[other]) and np.array_equal(flow[other], flow0[other])
    qch = np.flatnonzero(qpred != q0)
    allowed = np.zeros(NROWS, bool)
    for x in qc:
        allowed |= qrows & key.window_id.isin(set(wi.window_id[wi.condition == Q_COMP[x][0]])).to_numpy()
    assert allowed[qch].all() and not st_rows[qch].any()
    assert np.array_equal(new.path_flow.to_numpy(float), read_parent(usecols=["path_flow"]).path_flow.to_numpy(float))
    for x in a.extra_deltas.split(",") if a.extra_deltas else []:  # external queue delta files (vs C03), applied last
        S._in(x); dl = read_delta(x)
        assert set(dl.columns) <= {"submission_id", "queue_pred", "parent_queue_pred", "window_id", "timestamp", "link_id"}, f"{x}: extra deltas may only carry queue_pred"
        rows = np.searchsorted(key.submission_id.to_numpy(), dl.submission_id.to_numpy())
        assert (key.submission_id.to_numpy()[rows] == dl.submission_id.to_numpy()).all() and qrows[rows].all()
        if "parent_queue_pred" in dl.columns:  # chained delta (e.g. onset6 on onset5 S1): its parent must equal the queue rows built so far
            assert np.array_equal(dl.parent_queue_pred.to_numpy(float), qpred[rows]), f"{x}: parent_queue_pred differs from the current build"
        else:
            assert not np.isin(rows, qch).any(), f"{x}: rows overlap an earlier queue component"
        conds = set(wi.set_index("window_id").condition.reindex(key.window_id.to_numpy()[rows].astype(str)))
        assert len(conds) == 1, f"{x}: delta spans conditions {conds}"
        ch = dl.queue_pred.to_numpy(float) != qpred[rows]
        qpred[rows] = dl.queue_pred.to_numpy(float); new["queue_pred"] = qpred
        qlog["delta:" + x] = dict(condition=conds.pop(), rows=int(len(rows)), cells_changed=int(ch.sum()),
                                  windows_changed=sorted(set(key.window_id.to_numpy()[rows][ch].astype(str))))
        print("delta", x, qlog["delta:" + x]["cells_changed"], flush=True)
    assert not new.isna().any().any()
    if a.delta_vs_t1:
        assert a.t1_from and a.t1_from.endswith(".delta.parquet") and str(out).endswith(".delta.parquet")
        h, extra = write_out(new, out, speed, flow, q0, a.full_csv, base=a.t1_from)
    else:
        h, extra = write_out(new, out, speed0, flow0, q0, a.full_csv)
    del sub, new
    refchk = ref_checks(t1, stage_vals, done, a.s3tag)
    import lightgbm, scipy
    man = dict(output=str(out), sha256=h, sha256_note="sha256 of the full CSV bytes (CRLF, pandas to_csv) this build expands to", **extra, rows=NROWS, parent=str(PARENT.relative_to(ROOT)).replace("\\", "/"), components=comps,
               versions={x: VERSIONS[x] for x in comps}, b6_config=getattr(S, "b6", None), s3tag=a.s3tag, polish_chain=getattr(S, "chain", None),
               lsa_prm=getattr(S, "lsa_prm", None), task1_stage_changes=stats, queue=qlog, ref_checks=refchk, inputs=S.inputs,
               runtime_s=round(time.time() - t0), env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                                           lightgbm=lightgbm.__version__, scipy=scipy.__version__, platform=platform.platform()),
               built_utc=time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()))
    Path(str(out) + ".manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print("wrote", out, "sha256", h, "runtime", man["runtime_s"], "s", flush=True)


if __name__ == "__main__":
    main()
