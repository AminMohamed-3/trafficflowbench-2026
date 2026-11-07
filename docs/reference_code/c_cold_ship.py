"""C 2026-10-10 (ledger 0778): ship arm XSD (exp/C_semisup/SEMISUP.md section 6) for the ongoing queue members, laptop side.
Called by c_cold_ship.sh after the VM2 job (c_cold_official.sh) is pulled.
  python c_cold_ship.py merge OFFICIAL_DIR SHIPPED_ROWS OUT_ON
      member files ctl_s0..4, sc_s0..4 (c_cold_split.py official): checks (ft parity 0, poison 0 both splits, every member),
      serve parity vs the shipped adapted probabilities (ctl per seed: p_serve == on/cnn_s<i>.prob; sccnn: p_serve_mean of
      sc_s4 == on/sccnn_official.prob) and un parity vs the shipped unadapted ones (off/), all exact (COLD_PARITY_TOL);
      then OUT_ON = the shipped on/ files with the cold-date official windows replaced by XSD (ctl: p_xsd of seed i,
      queue_pred by eiou_mask as production; sccnn: shipped ensemble formula over the 5 seeds' p_xsd). Writes cold_windows.csv.
  python c_cold_ship.py build OFFICIAL_DIR SERVED_ON SERVED_OFF OUT_ON
      build entry step (asm2_run_all.sh, ADOPTED row 13): served on/ (any recipe) with cold windows replaced by XSD.
  python c_cold_ship.py table REF_ONGOING NEW_ONGOING COLD_WINDOWS.csv
      changed ongoing rows vs the parent's (only cold windows may change), per split x type, 0->1 / 1->0, windows.
  python c_cold_ship.py final PARENT.csv PROBE.csv COLD_WINDOWS.csv
      every row and column equals the parent except queue_pred on cold-window rows; 6,980,503 rows, no blanks, binary."""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
K = ["window_id", "timestamp", "link_id"]
KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"
TOL = float(os.environ.get("COLD_PARITY_TOL", "0"))


def rp(p):
    return pd.read_parquet(p, use_threads=False)


def merge(off_dir, rows, out_on):
    import torch
    from qcnn_shift import eiou_mask
    off_dir, rows, out_on = Path(off_dir), Path(rows), Path(out_on)
    M = {f"{k}_s{i}": rp(off_dir / f"{k}_s{i}.parquet") for k in ("ctl", "sc") for i in range(5)}
    bad = []
    for n in M:
        js = json.loads((off_dir / f"{n}.parquet.json").read_text()); ch = js["checks"]
        for sp in ("validation", "private"):
            po = [c for c in ch if c["split"] == sp and c["kind"] == "poison"]
            fp = [c for c in ch if c["split"] == sp and c["kind"] == "ft_parity"]
            if not po or po[0]["serve_xall"] != 0. or po[0]["xsd"] != 0.: bad.append(f"poison {n} {sp} {po}")
            if not fp or fp[0]["max_diff"] != 0.: bad.append(f"ft_parity {n} {sp} {fp}")
        print("CHECKS", n, json.dumps(ch), "minutes", js["minutes"], flush=True)
    ref = M["ctl_s0"]
    for n, d in M.items():
        assert len(d) == len(ref) and (d[K].to_numpy() == ref[K].to_numpy()).all(), f"row keys differ {n}"
    wins = ref.drop_duplicates("window_id")[["split", "panel", "window_id", "date", "t", "H", "pool"]]
    wins.to_csv(out_on.parent / "cold_windows.csv", index=False)
    print("cold windows", wins.groupby("split").window_id.nunique().to_dict(), "dates", wins.groupby("split").date.unique().map(list).to_dict(), flush=True)
    if out_on.exists(): shutil.rmtree(out_on)
    out_on.mkdir(parents=True)
    par = {}

    def put(fname, key_df, p_new, p_serve, p_un, decide):
        on = rp(rows / "on" / fname); of = rp(rows / "off" / fname)
        assert (on[K].to_numpy() == of[K].to_numpy()).all()
        idx = pd.Series(np.arange(len(on)), index=pd.MultiIndex.from_frame(on[K])).reindex(pd.MultiIndex.from_frame(key_df[K]))
        assert idx.notna().all(), f"cold rows missing in {fname}"; idx = idx.to_numpy(np.int64)
        assert set(on.window_id.iloc[idx]) == set(key_df.window_id) and on.window_id.isin(set(key_df.window_id)).sum() == len(idx), fname
        ds = float(np.abs(on.prob.to_numpy(np.float64)[idx] - p_serve.astype(np.float64)).max())
        du = float(np.abs(of.prob.to_numpy(np.float64)[idx] - p_un.astype(np.float64)).max())
        par[fname] = dict(serve_vs_shipped_on=ds, un_vs_shipped_off=du)
        new = on.copy(); pr = new.prob.to_numpy().copy(); pr[idx] = p_new.astype(pr.dtype); new["prob"] = pr
        if decide and "queue_pred" in new.columns:
            q = new.queue_pred.to_numpy().copy()
            for w, ix in pd.Series(idx).groupby(on.window_id.iloc[idx].to_numpy()).groups.items():
                ii = idx[np.asarray(ix)]
                assert np.array_equal(eiou_mask(on.prob.to_numpy()[ii]).astype(q.dtype), on.queue_pred.to_numpy()[ii]), "decoder parity"
                q[ii] = eiou_mask(pr[ii]).astype(q.dtype)
            new["queue_pred"] = q
        new.to_parquet(out_on / fname)
        if fname.endswith(".parquet") and (rows / "on" / fname.replace(".parquet", ".csv")).exists():
            new.drop(columns="prob").to_csv(out_on / fname.replace(".parquet", ".csv"), index=False)
        print("ON", fname, "cold rows", len(idx), "mean|dp| xsd vs shipped", round(float(np.abs(pr[idx] - on.prob.to_numpy()[idx]).mean()), 5),
              "serve parity", ds, "un parity", du, flush=True)
    for i in range(5):
        d = M[f"ctl_s{i}"]
        put(f"cnn_s{i}.parquet", d, d.p_xsd.to_numpy(), d.p_serve.to_numpy(), d.p_un.to_numpy(), True)
    ens = lambda c: torch.stack([torch.from_numpy(M[f"sc_s{j}"][c].to_numpy(np.float32).copy()) for j in range(5)]).mean(0).numpy()
    assert "p_serve_mean" in M["sc_s4"].columns
    put("sccnn_official.parquet", M["sc_s0"], ens("p_xsd"), M["sc_s4"].p_serve_mean.to_numpy(), ens("p_un"), False)
    serve_ens_from_members = float(np.abs(ens("p_serve").astype(np.float64) - M["sc_s4"].p_serve_mean.to_numpy(np.float64)).max())
    worst = max(max(v.values()) for v in par.values())
    (out_on.parent / "parity.json").write_text(json.dumps(dict(files=par, sc_serve_mean_from_member_files=serve_ens_from_members, bad=bad), indent=1))
    print("PARITY worst", worst, "sc serve mean from member files vs sc_s4", serve_ens_from_members, "check failures", len(bad), flush=True)
    for b in bad: print("CHECK_FAIL", b)
    if bad or worst > TOL or serve_ens_from_members > TOL:
        print("MERGE_FAIL"); sys.exit(3)
    print("MERGE_OK")


def build(off_dir, on_dir, un_dir, out_on):
    """Build entry step (asm2_run_all.sh, ADOPTED row 13, C36 form): OUT_ON = the served on/ files (current recipe, mh5)
    with the cold-date official windows replaced by XSD (prob; ctl queue_pred by eiou_mask), as c_cold_ship_c35.sh's rebase
    of merge's ship/on. Checks: ft parity 0 and poison 0 for every member and split, row keys, un parity vs the served off/
    (tol COLD_PARITY_TOL, default 1e-6: float32 sccnn seed mean), decoder parity of the served ctl rows. The serve arm of
    c_cold_split.py uses the 3,3e-4 recipe, so p_serve is reported against the served on/ but not asserted."""
    import torch
    from qcnn_shift import eiou_mask
    off_dir, on_dir, un_dir, out_on = Path(off_dir), Path(on_dir), Path(un_dir), Path(out_on)
    tol = float(os.environ.get("COLD_PARITY_TOL", "1e-6"))
    M = {f"{k}_s{i}": rp(off_dir / f"{k}_s{i}.parquet") for k in ("ctl", "sc") for i in range(5)}
    bad = []
    for n in M:
        ch = json.loads((off_dir / f"{n}.parquet.json").read_text())["checks"]
        for sp in ("validation", "private"):
            po = [c for c in ch if c["split"] == sp and c["kind"] == "poison"]
            fp = [c for c in ch if c["split"] == sp and c["kind"] == "ft_parity"]
            if not po or po[0]["serve_xall"] != 0. or po[0]["xsd"] != 0.: bad.append(f"poison {n} {sp} {po}")
            if os.environ.get("COLD_SKIP_FT_PARITY") == "1" and not fp:
                pass                                          # line proved parity at weight 0 in its own phase 0 (L_coldaux); poison still asserted
            elif not fp or fp[0]["max_diff"] != 0.: bad.append(f"ft_parity {n} {sp} {fp}")
    if os.environ.get("COLD_SKIP_FT_PARITY") == "1": print("NOTE ft_parity records not required (COLD_SKIP_FT_PARITY=1)", flush=True)
    ref = M["ctl_s0"]
    for n, d in M.items():
        assert len(d) == len(ref) and (d[K].to_numpy() == ref[K].to_numpy()).all(), f"row keys differ {n}"
    wins = ref.drop_duplicates("window_id")[["split", "panel", "window_id", "date", "t", "H", "pool"]]
    wins.to_csv(off_dir / "cold_windows.csv", index=False)
    if out_on.exists(): shutil.rmtree(out_on)
    out_on.mkdir(parents=True)
    par = {}

    def put(fname, key_df, p_new, p_serve, p_un, decide):
        on = rp(on_dir / fname); of = rp(un_dir / fname)
        assert (on[K].to_numpy() == of[K].to_numpy()).all()
        idx = pd.Series(np.arange(len(on)), index=pd.MultiIndex.from_frame(on[K])).reindex(pd.MultiIndex.from_frame(key_df[K]))
        assert idx.notna().all(), f"cold rows missing in {fname}"; idx = idx.to_numpy(np.int64)
        assert set(on.window_id.iloc[idx]) == set(key_df.window_id) and on.window_id.isin(set(key_df.window_id)).sum() == len(idx), fname
        du = float(np.abs(of.prob.to_numpy(np.float64)[idx] - p_un.astype(np.float64)).max())
        ds = float(np.abs(on.prob.to_numpy(np.float64)[idx] - p_serve.astype(np.float64)).max())
        par[fname] = dict(un_vs_served_off=du, serve33_vs_served_on_info=ds)
        new = on.copy(); pr = new.prob.to_numpy().copy(); pr[idx] = p_new.astype(pr.dtype); new["prob"] = pr
        if decide and "queue_pred" in new.columns:
            q = new.queue_pred.to_numpy().copy()
            for w, ix in pd.Series(idx).groupby(on.window_id.iloc[idx].to_numpy()).groups.items():
                ii = idx[np.asarray(ix)]
                assert np.array_equal(eiou_mask(on.prob.to_numpy()[ii]).astype(q.dtype), on.queue_pred.to_numpy()[ii]), "decoder parity"
                q[ii] = eiou_mask(pr[ii]).astype(q.dtype)
            new["queue_pred"] = q
        new.to_parquet(out_on / fname, index=False)
        print("XSD", fname, "cold rows", len(idx), "changed prob", int((pr != on.prob.to_numpy()).sum()), "un parity", du, flush=True)
    for i in range(5):
        d = M[f"ctl_s{i}"]
        put(f"cnn_s{i}.parquet", d, d.p_xsd.to_numpy(), d.p_serve.to_numpy(), d.p_un.to_numpy(), True)
    ens = lambda c: torch.stack([torch.from_numpy(M[f"sc_s{j}"][c].to_numpy(np.float32).copy()) for j in range(5)]).mean(0).numpy()
    put("sccnn_official.parquet", M["sc_s0"], ens("p_xsd"), ens("p_serve"), ens("p_un"), False)
    worst = max(v["un_vs_served_off"] for v in par.values())
    (off_dir / "build_parity.json").write_text(json.dumps(dict(files=par, bad=bad, tol=tol), indent=1))
    for b in bad: print("CHECK_FAIL", b)
    print("cold windows", wins.groupby("split").window_id.nunique().to_dict(), "un parity worst", worst, "check failures", len(bad), flush=True)
    if bad or worst > tol:
        print("COLD_XSD_BUILD_FAIL"); sys.exit(3)
    print("COLD_XSD_BUILD_OK")


def table(ref, new, cw):
    import asm_build as AB
    r = rp(ref).set_index("submission_id").queue_pred; n = rp(new).set_index("submission_id").queue_pred.reindex(r.index)
    assert len(r) == 87000 and n.notna().all()
    key = pd.concat([x for x in pd.read_csv(AB.KEY, usecols=["submission_id", "split", "window_id"], dtype=str, chunksize=500000)])
    key = key.assign(submission_id=key.submission_id.astype(np.int64)).set_index("submission_id").loc[r.index]
    ty = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv").set_index("window_id").typ
    cold = set(pd.read_csv(cw).window_id)
    d = pd.DataFrame(dict(split=key.split.values, typ=ty.reindex(key.window_id.values).values, window=key.window_id.values,
                          old=r.values, new=n.values)); d["ch"] = d.old != d.new; d["cold"] = d.window.isin(cold)
    out_cold = int((d.ch & ~d.cold).sum()); assert out_cold == 0, f"{out_cold} changed rows outside cold windows"
    g = d[d.cold].groupby(["split", "typ"])
    t = pd.DataFrame(dict(cold_cells=g.size(), cold_windows=g.window.nunique(), changed=g.ch.sum(),
                          to1=g.apply(lambda x: int((x.new > x.old).sum())), to0=g.apply(lambda x: int((x.new < x.old).sum())),
                          windows_changed=g.apply(lambda x: x[x.ch].window.nunique()), pos_old=g.old.sum(), pos_new=g.new.sum()))
    print("== XSD vs parent ongoing rows: changed", int(d.ch.sum()), "windows", d[d.ch].window.nunique(), "(none outside cold windows)")
    print(t.to_string())
    w = d[d.ch].groupby(["split", "typ", "window"]).agg(changed=("ch", "sum"), to1=("new", lambda s: int(s.sum())))
    w["to0"] = w.changed - w.to1; print("PER_WINDOW"); print(w.to_string())


def final(parent, probe, cw):
    a = pd.read_csv(parent, float_precision="round_trip", dtype={"task": str})
    b = pd.read_csv(probe, float_precision="round_trip", dtype={"task": str})
    assert len(a) == len(b) == 6_980_503 and list(a.columns) == list(b.columns)
    for c in a.columns:
        if c == "queue_pred": continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        assert ((x == y) | (pd.isna(x) & pd.isna(y))).all(), f"column {c} differs from the parent"
    cold = set(pd.read_csv(cw).window_id)
    key = pd.read_csv(KEY, usecols=["submission_id", "window_id"], dtype={"window_id": str})
    win = pd.Series(key.window_id.values, index=key.submission_id.values).reindex(b.submission_id.values).to_numpy()
    x, y = a.queue_pred.to_numpy(), b.queue_pred.to_numpy(); same = (x == y) | (pd.isna(x) & pd.isna(y))
    incold = pd.Series(win).isin(cold).to_numpy()
    assert same[~incold].all(), f"{int((~same & ~incold).sum())} queue rows changed outside cold windows"
    q = b.queue_pred[b.task == "queue"].to_numpy(); assert not np.isnan(q).any() and np.isin(q, [0, 1]).all()
    print("FINAL_OK rows", len(b), "queue_pred changed", int((~same).sum()), "all in cold windows; every other cell equals the parent")


if __name__ == "__main__":
    {"merge": merge, "build": build, "table": table, "final": final}[sys.argv[1]](*sys.argv[2:])
    sys.stdout.flush(); os._exit(0)
