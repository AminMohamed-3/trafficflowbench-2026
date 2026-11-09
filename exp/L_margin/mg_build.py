"""L_margin official decode and build (laptop CPU; exp/L_margin/PUSH.md 4-5). Ongoing rows of the 80 official windows only.
Delta form (LINE.md 5): p = clip(shipped + (NEW - BASE), 0, 1), NEW / BASE = two arms of the SAME Kaggle process (k1 / k0, or k0b / k0
for the noise reference).
  members ROOT NEW BASE NAME [--selftest]
        cold: per shipped A3 member file (artifacts/claude_ref/l_dose/a3/members/<tag>.parquet) p_xall, p_xsd += delta from
              ROOT/official/<arm>/<tag>_<split>.parquet -> build/members_NAME/<tag>.parquet (+ json with the shipped checks)
        warm: the served mh5 on/ files (asm2/e16/s01234_5_6e-4_none_out/rows/on): cnn_s<j>.parquet prob += delta of ctl_s<j> on the
              warm official windows (queue_pred re-decoded by eiou_mask on those windows, decoder parity asserted first);
              sccnn_official.parquet prob += mean_j delta of sc_s<j> -> build/on_NAME/
        --selftest: NEW = BASE = shipped (every delta 0): must reproduce the C59 rows exactly (rows step: 0 cells)
  chain NAME          exp/L_dose/a1_build.chain with COLD_MEMBERS = build/members_NAME and on/ = build/on_NAME -> build/NAME_chain/a0.75
  rows NAME [--no-parity]   decode with the C59 adapted Tiny (tools/rebase/rebase_c57.decode_all, P0 = P1 = A3 = the same probs,
        w8 .2); parity first (A3 chain + C59 Tiny == C59 rows, 0 cells); -> build/rows_NAME.parquet, build/rows_NAME.json
        (changed cells / windows per split, cold and warm, vs the C59 rows)
  delta PARENT.csv NAME OUT_NAME   parent's ongoing rows must equal the C59 rows (asserted), then a3.delta on PARENT (window rule
        >= 3 March windows, probe.py make/expand with sha, stream check, window_guard --no-record) -> build/delta/
        PARENT must be an ABSOLUTE Windows path (C:/...), not an MSYS /c/ path.
Then (C): exp/L_diag62/undo_check.py, window guard, manifest line, ONE board METHOD read (class A)."""
import sys, os, json, shutil, hashlib, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "docs/reference_code", ROOT / "tools/rebase"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
A = ROOT / "artifacts/claude_ref/l_margin"; SHIP = ROOT / "artifacts/claude_ref/l_dose/a3/members"; OUT = A / "build"
ON = ROOT / "artifacts/claude_ref/asm2/e16/s01234_5_6e-4_none_out/rows/on"
C59_ROWS = ROOT / "artifacts/claude_ref/l_tinyad/a1/on_c57/rows_c57.parquet"
TINY = ROOT / "artifacts/claude_ref/l_tinyad/a1/member/tinyad_member.parquet"
TAGS = [f"{k}_s{i}" for k in ("ctl", "sc") for i in range(5)]
K = ["window_id", "timestamp", "link_id"]
SPLITS = ("validation", "private")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arm(root, sub, a, t):
    return pd.concat([pd.read_parquet(Path(root) / sub / a / f"{t}_{s}.parquet") for s in SPLITS], ignore_index=True)


def members(root, new, base, name, selftest=False):
    from qcnn_shift import eiou_mask
    mem = OUT / f"members_{name}"; mem.mkdir(parents=True, exist_ok=True); info = {}
    for t in TAGS:                                                         # cold: A3 member files
        s = pd.read_parquet(SHIP / f"{t}.parquet"); out = s.copy()
        a1 = s if selftest else arm(root, "official", new, t); a0 = s if selftest else arm(root, "official", base, t)
        a1 = a1.set_index(K).reindex(pd.MultiIndex.from_frame(s[K])); a0 = a0.set_index(K).reindex(pd.MultiIndex.from_frame(s[K]))
        assert a1.p_xsd.notna().all() and a0.p_xsd.notna().all(), f"{t}: keys differ from the shipped member file"
        for c in ("p_xall", "p_xsd"):
            mv = a1[c].to_numpy(np.float64) - a0[c].to_numpy(np.float64)
            out[c] = np.clip(s[c].to_numpy(np.float64) + mv, 0., 1.).astype(s[c].dtype)
        out.to_parquet(mem / f"{t}.parquet")
        rj = json.loads((SHIP / f"{t}.parquet.json").read_text()); d = np.abs(out.p_xsd.to_numpy(np.float64) - s.p_xsd.to_numpy(np.float64))
        info[t] = dict(sha256=sha(mem / f"{t}.parquet"), cells_moved=int((d > 0).sum()), max_move=float(d.max()), mean_move=float(d.mean()))
        rj.update(margin=dict(form="delta: shipped + (new - base)", selftest=selftest, root=str(root), new=new, base=base, **info[t]))
        (mem / f"{t}.parquet.json").write_text(json.dumps(rj, indent=1))
    on = OUT / f"on_{name}"; on.mkdir(parents=True, exist_ok=True)               # warm: served mh5 on/ files
    cold_w = set(pd.read_parquet(SHIP / "ctl_s0.parquet", columns=["window_id"]).window_id)
    wi = {}
    for j in range(5):
        f = pd.read_parquet(ON / f"cnn_s{j}.parquet"); keys = pd.MultiIndex.from_frame(f[K])
        warm = ~f.window_id.isin(cold_w).to_numpy(); wi["warm_windows"] = int(f.window_id[warm].nunique())
        if selftest:
            mv = np.zeros(len(f))
        else:
            a1 = arm(root, "official_warm", new, f"ctl_s{j}").set_index(K).p_mh5; a0 = arm(root, "official_warm", base, f"ctl_s{j}").set_index(K).p_mh5
            mv = (a1.reindex(keys).to_numpy(np.float64) - a0.reindex(keys).to_numpy(np.float64))
            assert np.isfinite(mv[warm]).all() and np.isnan(mv[~warm]).all(), f"cnn_s{j}: warm keys differ from the on/ file"
            mv = np.where(warm, mv, 0.)
        pr = f.prob.to_numpy(np.float64); new_p = np.where(warm, np.clip(pr + mv, 0., 1.), pr).astype(f.prob.dtype)
        q = f.queue_pred.to_numpy().copy()
        for w, ix in f[warm].groupby("window_id").indices.items():
            ii = np.flatnonzero(warm)[ix]
            assert np.array_equal(eiou_mask(f.prob.to_numpy()[ii]).astype(q.dtype), q[ii]), f"decoder parity cnn_s{j} {w}"
            q[ii] = eiou_mask(new_p[ii]).astype(q.dtype)
        g = f.copy(); g["prob"] = new_p; g["queue_pred"] = q; g.to_parquet(on / f"cnn_s{j}.parquet", index=False)
        d = np.abs(new_p.astype(np.float64) - pr); info[f"warm_cnn_s{j}"] = dict(cells_moved=int((d > 0).sum()), max_move=float(d.max()))
    f = pd.read_parquet(ON / "sccnn_official.parquet"); keys = pd.MultiIndex.from_frame(f[K]); warm = ~f.window_id.isin(cold_w).to_numpy()
    if selftest:
        mv = np.zeros(len(f))
    else:
        mvs = []
        for j in range(5):
            a1 = arm(root, "official_warm", new, f"sc_s{j}").set_index(K).p_mh5; a0 = arm(root, "official_warm", base, f"sc_s{j}").set_index(K).p_mh5
            mvs.append(a1.reindex(keys).to_numpy(np.float64) - a0.reindex(keys).to_numpy(np.float64))
        mv = np.mean(mvs, 0); assert np.isfinite(mv[warm]).all(), "sccnn: warm keys differ from the on/ file"; mv = np.where(warm, mv, 0.)
    pr = f.prob.to_numpy(np.float64); g = f.copy(); g["prob"] = np.where(warm, np.clip(pr + mv, 0., 1.), pr).astype(f.prob.dtype)
    g.to_parquet(on / "sccnn_official.parquet", index=False)
    d = np.abs(g.prob.to_numpy(np.float64) - pr); info["warm_sccnn"] = dict(cells_moved=int((d > 0).sum()), max_move=float(d.max()), **wi)
    if selftest:
        for p in on.glob("*.parquet"):
            o = pd.read_parquet(ON / p.name); n = pd.read_parquet(p); assert o.equals(n), f"selftest: {p.name} changed"
    (mem / "members.json").write_text(json.dumps(info, indent=1))
    print("MG_MEMBERS", name, json.dumps({t: (v["cells_moved"], round(v["max_move"], 4)) for t, v in info.items()}), flush=True)


def load_b(name):
    sp = importlib.util.spec_from_file_location("a1_build", ROOT / "exp/L_dose/a1_build.py"); B = importlib.util.module_from_spec(sp); sp.loader.exec_module(B)
    B.OUT = OUT; B.REL = str(OUT.relative_to(ROOT)).replace("\\", "/"); B.A3.COLD_MEMBERS = str((OUT / f"members_{name}").relative_to(ROOT)).replace("\\", "/")
    return B


def chain(name):
    os.environ.setdefault("COLD_PARITY_TOL", "1e-5")   # p_un of the VM-made A3 files vs this CPU (~1e-6), as a3_finish.sh / dw_build
    B = load_b(name); OUT.mkdir(parents=True, exist_ok=True)
    B.chain(str((OUT / f"on_{name}").relative_to(ROOT)).replace("\\", "/"), f"{name}_chain"); print("MG_CHAIN_DONE", OUT / f"{name}_chain", flush=True)


def rows(name, parity=True):
    sp = importlib.util.spec_from_file_location("rebase_c57", ROOT / "tools/rebase/rebase_c57.py"); R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
    R.wait_ram(); cfg = json.loads(R.CFG.read_text()); w8 = float(cfg["a"]); assert w8 == .2
    c59 = R.A3.rp(C59_ROWS).set_index("submission_id").queue_pred.astype(float); R.BB = TINY
    if parity:
        P = R.probs(R.A3_CHAIN); D0, _ = R.decode_all({"P0": P, "P1": P, "A3": P}, w8); d0 = D0["A3"].reindex(c59.index)
        par = int((d0.to_numpy(float) != c59.to_numpy()).sum()); print("MG_ROWS_PARITY A3 chain + C59 Tiny vs C59 rows: cells", par, flush=True)
        assert par == 0, "decode parity"
    P1 = R.probs(OUT / f"{name}_chain/a0.75"); D1, _ = R.decode_all({"P0": P1, "P1": P1, "A3": P1}, w8); d1 = D1["A3"].reindex(c59.index)
    pd.DataFrame(dict(submission_id=c59.index.to_numpy(np.int64), queue_pred=d1.to_numpy())).to_parquet(OUT / f"rows_{name}.parquet", index=False)
    key = R.A3.ongoing_key().set_index("submission_id").loc[c59.index]; ch = d1.to_numpy() != c59.to_numpy()
    cold_w = set(pd.read_parquet(SHIP / "ctl_s0.parquet", columns=["window_id"]).window_id); isc = key.window_id.isin(cold_w).to_numpy(); cnt = {}
    for s in SPLITS:
        m = ch & (key.split.to_numpy() == s); wid = key.window_id.to_numpy()
        cnt[s] = dict(cells=int(m.sum()), windows=int(len(set(wid[m]))), cold_windows=int(len(set(wid[m & isc]))), warm_windows=int(len(set(wid[m & ~isc]))),
                      on=int((m & (d1.to_numpy() == 1)).sum()), off=int((m & (d1.to_numpy() == 0)).sum()))
    (OUT / f"rows_{name}.json").write_text(json.dumps(dict(name=name, counts=cnt, rows_sha=sha(OUT / f"rows_{name}.parquet")), indent=1))
    print("MG_ROWS", name, json.dumps(cnt), flush=True)


def delta(parent, name, out_name):
    assert not str(parent).startswith("/"), "pass an ABSOLUTE Windows path (C:/...), not an MSYS /c/ path"
    sp = importlib.util.spec_from_file_location("rebase_c57", ROOT / "tools/rebase/rebase_c57.py"); R = importlib.util.module_from_spec(sp); sp.loader.exec_module(R)
    c59 = R.A3.rp(C59_ROWS).set_index("submission_id").queue_pred.astype(float)
    par = pd.read_csv(parent, usecols=["submission_id", "queue_pred"], float_precision="round_trip").set_index("submission_id").queue_pred
    pr = par.reindex(c59.index); assert pr.notna().all() and (pr.to_numpy(float) == c59.to_numpy()).all(), "parent ongoing rows != C59 rows: rebase needed"
    ref = OUT / "ref" / "a0.75"; ref.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(submission_id=c59.index.to_numpy(np.int64), queue_pred=c59.to_numpy())).to_parquet(ref / "ongoing_bb.parquet", index=False)
    R.A3.C36 = str((OUT / "ref").relative_to(ROOT)).replace("\\", "/"); R.A3.PARENT = Path(parent)
    R.A3.delta(new_rows=OUT / f"rows_{name}.parquet", name=out_name, tmpdir=OUT / "tmp", outdir=OUT / "delta")
    print("MG_DELTA_DONE", out_name, flush=True)


if __name__ == "__main__":
    c = sys.argv[1]; a = [x for x in sys.argv[2:] if not x.startswith("--")]
    if c == "members":
        if "--selftest" in sys.argv:
            members(None, None, None, a[0] if a else "self", selftest=True)
        else:
            members(*a)
    elif c == "chain":
        chain(a[0])
    elif c == "rows":
        rows(a[0], parity="--no-parity" not in sys.argv)
    elif c == "delta":
        delta(a[0], a[1], a[2])
    sys.stdout.flush(); os._exit(0)
