"""Re-base an ongoing-rows candidate on C57 (C 2026-10-16). C57 = C55 + L_dose A3 (speed term in the cold-date XSD fits).
A line built for C55 (or for C50: its chain probabilities are decoded here with the VM Tiny member, the rebase_c55 path)
is put onto C57 by DIFFERENCE: parent + P1 minus P0, taken on the PROBABILITIES that enter the final decode (cnn_ens2 prob
on type B/C windows, sccnn prob on type A windows), then the shipped final decode (asm2_bbmember.decode: .8 x row prob +
.2 x VM Tiny member, eiou count, selcond .92 with t-12..t-1 persistence, band raise on type A) per window.
  parent = A3 chain (artifacts/claude_ref/l_dose/a3/a3_build/a0.75), P0 = the shipped chain the line was built against
  (default artifacts/claude_ref/c_cold/official/ship_c35/a0.75 = C55 rows with the VM member), P1 = the line's chain.
Cells where BOTH pieces move the probability: the moves add (p = P0 + dA3 + dLine, clipped to [0, 1]), except where the two
moves are identical (the same piece, e.g. A3 itself): counted once. So rebasing A3 onto C57 changes nothing, and in every
window where A3 moves no probability the result is exactly the line's own decode (the line's C55 delta cells).
Checks (exact): decode(P0) = C55 rows, decode(A3) = C57 rows (C55 rows + the C57 delta), decode(P1) = the line's rows when
known (CHAIN_DIR/ongoing_bb.parquet decoded with the VM member, or --line-rows). Overlap counts are printed and saved.
Output under OUTDIR: rows_c57.parquet (candidate ongoing rows), rebase_c57.json, delta/ (a3.delta on the C57 file:
window rule >= 3 March windows, tools/probe.py make/expand with sha check, stream check, window_guard --no-record).
  python tools/rebase/rebase_c57.py CHAIN_DIR NAME OUTDIR [--p0 DIR] [--line-rows PARQUET] [--no-file]
Waits while free RAM < 3 GiB (at most 4 h). Last line REBASE_C57_DONE NAME or a traceback."""
import sys, json, time, argparse, importlib.util, subprocess, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "docs/reference_code"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
spec = importlib.util.spec_from_file_location("a3", ROOT / "exp/L_lab/a3/a3.py"); A3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A3)
SW = ROOT / "artifacts/claude_ref/c_bb/vm_tiny_swap"
CFG, BB = SW / "cfg_vm.json", SW / "bb_member_vm.parquet"
C55_ROWS = SW / "ongoing_bb_vm.parquet"                                   # C55's ongoing rows
A3_CHAIN = ROOT / "artifacts/claude_ref/l_dose/a3/a3_build/a0.75"
A3_C55_ROWS = ROOT / "artifacts/claude_ref/l_dose/a3/rebase/ongoing_bb_c55.parquet"
C57_DELTA = ROOT / "artifacts/claude_ref/l_dose/a3/rebase/delta/c_a3xsd_on_c55.delta.csv.gz"
C57_CSV = ROOT / "artifacts/claude_ref/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv"
C57_SHA = "caaf4d8a3740a27db5cb5237164b3850c69761830002649ef4497f06ed7999a5"
C55_CSV = ROOT / "artifacts/claude_ref/probes/c_tinyvm_on_c54.csv"
SCRATCH = ROOT / "artifacts/claude_ref/rebase_c57"
P0_DEFAULT = ROOT / "artifacts/claude_ref/c_cold/official/ship_c35/a0.75"
K = ["window_id", "timestamp", "link_id"]
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
say = lambda *a: print(f"[{utc()}]", *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wait_ram(gib=3., hours=4.):
    import psutil
    t0 = time.time()
    while psutil.virtual_memory().available < gib * 2 ** 30:
        assert time.time() - t0 < hours * 3600, f"no {gib} GiB free RAM in {hours} h"
        time.sleep(30)
    say(f"free RAM {psutil.virtual_memory().available / 2 ** 30:.2f} GiB")


def c57_csv():
    """the C57 file (sha checked); rebuilt from C55 + the C57 delta under SCRATCH when the copy beside the delta is gone."""
    for p in (C57_CSV, SCRATCH / C57_CSV.name):
        if p.exists() and sha(p) == C57_SHA:
            return p
    SCRATCH.mkdir(parents=True, exist_ok=True); out = SCRATCH / C57_CSV.name
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/probe.py"), "expand", str(C55_CSV), str(C57_DELTA), str(out), "--sha", C57_SHA], cwd=str(ROOT), check=True)
    assert sha(out) == C57_SHA
    return out


def c57_rows():
    r = A3.rp(C55_ROWS).set_index("submission_id").queue_pred.astype(float)
    d = pd.read_csv(C57_DELTA, float_precision="round_trip"); assert list(d.columns) == ["submission_id", "queue_pred"] and d.submission_id.isin(r.index).all()
    r57 = r.copy(); r57.loc[d.submission_id.to_numpy()] = d.queue_pred.to_numpy(float)
    a3 = A3.rp(A3_C55_ROWS).set_index("submission_id").queue_pred.reindex(r.index)
    assert (a3.to_numpy(float) == r57.to_numpy()).all(), "C55 rows + C57 delta != the A3 rows rebased on C55"
    return r, r57


def probs(d):
    c = A3.rp(Path(d) / "cnn_ens2.parquet").set_index(K); s = A3.rp(Path(d) / "sccnn_official.parquet").set_index(K)
    return c.prob, c.queue_pred, s.prob


def decode_all(variants, w8):
    """asm2_bbmember.decode on several probability sets at once. variants: name -> (cnn prob, cnn queue_pred or None, sc prob)
    aligned on the same K index, or 'comb' (built per window from P0, A3, P1). Returns name -> Series(submission_id -> 0/1)
    and the per-window overlap table."""
    from tfb.data import load_cache
    from qcnn_data import cube
    from gapres_selcond_rule import constrained, iou, eiou_order
    from ong2_band_rule import raise_to

    def dec(f, pm, typ):
        k, o = eiou_order(f)
        m, _ = constrained(f, o, k, pm, .92)
        if typ == "A" and iou(m, pm) < .8:
            m = raise_to(m, f, pm, .8)
        return m
    key = A3.ongoing_key().set_index(K)
    mem = A3.rp(BB).set_index(K).prob
    types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv"); types = types[types.split != "train"].set_index("window_id")
    base = variants["P0"][0].reset_index()[K]
    out = {n: {} for n in list(variants) + ["comb"]}; wins = []
    for split in ("validation", "private"):
        for p in sorted(types[types.split == split].panel.unique()):
            c = load_cache(p, split); st = c.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
            c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
            obs, q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
            for wid, w in types[(types.split == split) & (types.panel == p)].iterrows():
                g = base[base.window_id == wid].sort_values(["timestamp", "link_id"])
                ts = sorted(g.timestamp.unique()); H = len(ts); L = len(g) // H
                idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
                d = c.day_index(str(w.date)); t = int(w.t)
                last = np.zeros(c3.shape[-1], bool)
                for k_ in range(t - 12, t):
                    last = np.where(obs[d, k_], q[d, k_], last)
                pm = np.broadcast_to(last[idx], (H, L)).ravel()
                ix = pd.MultiIndex.from_frame(g[K])
                fe = mem.loc[ix].to_numpy(np.float64); assert np.isfinite(fe).all()
                sid = key.loc[ix].submission_id.to_numpy(); assert (key.loc[ix].split == split).all()
                f = {}
                for n, (cp, cq, sp) in variants.items():
                    f[n] = (sp.loc[ix] if w.typ == "A" else cp.loc[ix]).to_numpy(np.float64)
                    if w.typ != "A" and cq is not None:
                        assert eiou_order(f[n])[0] == int(cq.loc[ix].sum()), (n, wid)
                dA, dL = f["A3"] - f["P0"], f["P1"] - f["P0"]
                same = dA == dL; raw = np.where(same, f["P1"], f["P0"] + dA + dL); f["comb"] = np.clip(raw, 0., 1.)
                for n, fv in f.items():
                    out[n][wid] = (sid, dec((1 - w8) * fv + w8 * fe, pm, w.typ))
                wins.append(dict(window_id=wid, split=split, typ=w.typ, a3_prob_cells=int((dA != 0).sum()), line_prob_cells=int((dL != 0).sum()),
                                 both_prob_cells=int(((dA != 0) & (dL != 0) & ~same).sum()), same_move_cells=int(((dA != 0) & same).sum()),
                                 clipped=int((raw != f["comb"]).sum())))
    res = {}
    for n, dd in out.items():
        s = pd.concat([pd.Series(m.astype(float), index=sid) for sid, m in dd.values()]).sort_index(); assert len(s) == 87000 and s.index.is_unique
        res[n] = s
    return res, pd.DataFrame(wins)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain"); ap.add_argument("name"); ap.add_argument("outdir")
    ap.add_argument("--p0", default=str(P0_DEFAULT)); ap.add_argument("--line-rows"); ap.add_argument("--no-file", action="store_true")
    a = ap.parse_args()
    ab = lambda x: str(Path(x) if Path(x).is_absolute() else ROOT / x)
    a.chain, a.p0 = ab(a.chain), ab(a.p0); a.line_rows = ab(a.line_rows) if a.line_rows else None
    out = Path(a.outdir); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=True)
    wait_ram(); A3.guard(30)
    cfg = json.loads(CFG.read_text()); w8 = float(cfg["a"]); cfg_sha = sha(CFG)
    r55, r57 = c57_rows()
    V = {"P0": probs(a.p0), "A3": probs(A3_CHAIN), "P1": probs(a.chain)}
    ix = V["P0"][0].index
    for n, (cp, cq, sp) in V.items():
        assert len(cp) == len(sp) == 87000 and set(cp.index) == set(ix) and set(sp.index) == set(ix), n
        V[n] = (cp.reindex(ix), cq.reindex(ix), sp.reindex(ix))
    say("decode P0, A3, P1, combined")
    D, wt = decode_all(V, w8)
    D = {n: s.reindex(r55.index) for n, s in D.items()}
    chk = dict(p0_vs_c55_cells=int((D["P0"] != r55).sum()), a3_vs_c57_cells=int((D["A3"] != r57).sum()))
    lr = a.line_rows
    if lr is None:
        j = Path(a.chain) / "ongoing_bb.parquet.json"
        if j.exists() and json.loads(j.read_text()).get("cfg_sha256") == cfg_sha:
            lr = str(Path(a.chain) / "ongoing_bb.parquet")
    chk["line_rows"] = lr
    chk["p1_vs_line_rows_cells"] = None if lr is None else int((D["P1"] != A3.rp(lr).set_index("submission_id").queue_pred.reindex(r55.index)).sum())
    say("parity", json.dumps(chk))
    assert chk["p0_vs_c55_cells"] == 0 and chk["a3_vs_c57_cells"] == 0 and chk["p1_vs_line_rows_cells"] in (None, 0), chk
    key = A3.ongoing_key().set_index("submission_id").loc[r55.index]
    wcol = key.window_id.to_numpy()
    a3c = r57.to_numpy() != r55.to_numpy(); lc = D["P1"].to_numpy() != r55.to_numpy(); cand = D["comb"].to_numpy()
    cc = cand != r57.to_numpy()
    cellrule = r57.to_numpy().copy(); cellrule[lc] = D["P1"].to_numpy()[lc]
    pov = set(wt.window_id[wt.both_prob_cells > 0]); a3pw = set(wt.window_id[wt.a3_prob_cells > 0]); lpw = set(wt.window_id[wt.line_prob_cells > 0])
    inpov, ina3p, inlp = np.isin(wcol, list(pov)), np.isin(wcol, list(a3pw)), np.isin(wcol, list(lpw))
    nw = lambda m: int(len(set(wcol[m])))
    ov = dict(
        prob_level=dict(a3_windows=len(a3pw), line_windows=len(lpw), both_windows=len(pov), both_cells_added=int(wt.both_prob_cells.sum()),
                        same_move_cells_counted_once=int(wt.same_move_cells.sum()), clipped_cells=int(wt.clipped.sum())),
        decoded=dict(a3_cells=int(a3c.sum()), a3_windows=nw(a3c), line_cells_on_c55=int(lc.sum()), line_windows_on_c55=nw(lc),
                     both_cells=int((a3c & lc).sum()), both_windows=len(set(wcol[a3c]) & set(wcol[lc])),
                     line_cells_in_a3_decoded_windows=int((lc & np.isin(wcol, list(set(wcol[a3c])))).sum())),
        candidate_vs_c57=dict(cells=int(cc.sum()), windows=nw(cc), cells_in_prob_overlap_windows=int((cc & inpov).sum()), windows_in_prob_overlap=nw(cc & inpov)),
        no_a3_move_windows_equal_line_c55_delta=bool(((cand != r57.to_numpy()) == lc)[~ina3p].all() and (cand[~ina3p] == D["P1"].to_numpy()[~ina3p]).all()),
        no_line_move_windows_equal_c57=bool((cand == r57.to_numpy())[~inlp].all()),
        cell_rule_alternative=dict(cells_vs_c57=int((cellrule != r57.to_numpy()).sum()), cells_differ_from_candidate=int((cellrule != cand).sum()),
                                   windows_differ=nw(cellrule != cand)))
    say("OVERLAP", json.dumps(ov))
    rows = pd.DataFrame(dict(submission_id=r55.index.to_numpy(np.int64), queue_pred=cand)); rows.to_parquet(out / "rows_c57.parquet", index=False)
    wt.to_csv(out / "windows_overlap.csv", index=False)
    res = dict(name=a.name, chain=str(a.chain), p0=str(a.p0), parent="C57 " + C57_SHA[:8], parity=chk, overlap=ov, made=utc())
    (out / "rebase_c57.json").write_text(json.dumps(res, indent=1))
    if a.no_file:
        A3.table(pd.DataFrame(dict(submission_id=r57.index.to_numpy(), queue_pred=r57.to_numpy())), rows, "candidate vs C57 (no file)")
        say("REBASE_C57_DONE", a.name, "(no file)"); return
    ref = out / "ref" / "a0.75"; ref.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(submission_id=r57.index.to_numpy(np.int64), queue_pred=r57.to_numpy())).to_parquet(ref / "ongoing_bb.parquet", index=False)
    parent = c57_csv(); wait_ram()
    A3.C36 = (out / "ref").resolve().relative_to(ROOT).as_posix(); A3.PARENT = parent
    A3.delta(new_rows=out / "rows_c57.parquet", name=a.name, tmpdir=out / "tmp", outdir=out / "delta")
    dl = json.loads((out / "delta" / "delta.json").read_text())
    say("COUNTS", json.dumps(dict(by_split={k: dict(windows=v["windows_changed"], cells=v["cells_changed"]) for k, v in dl["candidate"]["by_split"].items()},
                                  march_windows=dl["march_windows_changed"], april_windows=dl["april_windows_changed"], window_rule=dl["window_rule"],
                                  file=dl.get("file"), guard=(dl.get("window_guard") or {}).get("verdict"), full_sha256=dl.get("full_sha256"))))
    say("REBASE_C57_DONE", a.name)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); __import__("os")._exit(0)
