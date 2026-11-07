"""C hedge B 2026-10-11 (helper HB): final-pair file B (onset hedge) on ANY final file A, one command.

B = A except, in the hedged PRIVATE onset windows, the T+30 queue_pred row of every link is 1 on the second site's
cell set and 0 elsewhere. Validation rows and every other task are identical, so B's public score equals A's.
The selection logic is onset9_hedge.py (served, calib, window_probs, simulate, final_set, hedge_frame): imported, not
copied. This file adds: the rule variants, an exact (no Monte Carlo) gain, the build on any A, the checks, a receipt.

Rules (--rule):
  asbuilt : r = p2/p1 > .4 on the served Q0436 probabilities, union -> second site only, minus windows whose second
            site has zero online activity (kill rule). The .4 was set on 2026-10-05/07 with the POOLED March + April
            pseudo calibration (audit item U2: April labels after T took part in the choice). Kept for parity only.
  march   : same family, threshold chosen WITHOUT any April quantity: grid (.25, .4, .5, .75), truth probabilities by
            r bin from MARCH pseudo windows only, gain E[max(A,B)] - E[A] computed exactly on the MARCH official onset
            windows (their served probabilities at T, kill rule applied), highest gain wins, ties -> larger threshold.
            The threshold is then applied to each April window with that window's own r at T (online).
  split   : C_pair 2026-10-12 (exp/C_pair/PAIR.md): every PRIVATE onset window whose served decode is a union of two sites
            is split across the pair. Writes TWO deltas on A: N_A (file A' = the panel's train-dominant site only, META
            n_windows) and N_B (file B = the other site only). No threshold; same kill rule; same checks and receipts.
  --th T  : overrides the threshold of march / asbuilt (e.g. --rule march --th .9 = only D12_I5_S 003 -> second site).
Online: per window the rule reads the served site probabilities (Q0436, inputs <= T) and the S1 activity statistic
(stat2_off: days 0..d of the split, day d steps before T). Nothing after T is read for a window.

Checks (fail closed): A's onset rows equal the served decode in every official onset window (T+30 cells = a_cells,
no queue cell before T+30), else the onset method changed and the probabilities are stale; B differs from A only in
queue_pred, only on T+30 rows of the hedged private windows (checked on the key frame and again on the full files
through tools/probe.py make, which compares all value columns of all rows).

Usage (repo root, laptop: PYTHONPATH=src .venv_tfb/Scripts/python.exe):
  python docs/reference_code/c_hedge_b.py build A.csv --out DIR [--rule march|asbuilt|split] [--th T] [--name N] [--keep-full] [--no-diff]
         [--onset-delta D.csv]   # parity only: A = file + this delta (C19 + Q0436); the delta is still made on the file
         [--probs P.parquet]     # served probabilities of a NEW onset method (columns wkey, site, panel, split, p_candidate)
         [--allow-mismatch]      # only require the decode match in the hedged windows (mismatches go to the receipt)
  python docs/reference_code/c_hedge_b.py rule [--out DIR]          # threshold table for U2 (no A needed)
  python docs/reference_code/c_hedge_b.py compare NEW.delta OLD.delta
Outputs in DIR: N.delta.csv.gz + N.json (tools/probe.py make; parent = A file), N.receipt.json, N.hedge_set.csv,
N.rows_vs_A.csv, N.hedge_B_rows.csv (asm2_final.py --hedge-rows format), N.raw.csv, N.log. Runtime ~5 min, peak
disk one full csv (0.4 GB, removed unless --keep-full; refuses to run under 12.5 GiB free).
"""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/reference_code")); sys.path.insert(0, str(ROOT / "src"))
import onset9_hedge as H
KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"
PSEUDO = ROOT / "artifacts/claude_ref/big/hedge_pseudo.parquet"
META = ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json"
OLD_ROWS = ROOT / "artifacts/claude_ref/onset9/hedge_B_rows.csv"
PROBE = ROOT / "tools/probe.py"
GRID = (.25, .4, .5, .75)          # the thresholds tested on 10-05/10-07; no new values
ASBUILT_TH = .4
MIN_FREE_GIB = 12.5
PY = sys.executable


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def exact(o, probs, sel, union_opt="s2"):
    """Exact twin of H.simulate (same IoU model, same category cut): E[B-A], sd, E[max(A,B)]-E[A], P(gain >= .003).
    B - A per window is a multiple of A_IOU/2 * W, so the sum is a convolution on an integer grid."""
    pmf = {0: 1.}
    for i in sel:
        x = o.iloc[i]; pt, ps, pb, _ = probs[i]
        e0 = min(pt, 1.); e1 = min(pt + ps, 1.); e2 = min(pt + ps + pb, 1.)
        c0, c1, c2, c3 = e0, e1 - e0, e2 - e1, 1. - e2          # top only, second only, both, other
        if x.union:
            step = {-1: c0 + c2, 1: c1, 0: c3} if union_opt == "s2" else {1: c0, -1: c1 + c2, 0: c3}
        else:
            step = {-2: c0, 2: c1, 0: c2 + c3}
        new = {}
        for k, p in pmf.items():
            for d, q in step.items():
                new[k + d] = new.get(k + d, 0.) + p * q
        pmf = new
    u = H.A_IOU / 2 * H.W
    k = np.array(list(pmf)); p = np.array(list(pmf.values())); v = k * u
    m = float((v * p).sum())
    return m, float(np.sqrt(((v - m) ** 2 * p).sum())), float((np.maximum(v, 0) * p).sum()), float(p[v >= .003].sum())


def cands(o, split):
    return o[(o.split == split) & (o.r > .25)].sort_values("r", ascending=False).reset_index(drop=True)


def rule_table(o, w, boot=1000, seed=7):
    """Threshold table. Selection column = march calibration on the March official windows (no April quantity).
    Other columns are CONFIRM ONLY (they read April r values or April pseudo labels)."""
    val, pri = cands(o, "validation"), cands(o, "private")
    rows = []
    for th in GRID:
        kv, killv, _ = H.final_set(val, th); kp, killp, _ = H.final_set(pri, th)
        r = dict(th=th, n_val=len(kv), n_pri=len(kp), killed_val=";".join(killv), killed_pri=";".join(killp))
        for pop, fr, keep in (("val", val, kv), ("pri", pri, kp)):
            for mode in ("march", "pooled"):
                e, sd, g, p3 = exact(fr, H.window_probs(fr, w, mode), keep)
                r[f"gain_{mode}_on_{pop}"] = g; r[f"EBA_{mode}_on_{pop}"] = e
        rows.append(r)
    t = pd.DataFrame(rows)
    g = t.gain_march_on_val.to_numpy()
    best = max(i for i in range(len(GRID)) if g[i] >= g.max() - 1e-6)      # ties -> larger threshold
    # stability of the choice: bootstrap of the March pseudo windows (calibration sampling noise)
    rng = np.random.default_rng(seed); wm = w[w["pop"] == "val"].reset_index(drop=True)
    sets = [H.final_set(val, th)[0] for th in GRID]
    wins = np.zeros(len(GRID), int); gb = []
    for _ in range(boot):
        wb = wm.iloc[rng.integers(0, len(wm), len(wm))]
        try:
            P = H.window_probs(val, wb, "march")
        except KeyError:                                                    # an r bin left empty by the resample
            continue
        gg = np.array([exact(val, P, s)[2] for s in sets]); gb.append(gg)
        wins[max(i for i in range(len(GRID)) if gg[i] >= gg.max() - 1e-6)] += 1
    gb = np.array(gb)
    t["boot_win_share"] = wins / max(len(gb), 1)
    t["boot_gain_p05"] = np.quantile(gb, .05, axis=0); t["boot_gain_p95"] = np.quantile(gb, .95, axis=0)
    return t, GRID[best]


def select(rule, probs=None, th=None):
    o = H.served(probs); w = H.calib()
    table = None
    if rule == "asbuilt":
        th_used = ASBUILT_TH
        prov = "fixed .4 as built 2026-10-07 (chosen with pooled March+April pseudo calibration: U2)"
    else:
        table, th_used = rule_table(o, w)
        prov = "argmax of exact gain on March official windows, March pseudo calibration, grid .25/.4/.5/.75, ties -> larger"
    if th is not None:
        th_used, prov = th, f"manual --th {th} (overrides rule {rule})"
    pri = cands(o, "private")
    keep, killed, act = H.final_set(pri, th_used)
    fin = H.hedge_frame(pri, keep, act)
    gains = {}
    for mode, role in (("march", "selection evidence (train + March)"), ("pairM", "March only, pair-specific: information"),
                       ("pooled", "confirm only (April pseudo labels)"), ("pairA", "confirm only (April pseudo labels)")):
        P = H.window_probs(pri, w, mode)
        e, sd, g, p3 = exact(pri, P, keep); mc = H.simulate(pri, P, keep, "s2")
        gains[mode] = dict(role=role, E_B_minus_A=e, sd=sd, gain_Emax_minus_EA=g, P_gain_ge_003=p3, mc_gain_seed1=float(mc[2]))
    return dict(o=o, pri=pri, fin=fin, th=th_used, prov=prov, killed=killed, gains=gains, table=table)


def onset_frame(a_file, wins):
    key = pd.concat(k[k.window_id.isin(wins)] for k in pd.read_csv(
        KEY, chunksize=500000, usecols=["submission_id", "task", "split", "window_id", "timestamp", "link_id"], dtype=str))
    key["submission_id"] = key.submission_id.astype(int)
    assert key.window_id.nunique() == len(wins) and (key.task == "queue").all()
    ids = set(key.submission_id)
    c = pd.concat(x[x.submission_id.isin(ids)] for x in pd.read_csv(a_file, chunksize=500000, usecols=["submission_id", "queue_pred"]))
    return key.merge(c, on="submission_id", validate="one_to_one").rename(columns={"queue_pred": "file"})


def run(cmd, log):
    r = subprocess.run([PY] + [str(c) for c in cmd], capture_output=True, text=True)
    with open(log, "a", newline="\n") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n" + r.stdout + r.stderr + "\n")
    return r


def select_split(probs=None):
    """Rule split (C_pair, exp/C_pair/PAIR.md section 4): every PRIVATE onset window whose served decode is a union of
    two sites is split across the pair. File A' keeps the panel's train-dominant site only (META n_windows of the Q03
    fit, Jun-Dec; tie -> top site), file B keeps the other site only. Same kill rule as the threshold family.
    Reads per window: the served probabilities at T, the S1 statistic before T, train site counts. No April label."""
    import itertools
    o = H.served(probs)
    pri = o[(o.split == "private") & o.union].sort_values("r", ascending=False).reset_index(drop=True)
    assert len(pri), "no private onset window with a two-site decode: nothing to split"
    keep, killed, act = H.final_set(pri, 0.)
    fin = H.hedge_frame(pri, keep, act)
    nw = {(pn, s_["key"]): s_.get("n_windows", 0) for pn, ss in H.META["sites"].items() for s_ in ss if not s_.get("other")}
    fa, fb = fin.copy(), fin.copy(); dom = []
    for i, x in zip(fin.index, fin.itertuples()):
        d1 = nw.get((x.panel, x.s1), 0) >= nw.get((x.panel, x.s2), 0)
        cd, co = (x.b_s1only, x.b_s2only) if d1 else (x.b_s2only, x.b_s1only)
        sd, so = (x.s1, x.s2) if d1 else (x.s2, x.s1)
        fa.loc[i, "b_cells"] = cd; fa.loc[i, "b_site"] = sd; fb.loc[i, "b_cells"] = co; fb.loc[i, "b_site"] = so
        dom.append(dict(window_id=x.window_id, dominant=sd, n_windows_dominant=nw.get((x.panel, sd), 0), other=so, n_windows_other=nw.get((x.panel, so), 0)))
    # expected value of the pair, exact over the truth categories (top only, second only, both, neither), cells model
    gains = {}
    if len(fin) <= 5:
        import c_pair_sim as PS
        w = PS.pseudo_windows(); q = H.A_IOU * H.W; vec = []
        for x in fin.itertuples():
            n1, n2 = len(x.b_s1only.split(";")), len(x.b_s2only.split(";")); f1, f2 = n1 / (n1 + n2), n2 / (n1 + n2)
            d1 = nw.get((x.panel, x.s1), 0) >= nw.get((x.panel, x.s2), 0)
            un = q * np.array([f1, f2, 1, 0]); s1o = q * np.array([1, 0, f1, 0]); s2o = q * np.array([0, 1, f2, 0])
            vec.append((un, s1o if d1 else s2o, s2o if d1 else s1o))
        for mode, role in (("bin", "March r bin (HEDGE.md march): train + March"), ("dom", "March r bin x orientation: train + March (central)"),
                           ("pairM", "March pair-specific: train + March"), ("domA", "April pseudo: CONFIRM ONLY")):
            P = PS.probs(fin.reset_index(drop=True), w, mode)
            eu = ea = eb = em = emu = 0.
            for st in itertools.product(range(4), repeat=len(fin)):
                p = float(np.prod([P[j, c] for j, c in enumerate(st)]))
                u = sum(vec[j][0][c] for j, c in enumerate(st)); va = sum(vec[j][1][c] for j, c in enumerate(st)); vb = sum(vec[j][2][c] for j, c in enumerate(st))
                eu += p * u; ea += p * va; eb += p * vb; em += p * max(va, vb); emu += p * max(u, va)
            gains[mode] = dict(role=role, gain_Emax_pair_minus_E_parent=em - eu, E_Aprime_minus_parent=ea - eu, E_B_minus_parent=eb - eu,
                               gain_parent_plus_Aprime_file_only=emu - eu,
                               P_categories={x.window_id: [round(float(v), 3) for v in P[j]] for j, x in enumerate(fin.itertuples())})
    prov = "every private two-site decode (r >= close_ratio of the served decoder) is split; no threshold"
    mk = lambda f, role: dict(o=o, pri=pri, fin=f, th=None, prov=prov, killed=killed, gains=gains, table=None, role=role, dominant=dom)
    return mk(fa, "A'"), mk(fb, "B")


def emit(a, A, out, name, key, s, od, mism, o, t0, rule_name, partner=None):
    """Write one file of the pair as a delta on A, with checks and receipt."""
    fin = s["fin"]; key = key.copy(); log = out / f"{name}.log"; log.write_text("")
    key["B"] = key.A; wl = []
    for x in fin.itertuples():
        idx = key.index[(key.window_id == x.window_id) & key.t30]
        links = key.loc[idx, "link_id"]; b = set(x.b_cells.split(";"))
        assert b and b <= set(links), (x.window_id, "B cell not in the window's links")
        key.loc[idx, "B"] = links.isin(b).astype(float).to_numpy()
        ch = key.loc[idx, "B"] != key.loc[idx, "A"]
        wl.append(dict(window_id=x.window_id, t30=key.loc[idx, "timestamp"].iloc[0], panel=x.panel, r=float(x.r), p1=float(x.p1), p2=float(x.p2),
                       a_site=x.s1 + ("+" + x.s2 if x.union else ""), b_site=x.b_site, a_cells=x.a_cells, b_cells=x.b_cells,
                       s2_online_activity=float(x.A_s2_online), rows_changed_vs_A=int(ch.sum()),
                       cells_1_to_0=int((ch & (key.loc[idx, "A"] == 1)).sum()), cells_0_to_1=int((ch & (key.loc[idx, "A"] == 0)).sum())))
    vsA = key[key.B != key.A]
    assert vsA.window_id.isin(fin.window_id).all() and (vsA.split == "private").all() and vsA.t30.all() and (vsA.task == "queue").all()
    allowed = set(key.submission_id[key.window_id.isin(fin.window_id) & key.t30])
    vsA[["submission_id", "window_id", "timestamp", "link_id", "A", "B"]].to_csv(out / f"{name}.rows_vs_A.csv", index=False, lineterminator="\n")
    hb = vsA[["submission_id", "B", "A", "window_id"]].rename(columns={"B": "queue_pred", "A": "queue_pred_A"})
    hb.to_csv(out / f"{name}.hedge_B_rows.csv", index=False, lineterminator="\n")
    fin.to_csv(out / f"{name}.hedge_set.csv", index=False, lineterminator="\n")
    same_old = None
    if OLD_ROWS.exists():
        old = pd.read_csv(OLD_ROWS).sort_values("submission_id").reset_index(drop=True); nw = hb.sort_values("submission_id").reset_index(drop=True)
        same_old = bool(len(old) == len(nw) and (old.submission_id.to_numpy() == nw.submission_id.to_numpy()).all()
                        and (old.queue_pred.to_numpy() == nw.queue_pred.to_numpy()).all() and (old.queue_pred_A.to_numpy() == nw.queue_pred_A.to_numpy()).all())
    raw = key[key.B != key.file][["submission_id", "B"]].rename(columns={"B": "queue_pred"}).sort_values("submission_id")
    raw["queue_pred"] = raw.queue_pred.astype(int)
    assert len(raw), "nothing to hedge: B equals A (no window passes the rule)"
    raw.to_csv(out / f"{name}.raw.csv", index=False, lineterminator="\n")
    # ---- full file (temporary), delta in tools/probe.py make format, whole-file check
    full = Path(a.tmp or out) / f"{name}_full.csv"; delta = out / f"{name}.delta.csv.gz"
    r = run([PROBE, "expand", A, out / f"{name}.raw.csv", full], log); assert full.exists(), r.stderr[-400:]
    r = run([PROBE, "make", A, full, delta], log); assert r.returncode == 0, r.stderr[-400:]
    meta = json.loads((out / f"{name}.json").read_text())
    d = pd.read_csv(delta)
    assert list(d.columns) == ["submission_id", "queue_pred"] and meta["rows_by_task"] == {"queue": len(d)}, (list(d.columns), meta["rows_by_task"])
    assert (d.submission_id.to_numpy() == raw.submission_id.to_numpy()).all() and (d.queue_pred.to_numpy() == raw.queue_pred.to_numpy()).all()
    extra = set(d.submission_id) - allowed
    assert extra <= (set(od.submission_id) if od is not None else set()), f"{len(extra)} changed rows outside the hedged windows' T+30 rows"
    if od is None:
        assert set(d.submission_id) == set(vsA.submission_id) and len(d) == len(vsA)
    # every changed row of the whole file is a private onset row -> the public (validation) rows equal A in every column
    # (parity mode --onset-delta: A = file + onset delta, so the onset-delta rows are A itself and are not counted)
    ks = key.set_index("submission_id").split.reindex(d.submission_id[~d.submission_id.isin(od.submission_id)] if od is not None else d.submission_id)
    assert ks.notna().all() and (ks == "private").all(), "a changed row is not a private onset row: public rows would differ from A"
    diff_lines = None
    if not a.no_diff:
        r = run([PROBE, "diff", A, full], log); assert r.returncode == 0, r.stderr[-400:]
        diff_lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        assert [ln for ln in diff_lines if not ln.endswith("0 rows changed")] == [ln for ln in diff_lines if ln.startswith("queue_pred:")], diff_lines
    if not a.keep_full:
        full.unlink()
    code = {p: sha(ROOT / p) for p in ("docs/reference_code/c_hedge_b.py", "docs/reference_code/onset9_hedge.py", "tools/probe.py")
            + (("docs/reference_code/c_pair_sim.py",) if rule_name == "split" else ())}
    inputs = {str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/") if Path(p).resolve().is_relative_to(ROOT) else str(p): sha(p)
              for p in [a.probs or H.PROBS, PSEUDO, H.STAT, META, KEY] + ([a.onset_delta] if a.onset_delta else [])}
    if rule_name == "split":
        rule = dict(name="split", file_role=s["role"], partner=partner, threshold=None, threshold_provenance=s["prov"],
                    union="A' = the panel's train-dominant site only (META n_windows), B = the other site only", dominant=s["dominant"],
                    kill="second site zero online activity while some site has activity (stat2_off)",
                    april_inputs="none (served decode and S1 statistic at/before T of each window, train site counts)")
    else:
        rule = dict(name=rule_name, threshold=s["th"], threshold_provenance=s["prov"], union="second site only",
                    kill="second site zero online activity while some site has activity (stat2_off)", grid=list(GRID),
                    april_inputs="none (served probabilities and S1 statistic at/before T of each window only)" if rule_name == "march" and a.th is None
                    else ("none in the rule inputs; threshold set by hand (--th), not by the March fit" if rule_name == "march"
                          else "threshold chosen with April pseudo labels (pooled calibration) or by hand"))
    rec = dict(tool="docs/reference_code/c_hedge_b.py build", made_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), name=name,
               A=dict(file=str(A), sha256=meta["parent_sha256"], onset_delta=a.onset_delta),
               B=dict(full_sha256=meta["full_sha256"], delta=delta.name, delta_sha256=meta["delta_sha256"], rows_in_delta=len(d),
                      full_kept=str(full) if a.keep_full else None),
               rule=rule,
               windows=wl, n_windows=len(wl), rows_changed_vs_A=int(len(vsA)), cells_1_to_0=int((vsA.A == 1).sum()), cells_0_to_1=int((vsA.A == 0).sum()),
               changed_vs_A=dict(by_column={c: (int(len(d)) if c == "queue_pred" else 0) for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow")},
                                 by_task=meta["rows_by_task"], by_split=ks.value_counts().to_dict(),
                                 by_window=vsA.groupby("window_id").size().to_dict(), probe_diff=diff_lines),
               killed_by_kill_rule=s["killed"], private_candidates_r_gt_025=s["pri"][["window_id", "s1", "p1", "s2", "p2", "r", "union"]].to_dict("records"),
               expected=s["gains"], rule_table=None if s["table"] is None else s["table"].to_dict("records"),
               checks=dict(onset_windows=int(len(o)), decode_mismatch_windows=mism, B_differs_only_on_hedged_T30_queue_rows=True,
                           public_rows_identical_to_A=True,
                           whole_file_compare="tools/probe.py make (all value columns, all rows)" + ("" if a.no_diff else " + tools/probe.py diff"),
                           same_rows_as_onset9_hedge_B_rows=same_old),
               calibration_inputs_sha256=inputs, code_sha256=code, runtime_s=round(time.time() - t0))
    (out / f"{name}.receipt.json").write_text(json.dumps(rec, indent=1), newline="\n")
    print(f"== {name}" + (f" (file {s['role']} of the split pair)" if rule_name == "split" else ""))
    print(pd.DataFrame(wl)[["window_id", "t30", "r", "a_site", "b_site", "rows_changed_vs_A", "cells_1_to_0", "cells_0_to_1"]].to_string(index=False))
    print(f"rule {rule_name} th {s['th']}: {len(wl)} windows, {len(vsA)} rows vs A, delta rows {len(d)}; killed {s['killed']}")
    if rule_name == "split":
        print("pair E[max(A',B)] - E[A]: " + "  ".join(f"{k} {v['gain_Emax_pair_minus_E_parent']:+.5f}" for k, v in s["gains"].items())
              + "  (bin, dom, pairM = train + March; domA confirm only)")
    else:
        print(f"gain E[max(A,B)]-E[A]: " + "  ".join(f"{k} {v['gain_Emax_minus_EA']:+.5f}" for k, v in s["gains"].items()) + "  (march = evidence; pooled/pairA confirm only)")
    if diff_lines:
        print("probe.py diff vs A: " + "; ".join(diff_lines))
    print(f"public rows identical to A: True (all {len(d)} changed rows are private onset rows)")
    print(f"A sha256 {meta['parent_sha256']}\nfile sha256 {meta['full_sha256']}\ndelta {delta} sha256 {meta['delta_sha256']}")
    print(f"same rows as onset9/hedge_B_rows.csv: {same_old}; decode mismatches {len(mism)}; {round(time.time() - t0)} s")


def cmd_build(a):
    t0 = time.time()
    A = Path(a.a); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = A.name[:-4] if A.name.endswith(".csv") else A.name
    free = shutil.disk_usage(out).free / 2 ** 30
    assert free >= MIN_FREE_GIB, f"only {free:.1f} GiB free on the output drive (need {MIN_FREE_GIB}); free disk first"
    if a.rule == "split":
        assert a.th is None, "--th does not apply to --rule split"
        base = a.name or stem + "_pairsplit"
        sa, sb = select_split(a.probs)
        variants = [(base + "_A", sa, base + "_B"), (base + "_B", sb, base + "_A")]
    else:
        variants = [(a.name or stem + f"_hedgeB_{a.rule}", select(a.rule, a.probs, a.th), None)]
    o = variants[0][1]["o"]
    for _, s, _ in variants:
        assert s["fin"].window_id.str.contains("_private_").all() and (s["fin"].split == "private").all()
    hedged = set().union(*[set(s["fin"].window_id) for _, s, _ in variants])
    key = onset_frame(A, set(o.window_id))
    key["A"] = key.file
    od = None
    if a.onset_delta:
        od = pd.read_csv(a.onset_delta); assert od.submission_id.isin(key.submission_id).all(), "onset delta has non-onset rows"
        m = key.submission_id.isin(od.submission_id)
        key.loc[m, "A"] = key.loc[m, "submission_id"].map(od.set_index("submission_id").queue_pred).astype(float)
    assert key.A.isin([0, 1]).all()
    # ---- A must be the served decode (else the probabilities do not describe this A)
    tmax = key.groupby("window_id").timestamp.transform("max"); key["t30"] = key.timestamp == tmax
    mism = []
    for x in o.itertuples():
        g = key[key.window_id == x.window_id]; last = g[g.t30]
        cells = set(last.link_id[last.A == 1]); want = set(x.a_cells.split(";")) - {""}
        early = int(g[~g.t30].A.sum())
        if cells != want or early:
            mism.append(dict(window_id=x.window_id, file_cells=";".join(sorted(cells)), served_cells=";".join(sorted(want)), cells_before_T30=early))
    bad = {m_["window_id"] for m_ in mism}
    assert not (bad & hedged), f"A differs from the served decode in hedged windows {sorted(bad & hedged)}: recompute the probabilities (--probs)"
    assert a.allow_mismatch or not mism, f"A's onset rows differ from the served decode in {len(mism)} windows (first {mism[:2]}): onset method changed; pass --probs or --allow-mismatch"
    for name, s, partner in variants:
        emit(a, A, out, name, key, s, od, mism, o, t0, a.rule, partner)


def cmd_rule(a):
    o = H.served(a.probs); w = H.calib()
    t, th = rule_table(o, w)
    pd.set_option("display.width", 250)
    print("calibration by r bin (window level): march = pop val only; pooled adds April pseudo (pop pri)")
    for pops in (["val"], ["val", "pri"]):
        ww = w[w["pop"].isin(pops)]
        print(pops, ww.groupby("b").agg(n=("r", "size"), top=("top", "mean"), sec=("sec", "mean"), both=("both", "mean")).round(3).to_string().replace("\n", "\n   "))
    print(t.round(5).to_string(index=False))
    print(f"march rule threshold: {th}")
    for split in ("validation", "private"):
        c = cands(o, split); k, kl, _ = H.final_set(c, th)
        print(f"{split} windows hedged at th {th}: {[c.window_id[i] for i in k]} killed {kl}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(a.out) / "rule_table.csv", index=False, lineterminator="\n")
        o.to_csv(Path(a.out) / "served_windows.csv", index=False, lineterminator="\n")


def cmd_compare(a):
    n, o = pd.read_csv(a.new), pd.read_csv(a.old)
    m = n.merge(o, on="submission_id", how="outer", suffixes=("_new", "_old"), indicator=True)
    both = m[m._merge == "both"]
    dv = int((both.queue_pred_new != both.queue_pred_old).sum())
    print(f"rows new {len(n)} old {len(o)}; only new {(m._merge == 'left_only').sum()}, only old {(m._merge == 'right_only').sum()}, "
          f"common {len(both)}, value differs {dv}")
    print("PARITY EXACT" if len(n) == len(o) == len(both) and dv == 0 else "PARITY FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    b = sp.add_parser("build"); b.add_argument("a"); b.add_argument("--out", required=True)
    b.add_argument("--rule", choices=["march", "asbuilt", "split"], default="march"); b.add_argument("--th", type=float, default=None)
    b.add_argument("--no-diff", action="store_true", help="skip the extra tools/probe.py diff of the temporary full file")
    b.add_argument("--onset-delta", default=None); b.add_argument("--probs", default=None); b.add_argument("--name", default=None)
    b.add_argument("--tmp", default=None); b.add_argument("--keep-full", action="store_true"); b.add_argument("--allow-mismatch", action="store_true")
    r = sp.add_parser("rule"); r.add_argument("--out", default=None); r.add_argument("--probs", default=None)
    c = sp.add_parser("compare"); c.add_argument("new"); c.add_argument("old")
    a = ap.parse_args()
    {"build": cmd_build, "rule": cmd_rule, "compare": cmd_compare}[a.op](a)


if __name__ == "__main__":
    main()
