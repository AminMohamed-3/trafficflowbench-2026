"""K6 helper 2026-10-12 (ledger 0801, exp/C_b6long/B6LONG.md section 9): probe form of the adopted B6 12k set.

  weights   blend weights for the build: b6full_eval.fit_weights on the March I-S-OWN SSE grids of ALL days (as the shipped
            asm/b6w_s08.json = state5_probe_base.s08_weights), with the 12k OWN-view imputers (seeds 0,1,2) and the c11
            arrays of the upstream that ships (stage 2 3M rows on S08 l127_r4800 = arm 'n' of the chain cache)
            -> artifacts/claude_ref/c_b6long/b6w_12k.json (b6w_s08.json format; ONE file for both splits)
            + b6w_12k_fit.json (cross-parity fits and the ship-arm variants, for the reads)
  read      J(mix) on I-S-OWN in the probe form vs the exact parent (C43) form -> RESULT line in B6LONG.md
              candidate March: 12k, 3 seeds, weights fit on the other day parity (honest read of the all-days fit)
              candidate April: 12k, 3 seeds, the all-days March weights (= the file the probe uses for private rows)
              parent March: shipped OWN seeds 1,2 (the valown arm) with asm/b6w_s08.json
              parent April: shipped OWN seed 0 (stand-in for the one official-view priv_ seed) with asm/b6w_s08.json
  tune      exp/C_b6long/tune_b6l12k.json (the build change) and tune_b6ship.json (the shipped B6 through the same path)
  probe STATE.parquet NAME [CONTROL.parquet]   state rows on C43 -> artifacts/claude_ref/probes/NAME.csv (no delta, no manifest)
  control STATE.parquet                        shipped B6 through the tune path vs the C43 control build
Usage (repo root): PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python docs/reference_code/c_b6long_probe.py <cmd>"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "KNOB_THREADS"):
    os.environ.setdefault(k, "2")
import sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_b6long_fullscore as S
CK, L, EV, C5 = S.CK, S.L, S.EV, S.C5
ROOT = S.ROOT; CR = S.CR; B = CR / "c_b6long"; OUT = S.OUT / "P"; EXP = ROOT / "exp/C_b6long"
CFG = "it12000_w32"
WFILE = B / "b6w_12k.json"; WFIT = B / "b6w_12k_fit.json"; SHIPW = CR / "asm/b6w_s08.json"
PARENT = CR / "probes/c36_s08l127r4800_on_c36.csv"
CONTROL = CR / "c_budget/s08/official/l127_r4800/t1.csv.state.parquet"
ARM = "c_b6long/arm12k"
rel = lambda p: str(Path(p).relative_to(ROOT)).replace("\\", "/")


def weights():
    fit = {}
    for arm in ("n", "s"):
        st = {0: {}, 1: {}}
        for pn in L.PANELS:
            s, _ = S.panel_stats("validation", pn, S.imputer(CFG, "validation", pn), arm=arm)
            for par in (0, 1):
                st[par][pn] = s[par]
        cross = {par: EV.fit_weights(st[par])[0] for par in (0, 1)}
        alld = EV.fit_weights({pn: {k: st[0][pn][k] + st[1][pn][k] for k in st[0][pn]} for pn in st[0]})[0]
        fit[arm] = dict(cross=cross, all=alld)
    # the ship-arm fits are the R and M weights of the 10/10 RESULT line (c_b6long_fullscore.weights)
    assert json.dumps(fit["s"]["all"], sort_keys=True) == json.dumps(S.weights("M", CFG, "private")[0], sort_keys=True)
    assert json.dumps({str(k): v for k, v in fit["s"]["cross"].items()}, sort_keys=True) == json.dumps({str(k): v for k, v in S.weights("R", CFG, "validation").items()}, sort_keys=True)
    WFIT.write_text(json.dumps(fit, indent=1))
    src = ("c_b6long_probe.py weights: b6full_eval.fit_weights on I-S-OWN March, all days (as state5_probe_base.s08_weights), 12k OWN-view imputers "
           "artifacts/claude_ref/c_b6long/kaggle_full/*/b6long/it12000_w32/imp_own_<panel>_s{0,1,2}.npz, c11 = nv/nq of "
           f"artifacts/claude_ref/c_budget/s2/chain/c11_mar_{CK.TAGF}_<panel>.npz (stage 2 3M rows on S08 l127_r4800)")
    WFILE.write_text(json.dumps(dict(source=src, per_panel={pn: dict(weights_all_days=fit["n"]["all"][pn]) for pn in L.PANELS}), indent=1))
    ship = json.loads(SHIPW.read_text())["per_panel"]
    print("| panel | shipped b6w_s08 (v_nb v_blk q_nb q_blk) | 12k all days, c11 new arm = PROBE | 12k all days, c11 ship arm (M of the RESULT line) |"); print("|---|---|---|---|")
    f = lambda w: " ".join("%.3f" % w[k] for k in ("v_nb", "v_blk", "q_nb", "q_blk"))
    for pn in L.PANELS:
        print(f"| {pn} | {f(ship[pn]['weights_all_days'])} | {f(fit['n']['all'][pn])} | {f(fit['s']['all'][pn])} |")
    print("wrote", rel(WFILE), hashlib.sha256(WFILE.read_bytes()).hexdigest(), flush=True)


def _ship_imp(period, seeds):
    if period == "validation":
        return lambda pn: EV.imp("own", pn, list(seeds))
    return lambda pn: C5.imp_apr(pn, seeds=tuple(seeds))


def read():
    fit = json.loads(WFIT.read_text()); panels = list(L.PANELS)
    cross = {int(k): v for k, v in fit["n"]["cross"].items()}; alld = fit["n"]["all"]
    shipw = {pn: v["weights_all_days"] for pn, v in json.loads(SHIPW.read_text())["per_panel"].items()}
    both = lambda w: {0: w, 1: w}
    cand = lambda p: (lambda pn: S.imputer(CFG, p, pn))
    S.run_chain(OUT / "cand", "validation", panels, cand("validation"), cross)
    S.run_chain(OUT / "cand", "private", panels, cand("private"), both(alld))
    S.run_chain(OUT / "cand_all", "validation", panels, cand("validation"), both(alld))
    S.run_chain(OUT / "parent", "validation", panels, _ship_imp("validation", (1, 2)), both(shipw))
    S.run_chain(OUT / "parent", "private", panels, _ship_imp("private", (0,)), both(shipw))
    d = lambda name: {pn: OUT / name for pn in panels}
    r = dict(vs_parent={st: S.compare(d("cand"), st, base=d("parent")) for st in ("mar", "apr")},
             vs_knobs_base={st: S.compare(d("cand"), st) for st in ("mar", "apr")},
             parent_vs_knobs_base={st: S.compare(d("parent"), st) for st in ("mar", "apr")},
             all_days_vs_parent_mar=S.compare(d("cand_all"), "mar", base=d("parent")))
    (OUT / "read.json").write_text(json.dumps(r, indent=1))
    f = S.fmt; f6 = S.f6; x = r["vs_parent"]
    S.table("PROBE FORM: 12k (3 seeds, weights fit on March OWN with the shipping c11; March read = other-day parity) vs the C43 parent form", x)
    S.append(f"RESULT 0801 {S.now()} B6 12k PROBE FORM vs the C43 parent form (parent: validation = shipped seeds 1,2, private = one shipped seed, both with asm/b6w_s08.json; "
             f"candidate: 12k seeds 0,1,2, weights fit on March OWN with the c11 of S08 l127_r4800; March read = other-day-parity fit, April read = the all-days March file): "
             f"J(mix) March {f(x)['mar']} (state {f6(x['mar']['state_mix'])}, phys {f6(x['mar']['phys'])}), April {f(x)['apr']} (state {f6(x['apr']['state_mix'])}, phys {f6(x['apr']['phys'])}), "
             f"panels up {x['mar']['n_pos']}/10 and {x['apr']['n_pos']}/10. Same candidate vs the c_knobs base (shipped 3 seeds, weights A): March {f(r['vs_knobs_base'])['mar']} April {f(r['vs_knobs_base'])['apr']}. "
             f"Parent form vs the c_knobs base: March {f(r['parent_vs_knobs_base'])['mar']} April {f(r['parent_vs_knobs_base'])['apr']}. "
             f"March with the all-days file itself (in-sample fit of 4 weights per panel): {f6(r['all_days_vs_parent_mar']['J'])}. Per panel: exp/C_b6long/full_table.md")


def tune():
    t = dict(b6={sp: dict(arm=f"{ARM}:{pre}:0,1,2:", weights=rel(WFILE)) for sp, pre in (("validation", "valown_"), ("private", "privown_"))})
    c = dict(b6=dict(validation=dict(arm="b6full:valown_:1,2:", weights=rel(SHIPW)), private=dict(arm="b6full:priv_:0:", weights=rel(SHIPW))))
    (EXP / "tune_b6l12k.json").write_text(json.dumps(t)); (EXP / "tune_b6ship.json").write_text(json.dumps(c))
    for pn in L.PANELS:
        for pre in ("valown_", "privown_"):
            for k in (0, 1, 2):
                assert (CR / ARM / f"imp_{pre}{pn}_s{k}.npz").exists(), (pre, pn, k)
    print(json.dumps(t), "\n", json.dumps(c), flush=True)


def _cmp(new, ctl, label):
    assert new.submission_id.equals(ctl.submission_id)
    dv = np.abs(new.speed_kmh.to_numpy() - ctl.speed_kmh.to_numpy()); dq = np.abs(new.flow_vph.to_numpy() - ctl.flow_vph.to_numpy())
    res = dict(rows=len(new), speed_changed=int((dv > 1e-9).sum()), flow_changed=int((dq > 1e-9).sum()), mean_abs_dv=float(dv.mean()), max_abs_dv=float(dv.max()),
               mean_abs_dq=float(dq.mean()), max_abs_dq=float(dq.max()), mean_dv=float((new.speed_kmh.to_numpy() - ctl.speed_kmh.to_numpy()).mean()),
               mean_dq=float((new.flow_vph.to_numpy() - ctl.flow_vph.to_numpy()).mean()))
    print(label, json.dumps(res), flush=True)
    return res


def control(state_parquet):
    import pandas as pd
    r = _cmp(pd.read_parquet(state_parquet), pd.read_parquet(CONTROL), "CONTROL shipped B6 through the tune path vs the C43 control build:")
    print("CONTROL", "EXACT" if max(r["max_abs_dv"], r["max_abs_dq"]) == 0 else ("OK (<= 1e-6)" if max(r["max_abs_dv"], r["max_abs_dq"]) <= 1e-6 else "MISMATCH"), flush=True)


def probe(state_parquet, name):
    """As c_knobs.probe (pattern of c_s08_ship.probe), without the delta: C43 state rows must equal the control build
    (|d| <= 1e-6); full = C43 with the new state rows; every other cell untouched; tools/probe.py diff."""
    import pandas as pd
    new = pd.read_parquet(state_parquet); ctl = pd.read_parquet(CONTROL)
    assert new.submission_id.equals(ctl.submission_id)
    par = pd.read_csv(PARENT, float_precision="round_trip")
    pos = pd.Index(par.submission_id).get_indexer(ctl.submission_id); assert (pos >= 0).all() and (par.task.iloc[pos] == "state").all()
    assert (par.task == "state").sum() == len(ctl)
    d = max(float(np.max(np.abs(par.speed_kmh.to_numpy()[pos] - ctl.speed_kmh.to_numpy()))), float(np.max(np.abs(par.flow_vph.to_numpy()[pos] - ctl.flow_vph.to_numpy()))))
    print("control build vs parent (C43) state rows max|d|", d, flush=True); assert d <= 1e-6, "parent state rows are not the control build"
    res = _cmp(new, ctl, "new vs control:")
    before = {c: par[c].to_numpy().copy() for c in ("queue_pred", "path_flow")}; other = (par.task != "state").to_numpy()
    o0 = {c: par[c].to_numpy()[other].copy() for c in ("speed_kmh", "flow_vph")}
    old = {c: par[c].to_numpy()[pos].copy() for c in ("speed_kmh", "flow_vph")}
    par.loc[pos, "speed_kmh"] = new.speed_kmh.to_numpy(); par.loc[pos, "flow_vph"] = new.flow_vph.to_numpy()
    for c in before:
        assert np.array_equal(par[c].to_numpy(), before[c], equal_nan=True)
    for c in o0:
        assert np.array_equal(par[c].to_numpy()[other], o0[c], equal_nan=True)
    assert np.isfinite(par.speed_kmh.to_numpy()[pos]).all() and np.isfinite(par.flow_vph.to_numpy()[pos]).all() and other.sum() + len(pos) == len(par)
    assert (par.speed_kmh.to_numpy()[pos] > 0).all() and (par.flow_vph.to_numpy()[pos] >= 0).all()
    cells = {c: int((par[c].to_numpy()[pos] != old[c]).sum()) for c in old}
    eol = "\r\n" if open(PARENT, "rb").readline().endswith(b"\r\n") else "\n"
    full = CR / "probes" / f"{name}.csv"; assert not full.exists(), full
    par.to_csv(full, index=False, lineterminator=eol)
    h = hashlib.sha256()
    with open(full, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    info = dict(probe=rel(full), sha256=h.hexdigest(), parent=rel(PARENT), rows=len(par), state_rows=len(pos), changed_cells=dict(cells, queue_pred=0, path_flow=0),
                new_vs_control=res, state_parquet=rel(Path(state_parquet).resolve()), tune=rel(EXP / "tune_b6l12k.json"), weights=rel(WFILE),
                weights_sha256=hashlib.sha256(WFILE.read_bytes()).hexdigest())
    (B / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    print("PROBE", json.dumps(info), flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/probe.py"), "diff", str(PARENT), str(full)], check=True)
    print("PROBE DONE", full, flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(weights=weights, read=read, tune=tune, probe=probe, control=control)[cmd](*a)
