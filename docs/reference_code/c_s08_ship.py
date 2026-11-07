"""C budget helper 2026-10-10: ship path for "S08 at L leaves x R rounds" (exp/C_budget/S08_SHIP.md).

Official S08 = split-local residual members (3 seeds x speed/flow_lane) on the 273-day F4 parent, trained on each official
split's released observations (s08_delivery.build groups validation/private), equal3 candidate -> official_<split>.parquet,
read by asm2_build/asm2_final through AB.S08_DIR (artifacts/lane_s/0398_s08). Here (VM main root; nothing under
artifacts/lane_s is touched), outputs under artifacts/claude_ref/c_budget/s08/official/:
  data   <split>              J.write_seed_data for the 3 seeds (same recipe mask/block seeds), official observations view
  parity <split>              refit seed 4701 speed at 31 x 600 and compare its trees with 0398 models/<split> (data+fit path)
  fit    <split> <L> <R>      c_budget_s08.fit on the official data (num_leaves L, save at R; env C_SAVES overrides)
  export <split> <L> <R>      equal3 candidate -> l<L>_r<R>/official_<split>.parquet (D.template_frame layout); panel 0 is
                              also predicted with the shipped 0398 models and must equal 0398 official_<split>.parquet
  state  <L> <R>              official Task 1 state rows: c_budget_s2build.py (stage 2 = the 3M form, as C34/C35) with
                              TFB_ASM2_ART pointing the two S08 parquets at l<L>_r<R>; compares with the C34 control parquet
  clean  <split>              drop the data_seed* matrices (VM scratch; run only with C's ok)
  probe  <L> <R> PARENT.csv   LAPTOP: new state rows on PARENT (C35) -> c36_s08l<L>r<R>_on_c35.csv + delta (tools/probe.py make);
                              control parquet must equal PARENT state rows (|d| <= 1e-6)
Usage on the VM (C_THREADS default 6): python c_s08_ship.py data validation | fit validation 63 1200 | ...
Build use (asm2_run_all.sh --s08=L,R, KB 2026-10-12): env C_S08_OUT = output root relative to the repo root (default as above);
export checks against the shipped 0398 files only when they are present (0398 models are not in the serving package)."""
import os, sys, json, time, gc, subprocess
from pathlib import Path
# KB2 2026-10-13 (KBUILD.md 7c W5): inside the build (asm2_run_all.sh exports TFB_ROOT) the tree is always the one this file
# lives in; the /workspace/work/main switch is for stand-alone use on the project VM only.
ROOT = Path("/workspace/work/main") if Path("/workspace/work/main").is_dir() and not os.environ.get("TFB_ROOT") else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(Path(__file__).resolve().parent))   # append: docs/reference_code/tfb.py must not shadow the tfb package
os.environ.setdefault("C_THREADS", "6")
import numpy as np
OUT = ROOT / os.environ.get("C_S08_OUT", "artifacts/claude_ref/c_budget/s08/official")
SHIP = ROOT / "artifacts/lane_s/0398_s08"
F3M = "all_l63_m2000_n3000000_r600"
CONTROL = ROOT / "artifacts/claude_ref/c_budget/s2/build/t1_l63n3r600.csv.state.parquet"   # C34 state rows (3M stage 2, shipped S08)


def setup(split):
    from tfb.lane_s import s08_joint_diagnostics as J, split_residual as S
    from tfb.paths import panels
    assert split in ("validation", "private")
    p = S.read(ROOT / "exp/0305_s08_split_residual/protocol.json")
    assert p["seeds"] == list(S.SEEDS) and p["features"] == 179 and p["aggregation"] == "equal3"
    base = J.bundle(ROOT / "artifacts/lane_s/0182/production273/model", 273)
    names = panels(ROOT)
    return p, base, names, dict(zip(names, range(len(names))))


def data(split):
    from tfb.lane_s import s08_delivery as D, s08_joint_diagnostics as J, split_residual as S
    p, base, names, codes = setup(split)
    folder = OUT / split; folder.mkdir(parents=True, exist_ok=True)
    for seed in S.SEEDS:
        if (folder / f"data_seed{seed}" / "complete.json").exists():
            continue
        t0 = time.time(); J.write_seed_data(ROOT, folder, split, seed, base, names, codes, lambda pn: D.observations(ROOT, pn, split))
        print("data", split, seed, round((time.time() - t0) / 60, 1), "min", flush=True)


def _fit(split, leaves, rounds):
    import c_budget_s08 as CB
    CB.OUT = OUT; CB.THREADS = int(os.environ["C_THREADS"])
    def st(group):   # official: protocol only (fit does not use the parent); C thread cap
        p = setup(group)[0]; p["parameters"] = dict(p["parameters"], num_threads=CB.THREADS); return p, None, None, None, group
    CB.setup = st
    CB.fit(split, leaves, rounds)


def parity(split):
    """One member (seed 4701 speed) at 31 x 600 on the official data vs the shipped 0398 member: trees must match."""
    from tfb.lane_s.local_model import lgb
    from tfb.lane_s import split_residual as S
    p, _, _, _ = setup(split)
    spec = S.read(OUT / split / "data_seed4701" / "complete.json")
    n = sum(r["n_rows"] for r in spec["records"])
    from tfb.lane_s import s08_joint_diagnostics as J
    x = np.empty((n, 179), np.float32, order="F"); y = np.empty(n, np.float32); w = np.empty(n, np.float32); cur = 0
    for r in spec["records"]:
        f = OUT / split / "data_seed4701" / r["panel"]
        a = np.load(f / "X.npy", mmap_mode="r"); b = np.load(f / "y.npy", mmap_mode="r"); ww = np.array(np.load(f / "w.npy", mmap_mode="r"), copy=True)
        ww = J.pool_weights(ww, a[:, spec["columns"].index("regime")], n, 10)
        x[cur:cur + len(a)] = a; y[cur:cur + len(a)] = b[:, 0]; w[cur:cur + len(a)] = ww; cur += len(a)
    params = dict(p["parameters"], num_threads=int(os.environ["C_THREADS"]), seed=4701)
    ds = lgb.Dataset(x, label=y, weight=w, feature_name=spec["columns"], categorical_feature=list(S.CATEGORIES), params=params)
    bst = lgb.train(params, ds, num_boost_round=600)
    trees = lambda s: s.split("end of trees")[0].split("Tree=0", 1)[1]
    mine = trees(bst.model_to_string(num_iteration=600)); ship = trees((SHIP / "models" / split / "seed4701" / "speed.txt").read_text())
    ref = lgb.Booster(model_file=str(SHIP / "models" / split / "seed4701" / "speed.txt"))
    k = np.random.default_rng(0).choice(n, min(n, 200000), replace=False)
    d = float(np.max(np.abs(bst.predict(x[k], num_threads=4) - ref.predict(x[k], num_threads=4))))
    # not fatal: 0398 may have been fitted on another host (LightGBM float paths); C judges a nonzero difference
    print("PARITY", split, "seed4701 speed 31x600 vs 0398: trees identical", mine == ship, "pred max|d| on 200k train rows", d, "rows", n, flush=True)


def export(split, leaves, rounds):
    import pandas as pd
    import c_budget_s08 as CB
    from tfb.lane_s.local_model import lgb
    from tfb.lane_s import s08_delivery as D, s08_joint_diagnostics as J, split_residual as S
    CB.THREADS = int(os.environ["C_THREADS"])
    p, base, names, codes = setup(split)
    mem = lambda d: {seed: {t: CB._T(lgb.Booster(model_file=str(d / f"seed{seed}" / f"{t}.txt"))) for t in ("speed", "flow_lane")} for seed in S.SEEDS}
    members = mem(OUT / split / f"fit_l{leaves}" / f"r{rounds}")
    have_ref = (SHIP / f"official_{split}.parquet").exists(); have_models = (SHIP / "models" / split / "seed4701" / "speed.txt").exists()
    ref = pd.read_parquet(SHIP / f"official_{split}.parquet") if have_ref else None
    if not (have_ref and have_models):   # build without the 0398 reference files: guards skipped, said in the log
        print("export", split, "0398 reference: parquet", have_ref, "models", have_models, "(missing checks skipped)", flush=True)
    dest = OUT / f"l{leaves}_r{rounds}"; dest.mkdir(parents=True, exist_ok=True); frames = []
    for code, panel in enumerate(names):
        view = D.observations(ROOT, panel, split)
        fr = D.template_frame(ROOT, view, J.predict_arms(view, base, members, code)["candidate"])
        if ref is None:
            print("export", split, leaves, rounds, panel, len(fr), "rows", flush=True); frames.append(fr); del view; gc.collect(); continue
        r = ref[ref.panel == panel].reset_index(drop=True)
        assert fr[D.KEYS].equals(r[D.KEYS])
        if code == 0 and have_models:   # serving-path guard: the shipped members through this path reproduce the shipped rows
            f0 = D.template_frame(ROOT, view, J.predict_arms(view, base, mem(SHIP / "models" / split), code)["candidate"])
            d0 = float(np.max(np.abs(f0[["speed_kmh", "flow_vph"]].to_numpy() - r[["speed_kmh", "flow_vph"]].to_numpy())))
            print("SHIP-PATH PARITY", split, panel, "max|d|", d0, flush=True); assert d0 <= 1e-6
        dv = np.abs(fr.speed_kmh.to_numpy() - r.speed_kmh.to_numpy()); dq = np.abs(fr.flow_vph.to_numpy() - r.flow_vph.to_numpy())
        print("export", split, leaves, rounds, panel, len(fr), "rows; vs 0398 mean|dv|", round(dv.mean(), 4), "mean|dq|", round(dq.mean(), 2), flush=True)
        frames.append(fr); del view; gc.collect()
    out = dest / f"official_{split}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    print("wrote", out, flush=True)


def state(leaves, rounds):
    import pandas as pd
    dest = OUT / f"l{leaves}_r{rounds}"; rel = lambda p: str(p.relative_to(ROOT)).replace("\\", "/")
    art = {f"artifacts/lane_s/0398_s08/official_{s}.parquet": rel(dest / f"official_{s}.parquet") for s in ("validation", "private")}
    for v in art.values():
        assert (ROOT / v).exists(), v
    env = dict(os.environ, TFB_ASM2_ART=json.dumps(art), CUDA_VISIBLE_DEVICES="")
    code = ROOT / "docs/reference_code/c_budget_s2build.py"   # main mirror code (stage 2 = the 3M form dir, as C34)
    subprocess.run([sys.executable, str(code), F3M, str(dest / "t1.csv")], env=env, check=True, cwd=str(ROOT))
    new = pd.read_parquet(dest / "t1.csv.state.parquet"); ctl = pd.read_parquet(CONTROL)
    assert new.submission_id.equals(ctl.submission_id)
    dv = np.abs(new.speed_kmh.to_numpy() - ctl.speed_kmh.to_numpy()); dq = np.abs(new.flow_vph.to_numpy() - ctl.flow_vph.to_numpy())
    print("STATE vs C34 control: rows", len(new), "changed", int(((dv > 1e-9) | (dq > 1e-9)).sum()), "mean|dv|", dv.mean(), "mean|dq|", dq.mean(), flush=True)


def clean(split):
    import shutil
    assert str(ROOT) == "/workspace/work/main"
    for seed in (4701, 4702, 4703):
        shutil.rmtree(OUT / split / f"data_seed{seed}", ignore_errors=True)


def probe(leaves, rounds, parent):
    """Laptop: pulled l<L>_r<R>/t1.csv.state.parquet and the control parquet; full = PARENT with the new state rows."""
    import pandas as pd
    dest = OUT / f"l{leaves}_r{rounds}"
    new = pd.read_parquet(dest / "t1.csv.state.parquet"); ctl = pd.read_parquet(CONTROL)
    par = pd.read_csv(parent, float_precision="round_trip")
    pos = pd.Index(par.submission_id).get_indexer(ctl.submission_id); assert (pos >= 0).all() and (par.task.iloc[pos] == "state").all()
    assert (par.task == "state").sum() == len(ctl)
    d = max(float(np.max(np.abs(par.speed_kmh.to_numpy()[pos] - ctl.speed_kmh.to_numpy()))), float(np.max(np.abs(par.flow_vph.to_numpy()[pos] - ctl.flow_vph.to_numpy()))))
    print("control vs parent state rows max|d|", d, flush=True); assert d <= 1e-6, "parent state rows are not the C34 control build"
    pos2 = pd.Index(par.submission_id).get_indexer(new.submission_id); assert (pos2 >= 0).all()
    par.loc[pos2, "speed_kmh"] = new.speed_kmh.to_numpy(); par.loc[pos2, "flow_vph"] = new.flow_vph.to_numpy()
    probes = ROOT / "artifacts/claude_ref/probes"; name = f"c36_s08l{leaves}r{rounds}_on_{Path(parent).stem.split('_')[0]}"
    eol = "\r\n" if open(parent, "rb").readline().endswith(b"\r\n") else "\n"
    full = probes / f"{name}.csv"; par.to_csv(full, index=False, lineterminator=eol)
    py = sys.executable
    subprocess.run([py, str(ROOT / "tools/probe.py"), "make", str(parent), str(full), str(probes / f"{name}.delta.csv.gz")], check=True)
    subprocess.run([py, str(ROOT / "tools/probe.py"), "diff", str(parent), str(full)], check=True)


if __name__ == "__main__":
    a = sys.argv[1:]; t0 = time.time()
    if a[0] == "fit" and not os.environ.get("C_SAVES"):
        os.environ["C_SAVES"] = a[3]
    {"data": lambda: data(a[1]), "parity": lambda: parity(a[1]), "fit": lambda: _fit(a[1], int(a[2]), int(a[3])),
     "export": lambda: export(a[1], int(a[2]), int(a[3])), "state": lambda: state(int(a[1]), int(a[2])),
     "clean": lambda: clean(a[1]), "probe": lambda: probe(int(a[1]), int(a[2]), a[3])}[a[0]]()
    print("DONE", a, round((time.time() - t0) / 60, 1), "min", flush=True)
