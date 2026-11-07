"""C budget helper 2026-10-09: S08 split-local residual budget (headroom bet 4, budget_audit row 2), I-S-OWN first.

S08 (tfb.lane_s.s08_delivery / s08_joint_diagnostics, frozen recipe): per split, 3 seeds (4701-4703) of LightGBM residual
members (speed, flow_lane) on 179 features around the 214-day parent, trained on the split's own observed-adaptation rows,
600 rounds, 31 leaves, lr .05, min leaf 200, max_bin 127, deterministic; equal3 candidate. The OWN members shipped in 0398
are the retained 0305 members. Here (VM main root, nothing under artifacts/lane_s is touched; outputs under
artifacts/claude_ref/c_budget/s08/):
  data  <own_validation|own_private>             J.write_seed_data for the 3 seeds (same mask/block seeds as the recipe)
  fit   <group> <leaves> <rounds>                 = J.fit_member with the leaves changed and models saved at 600/1200/2400 <= rounds
                                                  (env C_SAVES=1800,2400 overrides the save list)
  clean <group>                                  drop the data_seed* matrices (VM scratch)
  pred  <group> <leaves> <it>                     J.predict_arms candidate -> own_<split>/<pn>.npz in the 0398 layout
                                                  (speed, flow, target, dates, station_id); parity for 31/600 vs 0398
Usage on the VM: python c_budget_s08.py data own_validation | fit own_validation 31 1200 | pred own_validation 31 600"""
import os, sys, json, time, gc
from pathlib import Path
# KB2 2026-10-13 (KBUILD.md 7c W5): inside the build (asm2_run_all.sh exports TFB_ROOT) the tree is always the one this file
# lives in; the /workspace/work/main switch is for stand-alone use on the project VM only.
ROOT = Path("/workspace/work/main") if Path("/workspace/work/main").is_dir() and not os.environ.get("TFB_ROOT") else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from tfb.lane_s import s08_delivery as D, split_residual as S, s08_joint_diagnostics as J
OUT = ROOT / "artifacts/claude_ref/c_budget/s08"
THREADS = int(os.environ.get("C_THREADS", "4"))


def setup(group):
    """D.recipe is not called: its source sha pins (SCIENTIFIC_SHA d63a8080) predate the committed split_residual (1c0a4c05);
    the 31/600 parity vs the 0398 delivery is the guard instead. Parent = the frozen 214-day 0305 parent (as D.build)."""
    from tfb.paths import panels
    p = S.read(ROOT / "exp/0305_s08_split_residual/protocol.json")
    assert p["seeds"] == list(S.SEEDS) and p["features"] == 179 and p["num_boost_round"] == 600 and p["aggregation"] == "equal3"
    p["parameters"] = dict(p["parameters"], num_threads=THREADS)
    base = J.bundle(ROOT / "artifacts/lane_s/0305/parent", 214)
    names = panels(ROOT); codes = dict(zip(names, range(len(names))))
    split = group.removeprefix("own_")
    return p, base, names, codes, split


class _T:
    """Booster wrapper: predict_arms hardcodes num_threads=8; keep the C thread cap."""
    def __init__(self, b): self.b = b
    def predict(self, z, num_threads=None): return self.b.predict(z, num_threads=THREADS)


def data(group):
    p, base, names, codes, split = setup(group)
    folder = OUT / group; folder.mkdir(parents=True, exist_ok=True)
    src = lambda panel: D.own_view(ROOT, panel, split)[0]
    for seed in S.SEEDS:
        if (folder / f"data_seed{seed}" / "complete.json").exists():
            continue
        t0 = time.time(); J.write_seed_data(ROOT, folder, group, seed, base, names, codes, src)
        print("data", group, seed, round((time.time() - t0) / 60, 1), "min", flush=True)


def fit(group, leaves, rounds):
    """J.fit_member with num_leaves changed and checkpointed saves (first k trees of an R-round run = a k-round run)."""
    from tfb.lane_s.local_model import lgb
    p, base, names, codes, split = setup(group)
    folder = OUT / group; its = [c for c in (600, 1200, 2400) if c <= rounds]
    if os.environ.get("C_SAVES"):   # 2026-10-10 ext: explicit save list (e.g. 1800,2400) so existing r600/r1200 saves are not rewritten
        its = [int(c) for c in os.environ["C_SAVES"].split(",") if int(c) <= rounds]
    params0 = dict(p["parameters"], num_leaves=leaves)
    for seed in S.SEEDS:
        data_ = folder / f"data_seed{seed}"; spec = S.read(data_ / "complete.json")
        assert spec["seed"] == seed and spec["phase"] == group and len(spec["records"]) == 10 and len(spec["columns"]) == 179
        n = sum(r["n_rows"] for r in spec["records"])
        x = np.empty((n, 179), np.float32, order="F"); y = np.empty((n, 2), np.float32); w = np.empty(n, np.float32); cur = 0
        for r in spec["records"]:
            f = data_ / r["panel"]
            a = np.load(f / "X.npy", mmap_mode="r"); b = np.load(f / "y.npy", mmap_mode="r"); ww = np.array(np.load(f / "w.npy", mmap_mode="r"), copy=True)
            reg = a[:, spec["columns"].index("regime")]
            ww = J.pool_weights(ww, reg, n, 10)
            x[cur:cur + len(a)] = a; y[cur:cur + len(a)] = b; w[cur:cur + len(a)] = ww; cur += len(a)
        assert cur == n
        params = dict(params0, seed=seed)
        ds = lgb.Dataset(x, label=np.ascontiguousarray(y[:, 0]), weight=w, feature_name=spec["columns"],
                         categorical_feature=list(S.CATEGORIES), params=params, free_raw_data=True)
        ds.construct()
        for k, target in enumerate(("speed", "flow_lane")):
            t0 = time.time(); ds.set_label(np.ascontiguousarray(y[:, k]))
            bst = lgb.train(params, ds, num_boost_round=rounds)
            for it in its:
                d = folder / f"fit_l{leaves}" / f"r{it}" / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
                bst.save_model(str(d / f"{target}.txt"), num_iteration=it)
            print("fit", group, leaves, seed, target, n, "rows", round((time.time() - t0) / 60, 1), "min", flush=True)
            del bst
        del x, y, w, ds; gc.collect()


def pred(group, leaves, it):
    from tfb.lane_s.local_model import lgb
    p, base, names, codes, split = setup(group)
    folder = OUT / group / f"fit_l{leaves}" / f"r{it}"
    members = {seed: {t: _T(lgb.Booster(model_file=str(folder / f"seed{seed}" / f"{t}.txt"))) for t in ("speed", "flow_lane")} for seed in S.SEEDS}
    dest = OUT / f"l{leaves}_r{it}" / group; dest.mkdir(parents=True, exist_ok=True); worst = 0.
    for code, panel in enumerate(names):
        view, target = D.own_view(ROOT, panel, split)
        v = J.predict_arms(view, base, members, code)["candidate"]
        np.savez_compressed(dest / f"{panel}.npz", speed=v[0], flow=v[1], target=target, dates=np.asarray(view.dates),
                            station_id=view.stations.station_id.to_numpy(str))
        ref = ROOT / "artifacts/lane_s/0398_s08" / group / f"{panel}.npz"
        if ref.exists():
            r = np.load(ref); assert (r["target"] == target).all()
            d = max(float(np.max(np.abs(r["speed"][target] - v[0][target]))), float(np.max(np.abs(r["flow"][target] - v[1][target]))))
            worst = max(worst, d); print("pred", group, leaves, it, panel, "vs 0398 max|d|", d, flush=True)
        del view, v; gc.collect()
    print("PARITY/DIFF vs 0398", group, leaves, it, "worst", worst, flush=True)


def clean(group):
    """VM scratch only: drop the training matrices of a group once its fits are saved (VM1 disk ~15 GB free)."""
    import shutil
    assert str(ROOT) == "/workspace/work/main"
    for seed in S.SEEDS:
        shutil.rmtree(OUT / group / f"data_seed{seed}", ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    t0 = time.time()
    {"data": lambda: data(a[1]), "fit": lambda: fit(a[1], int(a[2]), int(a[3])), "pred": lambda: pred(a[1], int(a[2]), int(a[3])), "clean": lambda: clean(a[1])}[a[0]]()
    print("DONE", a, round((time.time() - t0) / 60, 1), "min", flush=True)
