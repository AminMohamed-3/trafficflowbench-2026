"""Fixed S08 quick OPT and full ADV joint diagnostics; VM-only science."""
from pathlib import Path
from datetime import date, datetime, timezone
from types import SimpleNamespace
import argparse
import gc
import shutil
import time

from . import split_residual as S
from . import s08_own_mix as M

PHASES = {"quick_opt": 12, "adv": 60}
ARMS = ("parent", "candidate", "4701", "4702", "4703")
QUICK = ("D12_I5_N", "D7_I405_N")
WINDOW_SHA = "3bb6edec7dbb003dcf6d0bf7d69935009c867ad4a51413803407104c14e94670"


def verify(exp, root):
    p = S.read(exp / "joint_protocol.json")
    original = S.plan(exp, root)
    M.verify(exp, root)
    assert p["id"] == "0305" and p["phase_maximum_minutes"] == PHASES
    assert p["quick_panels"] == list(QUICK) and p["seeds"] == list(S.SEEDS)
    assert p["features"] == 179 and p["rounds"] == 600
    assert p["parameters"] == original["parameters"] and p["dose"] == 1 and p["aggregation"] == "equal3"
    assert p["mask_seeds"] == original["mask_seeds"] and p["block_seeds"] == original["block_seeds"]
    assert p["threads"] == 8 and p["ram_gib"] == 23 and not p["gpu"]
    assert p["diagnostic_reserve_minutes"] == 90 and p["total_minutes"] == 600
    assert p["ADV_window_sha256"] == WINDOW_SHA and p["physics_mode"] == "A"
    assert p["new_OWN_reads"] == 0 and p["ADV_joint_reads"] == 1
    for rel, digest in p["source_sha256"].items():
        assert S.sha(exp.parents[1] / rel) == digest, rel
    for rel, digest in p["main_source_sha256"].items():
        assert S.sha(Path(root) / rel) == digest, rel
    import importlib
    for name in ("metric", "split_constants", "data", "paths"):
        module = importlib.import_module("tfb."+name)
        assert S.sha(module.__file__) == p["main_source_sha256"]["src/tfb/"+name+".py"], "Imported shared API differs"
    base = Path(root) / "artifacts/lane_s/0305"
    for rel, digest in p["completed_sha256"].items():
        assert S.sha(base / rel) == digest, rel
    parent = S.read(base / "diagnostics/opt_parent/result.json")
    assert parent["verdict"] == "PASS" and parent["fit_dates"] == p["OPT_fit_dates"]
    assert parent["held_dates"] == p["OPT_dates"] and len(parent["fit_dates"]) == 171
    assert len(p["OPT_dates"]) == 43 and not set(p["OPT_dates"]) & set(p["OPT_fit_dates"])
    assert len(p["ADV_dates"]) == 59 and min(p["ADV_dates"]) == "2031-01-01"
    assert max(p["ADV_dates"]) == "2031-02-28"
    return p


def assert_observation_view(view):
    import numpy as np
    assert not any(k.endswith("_true") for k in vars(view)), "Truth reached predictor"
    omitted = view.target | view.gap
    for name in ("speed", "flow", "occ"):
        assert np.isnan(getattr(view, name)[omitted]).all(), "Hidden observation reached predictor"
    assert not view.pct[omitted].any(), "Hidden pct channel reached predictor"


def released_view(root, panel, dates, bindings):
    """Load only published observation arrays, never a *_true mmap."""
    import numpy as np
    import pandas as pd
    from tfb.data import VERSION
    from tfb.paths import cache_root
    from .own_instrument import OBSERVATION_NAMES, WINDOW_COLUMNS, official_windows
    folder = cache_root(root) / panel / "train"
    assert S.sha(folder / "meta.json") == bindings[panel]["meta.json"], "Cache metadata changed"
    for name in ("stations.parquet", "ramps.parquet"):
        assert S.sha(folder / name) == bindings[panel][name], "Cache station/ramp metadata changed"
    meta = S.read(folder / "meta.json")
    assert meta["verified"] and meta["format_version"] == VERSION
    ix = [meta["dates"].index(d) for d in dates]
    arrays = {}
    for name in OBSERVATION_NAMES:
        path = folder / (name + ".npy")
        assert S.sha(path) == bindings[panel][name + ".npy"], "Cache observation changed"
        mm = np.load(path, mmap_mode="r")
        expected = meta["ramp_shape"] if name.startswith("ramp_") else meta["shape"]
        assert list(mm.shape) == expected and str(mm.dtype) == meta["arrays"][name]
        arrays[name] = np.array(mm[ix], copy=True)
        mm._mmap.close()
    view = SimpleNamespace(meta=dict(meta), stations=pd.read_parquet(folder / "stations.parquet"),
                           ramps=pd.read_parquet(folder / "ramps.parquet"), dates=list(dates),
                           regimes=np.asarray(meta["regime_per_day"])[ix], **arrays)
    view.meta["shape"] = [len(dates), *meta["shape"][1:]]
    view.meta["ramp_shape"] = [len(dates), *meta["ramp_shape"][1:]]
    # The original query mask is already channel-blinded; do not expose its quality flag.
    view.pct[view.target | view.gap] = 0
    view.s_windows = official_windows(root, view)
    view.own_windows = pd.DataFrame(columns=WINDOW_COLUMNS)
    assert_observation_view(view)
    return view


def cache_bindings(root, names, dest):
    from tfb.paths import cache_root
    from .own_instrument import OBSERVATION_NAMES
    path = dest / "cache_bindings.json"
    assert not path.exists(), "Started source snapshot requires inspection"
    result = {}
    for panel in names:
        folder = cache_root(root) / panel / "train"
        result[panel] = {name + ".npy": S.sha(folder / (name + ".npy")) for name in OBSERVATION_NAMES}
        result[panel].update({name: S.sha(folder / name) for name in ("meta.json", "stations.parquet", "ramps.parquet")})
    S.write(path, result)
    return result


def bundle(folder, fit_count):
    from .local_model import lgb
    folder = Path(folder)
    m = S.read(folder / "metadata.json")
    assert len(m["columns"]) == 177 and len(m["fit_dates"]) == fit_count
    assert m["integrity"]["model_source"] == S.sha(Path(__file__).parent / "local_model.py")
    assert m["integrity"]["prior_adapter_source_sha256"] == S.sha(Path(__file__).parent / "local_priors.py")
    models = {}
    for target in ("speed", "flow_lane"):
        path = folder / (target + ".txt")
        record = m["model_cases"][target]
        assert record["best_iteration"] == 600 and S.sha(path) == record["model_sha256"]
        models[target] = lgb.Booster(model_file=str(path))
        assert models[target].current_iteration() == 600
    return dict(folder=folder, metadata=m, models=models,
                indices={t: [m["columns"].index(c) for c in b.feature_name()] for t, b in models.items()})


def pool_weights(weights, regime, n, panel_count):
    import numpy as np
    assert panel_count in (2, 10)
    output = np.array(weights, dtype=np.float32, copy=True)
    assert np.isfinite(output).all() and (output > 0).all()
    for r in (0, 1, 2):
        selected = regime == r
        assert selected.any(), "A panel/regime has no adaptation rows"
        output[selected] *= n / (panel_count * 3 * float(output[selected].sum()))
    return output


def write_seed_data(root, dest, phase, seed, base, names, codes, source):
    import numpy as np
    from .observed_adaptation import make_observed_adaptation
    from .observed_adaptation_features import day_levels, select_rows, feature_rows
    from .build_lgb_data import prepare_bases
    from .local_model import load_summaries
    from .vm_f4_reconstruct import check_space
    k = S.SEEDS.index(seed)
    columns = base["metadata"]["columns"]
    folder = dest / ("data_seed" + str(seed))
    assert not folder.exists(), "Started data must not be overwritten"
    folder.mkdir()
    records = []
    for panel in names:
        view = source(panel)
        assert_observation_view(view)
        augmented, labels_v, labels_q, augmentation = make_observed_adaptation(view, 16853+k, 16854+k)
        del labels_v, labels_q
        assert_observation_view(augmented)
        selected = [select_rows(view, augmented, d) for d in range(len(view.dates))]
        n = sum(int(mask.sum()) for mask, _ in selected)
        assert n > 0
        case = folder / panel
        case.mkdir()
        check_space(case, n * (179*4+2*4+4) + 2**20)
        xout = np.lib.format.open_memmap(case / "X.npy", mode="w+", dtype=np.float32, shape=(n, 179))
        yout = np.lib.format.open_memmap(case / "y.npy", mode="w+", dtype=np.float32, shape=(n, 2))
        wout = np.lib.format.open_memmap(case / "w.npy", mode="w+", dtype=np.float32, shape=(n,))
        bases = prepare_bases(augmented, base["metadata"]["asm_parameters"])
        assert S.sha(base["folder"] / (panel+"_summaries.npz")) == base["metadata"]["summary_sha256"][panel]
        structural = load_summaries(base["folder"], panel)
        cursor = 0
        for d, day in enumerate(view.dates):
            mask, weights = selected[d]
            assert not (mask & (view.target[d] | view.hist[d] | view.gap[d])).any()
            constants = day_levels(augmented, day)
            x, meta, _ = feature_rows(augmented, d, bases, columns, codes[panel], structural, constants, mask)
            si = meta[:, 2]
            lanes = view.stations.lanes.to_numpy(np.float32)[si]
            capacity = view.stations.capacity_vph.to_numpy(float)[si]
            v, total = S.parent_rows(x, base, lanes, capacity)
            q = total / lanes
            part = slice(cursor, cursor+len(x))
            xout[part] = S.appended(x, v, q)
            yout[part, 0] = view.speed[d][mask] - v
            yout[part, 1] = view.flow[d][mask] / lanes - q
            wout[part] = weights[mask]
            cursor += len(x)
        assert cursor == n and np.isfinite(yout).all() and np.isfinite(wout).all()
        xout.flush(); yout.flush(); wout.flush()
        del xout, yout, wout, bases, view, augmented, selected, structural
        records.append(dict(panel=panel, n_rows=n, files={n: S.sha(case/n) for n in ("X.npy", "y.npy", "w.npy")}, augmentation=augmentation))
        gc.collect()
        print(dict(phase=phase, seed=seed, panel=panel, training_rows=n), flush=True)
    S.write(folder / "complete.json", dict(records=records, columns=columns+list(S.EXTRA), seed=seed, phase=phase,
            source_sha256=S.sha(__file__), scientific_source_sha256=S.sha(S.__file__)))
    return folder


def fit_member(dest, phase, seed, data, parameters, panel_count):
    """Original fixed S08 trainer, with declared quick-panel mass denominator."""
    import numpy as np
    from .local_model import lgb
    spec = S.read(data / "complete.json")
    assert spec["seed"] == seed and spec["phase"] == phase
    assert spec["source_sha256"] == S.sha(__file__)
    assert len(spec["records"]) == panel_count and len(spec["columns"]) == 179
    n = sum(r["n_rows"] for r in spec["records"])
    x = np.empty((n, 179), np.float32, order="F")
    y = np.empty((n, 2), np.float32)
    w = np.empty(n, np.float32)
    cursor = 0
    for r in spec["records"]:
        folder = data / r["panel"]
        for name, digest in r["files"].items():
            assert S.sha(folder/name) == digest
        a = np.load(folder/"X.npy", mmap_mode="r")
        b = np.load(folder/"y.npy", mmap_mode="r")
        weights = np.array(np.load(folder/"w.npy", mmap_mode="r"), copy=True)
        assert len(a) == r["n_rows"]
        reg = a[:, spec["columns"].index("regime")]
        weights = pool_weights(weights, reg, n, panel_count)
        part = slice(cursor, cursor+len(a))
        x[part] = a; y[part] = b; w[part] = weights
        cursor += len(a)
        del a, b, weights
    assert cursor == n
    model = dest / ("seed" + str(seed))
    assert not model.exists(), "Partial/completed model requires inspection"
    model.mkdir()
    params = dict(parameters, seed=seed)
    cases = {}
    dataset = lgb.Dataset(x, label=np.ascontiguousarray(y[:, 0]), weight=w, feature_name=spec["columns"],
                          categorical_feature=list(S.CATEGORIES), params=params, free_raw_data=True)
    dataset.construct()
    for k, target in enumerate(("speed", "flow_lane")):
        dataset.set_label(np.ascontiguousarray(y[:, k]))
        booster = lgb.train(params, dataset, num_boost_round=600)
        assert booster.current_iteration() == 600
        indices = [spec["columns"].index(c) for c in S.CATEGORIES]
        assert "[categorical_feature: " + ",".join(map(str, indices)) + "]" in booster.model_to_string().splitlines()
        path = model / (target+".txt")
        booster.save_model(str(path), num_iteration=600)
        cases[target] = dict(sha256=S.sha(path), rounds=600)
        del booster
    S.write(model/"metadata.json", dict(id="0305", phase=phase, seed=seed, columns=spec["columns"], parameters=params,
            models=cases, data_sha256=S.sha(data/"complete.json"), source_sha256=S.sha(__file__),
            scientific_source_sha256=S.sha(S.__file__), training_panel_count=panel_count))
    del x, y, w, dataset
    gc.collect()


def load_members(dest, phase, columns, parameters, panel_count):
    from .local_model import lgb
    result = {}
    for seed in S.SEEDS:
        folder = dest / ("seed"+str(seed))
        meta = S.read(folder/"metadata.json")
        assert meta["seed"] == seed and meta["phase"] == phase and meta["training_panel_count"] == panel_count
        assert meta["columns"] == columns+list(S.EXTRA) and meta["parameters"] == dict(parameters, seed=seed)
        assert meta["source_sha256"] == S.sha(__file__) and meta["scientific_source_sha256"] == S.sha(S.__file__)
        assert meta["data_sha256"] == S.sha(dest/("data_seed"+str(seed))/"complete.json")
        result[seed] = {}
        for name, entry in meta["models"].items():
            assert entry["rounds"] == 600 and entry["sha256"] == S.sha(folder/(name+".txt"))
            model = lgb.Booster(model_file=str(folder/(name+".txt")))
            assert model.current_iteration() == 600 and model.feature_name() == meta["columns"]
            result[seed][name] = model
    return result


def predict_arms(view, base, members, code):
    import numpy as np
    from tfb.split_constants import own_month_constants
    from .build_lgb_data import prepare_bases
    from .local_model import load_summaries
    from .travel_features import day_features
    from .own_instrument import own_context_for_day
    from .local_priors import fill_scored_dynamic
    assert_observation_view(view)
    constants = own_month_constants(view, hidden=view.target|view.gap)
    bases = prepare_bases(view, base["metadata"]["asm_parameters"])
    columns = base["metadata"]["columns"]
    assert S.sha(base["folder"]/(view.meta["panel"]+"_summaries.npz")) == base["metadata"]["summary_sha256"][view.meta["panel"]]
    structural = load_summaries(base["folder"], view.meta["panel"])
    values = {name: [a.copy() for a in bases["time"]] for name in ARMS}
    for d, day in enumerate(view.dates):
        fields, override = day_features(view, d, bases, code, own_context_for_day(view, d))
        assert list(fields) == columns
        mask = view.target[d]; ti, si = np.nonzero(mask)
        if not len(ti):
            continue
        x = np.column_stack([a[mask] for a in fields.values()]).astype(np.float32)
        weekend = np.full(len(ti), int(date.fromisoformat(day).weekday() >= 5))
        fill_scored_dynamic(x, columns, si, ti, weekend, override[mask], constants[day[:7]], structural)
        lanes = view.stations.lanes.to_numpy(np.float32)[si]
        capacity = view.stations.capacity_vph.to_numpy(float)[si]
        v, total = S.parent_rows(x, base, lanes, capacity)
        q = total / lanes
        z = S.appended(x, v, q)
        values["parent"][0][d][mask] = v; values["parent"][1][d][mask] = total
        meanv = np.zeros(len(x), np.float64); meanq = meanv.copy()
        for seed in S.SEEDS:
            models = members[seed]
            a, b = S.bound_rows(v+models["speed"].predict(z, num_threads=8),
                                q+models["flow_lane"].predict(z, num_threads=8), lanes, capacity)
            values[str(seed)][0][d][mask] = a; values[str(seed)][1][d][mask] = b
            meanv += a.astype(float)/3; meanq += b.astype(float)/3
        a, b = S.bound_total(meanv, meanq, capacity)
        values["candidate"][0][d][mask] = a; values["candidate"][1][d][mask] = b
    return values


def joint_gate(comparison, member_deltas):
    from .bootstrap import gate
    result = S.positive(comparison["delta_J_S"], comparison["ci95"], member_deltas)
    result["delta_J_S"] = result.pop("delta_S_state")
    result["original_noise_floor_verdict"] = gate(comparison, result["seed_delta_sd"])
    result["delta_S_state"] = comparison["candidate"]["S_state"] - comparison["incumbent"]["S_state"]
    result["state_loss_guard"] = result["delta_S_state"] >= 0
    result["candidate_qualified"] = result["positive"] and result["original_noise_floor_verdict"] == "PASS" and result["state_loss_guard"]
    return result


def score_phase(root, dest, phase, p, names, codes, source, base, windows):
    import numpy as np
    import pandas as pd
    from tfb.data import load_cache
    from tfb import metric
    from tfb.paths import release_root
    from .evaluate_baselines import subset_cache, raw_truth
    from .instruments import apply_frozen_gaps
    from .day_stats import day_statistics
    from .bootstrap import aggregate, paired_ci
    members = load_members(dest, phase, base["metadata"]["columns"], p["parameters"], len(names))
    marker = dest / "read_started.json"
    assert not marker.exists(), "Started joint read must never be overwritten/repeated"
    S.write(marker, dict(phase=phase, joint_OPT_reads=int(phase=="quick_opt"), joint_ADV_reads=int(phase=="adv"),
            OWN_reads=0, seeds=list(S.SEEDS), utc=datetime.now(timezone.utc).isoformat()))
    rows = {a: [] for a in ARMS}
    dates = p["OPT_dates"] if phase == "quick_opt" else p["ADV_dates"]
    families = {a["corridor_id"]: a["family_id"] for a in S.read(release_root(root)/"config/corridors.json")["panels"]}
    for panel in names:
        # The source closure returns observations only and never sees scorer arrays.
        view = source(panel)
        forecasts = predict_arms(view, base, members, codes[panel])
        raw = load_cache(panel, "train", root=root)
        cache = subset_cache(raw, [raw.day_index(d) for d in dates])
        for a in raw.arrays.values():
            if isinstance(a, np.memmap):
                a._mmap.close()
        del raw
        tv, tq = raw_truth(cache, root)
        if phase == "adv":
            apply_frozen_gaps(cache, tv, tq, windows)
            cache.s_windows = windows
        assert cache.dates == view.dates and (cache.target <= view.target).all()
        part = {}
        for name, (v, q) in forecasts.items():
            assert np.isfinite(v[cache.target]).all() and np.isfinite(q[cache.target]).all()
            frame = day_statistics(cache, v, q, tv, tq, root=root, mode="A")
            direct = aggregate(frame, families)
            state = metric.state_score(tv, tq, v, q, target=cache.target, lanes=cache.stations.lanes, regimes=cache.regimes)
            physics = metric.physics_score(cache, v, q, truth_speed=tv, truth_flow=tq, mode="A", root=root, ramp_filter=True)
            assert abs(direct["S_state"]-state["S_state"]) < 1e-9
            assert abs(direct["S_physics"]-physics["S_physics"]) < 1e-9
            part[name] = frame.to_dict("records")
            rows[name].extend(part[name])
        S.write(dest/(panel+"_rows.json"), dict(panel=panel, n_targets=int(cache.target.sum()), rows=part))
        del view, forecasts, cache, tv, tq, frame
        gc.collect()
        print(dict(phase=phase, panel=panel, scored=True), flush=True)
    frames = {a: pd.DataFrame(r) for a, r in rows.items()}
    comparison = paired_ci(frames["parent"], frames["candidate"], families)
    member_deltas = [paired_ci(frames["parent"], frames[str(seed)], families)["delta_J_S"] for seed in S.SEEDS]
    gates = joint_gate(comparison, member_deltas)
    S.write(dest/"result.json", dict(id="0305", phase=phase, comparison=comparison, gate=gates, member_deltas=member_deltas,
            points={a: aggregate(frame, families) for a, frame in frames.items()}, seeds=list(S.SEEDS),
            reference="Original F4 OPT171" if phase=="quick_opt" else "Original F4fit214 on matched conservative ADV source",
            panels=names, training_panel_count=len(names), full10_reproduction=phase=="adv", physics_mode="A", ramp_filter=True,
            source_extra_OWN_blinding=phase=="adv", scorer_population="Original OPT0" if phase=="quick_opt" else "Original ADV160 windows",
            joint_OPT_reads=int(phase=="quick_opt"), joint_ADV_reads=int(phase=="adv"), OWN_reads=0,
            protocol_sha256=S.sha(Path(__file__).parents[3]/"exp/0305_s08_split_residual/joint_protocol.json"),
            verdict="PASS" if gates["candidate_qualified"] else "NULL"))


def run_phase(root, dest, phase, p, maximum, job):
    from tfb.paths import panels
    from .vm_s07_stages import load_manifest, source_view
    names = list(QUICK) if phase=="quick_opt" else panels()
    codes = {name: i for i, name in enumerate(panels())}
    bindings = cache_bindings(root, names, dest)
    windows = None
    if phase == "quick_opt":
        base = bundle(root/"artifacts/lane_s/0305/diagnostics/opt_parent", 171)
        source = lambda panel: released_view(root, panel, p["OPT_dates"], bindings)
    else:
        import pandas as pd
        from .vm_s07_adv_windows import main as rebuild_windows
        rebuild_windows()  # Pure prerequisite producer; never an old controller/score.
        wp = root/"artifacts/lane_s/0222/prerequisites/ADV/windows.csv"
        assert S.sha(wp) == WINDOW_SHA
        windows = pd.read_csv(wp)
        assert len(windows) == 160
        manifest = load_manifest(root)
        items = {a["panel"]: a for a in manifest["items"] if a["period"]=="jan_feb"}
        assert set(items)==set(names) and all(a["dates"]==p["ADV_dates"] for a in items.values())
        stage = dict(root=root, items=items, phase="ADV1", dates=p["ADV_dates"], windows=windows)
        def source(panel):
            value, receipt = source_view(stage, panel)
            assert "no hidden labels" in receipt["source"]
            assert_observation_view(value)
            return value
        base = bundle(root/"artifacts/lane_s/0305/parent", 214)
    for seed in S.SEEDS:
        data = write_seed_data(root, dest, phase, seed, base, names, codes, source)
        fit_member(dest, phase, seed, data, p["parameters"], len(names))
        assert M.actual_job_minutes(job, current=True) < maximum, "Registered cap exceeded"
    score_phase(root, dest, phase, p, names, codes, source, base, windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-plan", action="store_true")
    ap.add_argument("--phase", choices=PHASES)
    ap.add_argument("--maximum-minutes", type=float)
    args = ap.parse_args()
    from tfb.paths import project_root
    root = project_root().resolve()
    exp = Path(__file__).parents[3]/"exp/0305_s08_split_residual"
    if args.verify_plan:
        assert args.phase is None and args.maximum_minutes is None
        verify(exp, root)
        print(dict(joint_source_plan="PASS", fits=0, quality_reads=0))
        return
    S.vm_only(root)
    assert args.phase and args.maximum_minutes and 0 < args.maximum_minutes <= PHASES[args.phase]
    p = S.read(exp/"joint_protocol.json")
    deadline = datetime.fromisoformat("2026-10-06T23:59:00+00:00")
    assert args.maximum_minutes <= (deadline-datetime.now(timezone.utc)).total_seconds()/60
    parent = root/"artifacts/lane_s/0305"
    dest = parent/"diagnostics"/args.phase
    budget_path = parent/"diagnostics/active_budget.json"
    budget = S.read(budget_path)
    assert not any(a["phase"]==args.phase for a in budget["attempts"]), "Inspect previous phase; no repetition/reset"
    if args.phase == "adv":
        assert any(a["phase"]=="quick_opt" and a["state"]=="PASS" for a in budget["attempts"])
    for a in budget["attempts"]:
        status = S.read(Path('/workspace/work/jobs')/a["job"]/"status.json")
        assert status["state"]=="done" and status["exit"]==0, "Collect and reconcile prior failed job first"
        assert a["state"]=="PASS" and a["result_sha256"]==S.sha(parent/"diagnostics"/a["phase"]/"result.json")
    base, charged = M.reserve(S.read(exp/"diagnostics_protocol.json"), budget["attempts"], args.maximum_minutes)
    assert shutil.disk_usage(root).free >= 10*2**30
    assert not dest.exists(), "Partial/completed diagnostic needs inspection"
    job = M.launcher_job()
    attempt = dict(phase=args.phase, job=job, state="RUNNING", maximum_minutes=args.maximum_minutes,
                   base_actual_minutes=base, prior_diagnostic_actual_minutes=charged)
    budget["attempts"].append(attempt); S.write(budget_path, budget)
    passed = False
    try:
        verify(exp, root)
        attempt["actual_source_plan"] = "PASS"
        S.write(budget_path, budget)
        print(dict(actual_joint_source_plan="PASS", phase=args.phase), flush=True)
        import pyarrow as pa
        pa.set_cpu_count(1); pa.set_io_thread_count(1)
        from threadpoolctl import threadpool_limits
        from .jobs import owned_job
        dest.mkdir()
        with threadpool_limits(8), owned_job("0305_joint_"+args.phase, dest, eta_hours=args.maximum_minutes/60, ram_gib=23):
            run_phase(root, dest, args.phase, p, args.maximum_minutes, job)
            assert M.actual_job_minutes(job, current=True) < args.maximum_minutes
        passed = True
    finally:
        attempt.update(state="PASS" if passed else "FAIL", charged_minutes_so_far=M.actual_job_minutes(job, current=True))
        if passed:
            attempt["result_sha256"] = S.sha(dest/"result.json")
        S.write(budget_path, budget)


if __name__ == "__main__":
    main()
