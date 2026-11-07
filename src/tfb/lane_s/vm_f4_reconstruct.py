"""Rebuild the carried F4 state component on a fresh VM, without quality reads."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import shutil
import time

ID = "0301"
PHASES = {"raw": (135, 1), "fit": (90, 8), "native": (60, 2)}
HEADROOM = 10 * 2**30


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(values):
    import numpy as np
    return {key: dict(dtype=str(array.dtype), shape=list(array.shape),
                     sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
            for key, array in values.items()}


def check_plan(exp, lane, root):
    protocol = read(exp / "protocol.json")
    assert protocol["id"] == ID and protocol["purpose"] == "carried_F4_reconstruction"
    assert protocol["phases"] == {key: dict(maximum_minutes=value[0], threads=value[1])
                                  for key, value in PHASES.items()}
    assert protocol["RAM_gib"] == 7 and protocol["GPU"] is False
    assert protocol["quality_reads"] == 0 and protocol["maximum_attempts_per_phase"] == 2
    assert protocol["destination"] == "artifacts/lane_s/0301"
    for name, digest in protocol["lane_source_sha256"].items():
        assert sha(lane / name) == digest, ("Lane source differs", name)
    for name, digest in protocol["main_source_sha256"].items():
        assert sha(root / name) == digest, ("Main source differs", name)
    assert sha(exp / "reference.json") == protocol["reference_sha256"]
    reference = read(exp / "reference.json")
    assert reference["id"] == ID and reference["original_parent_id"] == "0182"
    assert "conditional_candidate_id" not in reference
    assert len(reference["fit_dates"]) == 273 and len(reference["columns"]) == 177
    assert reference["fit_dates"][0] == "2030-06-01" and reference["fit_dates"][-1] == "2031-02-28"
    assert reference["configuration"]["drop_features"] == []
    assert reference["configuration"]["num_boost_round"] == 600
    assert reference["configuration"]["parameters"]["num_threads"] == 8
    assert reference["configuration"]["parameters"]["seed"] == 4701
    assert len(reference["native_all20"]) == 20
    assert sum(case["shape"][0] for case in reference["native_all20"].values()) == 6735795
    return protocol, reference


def check_space(path, expected_bytes):
    free = shutil.disk_usage(path).free
    assert free >= expected_bytes + HEADROOM, ("Projected output violates disk headroom", free, expected_bytes)
    return free


def begin_attempt(dest, phase, maximum):
    assert math.isfinite(maximum) and maximum > 0
    path = dest / "active_budget.json"
    budget = read(path) if path.exists() else dict(id=ID, attempts=[])
    assert budget["id"] == ID
    assert not any(a["state"] == "RUNNING" for a in budget["attempts"]), "Interrupted attempt needs actual VM-status reconciliation"
    previous = [a for a in budget["attempts"] if a["phase"] == phase]
    assert not any(a["state"] == "PASS" for a in previous), "Completed phase must be collected, not relaunched"
    assert len(previous) < 2, "One build plus one repair only"
    charged = sum(a["charged_minutes"] for a in previous)
    assert maximum <= PHASES[phase][0] - charged + 1e-9, "Original phase cap exceeded"
    attempt = dict(phase=phase, state="RUNNING", started_epoch=time.time(), maximum_minutes=maximum)
    budget["attempts"].append(attempt)
    write(path, budget)
    return path, budget, attempt


def finish_attempt(path, budget, attempt, started, passed):
    attempt.update(state="PASS" if passed else "FAIL", ended_epoch=time.time(),
                   charged_minutes=(time.monotonic() - started) / 60)
    if passed:
        attempt["result_sha256"] = sha(path.parent / attempt["phase"] / "result.json")
    write(path, budget)


def require_phase(dest, phase, protocol):
    path = dest / phase / "result.json"
    result = read(path)
    assert result["id"] == ID and result["verdict"] == "PASS"
    assert result["protocol_sha256"] == protocol
    attempts = [a for a in read(dest / "active_budget.json")["attempts"] if a["phase"] == phase]
    assert attempts and attempts[-1]["state"] == "PASS", "Phase receipt lacks completed active-budget success"
    assert attempts[-1]["result_sha256"] == sha(path), "Completed phase result differs"
    assert sum(a["charged_minutes"] for a in attempts) <= PHASES[phase][0], "Cumulative phase cap exceeded"
    return result


def raw_phase(root, dest, ref, protocol_sha):
    import numpy as np
    import pandas as pd
    from tfb.data import load_cache, guard
    from tfb.paths import panels
    from tfb.split_constants import own_month_constants
    from . import q01_training_snapshot as q01
    from .build_production_travel import build_panel, select_new_windows
    from .evaluate_baselines import subset_cache
    from .features import fit_summaries
    from .instruments import apply_frozen_gaps
    from .self_supervision import observation_view

    assert sha(q01.__file__) == ref["q01_source_sha256"]
    assert sha(root / "src/tfb/split_constants.py") == ref["provider_sha256"]
    assert sha(root / "configs/date_folds.csv") == ref["date_folds_sha256"]
    folds = pd.read_csv(root / "configs/date_folds.csv")
    geometry_dates = folds.loc[folds.fold.ne(0), "date"].tolist()
    assert len(geometry_dates) == 171
    outstanding = sum(sum(case["file_size_bytes"].values()) for panel, case in ref["panels"].items()
                      if not (dest / "data" / panel / "array_parity.json").exists())
    initial_free = check_space(dest, outstanding)
    path = dest / "windows.csv"
    if not path.exists():
        frames = []
        for panel in q01.TASK2_PANELS:
            cache = load_cache(panel, "train")
            assert cache.dates == ref["fit_dates"]
            candidates = []
            for d, day in enumerate(cache.dates):
                speed = cache.speed_true[d]
                observed = np.isfinite(speed)
                grid = q01.DayGrid(panel, day, tuple(cache.stations.link_id), speed, observed,
                                  cache.elig[d], observed & (speed <= cache.stations.v_cut.to_numpy()))
                candidates.append(q01.candidate_windows(grid))
            frames.append(select_new_windows(pd.concat(candidates, ignore_index=True)))
            del cache, candidates
            gc.collect()
        windows = pd.concat(frames, ignore_index=True).sort_values(["panel", "forecast_origin", "condition"]).reset_index(drop=True)
        assert len(windows) == ref["n_production_windows"] == 710
        encoded = windows.to_csv(index=False, lineterminator="\r\n").encode()
        if hashlib.sha256(encoded).hexdigest() != ref["production_windows_sha256"]:
            encoded = windows.to_csv(index=False, lineterminator="\n").encode()
        assert hashlib.sha256(encoded).hexdigest() == ref["production_windows_sha256"]
        guard(root, path, expected_bytes=len(encoded))
        path.write_bytes(encoded)
    assert sha(path) == ref["production_windows_sha256"]
    windows = pd.read_csv(path)
    cfg = dict(ref["configuration"], training_windows_path=str(path.relative_to(root)),
               training_windows_sha256=sha(path))
    write(dest / "configuration.json", cfg)
    geometry = dest / "structural171"
    geometry.mkdir(exist_ok=True)
    geometry_hashes = {}
    records = {}
    for code, panel in enumerate(panels()):
        expected = ref["panels"][panel]
        folder = dest / "data" / panel
        if not (folder / "array_parity.json").exists():
            check_space(dest, sum(expected["file_size_bytes"].values()))
        legacy = build_panel(panel, code, ref["fit_dates"], cfg, windows, folder)
        assert legacy["n_rows"] == expected["n_rows"] and legacy["columns"] == ref["columns"]
        actual = {name: sha(folder / name) for name in expected["file_sha256"]}
        assert actual == expected["file_sha256"], ("Raw production arrays differ", panel)
        write(folder / "array_parity.json", dict(verdict="PASS", file_sha256=actual,
              original_rows=expected["n_rows"], protocol_sha256=protocol_sha))
        raw = load_cache(panel, "train")
        assert raw.dates == ref["fit_dates"]
        summary = fit_summaries(raw, geometry_dates)
        structural = {key: np.asarray(summary[key]) for key in ("distance_bottleneck_km", "bottleneck_positions_km")}
        assert fingerprint(structural) == expected["structural"], ("Geometry differs", panel)
        np.savez_compressed(geometry / (panel + "_summaries.npz"), **structural)
        geometry_hashes[panel] = sha(geometry / (panel + "_summaries.npz"))
        del summary, structural
        view = subset_cache(raw, list(range(273)))
        del raw
        apply_frozen_gaps(view, view.speed_true, view.flow_true, windows)
        view.s_windows = windows
        view = observation_view(view)
        assert not any(key.endswith("_true") for key in vars(view))
        values = own_month_constants(view, hidden=view.target | view.gap)
        assert len(values) == 9
        prior = dest / "priors" / panel
        prior.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for month, arrays in values.items():
            name = month + ".npz"
            assert fingerprint(arrays) == expected["months"][name], ("Own-month constants differ", panel, month)
            guard(root, prior, expected_bytes=sum(a.nbytes for a in arrays.values()))
            np.savez_compressed(prior / name, **arrays)
            hashes[name] = sha(prior / name)
        assert set(hashes) == set(expected["months"])
        write(prior / "manifest.json", dict(contract=dict(panel=panel, period_dates=ref["fit_dates"],
              provider_source_sha256=ref["provider_sha256"], windows_sha256=sha(path),
              parent_manifest_sha256=sha(folder / "manifest.json"), builder_source_sha256=sha(__file__),
              scope="Own UTC month, production273 gap-shaped observations only; no truth fallback"),
              file_sha256=hashes, hidden_cells=int((view.target | view.gap).sum()), truth_arrays_in_provider_view=False))
        records[panel] = dict(n_rows=legacy["n_rows"], data_manifest_sha256=sha(folder / "manifest.json"),
                             prior_manifest_sha256=sha(prior / "manifest.json"))
        print(json.dumps(dict(panel=panel, raw_arrays="EXACT", geometry171="EXACT", month_tables9="EXACT")), flush=True)
        del view, values
        gc.collect()
    write(geometry / "metadata.json", dict(fit_dates=geometry_dates, panels=panels(), summary_sha256=geometry_hashes))
    write(dest / "raw/result.json", dict(id=ID, verdict="PASS", protocol_sha256=protocol_sha,
          panels=records, windows_sha256=sha(path), initial_free_bytes=initial_free,
          n_raw_arrays=60, n_month_tables=90, quality_reads=0, fits=0))


def fit_phase(root, dest, ref, protocol_sha):
    from tfb.paths import panels
    from .local_model import train_case, matrix, save_summaries
    from .vm_f4_parent import controls, assert_categorical_recipe
    from .vm_model_numeric_fingerprint import fingerprints
    from .lgb_runtime import load_lightgbm

    raw_receipt = require_phase(dest, "raw", protocol_sha)
    for panel, record in raw_receipt["panels"].items():
        assert sha(dest / "data" / panel / "manifest.json") == record["data_manifest_sha256"]
        assert sha(dest / "priors" / panel / "manifest.json") == record["prior_manifest_sha256"]
    check_space(dest, 100 * 2**20)
    cfg = read(dest / "configuration.json")
    assert cfg["parameters"] == ref["configuration"]["parameters"]
    model = dest / "model"
    model.mkdir(exist_ok=True)
    runtime_control = model / "runtime_controls.json"
    if not runtime_control.exists():
        summaries = save_summaries(model, panels(), ref["fit_dates"], dest / "priors", dest / "structural171")
        x, y, weights, columns = matrix(dest / "data", panels(), set(ref["fit_dates"]), summaries)
        assert columns == ref["columns"]
        controls(x, y, weights, columns, cfg["parameters"], model)
        del x, y, weights, summaries
        gc.collect()
    control = read(runtime_control)
    assert control["verdict"] == "PASS" and control["categorical_indices"] == [83, 84, 87]
    train_case(model, dest / "data", panels(), ref["fit_dates"], [], cfg, "production273",
               dest / "priors", dest / "structural171")
    metadata = read(model / "metadata.json")
    assert metadata["fit_dates"] == ref["fit_dates"] and metadata["columns"] == ref["columns"]
    lgb = load_lightgbm()
    compared = {}
    for target in ("speed", "flow_lane"):
        path = model / (target + ".txt")
        case = metadata["model_cases"][target]
        assert case["best_iteration"] == 600 and case["model_sha256"] == sha(path)
        booster = lgb.Booster(model_file=str(path))
        assert booster.current_iteration() == 600
        assert_categorical_recipe(booster, ref["columns"])
        values = fingerprints(path.read_text())
        for key in ("numeric_leaf11_canonical_sha256", "numeric_leaf11_trees_sha256"):
            assert values[key] == ref["models"][target][key], ("Original F4 numeric model differs", target, key)
        compared[target] = dict(actual=values, model_sha256=sha(path),
                                original_model_sha256=ref["models"][target]["file_sha256"])
        del booster
    write(dest / "fit/result.json", dict(id=ID, verdict="PASS", protocol_sha256=protocol_sha,
          models=compared, metadata_sha256=sha(model / "metadata.json"), full600=True,
          list_categories=[83, 84, 87], all20_native_parity_pending=True, quality_reads=0,
          parent_target_fits=2, diagnostic_fits=4))


def native_phase(root, dest, ref, protocol_sha):
    import numpy as np
    import pandas as pd
    from tfb.data import guard
    from tfb.paths import panels, release_root
    from .local_predict import predict_state

    fit_receipt = require_phase(dest, "fit", protocol_sha)
    assert sha(dest / "model/metadata.json") == fit_receipt["metadata_sha256"]
    folder = dest / "native"
    folder.mkdir(exist_ok=True)
    check_space(dest, 6735795 * 8 + 20 * 4096)
    records = []
    columns = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    for split in ("validation", "private"):
        for panel in panels():
            expected = ref["native_all20"][panel + "|" + split]
            path = folder / (panel + "_" + split + ".npy")
            tick = time.monotonic()
            output = predict_state(panel, split, dict(model_dir=str(dest / "model"), prediction_threads=2))
            keys = pd.util.hash_pandas_object(output[columns], index=False).to_numpy(np.uint64)
            key_hash = hashlib.sha256(keys.tobytes()).hexdigest()
            template = release_root(root) / "task1" / panel / split / "sample_submission_state.csv"
            values = output[["speed_kmh", "flow_vph"]].to_numpy(np.float32)
            assert list(values.shape) == expected["shape"]
            assert key_hash == expected["output_key_sha256"] and sha(template) == expected["template_sha256"]
            guard(root, path, expected_bytes=values.nbytes + 4096)
            np.save(path, values)
            assert sha(path) == expected["prediction_sha256"], ("Original F4 native prediction differs", panel, split)
            record = dict(panel=panel, split=split, n_rows=len(values), prediction_sha256=sha(path),
                          output_key_sha256=key_hash, original_prediction="EXACT0",
                          runtime_min=(time.monotonic() - tick) / 60)
            write(path.with_suffix(".json"), record)
            records.append(record)
            print(json.dumps(dict(panel=panel, split=split, native="EXACT0", n_rows=len(values))), flush=True)
            del output, values, keys
            gc.collect()
    elapsed = sum(a["runtime_min"] for a in records)
    assert len(records) == 20 and sum(a["n_rows"] for a in records) == 6735795 and elapsed < 60
    write(folder / "result.json", dict(id=ID, verdict="PASS", protocol_sha256=protocol_sha,
          records=records, n_cases=20, n_rows=6735795, native_prediction_parity="EXACT0",
          total_case_runtime_min=elapsed, quality_reads=0, candidate=False,
          final_laptop_and_independent_Lead_review_pending=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--maximum-minutes", type=float)
    parser.add_argument("--verify-plan", action="store_true")
    args = parser.parse_args()
    lane = Path(__file__).resolve().parent
    exp = lane.parents[2] / "exp/0301_s_f4_fresh_host"
    from tfb.paths import project_root
    root = project_root().resolve()
    protocol, reference = check_plan(exp, lane, root)
    if args.verify_plan:
        assert args.phase is None and args.maximum_minutes is None
        print(json.dumps(dict(id=ID, verdict="PASS_SOURCE_PLAN_ONLY", VM_jobs=0, fits=0, quality_reads=0)))
        return
    assert str(root) == "/workspace/work/main", "Compute entry is VM-only"
    assert args.phase is not None and args.maximum_minutes is not None
    dest = root / protocol["destination"]
    dest.mkdir(parents=True, exist_ok=True)
    budget_path, budget, attempt = begin_attempt(dest, args.phase, args.maximum_minutes)
    started = time.monotonic()
    passed = False
    try:
        import pyarrow as pa
        from threadpoolctl import threadpool_limits
        from .jobs import owned_job
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        with threadpool_limits(PHASES[args.phase][1]), owned_job(ID + "_" + args.phase, dest / args.phase,
                                                              eta_hours=args.maximum_minutes / 60, ram_gib=7):
            {"raw": raw_phase, "fit": fit_phase, "native": native_phase}[args.phase](
                root, dest, reference, sha(exp / "protocol.json"))
            assert (time.monotonic() - started) / 60 < args.maximum_minutes, "Original active phase cap exceeded"
        passed = True
    finally:
        finish_attempt(budget_path, budget, attempt, started, passed)


if __name__ == "__main__":
    main()
