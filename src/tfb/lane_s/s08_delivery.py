"""C's single frozen S08 prediction build: no ADV or quality scoring.

The user explicitly authorized this narrow build locally on 2026-10-04.
Local fitting stays at CPU8/RAM7; --reproduce does inference only.
The original S0305 controllers, protocols, budgets and read markers are untouched.
"""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import time

from . import split_residual as S
from . import s08_joint_diagnostics as J

DEST = "artifacts/lane_s/0398_s08"
KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
SCIENTIFIC_SHA = "d63a8080bc0ae3522c78381c6dcd2419f4da18ac85a2c356d28dff0332ffd53e"
JOINT_SHA = "0de20d5de69c28aa057d0a791178da26d1a5af5b097a5ef9842bd258e5133fa2"


def recipe(root):
    exp = Path(__file__).parents[3] / "exp/0305_s08_split_residual"
    p = S.read(exp / "protocol.json")
    j = S.read(exp / "joint_protocol.json")
    assert S.sha(S.__file__) == SCIENTIFIC_SHA and S.sha(J.__file__) == JOINT_SHA
    assert p["seeds"] == list(S.SEEDS) == [4701, 4702, 4703]
    assert p["features"] == 179 and p["num_boost_round"] == 600
    assert p["correction_dose"] == 1 and p["aggregation"] == "equal3"
    assert p["mask_seeds"] == [16853, 16854, 16855]
    assert p["block_seeds"] == [16854, 16855, 16856]
    assert j["parameters"] == p["parameters"]
    for rel, digest in j["source_sha256"].items():
        assert S.sha(exp.parents[1] / rel) == digest, rel
    for rel, digest in j["main_source_sha256"].items():
        assert S.sha(root / rel) == digest, rel
    return exp, p, j


def observations(root, panel, split, dates=None):
    """Observation-only cache loading, including an explicit hidden-channel blinder."""
    import numpy as np
    import pandas as pd
    from tfb.paths import cache_root
    from tfb.data import VERSION
    from .own_instrument import OBSERVATION_NAMES, WINDOW_COLUMNS, official_windows
    folder = cache_root(root) / panel / split
    meta = S.read(folder / "meta.json")
    assert meta["verified"] and meta["format_version"] == VERSION
    assert meta["panel"] == panel and meta["split"] == split
    dates = list(meta["dates"] if dates is None else dates)
    assert dates == sorted(set(dates)) and dates
    ix = [meta["dates"].index(day) for day in dates]
    arrays = {}
    for name in OBSERVATION_NAMES:
        path = folder / (name + ".npy")
        with_array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = meta["ramp_shape"] if name.startswith("ramp_") else meta["shape"]
        assert list(with_array.shape) == expected and str(with_array.dtype) == meta["arrays"][name]
        arrays[name] = np.array(with_array[ix], copy=True)
        with_array._mmap.close()
    view = SimpleNamespace(meta=dict(meta), stations=pd.read_parquet(folder / "stations.parquet"),
                           ramps=pd.read_parquet(folder / "ramps.parquet"), dates=dates,
                           regimes=np.asarray(meta["regime_per_day"])[ix], **arrays)
    view.meta["shape"] = list(view.speed.shape)
    view.meta["ramp_shape"] = list(view.ramp_flow.shape)
    for name in ("speed", "flow", "occ"):
        getattr(view, name)[view.target | view.gap] = np.nan
    view.pct[view.target | view.gap] = 0
    view.s_windows = official_windows(root, view)
    view.own_windows = pd.DataFrame(columns=WINDOW_COLUMNS)
    J.assert_observation_view(view)
    return view


def logical(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def own_masks(view, cfg):
    """Exact original OWN mask draw; no JanFeb generation and no label scoring."""
    import numpy as np
    observed = (view.elig & np.isfinite(view.speed) & np.isfinite(view.flow)
                & (view.speed >= 1) & (view.flow >= 0) & ~view.target)
    protected = view.hist | view.gap
    candidates = observed & ~protected
    suffix = view.meta["panel"] + "|" + view.meta["split"]
    seed = cfg["seed"] + int(hashlib.sha256(suffix.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    hidden = np.zeros(view.speed.shape, bool)
    for regime in ("R1", "R2", "R3"):
        rows = np.flatnonzero((candidates & (view.regimes == regime)[:, None, None]).reshape(-1))
        selected = rng.choice(rows, size=int(np.floor(cfg["random_fraction"] * len(rows))), replace=False)
        hidden.reshape(-1)[selected] = True
    blocks = []
    for d, day in enumerate(view.dates):
        choices = [t for t in range(12, 270) if not protected[d, t+1:t+19].any()]
        assert choices
        origin = int(rng.choice(choices))
        hidden[d, origin+1:origin+19] = True
        blocks.append(dict(date=day, origin_step=origin, start_step=origin+1, end_step=origin+18))
    target = hidden & candidates
    assert not (hidden & protected).any() and not (target & view.target).any()
    return hidden, target, blocks


def own_view(root, panel, split):
    import numpy as np
    import pandas as pd
    from .own_instrument import blind_view
    view = observations(root, panel, split)
    cfgpath = root / "exp/0184_l_s_own_instrument/config.yaml"
    reference = S.read(Path(__file__).parents[3] / "exp/0222_s07_vm_continuation/OWN_mask_reference.json")
    assert S.sha(cfgpath) == reference["config_sha256"]
    hidden, target, blocks = own_masks(view, S.read(cfgpath))
    expected = reference["items"][panel + "|" + split]["arrays"]
    assert {k: logical(a) for k, a in dict(hidden=hidden, target=target,
                                        dates=np.asarray(view.dates)).items()} == expected
    result = blind_view(view, hidden, target, blocks)
    result.s_windows = pd.concat([view.s_windows, result.own_windows], ignore_index=True)
    J.assert_observation_view(result)
    return result, target


def template_frame(root, view, values):
    import numpy as np
    import pandas as pd
    from tfb.data import indices
    from tfb.paths import release_root
    path = release_root(root) / "task1" / view.meta["panel"] / view.meta["split"] / "sample_submission_state.csv"
    out = pd.read_csv(path, usecols=KEYS, dtype=str)[KEYS]
    assert len(out) == int(view.target.sum()) and not out.duplicated(KEYS).any()
    key = indices(out, {d: i for i, d in enumerate(view.dates)},
                  dict(zip(view.stations.station_id, view.stations.idx)), "station_id")
    assert view.target[key].all() and out.panel.eq(view.meta["panel"]).all()
    station = view.stations.set_index("station_id")
    assert np.array_equal(out.link_id, out.station_id.map(station.link_id))
    assert np.array_equal(out.mask_regime, view.regimes[key[0]])
    out["speed_kmh"] = values[0][key].astype(float)
    out["flow_vph"] = values[1][key].astype(float)
    assert np.isfinite(out[["speed_kmh", "flow_vph"]]).all().all()
    assert out.speed_kmh.between(3, 120.75).all() and out.flow_vph.ge(50).all()
    assert (out.flow_vph <= 1.3 * out.station_id.map(station.capacity_vph)).all()
    return out


def saved_parents(root, exp):
    """Reuse the qualified frozen parents; never repeat successful fits."""
    from .vm_model_numeric_fingerprint import fingerprints
    refs = {273: S.read(exp.parents[1] / "exp/0301_s_f4_fresh_host/reference.json"),
            214: S.read(exp / "parent_numeric_reference.json"),
            171: S.read(exp / "opt_parent_reference.json")}
    folders = {273: root / "artifacts/lane_s/0182/production273/model",
               214: root / "artifacts/lane_s/0305/parent",
               171: root / "artifacts/lane_s/0305/diagnostics/opt_parent"}
    for count, folder in folders.items():
        base = J.bundle(folder, count)
        assert len(base["metadata"]["panels"]) == 10
        for target in ("speed", "flow_lane"):
            actual = fingerprints((folder / (target + ".txt")).read_text())
            expected = refs[count]["models"][target]
            if count == 273:
                for key in ("numeric_leaf11_canonical_sha256", "numeric_leaf11_trees_sha256"):
                    assert actual[key] == expected[key]
            else:
                assert actual == expected["numeric"]
        del base
    return folders


def saved_own_members(root, split, base, parameters):
    """Read the retained original S0305 full-ten-panel members without refitting."""
    from .local_model import lgb
    result = {}
    folder = root / "artifacts/lane_s/0305" / split
    for seed in S.SEEDS:
        case = folder / ("seed" + str(seed))
        meta = S.read(case / "metadata.json")
        assert meta["seed"] == seed and meta["period"] == split
        assert meta["source_sha256"] == SCIENTIFIC_SHA
        assert meta["columns"] == base["metadata"]["columns"] + list(S.EXTRA)
        assert meta["parameters"] == dict(parameters, seed=seed)
        complete = folder / ("data_seed" + str(seed)) / "complete.json"
        assert S.sha(complete) == meta["data_sha256"]
        data = S.read(complete)
        assert len(data["records"]) == 10 and data["columns"] == meta["columns"]
        result[seed] = {}
        for target in ("speed", "flow_lane"):
            path = case / (target + ".txt")
            assert meta["models"][target]["rounds"] == 600
            assert S.sha(path) == meta["models"][target]["sha256"]
            model = lgb.Booster(model_file=str(path))
            assert model.current_iteration() == 600 and model.feature_name() == meta["columns"]
            result[seed][target] = model
    return result


def train_group(root, folder, group, base, source, p, names):
    codes = dict(zip(names, range(len(names))))
    folder.mkdir(parents=True)
    for seed in S.SEEDS:
        data = J.write_seed_data(root, folder, group, seed, base, names, codes, source)
        J.fit_member(folder, group, seed, data, p["parameters"], 10)
    return J.load_members(folder, group, base["metadata"]["columns"], p["parameters"], 10)


def export_group(root, dest, group, base, members, source, names):
    import numpy as np
    import pandas as pd
    frames, records = [], []
    is_own = group.startswith("own_")
    for code, panel in enumerate(names):
        loaded = source(panel)
        view, target = loaded if is_own else (loaded, loaded.target)
        values = J.predict_arms(view, base, members, code)["candidate"]
        if group in ("validation", "private"):
            frame = template_frame(root, view, values)
            frames.append(frame)
            record = dict(panel=panel, rows=len(frame))
        else:
            path = dest / group / (panel + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists()
            np.savez_compressed(path, speed=values[0], flow=values[1], target=target,
                                dates=np.asarray(view.dates), station_id=view.stations.station_id.to_numpy(str))
            record = dict(panel=panel, path=str(path.relative_to(root)), sha256=S.sha(path), targets=int(target.sum()))
        if code == 0:
            repeated = J.predict_arms(view, base, members, code)["candidate"]
            assert all(np.array_equal(a[target], b[target]) for a, b in zip(values, repeated))
            record["repeated_inference_max_abs"] = 0.0
        records.append(record)
        del view, values, loaded
        gc.collect()
    if frames:
        path = dest / ("official_" + group + ".parquet")
        assert not path.exists()
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return records


def build(root, deadline, local=False):
    import pyarrow as pa
    from threadpoolctl import threadpool_limits
    from tfb.paths import panels
    if local:
        assert __import__("os").name == "nt", "User authorized the current Windows laptop"
    else:
        S.vm_only(root)
    assert datetime.now(timezone.utc) < deadline
    exp, p, j = recipe(root)
    dest = root / DEST
    assert not dest.exists(), "This one build must not overwrite an earlier attempt"
    dest.mkdir(parents=True)
    names = panels(root)
    assert len(names) == 10
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    start = time.monotonic()
    records = {}
    from .jobs import owned_job
    import threading
    import os
    done = threading.Event()
    def deadline_watch():
        while not done.wait(1):
            if datetime.now(timezone.utc) >= deadline:
                print("S08 six-hour deadline reached; preserving completed outputs", flush=True)
                os._exit(124)
    watcher = threading.Thread(target=deadline_watch, daemon=True)
    watcher.start()
    job_name = "0400_S08_local" if local else "0400_S08_delivery"
    with owned_job(job_name, dest, eta_hours=max(0, (deadline-datetime.now(timezone.utc)).total_seconds()/3600),
                   ram_gib=7 if local else 23), threadpool_limits(8):
        if local:
            import psutil
            process = psutil.Process()
            process.cpu_affinity(process.cpu_affinity()[:8])
        parents = saved_parents(root, exp)
        for group in ("validation", "private", "opt", "own_validation", "own_private"):
            assert datetime.now(timezone.utc) < deadline
            count = 171 if group == "opt" else (214 if group.startswith("own_") else 273)
            base = J.bundle(parents[count], count)
            if group == "opt":
                source = lambda panel: observations(root, panel, "train", j["OPT_dates"])
                assert len(j["OPT_dates"]) == 43 and not set(j["OPT_dates"]) & set(j["OPT_fit_dates"])
            elif group.startswith("own_"):
                split = group.removeprefix("own_")
                source = lambda panel, split=split: own_view(root, panel, split)
            else:
                source = lambda panel, split=group: observations(root, panel, split)
            training_source = ((lambda panel, s=source: s(panel)[0]) if group.startswith("own_") else source)
            if group.startswith("own_"):
                members = saved_own_members(root, split, base, p["parameters"])
            else:
                members = train_group(root, dest / "models" / group, group, base, training_source, p, names)
            print(dict(group=group, stage="prediction_export"), flush=True)
            records[group] = export_group(root, dest, group, base, members, source, names)
            del base, members
            gc.collect()
    done.set(); watcher.join()
    assert datetime.now(timezone.utc) <= deadline
    S.write(dest / "result.json", dict(id="0400", purpose="C requested frozen S08 prediction delivery",
            records=records, parents={str(k): str(v.relative_to(root)) for k, v in parents.items()},
            host="local" if local else "vm", reused_OWN_models=True, new_parent_fits=0,
            seeds=list(S.SEEDS), rounds=600, features=179, dose=1,
            aggregation="equal3", new_quality_reads=0, ADV_jobs=0,
            duration_minutes=(time.monotonic()-start)/60,
            source_sha256=S.sha(__file__), scientific_source_sha256=SCIENTIFIC_SHA,
            joint_source_sha256=JOINT_SHA, deadline_utc=deadline.isoformat()))


def reproduce(root):
    import numpy as np
    import pandas as pd
    from tfb.paths import panels
    exp, p, _ = recipe(root)
    dest = root / DEST
    panel = panels(root)[0]
    parent = S.read(dest / "result.json")["parents"]["273"]
    base = J.bundle(root / parent, 273)
    members = J.load_members(dest / "models/validation", "validation", base["metadata"]["columns"], p["parameters"], 10)
    view = observations(root, panel, "validation")
    values = J.predict_arms(view, base, members, 0)["candidate"]
    out = template_frame(root, view, values)
    saved = pd.read_parquet(dest / "official_validation.parquet", filters=[("panel", "==", panel)])
    assert out[KEYS].equals(saved[KEYS].reset_index(drop=True))
    difference = float(np.max(np.abs(out[["speed_kmh", "flow_vph"]].to_numpy() - saved[["speed_kmh", "flow_vph"]].to_numpy())))
    assert difference <= 1e-6
    print(dict(panel=panel, rows=len(out), prediction_max_abs=difference, fits=0, quality_reads=0), flush=True)


def main():
    from tfb.paths import project_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-utc")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    root = project_root().resolve()
    if args.reproduce:
        assert args.deadline_utc is None and not args.local
        reproduce(root)
    else:
        assert args.deadline_utc is not None
        deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        assert deadline.tzinfo is not None
        build(root, deadline, local=args.local)


if __name__ == "__main__":
    main()
