"""Fixed official-mix replay of completed S08 forecasts; no fits or model choices."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import shutil
import time

from . import split_residual as S

SEEDS = (4701, 4702, 4703)
ARMS = ("parent", "candidate", "4701", "4702", "4703")
REGIMES = ("R1", "R2", "R3")


def verify(exp, root):
    p = S.read(exp / "diagnostics_protocol.json")
    S.plan(exp, root)
    assert p["id"] == "0305" and p["recipe"] == "UNCHANGED_S08_EQUAL3_DOSE1"
    assert p["diagnostic_reserve_minutes"] == 90 and p["own_mix_maximum_minutes"] == 25
    assert p["deadline_utc"] == "2026-10-06T23:59:00Z"
    assert p["periods"] == ["validation", "private"] and p["weight_labels_used"] is False
    assert p["additional_joint_OWN_reads_per_period"] == 1
    for name, digest in p["source_sha256"].items():
        assert S.sha(exp.parents[1] / name) == digest, name
    dest = Path(root) / "artifacts/lane_s/0305"
    for rel, digest in p["completed_sha256"].items():
        assert S.sha(dest / rel) == digest, "Completed input/model/result changed: " + rel
    b = S.read(dest / "active_budget.json")
    assert [a["phase"] for a in b["attempts"]] == ["inputs", "parent", "validation", "private"]
    assert all(a["state"] == "PASS" and a["result_sha256"] == S.sha(dest/a["phase"]/"result.json") for a in b["attempts"])
    return p


def actual_job_minutes(job, current=False):
    status = S.read(Path("/workspace/work/jobs") / job / "status.json")
    if current:
        assert status["state"] == "running"
        return (time.time() - status["start"]) / 60
    assert status["state"] != "running" and "end" in status
    assert status["end"] >= status["start"]
    return (status["end"] - status["start"]) / 60


def launcher_job():
    paths = []
    for f in Path("/workspace/work/jobs").glob("S_0305_*/status.json"):
        a = S.read(f)
        if a["state"] == "running":
            meta = S.read(f.parent / "meta.json")
            assert meta["threads"] == 8 and not meta["gpu"]
            assert meta["cwd"] == "/workspace/work/tfb-wt/lane-s"
            paths.append(f.parent.name)
    assert len(paths) == 1, "Bind the one actual S0305 launcher"
    return paths[0]


def reserve(p, attempts, maximum):
    base = sum(actual_job_minutes(j) for j in p["original_jobs"])
    charged = sum(actual_job_minutes(a["job"]) for a in attempts)
    assert all(a["state"] != "RUNNING" for a in attempts), "Reconcile interrupted diagnostic before any retry"
    assert sum(a["state"] == "FAIL" for a in attempts) < 2, "One implementation repair total"
    assert base + charged + maximum <= 600 and charged + maximum <= 90
    return base, charged


def reduce_mix(data, shares, weights, groups):
    import numpy as np
    totals = np.einsum("prdsc,d->prsc", data, weights)
    n, sv, sq = (totals[..., i] for i in range(3))
    valid = n.sum(axis=-1) > 0
    assert not ((shares > 0) & (n == 0) & valid[..., None]).any(), "Missing supported stratum"
    mv = (shares * np.divide(sv, n, out=np.zeros_like(sv), where=n > 0)).sum(axis=-1)
    mq = (shares * np.divide(sq, n, out=np.zeros_like(sq), where=n > 0)).sum(axis=-1)
    scores = .54 * np.maximum(0, 1 - np.sqrt(mv) / 25)
    scores += .46 * np.maximum(0, 1 - np.sqrt(mq) / 600)
    per_panel = np.where(valid, scores, 0).mean(axis=1)
    return float(np.mean([per_panel[g].mean() for g in groups]))


def mix_comparison(data, shares, groups):
    import numpy as np
    nd = data["parent"].shape[2]
    point = {a: reduce_mix(v, shares, np.ones(nd), groups) for a, v in data.items()}
    rng = np.random.default_rng(4701)
    delta = np.empty(1000)
    for k in range(1000):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        delta[k] = reduce_mix(data["candidate"], shares, w, groups) - reduce_mix(data["parent"], shares, w, groups)
    interval = np.quantile(delta, [.025, .975]).tolist()
    members = [point[str(s)] - point["parent"] for s in SEEDS]
    d = point["candidate"] - point["parent"]
    return dict(parent_S_state=point["parent"], candidate_S_state=point["candidate"],
                delta_S_state=d, ci95=interval, member_deltas=members,
                gate=S.positive(d, interval, members), points=point, n_dates=nd, bootstrap_samples=1000)


def freeze_weights(root, items, folder):
    """Geometry only: official template targets grouped by regime and original gap."""
    import numpy as np
    from tfb.paths import cache_root
    weights = {}
    for item in items:
        path = cache_root(root) / item["panel"] / item["split"]
        meta = S.read(path / "meta.json")
        ix = [meta["dates"].index(d) for d in item["dates"]]
        arrays = {}
        for name in ("target", "gap"):
            source = path / (name + ".npy")
            assert S.sha(source) == item["source"][name + ".npy"]
            mm = np.load(source, mmap_mode="r")
            arrays[name] = np.array(mm[ix], copy=True)
            mm._mmap.close()
        reg = np.asarray(meta["regime_per_day"])[ix]
        records = {}
        for r in REGIMES:
            selected = reg == r
            target = arrays["target"][selected]
            gap = arrays["gap"][selected]
            total = int(target.sum())
            block = int((target & gap).sum())
            assert total > 0 and 0 <= block <= total
            records[r] = dict(official_targets=total, official_gap_targets=block,
                              shares=[1-block/total, block/total])
        weights[item["panel"]] = records
    S.write(folder / "weights_frozen_before_labels.json",
            dict(weight_labels_used=False, definition="Per panel/regime official target gap/non-gap cell proportions, applied to OWN block/random MSE before RMSE and equal regime/family aggregation.",
                 weights=weights))
    return weights


def replay(root, dest, period, bundle, items, weights, parent_output, maximum, start):
    import numpy as np
    from datetime import date
    from tfb.paths import panels, release_root
    from tfb.split_constants import estimate_constants
    from .own_instrument import load_item, own_context_for_day, state_rows
    from .build_lgb_data import prepare_bases
    from .travel_features import day_features
    from .local_priors import fill_scored_dynamic
    from .local_model import lgb, load_summaries
    names = panels()
    nd = len(items[0]["dates"])
    arrays = {a: np.zeros((10, 3, nd, 2, 3), dtype=np.float64) for a in ARMS}
    members = {}
    for seed in SEEDS:
        f = parent_output / period / ("seed" + str(seed))
        m = S.read(f / "metadata.json")
        assert m["source_sha256"] == S.sha(S.__file__) and m["seed"] == seed
        assert m["columns"] == bundle["metadata"]["columns"] + list(S.EXTRA)
        assert m["parameters"] == dict(S.read(Path(S.__file__).parents[3]/"exp/0305_s08_split_residual/protocol.json")["parameters"], seed=seed)
        members[seed] = {}
        for t, entry in m["models"].items():
            assert entry["rounds"] == 600 and entry["sha256"] == S.sha(f/(t+".txt"))
            members[seed][t] = lgb.Booster(model_file=str(f/(t+".txt")))
    marker = dest / period / "read_started.json"
    assert not marker.exists(), "Never repeat a started diagnostic read"
    S.write(marker, dict(period=period, original_joint_OWN_reads=1, additional_joint_diagnostic_reads=1,
                         total_joint_OWN_reads=2, model_choice=False, utc=datetime.now(timezone.utc).isoformat()))
    cols = bundle["metadata"]["columns"]
    for code, item in enumerate(items):
        assert item["panel"] == names[code]
        view, tv, tq, target, hidden = load_item(root, item)
        # Model layer consumes view only; truth arrays go directly to this scorer.
        constants = estimate_constants(view, hidden=view.target | view.gap)
        bases = prepare_bases(view, bundle["metadata"]["asm_parameters"])
        structural = load_summaries(bundle["folder"], item["panel"])
        values = {name: [a.copy() for a in bases["time"]] for name in ARMS}
        for d, day in enumerate(view.dates):
            assert (time.time() - start) / 60 < maximum
            fields, override = day_features(view, d, bases, code, own_context_for_day(view, d))
            assert list(fields) == cols
            mask = target[d]
            ti, si = np.nonzero(mask)
            x = np.column_stack([a[mask] for a in fields.values()]).astype(np.float32)
            weekend = np.full(len(ti), int(date.fromisoformat(day).weekday() >= 5))
            fill_scored_dynamic(x, cols, si, ti, weekend, override[mask], constants, structural)
            lanes = view.stations.lanes.to_numpy(np.float32)[si]
            cap = view.stations.capacity_vph.to_numpy(float)[si]
            v, qtotal = S.parent_rows(x, bundle, lanes, cap)
            qlane = qtotal / lanes
            z = S.appended(x, v, qlane)
            values["parent"][0][d][mask] = v
            values["parent"][1][d][mask] = qtotal
            mv = np.zeros(len(x), dtype=float)
            mq = mv.copy()
            for seed in SEEDS:
                models = members[seed]
                a, b = S.bound_rows(v + models["speed"].predict(z, num_threads=8),
                                    qlane + models["flow_lane"].predict(z, num_threads=8), lanes, cap)
                values[str(seed)][0][d][mask] = a
                values[str(seed)][1][d][mask] = b
                mv += a.astype(float) / 3
                mq += b.astype(float) / 3
            a, b = S.bound_total(mv, mq, cap)
            values["candidate"][0][d][mask] = a
            values["candidate"][1][d][mask] = b
        original = S.read(parent_output/period/(item["panel"]+"_rows.json"))["rows"]
        raw_check = {}
        for name, (pv, pq) in values.items():
            raw, direct = state_rows(view, tv, tq, pv, pq, target)
            assert len(raw) == len(original[name])
            for row, ref in zip(raw, original[name]):
                for k in ("panel", "date", "regime", "n"):
                    assert row[k] == ref[k]
                for k in ("sq_v", "sq_q"):
                    assert math.isclose(row[k], ref[k], rel_tol=1e-13, abs_tol=1e-8), "Unchanged prediction replay differs"
            raw_check[name] = "PASS"
            lanes = view.stations.lanes.to_numpy(float)
            for d in range(nd):
                r = REGIMES.index(str(view.regimes[d]))
                for group in (0, 1):
                    mask = target[d] & (view.gap[d] if group else ~view.gap[d])
                    dv = pv[d][mask].astype(float) - tv[d][mask].astype(float)
                    dq = (pq[d].astype(float)-tq[d].astype(float))/lanes
                    arrays[name][code, r, d, group] = [int(mask.sum()), float(np.square(dv).sum()), float(np.square(dq[mask]).sum())]
        assert np.array_equal(arrays["parent"][code,...,0], arrays["candidate"][code,...,0])
        S.write(dest/period/(item["panel"]+"_groups.json"),
                dict(panel=item["panel"],dates=item["dates"],raw_prediction_rows_match=raw_check,
                     weights=weights[item["panel"]], statistics={a: arrays[a][code].tolist() for a in ARMS}))
        print(json.dumps(dict(period=period,panel=item["panel"],unchanged_forecasts="PASS",groups=True)), flush=True)
        del view, tv, tq, target, hidden, constants, bases, structural, values
        gc.collect()
    families = {a["corridor_id"]:a["family_id"] for a in S.read(release_root(root)/"config/corridors.json")["panels"]}
    groups = [[i for i,panel in enumerate(names) if families[panel] == f] for f in sorted(set(families.values()))]
    shares = np.asarray([[weights[p][r]["shares"] for r in REGIMES] for p in names])
    result = mix_comparison(arrays, shares, groups)
    result.update(period=period, instrument="I-S-OWN official regime/block mix diagnostic",
                  reference="Original F4fit214; unchanged S08 equal3/dose1", fits=0, original_joint_OWN_reads=1,
                  additional_joint_diagnostic_reads=1, total_joint_OWN_reads=2,
                  n_targets=int(arrays["parent"][...,0].sum()), original_raw_read_not_replaced=True,
                  physics="UNKNOWN", verdict="PASS" if result["gate"]["positive"] else "NULL")
    S.write(dest/period/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-plan", action="store_true")
    parser.add_argument("--maximum-minutes", type=float)
    args = parser.parse_args()
    from tfb.paths import project_root, panels
    root = project_root().resolve()
    exp = Path(__file__).parents[3] / "exp/0305_s08_split_residual"
    p = verify(exp, root)
    if args.verify_plan:
        assert args.maximum_minutes is None
        print(json.dumps(dict(id="0305",diagnostic_source_plan="PASS",fits=0,quality_reads=0)))
        return
    S.vm_only(root)
    assert args.maximum_minutes and 0 < args.maximum_minutes <= p["own_mix_maximum_minutes"]
    deadline = datetime.fromisoformat(p["deadline_utc"].replace("Z","+00:00"))
    assert args.maximum_minutes <= (deadline-datetime.now(timezone.utc)).total_seconds()/60
    parent = root / "artifacts/lane_s/0305"
    dest = parent / "diagnostics/official_mix"
    budget_path = parent / "diagnostics/active_budget.json"
    budget = S.read(budget_path) if budget_path.exists() else dict(id="0305",reserve_minutes=90,attempts=[])
    assert not any(a["phase"] == "official_mix" for a in budget["attempts"]), "Inspect interrupted/complete replay; never repeat reads"
    base, charged = reserve(p, budget["attempts"], args.maximum_minutes)
    assert shutil.disk_usage(root).free >= 10*2**30
    job = launcher_job()
    launch = S.read(Path("/workspace/work/jobs")/job/"status.json")["start"]
    attempt = dict(phase="official_mix",job=job,state="RUNNING",maximum_minutes=args.maximum_minutes,
                   started_utc=datetime.now(timezone.utc).isoformat(),base_actual_minutes=base,prior_diagnostic_actual_minutes=charged)
    budget["attempts"].append(attempt)
    S.write(budget_path,budget)
    passed = False
    try:
        import pyarrow as pa
        pa.set_cpu_count(1); pa.set_io_thread_count(1)
        from threadpoolctl import threadpool_limits
        from .jobs import owned_job
        from .vm_s07_stages import load_manifest
        with threadpool_limits(8), owned_job("0305_official_mix",dest,eta_hours=args.maximum_minutes/60,ram_gib=23):
            bundle = S.parent_bundle(parent/"parent")
            manifest = load_manifest(root)
            all_items = {}
            frozen = {}
            # Both periods' geometry weights are written before either period's labels are opened.
            for period in p["periods"]:
                by_panel = {a["panel"]:a for a in manifest["items"] if a["period"] == period}
                items = [by_panel[x] for x in panels()]
                all_items[period] = items
                frozen[period] = freeze_weights(root,items,dest/period)
            result = {}
            for period in p["periods"]:
                result[period] = replay(root,dest,period,bundle,all_items[period],frozen[period],parent,args.maximum_minutes,launch)
            assert actual_job_minutes(job,current=True) < args.maximum_minutes
            S.write(dest/"result.json",dict(id="0305",phase="official_mix",fits=0,reports=result,
                    original_OWN_state_results_preserved=True,recipe="UNCHANGED_S08_EQUAL3_DOSE1",
                    weights_labels_used=False,physics="UNKNOWN",joint_JS_gate="PENDING_OPT_ADV",
                    protocol_sha256=S.sha(exp/"diagnostics_protocol.json"),
                    verdict="PASS" if all(x["gate"]["positive"] for x in result.values()) else "NULL"))
        passed = True
    finally:
        attempt.update(state="PASS" if passed else "FAIL",charged_minutes_so_far=actual_job_minutes(job,current=True))
        if passed:
            attempt["result_sha256"] = S.sha(dest/"result.json")
        S.write(budget_path,budget)


if __name__ == "__main__":
    main()
