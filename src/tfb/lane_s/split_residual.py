"""Fixed S08: same-split residual learning around qualified F4; VM compute only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import shutil
import time

ID = "0305"
SEEDS = (4701, 4702, 4703)
CATEGORIES = ("panel", "regime", "weekday")
PHASES = {"inputs":150, "parent":90, "validation":135, "private":135}
EXTRA = ("F4_speed", "F4_flow_lane")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def vm_only(root):
    assert str(Path(root).resolve()) == "/workspace/work/main", "S08 compute is VM-only"


def plan(exp, root):
    p = read(exp / "protocol.json")
    assert p["id"] == ID and p["features"] == 179
    assert p["seeds"] == list(SEEDS) and p["num_boost_round"] == 600
    assert p["correction_dose"] == 1 and p["aggregation"] == "equal3"
    assert p["maximum_minutes"] == 600 and p["threads"] == 8
    assert p["ram_gib"] == 23 and p["gpu"] is False
    assert p["phase_maximum_minutes"] == dict(PHASES, OPT_ADV=90)
    assert p["build_limit"] == 1 and p["repair_limit"] == 1
    assert p["deadline_utc"] == "2026-10-06T23:59:00Z"
    assert p["own_order"] == ["validation","private_if_validation_positive"]
    expected = dict(objective="regression",metric="l2",learning_rate=.05,num_leaves=31,
                    min_data_in_leaf=200,max_bin=127,feature_fraction=1.,bagging_fraction=1.,
                    bagging_freq=0,deterministic=True,force_col_wise=True,num_threads=8,
                    histogram_pool_size=256,verbosity=-1)
    assert p["parameters"] == expected, "Frozen strong recipe differs"
    assert p["mask_seeds"] == [16853,16854,16855]
    assert p["block_seeds"] == [16854,16855,16856]
    for relative, digest in p["lane_source_sha256"].items():
        assert sha(exp.parents[1] / relative) == digest, relative
    for relative, digest in p["main_source_sha256"].items():
        assert sha(Path(root) / relative) == digest, relative
    return p


def positive(delta, interval, member_deltas):
    import numpy as np
    assert len(member_deltas) == 3 and len(interval) == 2
    values = [delta, *interval, *member_deltas]
    assert np.isfinite(values).all() and interval[0] <= interval[1]
    sd = float(np.std(member_deltas, ddof=1))
    return dict(delta_S_state=float(delta), ci95=list(interval), seed_delta_sd=sd,
                positive=bool(delta > 0 and interval[0] > 0 and sd < delta / 2))


def appended(x, pv, qlane):
    import numpy as np
    assert x.ndim == 2 and x.shape[1] == 177
    assert len(pv) == len(qlane) == len(x)
    assert np.isfinite(pv).all() and np.isfinite(qlane).all()
    out = np.empty((len(x),179), dtype=np.float32)
    out[:,:177] = x
    out[:,177] = pv
    out[:,178] = qlane
    return out


def bound_total(v, total, capacity):
    import numpy as np
    assert np.isfinite(capacity).all() and (capacity * 1.3 >= 50).all()
    v = np.clip(v,3.,120.75).astype(np.float32)
    upper = capacity.astype(float) * 1.3
    upper32 = upper.astype(np.float32)
    upper32 = np.where(upper32.astype(float)>upper,
                       np.nextafter(upper32,np.float32(-np.inf)),upper32)
    total = np.minimum(np.maximum(np.asarray(total,dtype=np.float32),50.),upper32)
    assert np.isfinite(v).all() and np.isfinite(total).all()
    return v, total


def bound_rows(v, qlane, lanes, capacity):
    import numpy as np
    assert np.isfinite(lanes).all() and (lanes > 0).all()
    return bound_total(v, qlane * lanes, capacity)


def parent_bundle(folder):
    from .local_model import lgb, load_summaries
    folder = Path(folder)
    m = read(folder / "metadata.json")
    assert len(m["columns"]) == 177 and len(m["fit_dates"]) == 214
    assert m["integrity"]["model_source"] == sha(Path(__file__).parent / "local_model.py")
    assert m["integrity"]["prior_adapter_source_sha256"] == sha(Path(__file__).parent / "local_priors.py")
    models = {}
    for target in ("speed", "flow_lane"):
        item = m["model_cases"][target]
        path = folder / (target + ".txt")
        assert item["best_iteration"] == 600 and sha(path) == item["model_sha256"]
        models[target] = lgb.Booster(model_file=str(path))
        assert models[target].current_iteration() == 600
    return dict(folder=folder, metadata=m, models=models,
                indices={t:[m["columns"].index(c) for c in b.feature_name()] for t,b in models.items()})


def parent_rows(x, bundle, lanes, capacity):
    columns = bundle["metadata"]["columns"]
    models, indices = bundle["models"], bundle["indices"]
    v = x[:,columns.index("base_B2_speed")] + models["speed"].predict(x[:,indices["speed"]],num_threads=8)
    q = x[:,columns.index("base_time_flow_lane")] + models["flow_lane"].predict(x[:,indices["flow_lane"]],num_threads=8)
    return bound_rows(v,q,lanes,capacity)


def seed_data(root, dest, period, seed, bundle, items):
    import numpy as np
    from .own_instrument import load_item
    from .observed_adaptation import make_observed_adaptation
    from .observed_adaptation_features import day_levels, select_rows, feature_rows
    from .build_lgb_data import prepare_bases
    from .local_model import load_summaries
    from .vm_f4_reconstruct import check_space
    vm_only(root)
    k = SEEDS.index(seed)
    columns = bundle["metadata"]["columns"]
    folder = dest / period / ("data_seed" + str(seed))
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder / "complete.json").exists(), "Collect completed data; do not repeat"
    records = []
    for code,item in enumerate(items):
        panel = item["panel"]
        view,tv,tq,target,hidden = load_item(root,item)
        del tv,tq,target,hidden
        assert not any(n.endswith("_true") for n in vars(view))
        augmented,labels_v,labels_q,augmentation = make_observed_adaptation(view,16853+k,16854+k)
        del labels_v,labels_q
        selected = [select_rows(view,augmented,d) for d in range(len(view.dates))]
        n = sum(int(mask.sum()) for mask,w in selected)
        assert n > 0
        case = folder / panel
        case.mkdir(exist_ok=True)
        check_space(case,n*(179*4+2*4+4)+2**20)
        assert not any(case.iterdir()), "Partial data requires inspection; never replace silently"
        xout = np.lib.format.open_memmap(case / "X.npy",mode="w+",dtype=np.float32,shape=(n,179))
        yout = np.lib.format.open_memmap(case / "y.npy",mode="w+",dtype=np.float32,shape=(n,2))
        wout = np.lib.format.open_memmap(case / "w.npy",mode="w+",dtype=np.float32,shape=(n,))
        bases = prepare_bases(augmented,bundle["metadata"]["asm_parameters"])
        assert sha(bundle["folder"] / (panel+"_summaries.npz")) == bundle["metadata"]["summary_sha256"][panel]
        structural = load_summaries(bundle["folder"],panel)
        cursor = 0
        for d,date in enumerate(view.dates):
            mask,weights = selected[d]
            constants = day_levels(augmented,date)
            x,meta,override = feature_rows(augmented,d,bases,columns,code,structural,constants,mask)
            si = meta[:,2]
            lanes = view.stations.lanes.to_numpy(np.float32)[si]
            capacity = view.stations.capacity_vph.to_numpy(float)[si]
            v,total = parent_rows(x,bundle,lanes,capacity)
            q = total / lanes
            part = slice(cursor,cursor+len(x))
            xout[part] = appended(x,v,q)
            assert not (mask & view.target[d]).any()
            yout[part,0] = view.speed[d][mask] - v
            yout[part,1] = view.flow[d][mask] / lanes - q
            wout[part] = weights[mask]
            cursor += len(x)
        assert cursor == n and np.isfinite(yout).all() and np.isfinite(wout).all()
        xout.flush();yout.flush();wout.flush()
        del xout,yout,wout,bases,view,augmented,selected,structural
        files = {name:sha(case/name) for name in ("X.npy","y.npy","w.npy")}
        records.append(dict(panel=panel,n_rows=n,files=files))
        gc.collect()
        print(json.dumps(dict(phase=period,seed=seed,panel=panel,training_rows=n)),flush=True)
    write(folder / "complete.json",dict(records=records,columns=columns+list(EXTRA),seed=seed,period=period))
    return folder


def fit_seed(root, dest, period, seed, data, parameters):
    import numpy as np
    from .local_model import lgb
    from .vm_f4_reconstruct import check_space
    vm_only(root)
    spec = read(data / "complete.json")
    n = sum(r["n_rows"] for r in spec["records"])
    assert len(spec["records"]) == 10 and len(spec["columns"]) == 179
    x = np.empty((n,179),np.float32,order="F")
    y = np.empty((n,2),np.float32)
    w = np.empty(n,np.float32)
    cursor = 0
    for r in spec["records"]:
        folder = data / r["panel"]
        for name,digest in r["files"].items():
            assert sha(folder/name) == digest
        a = np.load(folder/"X.npy",mmap_mode="r")
        b = np.load(folder/"y.npy",mmap_mode="r")
        weights = np.array(np.load(folder/"w.npy",mmap_mode="r"),copy=True)
        assert len(a) == r["n_rows"]
        reg = a[:,spec["columns"].index("regime")]
        for regime in (0,1,2):
            group = reg == regime
            assert group.any()
            weights[group] *= n / (30 * float(weights[group].sum()))
        part = slice(cursor,cursor+len(a))
        x[part] = a;y[part] = b;w[part] = weights
        cursor += len(a)
        del a,b,weights
    model = dest / period / ("seed" + str(seed))
    model.mkdir(parents=True,exist_ok=True)
    assert not any(model.iterdir()), "Partial model requires inspection; never overwrite"
    check_space(model,100*2**20)
    params = dict(parameters,seed=seed)
    cases = {}
    # Reuse bins only: each target receives its own exact residual label.
    dataset = lgb.Dataset(x,label=np.ascontiguousarray(y[:,0]),weight=w,
                          feature_name=spec["columns"],categorical_feature=list(CATEGORIES),params=params,free_raw_data=True)
    dataset.construct()
    for k,target in enumerate(("speed","flow_lane")):
        dataset.set_label(np.ascontiguousarray(y[:,k]))
        booster = lgb.train(params,dataset,num_boost_round=600)
        assert booster.current_iteration() == 600
        indices = [spec["columns"].index(c) for c in CATEGORIES]
        assert "[categorical_feature: " + ",".join(map(str,indices)) + "]" in booster.model_to_string().splitlines()
        path = model / (target+".txt")
        booster.save_model(str(path),num_iteration=600)
        cases[target] = dict(sha256=sha(path),rounds=600)
        del booster
    write(model/"metadata.json",dict(id=ID,period=period,seed=seed,columns=spec["columns"],
          parameters=params,data_sha256=sha(data/"complete.json"),models=cases,source_sha256=sha(__file__)))
    del x,y,w,dataset
    gc.collect()


def score_period(root, dest, period, bundle, items):
    import numpy as np
    from datetime import date
    from tfb.paths import panels,release_root
    from .own_instrument import load_item,own_context_for_day,state_rows,paired_state
    from .build_lgb_data import prepare_bases
    from .travel_features import day_features
    from .local_priors import fill_scored_dynamic
    from .local_model import lgb,load_summaries
    from tfb.split_constants import estimate_constants
    vm_only(root)
    folder = dest/period
    assert not (folder/"read_started.json").exists(), "Started read needs inspection; never reveal twice"
    members = {}
    for seed in SEEDS:
        path = folder/("seed"+str(seed));m = read(path/"metadata.json")
        assert m["seed"] == seed and m["source_sha256"] == sha(__file__)
        assert m["columns"] == bundle["metadata"]["columns"]+list(EXTRA)
        members[seed] = {}
        for target,record in m["models"].items():
            assert sha(path/(target+".txt")) == record["sha256"] and record["rounds"] == 600
            members[seed][target] = lgb.Booster(model_file=str(path/(target+".txt")))
    write(folder/"read_started.json",dict(id=ID,period=period,OWN_read=1,utc=datetime.now(timezone.utc).isoformat()))
    rows = {name:[] for name in ("parent","candidate",*[str(s) for s in SEEDS])}
    cols = bundle["metadata"]["columns"]
    for code,item in enumerate(items):
        view,tv,tq,target,hidden = load_item(root,item)
        constants = estimate_constants(view,hidden=view.target|view.gap)
        bases = prepare_bases(view,bundle["metadata"]["asm_parameters"])
        structural = load_summaries(bundle["folder"],item["panel"])
        values = {name:[a.copy() for a in bases["time"]] for name in rows}
        for d,day in enumerate(view.dates):
            fields,override = day_features(view,d,bases,code,own_context_for_day(view,d))
            assert list(fields) == cols
            mask = view.target[d];ti,si = np.nonzero(mask)
            x = np.column_stack([a[mask] for a in fields.values()]).astype(np.float32)
            weekend = np.full(len(ti),int(date.fromisoformat(day).weekday()>=5))
            fill_scored_dynamic(x,cols,si,ti,weekend,override[mask],constants,structural)
            lanes = view.stations.lanes.to_numpy(np.float32)[si]
            capacity = view.stations.capacity_vph.to_numpy(float)[si]
            v,total = parent_rows(x,bundle,lanes,capacity)
            q = total / lanes
            z = appended(x,v,q)
            values["parent"][0][d][mask]=v;values["parent"][1][d][mask]=total
            meanv = np.zeros(len(x),np.float64);meanq = meanv.copy()
            for seed in SEEDS:
                models = members[seed]
                a,b = bound_rows(v+models["speed"].predict(z,num_threads=8),
                                 q+models["flow_lane"].predict(z,num_threads=8),lanes,capacity)
                values[str(seed)][0][d][mask]=a;values[str(seed)][1][d][mask]=b
                meanv += a.astype(float)/3;meanq += b.astype(float)/3
            a,b = bound_total(meanv,meanq,capacity)
            values["candidate"][0][d][mask]=a;values["candidate"][1][d][mask]=b
        part = {}
        for name,(v,q) in values.items():
            part[name],direct = state_rows(view,tv,tq,v,q,target)
            rows[name].extend(part[name])
        write(folder/(item["panel"]+"_rows.json"),dict(n_targets=int(target.sum()),rows=part))
        del view,tv,tq,target,hidden,constants,bases,values,structural
        gc.collect()
        print(json.dumps(dict(phase=period,panel=item["panel"],scored=True)),flush=True)
    names = panels();dates = items[0]["dates"]
    families = {a["corridor_id"]:a["family_id"] for a in read(release_root(root)/"config/corridors.json")["panels"]}
    groups = [[i for i,p in enumerate(names) if families[p]==f] for f in sorted(set(families.values()))]
    primary = paired_state(rows["parent"],rows["candidate"],names,dates,groups)
    member_deltas = [paired_state(rows["parent"],rows[str(s)],names,dates,groups)["delta_S_state"] for s in SEEDS]
    gate = positive(primary["delta_S_state"],primary["ci95"],member_deltas)
    write(folder/"result.json",dict(id=ID,period=period,comparison=primary,gate=gate,
          seeds=list(SEEDS),member_deltas=member_deltas,OWN_read=1,physics="UNKNOWN",
          n_targets=sum(r["n"] for r in rows["parent"]),verdict="PASS" if gate["positive"] else ("FAIL" if primary["verdict"]=="FAIL" else "NULL"),
          protocol_sha256=sha(Path(__file__).parents[3]/"exp/0305_s08_split_residual/protocol.json")))


def inputs_phase(root,dest,exp):
    from .vm_f4_reconstruct import check_plan,raw_phase
    from .vm_s07_own_masks import main as rebuild_own
    lane = Path(__file__).parent
    oldexp = exp.parents[1]/"exp/0301_s_f4_fresh_host"
    _,ref = check_plan(oldexp,lane,root)
    (dest/"inputs").mkdir(parents=True,exist_ok=True)
    raw_phase(root,dest/"inputs",ref,sha(exp/"protocol.json"))
    # This is regeneration of a frozen population, not another old S07 screen.
    rebuild_own()
    write(dest/"inputs/result.json",dict(id=ID,verdict="PASS",quality_reads=0,
          raw_producer="0301 qualified helper only; no native production/controller",
          OWN_producer="0222 frozen logical population regeneration; no old budget reset"))


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def require_completed(dest, budget, phase):
    attempts = [a for a in budget["attempts"] if a["phase"]==phase]
    assert attempts and attempts[-1]["state"]=="PASS", "Prior phase has no successful actual budget completion"
    assert attempts[-1]["result_sha256"] == sha(dest/phase/"result.json"), "Prior result changed"
    assert sum(a["charged_minutes"] for a in attempts) <= PHASES[phase]
    return read(dest/phase/"result.json")


def parent_phase(root,dest,exp):
    import yaml
    from tfb.paths import panels
    from .local_model import train_case
    from .vm_model_numeric_fingerprint import fingerprints
    assert read(dest/"inputs/result.json")["verdict"] == "PASS"
    reference = read(exp.parents[1]/"exp/0222_s07_vm_continuation/F4_parent214_reference.json")
    cfg = yaml.safe_load((exp.parents[1]/"exp/0182_s03_split_local/config.yaml").read_text())
    cfg["training_windows_path"] = str((dest/"inputs/windows.csv").relative_to(root))
    cfg["training_windows_sha256"] = sha(dest/"inputs/windows.csv")
    folder = dest/"parent"
    assert not (folder/"metadata.json").exists(), "Completed parent must not be refitted"
    train_case(folder,dest/"inputs/data",panels(),reference["fit_dates"],[],cfg,"fit214",
               dest/"inputs/priors",dest/"inputs/structural171")
    m = read(folder/"metadata.json")
    assert m["columns"] == reference["columns"]
    for target in ("speed","flow_lane"):
        actual = fingerprints((folder/(target+".txt")).read_text())
        expected = read(exp/"parent_numeric_reference.json")["models"][target]
        assert actual == expected["numeric"], "F4 full600 structure/numeric leaf11 differs"
        text = (folder/(target+".txt")).read_text()
        canonical = "\n".join(line for line in text.splitlines() if not line.startswith("[num_threads:"))+"\n"
        trees = text.split("Tree=0\n",1)[1].split("\nend of trees",1)[0]
        strict = dict(canonical_threads_sha256=sha_text(canonical),tree_values_sha256=sha_text(trees))
        write(folder/(target+"_reference_check.json"),dict(numeric=actual,strict=strict,
              strict_exact=all(strict[k]==reference["models"][target][k] for k in strict),
              original_model_sha256=reference["models"][target]["file_sha256"],actual_model_sha256=sha(folder/(target+".txt"))))
    write(folder/"result.json",dict(id=ID,verdict="PASS",reference="Original F4fit214",quality_reads=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",choices=PHASES)
    parser.add_argument("--maximum-minutes",type=float)
    parser.add_argument("--verify-plan",action="store_true")
    args = parser.parse_args()
    from tfb.paths import project_root
    root = project_root().resolve()
    exp = Path(__file__).parents[3]/"exp/0305_s08_split_residual"
    p = plan(exp,root)
    if args.verify_plan:
        assert args.phase is None and args.maximum_minutes is None
        print(json.dumps(dict(id=ID,status="SOURCE_PLAN_PASS",fits=0,quality_reads=0)))
        return
    vm_only(root)
    assert args.phase and args.maximum_minutes and math.isfinite(args.maximum_minutes)
    assert 0 < args.maximum_minutes <= PHASES[args.phase]
    deadline = datetime.fromisoformat(p["deadline_utc"].replace("Z","+00:00"))
    assert args.maximum_minutes <= (deadline-datetime.now(timezone.utc)).total_seconds()/60, "Phase would exceed S08 deadline"
    dest = root/"artifacts/lane_s/0305"
    dest.mkdir(parents=True,exist_ok=True)
    budget_path = dest/"active_budget.json"
    budget = read(budget_path) if budget_path.exists() else dict(id=ID,attempts=[])
    assert budget["id"] == ID and not any(a["state"]=="RUNNING" for a in budget["attempts"]), "Reconcile interrupted actual launcher charge"
    attempts = budget["attempts"]
    assert sum(a["state"]=="FAIL" for a in attempts) < 2, "At most one implementation repair"
    assert not any(a["phase"]==args.phase and a["state"]=="PASS" for a in attempts), "Completed work must be collected"
    charged = sum(a["charged_minutes"] for a in attempts)
    phase_charge = sum(a["charged_minutes"] for a in attempts if a["phase"]==args.phase)
    assert charged+args.maximum_minutes <= 600 and phase_charge+args.maximum_minutes <= PHASES[args.phase]
    if args.phase in ("parent","validation","private"):
        require_completed(dest,budget,"inputs")
    if args.phase in ("validation","private"):
        require_completed(dest,budget,"parent")
    if args.phase == "private":
        previous = require_completed(dest,budget,"validation")
        assert previous["gate"]["positive"] and previous["protocol_sha256"] == sha(exp/"protocol.json"), "March must pass before unchanged April"
    start = time.monotonic()
    attempt = dict(phase=args.phase,state="RUNNING",started_utc=datetime.now(timezone.utc).isoformat(),maximum_minutes=args.maximum_minutes)
    attempts.append(attempt);write(budget_path,budget)
    passed = False
    try:
        import pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        from threadpoolctl import threadpool_limits
        from .jobs import owned_job
        with threadpool_limits(8), owned_job(ID+"_"+args.phase,dest/args.phase,eta_hours=args.maximum_minutes/60,ram_gib=23):
            if args.phase == "inputs":
                inputs_phase(root,dest,exp)
            elif args.phase == "parent":
                parent_phase(root,dest,exp)
            else:
                assert read(dest/"parent/result.json")["verdict"] == "PASS"
                from .vm_s07_stages import load_manifest
                from tfb.paths import panels
                items_by_panel = {a["panel"]:a for a in load_manifest(root)["items"] if a["period"]==args.phase}
                items = [items_by_panel[name] for name in panels()]
                bundle = parent_bundle(dest/"parent")
                for seed in SEEDS:
                    model_path = dest/args.phase/("seed"+str(seed))/"metadata.json"
                    data = dest/args.phase/("data_seed"+str(seed))
                    if model_path.exists():
                        finished = read(model_path)
                        assert finished["source_sha256"] == sha(__file__) and finished["seed"] == seed
                        assert finished["parameters"] == dict(p["parameters"],seed=seed)
                        assert finished["data_sha256"] == sha(data/"complete.json")
                        for target,record in finished["models"].items():
                            assert record["rounds"] == 600 and record["sha256"] == sha(model_path.parent/(target+".txt"))
                        continue
                    if not (data/"complete.json").exists():
                        data = seed_data(root,dest,args.phase,seed,bundle,items)
                    fit_seed(root,dest,args.phase,seed,data,p["parameters"])
                score_period(root,dest,args.phase,bundle,items)
            assert (time.monotonic()-start)/60 < args.maximum_minutes, "Active phase cap exceeded"
        passed = True
    finally:
        attempt.update(state="PASS" if passed else "FAIL",charged_minutes=(time.monotonic()-start)/60)
        if passed:
            attempt["result_sha256"] = sha(dest/args.phase/"result.json")
        write(budget_path,budget)


if __name__ == "__main__":
    main()
