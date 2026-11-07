"""Frozen observed-only OWN views, unknown block context and exact state rows."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import numpy as np
import pandas as pd
from tfb.data import VERSION
from tfb.paths import cache_root, release_root
from tfb import metric
from .build_lgb_data import sha, prepare_bases
from .gap_features import context_for_day, edge_values, extent_distance
from .features import geometry
from .travel_features import day_features
from .local_priors import fill_scored_dynamic
from .local_model import load_summaries, lgb

MANIFEST_SHA = "39f2f12433ab953af3ac8290e3abe122cc2c959da5df024969294dd921f65c99"
PROVIDER_SHA = "17cdfd1aca08a2500d6ec88a4357d130c686c047ce9cc0a838effdc312b89390"
PERIODS = ("jan_feb", "validation", "private")
OBSERVATION_NAMES = ("speed", "flow", "occ", "pct", "elig", "target", "gap", "hist", "ramp_flow")
WINDOW_COLUMNS = ("panel", "date", "forecast_origin", "condition", "gap_start", "gap_end")


def load_manifest(root):
    path = Path(root) / "artifacts/lead/0184/manifest.json"
    assert sha(path) == MANIFEST_SHA, "OWN population changed"
    return json.loads(path.read_text())


def official_windows(root, view):
    path = release_root(root) / "task2" / view.meta["panel"] / view.meta["split"] / "window_index.csv"
    if not path.exists():
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    rows = pd.read_csv(path)
    rows["panel"] = view.meta["panel"]
    rows["date"] = rows.forecast_origin.astype(str).str[:10]
    selected = []
    for row in rows.itertuples(index=False):
        if row.date not in view.dates:
            continue
        t = pd.Timestamp(row.forecast_origin)
        step = t.hour * 12 + t.minute // 5
        day = view.dates.index(row.date)
        # Train windows publish histories but do not blank their horizons.
        # Never synthesize an official gap absent from the released Task1 view.
        if view.gap[day, step+1:step+19].all():
            selected.append(dict(panel=row.panel, date=row.date,
                                 forecast_origin=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 condition=row.condition,
                                 gap_start=(t+pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 gap_end=(t+pd.Timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    return pd.DataFrame(selected, columns=WINDOW_COLUMNS)


def blind_view(view, hidden, score_target, blocks):
    """Copy observations only; hidden labels never enter any feature channel."""
    result = SimpleNamespace(meta=deepcopy(view.meta), stations=view.stations.copy(),
                             ramps=view.ramps.copy(), dates=list(view.dates),
                             regimes=np.asarray(view.regimes).copy(),
                             **{k:np.asarray(getattr(view,k)).copy() for k in OBSERVATION_NAMES})
    assert hidden.shape == score_target.shape == result.speed.shape
    assert not (hidden & (result.hist | result.gap)).any()
    assert not (score_target & (result.target | result.hist | result.gap)).any()
    assert (score_target <= hidden).all()
    result.target |= hidden
    for name in ("speed", "flow", "occ"):
        getattr(result,name)[result.target] = np.nan
    # pct is not one of the177 features; blank for future channel consumers too.
    result.pct[result.target] = 0
    artificial = []
    for block in blocks:
        d = result.dates.index(block["date"])
        origin = int(block["origin_step"])
        assert block["start_step"] == origin+1 and block["end_step"] == origin+18
        assert 12 <= origin <= 269 and hidden[d,origin+1:origin+19].all()
        assert not result.gap[d,origin+1:origin+19].any()
        result.gap[d,origin+1:origin+19] = True
        t = pd.Timestamp(block["date"],tz="UTC") + pd.Timedelta(minutes=5*origin)
        artificial.append(dict(panel=result.meta["panel"],date=block["date"],
                               forecast_origin=t.strftime("%Y-%m-%dT%H:%M:%SZ"),condition="unknown",
                               gap_start=(t+pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               gap_end=(t+pd.Timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    result.own_windows = pd.DataFrame(artificial,columns=WINDOW_COLUMNS)
    assert not any(k.endswith("_true") for k in vars(result))
    assert np.isnan(result.speed[result.target]).all()
    assert np.isnan(result.flow[result.target]).all()
    assert np.isnan(result.occ[result.target]).all()
    return result



def _rebuilt_ok(actual,digest):
    """C 2026-10-11: artifacts/cache was rebuilt (arrays EXACT, meta.json differs only by runtime_seconds); the receipt
    artifacts/lane_s/0222/prerequisites/OWN_v2/REBUILT_2026-10-11.json maps rebuilt meta digests to the pinned originals."""
    if actual==digest:
        return True
    import json
    r=Path(r"C:\Data\Projects\Money Making\work\IEEE Big Data-Traffic")/"artifacts/lane_s/0222/prerequisites/OWN_v2/REBUILT_2026-10-11.json"
    if not r.exists():
        return False
    return any(actual==f["rebuilt"] and digest==f["original"] for f in json.loads(r.read_text()).get("cache_meta",{}).values())

def load_item(root, item, hidden_poison=None):
    """Map only explicitly released inputs; do not open any *_true array."""
    root=Path(root);path=cache_root(root)/item["panel"]/item["split"]
    for name,digest in item["source"].items():
        assert _rebuilt_ok(sha(path/name),digest), "OWN source changed "+str(path/name)
    metadata=json.loads((path/"meta.json").read_text())
    assert metadata["verified"] and metadata["format_version"]==VERSION
    indices=[metadata["dates"].index(d) for d in item["dates"]]
    arrays={}
    for name in OBSERVATION_NAMES:
        data=np.load(path/(name+".npy"),mmap_mode="r")
        expected=metadata["ramp_shape"] if name.startswith("ramp_") else metadata["shape"]
        assert list(data.shape)==expected and str(data.dtype)==metadata["arrays"][name]
        arrays[name]=np.array(data[indices],copy=True)
        data._mmap.close()
    original=SimpleNamespace(meta=dict(metadata),stations=pd.read_parquet(path/"stations.parquet"),
                             ramps=pd.read_parquet(path/"ramps.parquet"),dates=list(item["dates"]),
                             regimes=np.asarray(metadata["regime_per_day"])[indices],**arrays)
    original.meta["shape"]=[len(indices),*arrays["speed"].shape[1:]]
    assert sha(root/item["path"])==item["sha256"]
    with np.load(root/item["path"],allow_pickle=False) as masks:
        hidden,target,dates=masks["hidden"],masks["target"],masks["dates"].tolist()
    assert dates==original.dates and int(target.sum())==item["n_targets"]
    assert np.isfinite(original.speed[target]).all() and np.isfinite(original.flow[target]).all()
    truth_v,truth_q=original.speed.copy(),original.flow.copy()
    windows=official_windows(root,original)
    if hidden_poison is not None:
        # Fault fixture only: alter hidden inputs before the production blinder.
        omitted=hidden|original.target
        for name in ("speed","flow","occ"):
            getattr(original,name)[omitted]=np.float32(hidden_poison)
        original.pct[omitted]=np.uint8(99)
    view=blind_view(original,hidden,target,item["blocks"])
    view.s_windows=pd.concat([windows,view.own_windows],ignore_index=True)
    return view,truth_v,truth_q,target,hidden


def own_context_for_day(view,day):
    """Observed edges of uniform gaps; condition0 and noT+30 onset assumption."""
    # The existing official onset/ongoing context remains byte-equivalent.
    official=view.s_windows[view.s_windows.condition.ne("unknown")]
    out=context_for_day(view,day,official)
    g=geometry(view);nt,ns=view.speed.shape[1:]
    for row in view.own_windows[view.own_windows.date.eq(view.dates[day])].itertuples(index=False):
        origin=pd.Timestamp(row.forecast_origin);t=origin.hour*12+origin.minute//5
        sl=slice(t+1,t+19);assert view.gap[day,sl].all()
        before_v=edge_values(view.speed[day],t,"before")
        before_q=edge_values(view.flow[day],t,"before")/g["lanes"]
        if t+19<nt:
            after_v=edge_values(view.speed[day],t+19,"after")
            after_q=edge_values(view.flow[day],t+19,"after")/g["lanes"]
            observed_after=float(np.isfinite(view.speed[day,t+19]).mean())
        elif day+1<len(view.dates) and pd.Timestamp(view.dates[day+1])-pd.Timestamp(view.dates[day])==pd.Timedelta(days=1):
            # Four frozen blocks end at23:55: the postgap edge is next00:00.
            after_v=edge_values(view.speed[day+1],0,"after")
            after_q=edge_values(view.flow[day+1],0,"after")/g["lanes"]
            observed_after=float(np.isfinite(view.speed[day+1,0]).mean())
        else:
            after_v=np.full(ns,np.nan);after_q=after_v.copy();observed_after=0.
        before_k=before_q/np.maximum(before_v,1);after_k=after_q/np.maximum(after_v,1)
        tail0,head0,queue0=extent_distance(before_v,g)
        tail1,head1,queue1=extent_distance(after_v,g)
        alpha=np.arange(1,19,dtype=float)[:,None]/19.
        out["minutes_since_gap_start"][sl]=(np.arange(18)*5)[:,None]
        out["minutes_until_gap_end"][sl]=(np.arange(17,-1,-1)*5)[:,None]
        out["window_condition"][sl]=0
        out["onset_minutes_from_T30"][sl]=np.nan
        for name,value in (("speed_before",before_v),("speed_after",after_v),
                           ("flow_lane_before",before_q),("flow_lane_after",after_q),
                           ("density_before",before_k),("density_after",after_k),
                           ("queued_before",queue0),("queued_after",queue1),
                           ("tail_distance_before_km",tail0),("tail_distance_after_km",tail1),
                           ("head_distance_before_km",head0),("head_distance_after_km",head1)):
            out[name][sl]=value
        tail=(1-alpha)*tail0+alpha*tail1;head=(1-alpha)*head0+alpha*head1
        out["tail_distance_linear_km"][sl]=tail;out["head_distance_linear_km"][sl]=head
        out["queue_linear"][sl]=np.where(np.isfinite(tail)&np.isfinite(head),(tail>=0)&(head>=0),np.nan)
        for name,a,b in (("speed_linear",before_v,after_v),("flow_lane_linear",before_q,after_q),
                         ("density_linear",before_k,after_k)):
            out[name][sl]=(1-alpha)*a+alpha*b
        out["progress"][sl]=alpha
        out["edge_observed_fraction_before"][sl]=float(np.isfinite(view.speed[day,t]).mean())
        out["edge_observed_fraction_after"][sl]=observed_after
    return out


def predict_own_arms(view,folder,panel_code,constants_by_arm,bases=None):
    """Frozen F4 boosters and177 fields; only artificial-gap context is adapted."""
    folder=Path(folder);metadata=json.loads((folder/"metadata.json").read_text())
    columns=metadata["columns"];panel=view.meta["panel"]
    assert metadata["integrity"]["model_source"]==sha(Path(__file__).parent/"local_model.py")
    assert metadata["integrity"]["prior_adapter_source_sha256"]==sha(Path(__file__).parent/"local_priors.py")
    assert sha(folder/(panel+"_summaries.npz"))==metadata["summary_sha256"][panel]
    models={k:lgb.Booster(model_file=str(folder/(k+".txt"))) for k in ("speed","flow_lane")}
    for name,model in models.items():
        assert sha(folder/(name+".txt"))==metadata["model_cases"][name]["model_sha256"]
    structural=load_summaries(folder,panel)
    if bases is None:bases=prepare_bases(view,metadata["asm_parameters"])
    outputs={name:(bases["time"][0].copy(),bases["time"][1].copy()) for name in constants_by_arm}
    lanes=view.stations.lanes.to_numpy(np.float32)
    used={name:[columns.index(c) for c in model.feature_name()] for name,model in models.items()}
    for d,date in enumerate(view.dates):
        features,override=day_features(view,d,bases,panel_code,own_context_for_day(view,d))
        assert list(features)==columns
        target=view.target[d];ti,si=np.nonzero(target)
        if not len(ti):continue
        x=np.column_stack([a[target] for a in features.values()])
        weekend=np.full(len(ti),int(__import__("datetime").date.fromisoformat(date).weekday()>=5))
        index={k:i for i,k in enumerate(columns)}
        for arm,constants in constants_by_arm.items():
            fill_scored_dynamic(x,columns,si,ti,weekend,override[target],constants,structural)
            pv,pq=outputs[arm]
            pv[d][target]=x[:,index["base_B2_speed"]]+models["speed"].predict(x[:,used["speed"]],num_threads=8)
            pq[d][target]=lanes[si]*(x[:,index["base_time_flow_lane"]]+models["flow_lane"].predict(x[:,used["flow_lane"]],num_threads=8))
    upper=1.3*view.stations.capacity_vph.to_numpy(float);upper32=upper.astype(np.float32)
    upper32=np.where(upper32.astype(float)>upper,np.nextafter(upper32,np.float32(-np.inf)),upper32)
    for pv,pq in outputs.values():
        np.clip(pv,3.,120.75,out=pv)
        np.maximum(pq,50.,out=pq);np.minimum(pq,upper32,out=pq)
        assert np.isfinite(pv[view.target]).all() and np.isfinite(pq[view.target]).all()
    return outputs


def predict_own_arrays(view,folder,panel_code,constants,bases=None):
    return predict_own_arms(view,folder,panel_code,{"candidate":constants},bases)["candidate"]


def state_rows(view,tv,tq,pv,pq,target):
    direct=metric.state_score(tv,tq,pv,pq,target=target,lanes=view.stations.lanes,regimes=view.regimes)
    assert direct["n_target_cells"]==int(target.sum())
    rows=[]
    for d,date in enumerate(view.dates):
        regime=str(view.regimes[d])
        part=metric.state_score(tv[d:d+1],tq[d:d+1],pv[d:d+1],pq[d:d+1],
                                target=target[d:d+1],lanes=view.stations.lanes,
                                regimes=[regime])["regimes"][regime]
        assert part["n_missing_predictions"]==part["n_invalid_truth"]==0
        rows.append(dict(panel=view.meta["panel"],date=date,regime=regime,
                         n=part["n_target_cells"],sq_v=part["sum_sq_speed"],sq_q=part["sum_sq_flow_per_lane"]))
    return rows,direct


def encode(rows,panels,dates):
    data=np.zeros((len(panels),3,len(dates),3))
    for row in rows:
        data[panels.index(row["panel"]),("R1","R2","R3").index(row["regime"]),dates.index(row["date"])]=[row[k] for k in ("n","sq_v","sq_q")]
    return data


def reduce_state(data,weights,groups):
    totals=np.einsum("prdc,d->prc",data,weights)
    n,sv,sq=np.moveaxis(totals,-1,0)
    score=.54*np.maximum(0,1-np.sqrt(np.divide(sv,n,out=np.zeros_like(n),where=n>0))/25)
    score+=.46*np.maximum(0,1-np.sqrt(np.divide(sq,n,out=np.zeros_like(n),where=n>0))/600)
    per_panel=np.where(n>0,score,0).mean(axis=1)
    return float(np.mean([per_panel[group].mean() for group in groups]))


def paired_state(reference,candidate,panels,dates,groups):
    a=encode(reference,panels,dates);b=encode(candidate,panels,dates)
    assert np.array_equal(a[...,0],b[...,0]),"OWN reference population mismatch"
    weights=np.ones(len(dates))
    inc=reduce_state(a,weights,groups);cand=reduce_state(b,weights,groups)
    rng=np.random.default_rng(4701);deltas=np.empty(1000);inc_scores=np.empty(1000)
    for i in range(1000):
        weights=np.bincount(rng.integers(0,len(dates),len(dates)),minlength=len(dates))
        inc_scores[i]=reduce_state(a,weights,groups)
        deltas[i]=reduce_state(b,weights,groups)-inc_scores[i]
    lo,hi=np.quantile(deltas,[.025,.975]);inc_ci=np.quantile(inc_scores,[.025,.975])
    return dict(incumbent_S_state=inc,candidate_S_state=cand,delta_S_state=cand-inc,
                ci95=[float(lo),float(hi)],n_dates=len(dates),bootstrap_samples=1000,seed=4701,
                incumbent_ci95=inc_ci.tolist(),incumbent_half_width=float((inc_ci[1]-inc_ci[0])/2),
                verdict="PASS" if lo>0 else ("FAIL" if hi<0 else "NULL"))
