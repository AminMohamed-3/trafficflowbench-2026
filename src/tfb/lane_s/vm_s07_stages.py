"""Portable S07 stages with freshly regenerated, explicitly qualified VM inputs."""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
import yaml
from tfb.paths import project_root,panels
from tfb.data import load_cache
from .build_lgb_data import sha
from .evaluate_baselines import subset_cache
from .self_supervision import observation_view
from .own_instrument import load_item,OBSERVATION_NAMES,WINDOW_COLUMNS
from .instruments import apply_frozen_gaps
PHASES=("OWN1_jan_feb","OWN1_validation","OWN1_private","ADV1","OWN2_validation","OWN2_private")
CONFIG_SHA="89801f044b335ca5ab7cc7ba8cefdf99286c2195054e7e7c9f1137d01217658a"
def read(path):return json.loads(Path(path).read_text())
def base():return project_root()/"artifacts/lane_s/0222"
def load_manifest(root):
    folder=Path(root)/"artifacts/lane_s/0222/prerequisites/OWN_v2"
    result=read(folder/"result.json");manifest=read(folder/"manifest.json")
    assert result["verdict"]=="PASS" and result["manifest_sha256"]==sha(folder/"manifest.json")
    assert manifest["comparison"]=="EXACT_LOGICAL_POPULATION" and len(manifest["items"])==30
    return manifest
def spec(phase):
    assert phase in PHASES
    root=project_root();lane=Path(__file__).parent;wt=lane.parents[2]
    config=wt/"exp/0189_s07_split_local/config.yaml";assert sha(config)==CONFIG_SHA
    cfg=read(config);production=phase.startswith("OWN2");days=273 if production else 214
    if production:
        gate=read(base()/"transfer_gate/result.json");assert gate["numeric_transfer_gate"]=="PASS"
    parent=base()/f"F4_parent{days}_v2";metadata=read(parent/"metadata.json");fit=metadata["fit_dates"]
    assert len(fit)==days and len(metadata["columns"])==177
    assert all(a["best_iteration"]==600 for a in metadata["model_cases"].values())
    assert read(parent/"laptop_parent_parity.json")["verdict"]=="PASS"
    assert read(parent/"runtime_controls.json")["verdict"]=="PASS"
    period="jan_feb" if phase=="ADV1" else phase.split("_",1)[1]
    items={a["panel"]:a for a in load_manifest(root)["items"] if a["period"]==period}
    assert set(items)==set(panels());dates=next(iter(items.values()))["dates"]
    assert all(a["dates"]==dates for a in items.values())
    assert max(fit)<min(dates) and not set(fit)&set(dates)
    prereq=base()/"prerequisites";window_path=None;windows=None
    if phase=="ADV1":
        window_path=prereq/"ADV/windows.csv"
        instruments=yaml.safe_load((root/"configs/instruments.yaml").read_text(encoding="utf-8-sig"))
        assert sha(window_path)==instruments["I-S-ADV"]["windows"]["sha256"]
        assert read(prereq/"ADV/result.json")["verdict"]=="PASS"
        windows=pd.read_csv(window_path);assert windows.date.isin(dates).all()
    return dict(phase=phase,root=root,cfg=cfg,config=config,parent=parent,metadata=metadata,fit_dates=fit,dates=dates,items=items,original=prereq/f"data{days}",priors=prereq/f"priors{days}",observed=base()/f"data_{phase}",windows=windows,windows_path=window_path)

def source_view(stage,panel,hidden_poison=None):
    root=stage["root"];item=stage["items"][panel]
    own,tv,tq,target,hidden=load_item(root,item,hidden_poison=hidden_poison)
    del tv,tq,target
    if stage["phase"]!="ADV1":
        assert not any(name.endswith("_true") for name in vars(own))
        assert np.isnan(own.speed[hidden]).all() and np.isnan(own.flow[hidden]).all()
        return own,dict(hidden_OWN_cells=int(hidden.sum()),simulated_history_OWN_overlap=0,
            source="Same released OWN-blinded period; frozen hidden labels/features remain unavailable")
    raw=load_cache(panel,"train");cache=subset_cache(raw,[raw.day_index(day) for day in stage["dates"]])
    for value in raw.arrays.values():
        if isinstance(value,np.memmap):value._mmap.close()
    del raw;gc.collect()
    assert cache.dates==own.dates
    if hidden_poison is not None:
        # In-memory fault fixture, including the raw values that synthetic histories would copy.
        for name in ("speed","flow","occ"):
            getattr(cache,name)[hidden & ~cache.target]=np.float32(hidden_poison)
        for name in ("speed_true","flow_true","occ_true"):
            getattr(cache,name)[hidden]=np.float32(hidden_poison)
        cache.pct[hidden & ~cache.target]=99
        assert np.isnan(cache.speed[cache.target]).all() and np.isnan(cache.flow[cache.target]).all()
    apply_frozen_gaps(cache,cache.speed_true,cache.flow_true,stage["windows"])
    cache.s_windows=stage["windows"];overlap=int((cache.hist&hidden).sum())
    view=observation_view(cache);del cache
    # Reblind AFTER simulated-history restoration, including all original OWN omissions.
    view.target|=hidden
    for name in ("speed","flow","occ"):getattr(view,name)[view.target|view.gap]=np.nan
    view.pct[view.target|view.gap]=0
    view.own_windows=own.own_windows.copy()
    for row in view.own_windows.itertuples(index=False):
        origin=pd.Timestamp(row.forecast_origin);d=view.dates.index(row.date);step=origin.hour*12+origin.minute//5
        view.gap[d,step+1:step+19]=True
    view.s_windows=pd.concat([view.s_windows.reindex(columns=WINDOW_COLUMNS),view.own_windows],ignore_index=True)
    assert np.isnan(view.speed[hidden]).all() and np.isnan(view.flow[hidden]).all()
    assert np.isnan(view.occ[hidden]).all() and not view.pct[hidden].any()
    assert not any(name.endswith("_true") for name in vars(view))
    del own;gc.collect()
    return view,dict(hidden_OWN_cells=int(hidden.sum()),simulated_history_OWN_overlap=overlap,
        source="ADV-observed inputs; frozen OWN cells reblinded after surrogate history fill; no hidden labels")
