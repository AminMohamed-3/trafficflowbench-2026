"""Rebuild the fixed qualified F4 parent from regenerated VM training inputs."""
from pathlib import Path
import argparse,gc,json,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from tfb.paths import project_root,panels
from tfb.data import guard
from .local_model import matrix,lgb
from .build_lgb_data import sha
from .features import CATEGORICAL
from .jobs import owned_job

def write(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False)+"\n")

def assert_categorical_recipe(model,columns):
    indices=sorted(columns.index(name) for name in CATEGORICAL)
    assert indices==[83,84,87],("Original categorical feature order differs",indices)
    line="[categorical_feature: "+",".join(map(str,indices))+"]"
    assert line in model.model_to_string().splitlines(),"Original three categorical columns missing"
    return indices

def controls(X,y,w,columns,parameters,folder):
    x=np.asfortranarray(X[:210001]);labels=y[:len(x)].copy();weights=w[:len(x)].copy()
    assert len(x)==210001
    ordinary=[]
    for k in (0,1):
        ds=lgb.Dataset(x,label=labels[:,k],weight=weights,feature_name=columns,categorical_feature=list(CATEGORICAL),params=parameters,free_raw_data=True)
        model=lgb.train(parameters,ds,num_boost_round=32)
        assert_categorical_recipe(model,columns)
        ordinary.append((model.model_to_string(),model.predict(x[:4096],num_threads=3)))
        del model,ds;gc.collect()
    ds=lgb.Dataset(x,label=labels[:,0],weight=weights,feature_name=columns,categorical_feature=list(CATEGORICAL),params=parameters,free_raw_data=True)
    ds.construct();assert ds.data is None
    for k in (0,1):
        ds.set_label(labels[:,k]);model=lgb.train(parameters,ds,num_boost_round=32)
        assert_categorical_recipe(model,columns)
        assert model.model_to_string()==ordinary[k][0]
        np.testing.assert_array_equal(model.predict(x[:4096],num_threads=3),ordinary[k][1])
        del model;gc.collect()
    del ds,x,labels,weights,ordinary;gc.collect()
    proof=dict(verdict="PASS",rows=210001,rounds=32,control_fits=4,features=177,categorical_indices=[83,84,87],categorical_argument_type="list",scope="Actual fully populated original F4 monthly/structural features and residual labels; ordinary vs reused bins exact model text and prediction",source_sha256=sha(__file__),native_library_sha256=sha(lgb.basic._LIB._name),OWN_reads=0,ADV_reads=0)
    write(folder/"runtime_controls.json",proof)
    return proof

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--days",type=int,choices=[214,273],default=214);args=parser.parse_args()
    root=project_root();wt=Path(__file__).parents[3]
    cfgpath=wt/"exp/0182_s03_split_local/config.yaml"
    original=yaml.safe_load(cfgpath.read_text());cfg=dict(original,parameters=dict(original["parameters"],num_threads=3))
    assert cfg["num_boost_round"]==600 and cfg["drop_features"]==[]
    assert cfg["parameters"]["seed"]==4701 and cfg["parameters"]["bagging_fraction"]==1.
    dates=pd.date_range("2030-06-01","2030-12-31" if args.days==214 else "2031-02-28").strftime("%Y-%m-%d").tolist()
    assert len(dates)==args.days
    prereq=root/"artifacts/lane_s/0222/prerequisites"
    data=prereq/f"data{args.days}";priors=prereq/f"priors{args.days}";structure=prereq/"structural171"
    folder=root/f"artifacts/lane_s/0222/F4_parent{args.days}_v2";folder.mkdir(parents=True,exist_ok=True)
    selected=panels();records={}
    for panel in selected:
        a=json.loads((data/panel/"manifest.json").read_text())
        assert a["comparison"]=="EXACT" and a["contract"]["dates"]==dates
        for name,digest in a["file_sha256"].items():assert sha(data/panel/name)==digest
        b=json.loads((priors/panel/"manifest.json").read_text())
        assert b["verdict"]=="PASS" and b["structural"]=="EXACT"
        for name,r in b["months"].items():assert sha(priors/panel/name)==r["file_sha256"]
        records[panel]=dict(data_manifest_sha256=sha(data/panel/"manifest.json"),prior_manifest_sha256=sha(priors/panel/"manifest.json"),structural_sha256=sha(structure/f"{panel}_summaries.npz"),n_rows=a["n_rows"])
    contract=dict(id="0222",host="vm",scientific_parent_id="0182",fit_dates=dates,original_config_sha256=sha(cfgpath),configuration=cfg,actual_parameters=cfg["parameters"],records=records,source_sha256={n:sha(Path(__file__).parent/n) for n in ["vm_f4_parent.py","local_model.py","local_priors.py","lgb_runtime.py","features.py","travel_features.py","characteristics.py","build_lgb_data.py"]},qualification_reference_sha256=sha(wt/"exp/0222_s07_vm_continuation/F4_parent214_reference.json") if args.days==214 else None,native_library_sha256=sha(lgb.basic._LIB._name),runtime_only_change="3 CPU cores instead of original8, deterministic native control before complete600 refit; original geometry, sampling, levels, weights and labels unchanged")
    cp=folder/"contract.json"
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:write(cp,contract)
    if (folder/"metadata.json").exists():
        a=json.loads((folder/"metadata.json").read_text());assert a["integrity"]==contract
        for target,r in a["model_cases"].items():assert r["best_iteration"]==600 and r["model_sha256"]==sha(folder/f"{target}.txt")
        return
    reference=json.loads((wt/"exp/0222_s07_vm_continuation/F4_parent214_reference.json").read_text()) if args.days==214 else None
    if reference is not None:assert reference["fit_dates"]==dates
    start=time.monotonic()
    with owned_job(f"0222_F4_parent{args.days}_v2",folder,eta_hours=1.5,ram_gib=7),threadpool_limits(3):
        summaries={}
        for panel in selected:
            b=json.loads((priors/panel/"manifest.json").read_text())
            monthly={}
            for name in b["months"]:
                with np.load(priors/panel/name) as arrays:monthly[name.removesuffix(".npz")]={k:arrays[k].copy() for k in arrays.files}
            with np.load(structure/f"{panel}_summaries.npz") as arrays:structural={k:arrays[k].copy() for k in arrays.files}
            summaries[panel]=dict(monthly=monthly,structural=structural)
            np.savez_compressed(folder/f"{panel}_summaries.npz",**structural)
        X,y,w,columns=matrix(data,selected,set(dates),summaries)
        assert len(columns)==177 and len(X)==sum(r["n_rows"] for r in records.values())
        control=controls(X,y,w,columns,cfg["parameters"],folder)
        complete={}
        for k,target in enumerate(["speed","flow_lane"]):
            checkpoint=folder/f"{target}_complete.json"
            if checkpoint.exists():
                r=json.loads(checkpoint.read_text());assert r["best_iteration"]==600 and r["model_sha256"]==sha(folder/f"{target}.txt");complete[target]=r;continue
            ds=lgb.Dataset(X,label=np.ascontiguousarray(y[:,k]),weight=w,feature_name=columns,categorical_feature=list(CATEGORICAL),free_raw_data=True)
            def progress(env):
                if (env.iteration+1)%100==0:print(json.dumps(dict(target=target,round=env.iteration+1)),flush=True)
            progress.order=50
            model=lgb.train(cfg["parameters"],ds,num_boost_round=600,callbacks=[progress])
            assert model.current_iteration()==600
            assert_categorical_recipe(model,columns)
            guard(root,folder,expected_bytes=20*1024**2);model.save_model(str(folder/f"{target}.txt"))
            r=dict(best_iteration=600,model_sha256=sha(folder/f"{target}.txt"));write(checkpoint,r);complete[target]=r
            del model,ds;gc.collect()
        parent_parity={}
        for target in ["speed","flow_lane"]:
            model_text=(folder/(target+".txt")).read_text()
            canonical="\n".join(line for line in model_text.splitlines() if not line.startswith("[num_threads:"))+"\n"
            trees=model_text.split("Tree=0\n",1)[1].split("\nend of trees",1)[0]
            import hashlib
            actual=dict(canonical_threads_sha256=hashlib.sha256(canonical.encode()).hexdigest(),tree_values_sha256=hashlib.sha256(trees.encode()).hexdigest())
            parent_parity[target]=actual
            if reference is not None:
                assert actual["tree_values_sha256"]==reference["models"][target]["tree_values_sha256"],("Full600 parent tree values differ from laptop",target)
                assert actual["canonical_threads_sha256"]==reference["models"][target]["canonical_threads_sha256"],("Full600 parent model differs beyond thread-count metadata",target)
        write(folder/"laptop_parent_parity.json",dict(verdict="PASS" if reference is not None else "PENDING",models=parent_parity,reference_sha256=sha(wt/"exp/0222_s07_vm_continuation/F4_parent214_reference.json") if reference is not None else None,normalization="Only native num_threads metadata line and text newline convention; all600 tree values byteexact",held_reads=0))
        metadata=dict(columns=columns,fit_dates=dates,panels=selected,asm_parameters=cfg["asm_parameters"],model_cases=complete,integrity=contract,summary_sha256={panel:sha(folder/f"{panel}_summaries.npz") for panel in selected})
        write(folder/"metadata.json",metadata)
        result=dict(id="0222",phase=f"Fresh fixed F4 parent{args.days} reconstruction; serving parity remains due",verdict="REF",runtime_min=(time.monotonic()-start)/60,candidate_fits=0,parent_target_fits=2,diagnostic_fits=4,OWN_reads=0,ADV_reads=0,board_reads=0,metadata_sha256=sha(folder/"metadata.json"),runtime_control_sha256=sha(folder/"runtime_controls.json"),utc=datetime.now(timezone.utc).isoformat())
        write(folder/"result.json",result)
        print(json.dumps(result),flush=True)
if __name__=="__main__":main()
