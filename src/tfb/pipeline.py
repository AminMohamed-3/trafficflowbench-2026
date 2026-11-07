"""Build the canonical submission from pinned lane configurations."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from .data import guard, load_cache
from .paths import project_root, release_root, panels
from .submit import assemble, sha256, validate
from .provenance import additional_sources

def verify_instruments(root):
    path=root/"configs/instruments.yaml"
    manifest=yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if manifest["status"]!="FROZEN":
        raise ValueError("Instruments must be frozen")
    items=[manifest["date_folds"],manifest["I-S-ADV"]["windows"],
           manifest["I-Q-ADV"]["all_candidates"],manifest["I-Q-ADV"]["frozen_subset"],
           manifest["I-Q-ADV"]["first_per_day_diagnostic"]]
    for item in items:
        if sha256(root/item["path"])!=item["sha256"]:
            raise ValueError(f"Changed frozen instrument: {item['path']}")
    return sha256(path)

def verify_release_caches(root):
    verified={}
    for panel in panels(root):
        for split in ["train","validation","private"]:
            cache=load_cache(panel,split,root=root)
            if not cache.meta["verified"]:
                raise ValueError(f"Unaudited cache {panel}/{split}")
            for item in cache.meta["sources"]:
                if item["path"] in verified:
                    if verified[item["path"]]!=item["sha256"]:
                        raise ValueError("Inconsistent cache source receipts")
                    continue
                actual=sha256(release_root(root)/item["path"])
                if actual!=item["sha256"]:
                    raise ValueError(f"Raw source changed: {item['path']}")
                verified[item["path"]]=actual
        print("verified raw cache sources",panel,flush=True)
    return verified

def prepare_queue_prior(root, cfg, destination):
    from .lane_q.build_baseline import collect_onsets
    from .lane_q.predict import MODEL_VERSION, fit_panel_prior
    from .lane_q.windows import Release, TASK2_PANELS
    release=Release(release_root(root))
    fit=cfg["fit"]
    model={"model_version":MODEL_VERSION,"fit":fit,"panels":{}}
    records={}
    for panel in TASK2_PANELS:
        rows=collect_onsets(release,panel,fit["start"],fit["end"])
        model["panels"][panel]=fit_panel_prior(rows,release.network(panel).links,
                                               max_position_gap=fit["max_position_gap"])
        records[panel]=rows
        print("raw prior refit",panel,len(rows),"onsets",flush=True)
    guard(root,destination.parent,2*1024**2)
    destination.write_text(json.dumps(model,sort_keys=True,indent=2),encoding="utf-8")
    if cfg.get("expected_prior_sha256") and sha256(destination)!=cfg["expected_prior_sha256"]:
        raise ValueError("Train-only prior refit differs from reviewed builder")
    return destination,records

def prepare_onset_model(root, cfg, prior, records, artifact, threads):
    """Delegate raw feature extraction and fitting to the reviewed Q implementation."""
    import numpy as np
    import pandas as pd
    import lightgbm
    from .lane_q.build_onset import extract
    from .lane_q.evaluation_cache import EvaluationRelease
    from .lane_q.onset_features import NUMERIC_FEATURES, site_table
    from .lane_q.onset_model import OnsetModel, train
    from .rebuild_contract import load_contract,verify_model
    portable=load_contract(root,cfg,"onset") if cfg.get("onset_contract") else None
    expected=portable["reviewed_artifact_sha256"] if portable else cfg["onset_model_sha256"]
    if not portable:
        reference_dir=root/cfg["onset_model_dir"]
        for name,digest in expected.items():
            if sha256(reference_dir/name)!=digest:
                raise ValueError(f"Reviewed onset artifact changed: {name}")
    if lightgbm.__version__!=cfg["lightgbm_version"]:
        raise ValueError("Onset runtime differs from pinned shared dependency")
    folds=pd.read_csv(root/"configs/date_folds.csv")
    fold_map=dict(zip(folds.date,folds.fold))
    release=EvaluationRelease(release_root(root))
    fitted_prior=json.loads(prior.read_text(encoding="utf-8"))
    if portable and sha256(prior)!=portable["prior_sha256"]:
        raise ValueError("Raw onset prior differs from portable contract")
    features=[]
    sites={}
    for panel,rows in records.items():
        sites[panel]=site_table(fitted_prior["panels"][panel])
        frame,_=extract(release,panel,pd.DataFrame(rows),sites[panel],fold_map)
        guard(root,artifact,max(1,len(frame))*1000)
        frame.to_parquet(artifact/f"onset_fit_{panel}.parquet",index=False)
        features.append(frame)
        print("raw onset features",panel,len(frame),flush=True)
    frame=pd.concat(features,ignore_index=True)
    booster,categories=train(frame,NUMERIC_FEATURES,num_threads=threads)
    reference=None if portable else OnsetModel.load(reference_dir,num_threads=threads)
    reviewed_metadata=portable["metadata"] if portable else reference.metadata
    if categories!=reviewed_metadata["categories"] or sites!=reviewed_metadata["sites"]:
        raise ValueError("Raw onset category/site reconstruction changed")
    if NUMERIC_FEATURES!=reviewed_metadata["features"]:
        raise ValueError("Raw onset feature family changed")
    metadata={**reviewed_metadata,"categories":categories,"sites":sites,
              "pipeline_raw_refit":True,"num_threads":threads}
    model=OnsetModel(booster,metadata,fitted_prior,num_threads=threads)
    tree_sha=None
    if portable:
        tree_sha=verify_model(booster,portable["learned_model_sha256"]["model.txt"])
    else:
        actual=model.probabilities(frame)
        baseline=reference.probabilities(frame)
        if not np.array_equal(actual,baseline):
            raise ValueError(f"Raw classifier refit probability delta {np.max(np.abs(actual-baseline))}")
    directory=artifact/"onset_model"
    guard(root,directory,10*1024**2)
    model.save(directory)
    receipt={"raw_refit":True,"site_rows":len(frame),"probability_max_delta":None if portable else 0.,
             "reference_sha256":expected,"runtime_version":lightgbm.__version__,
             "num_threads":threads,"sha256":{name:sha256(directory/name) for name in expected}}
    if portable:
        receipt.update(portable_contract=cfg["onset_contract"],portable_contract_sha256=cfg["onset_contract_sha256"],
                       learned_model_sha256=tree_sha,verification="Exact learned model text except num_threads; no saved teacher loaded")
    return model,receipt

def prepare_ongoing_model(root, cfg, prior, artifact, threads):
    """Use Q-owned raw extraction and fitting for the reviewed ongoing model."""
    import gc
    import numpy as np
    import pandas as pd
    import lightgbm
    from .lane_q.build_ongoing import extract_panel
    from .lane_q.evaluation_cache import EvaluationRelease
    from .lane_q.ongoing_features import FEATURES as BASE_FEATURES,static_sites
    from .lane_q.ongoing_runtime import OngoingPredictor,train
    from .lane_q.onset_model import model_frame
    from .lane_q.predict import load_ongoing_model
    from .rebuild_contract import load_contract,verify_model,learned_model_sha256
    portable=load_contract(root,cfg,"ongoing") if cfg.get("ongoing_contract") else None
    expected=portable["reviewed_artifact_sha256"] if portable else cfg["ongoing_model_sha256"]
    if not portable:
        reference_dir=root/cfg["ongoing_model_dir"]
        for name,digest in expected.items():
            if sha256(reference_dir/name)!=digest:
                raise ValueError(f"Reviewed ongoing artifact changed: {name}")
    if lightgbm.__version__!=cfg["lightgbm_version"]:
        raise ValueError("Ongoing runtime differs from pinned shared dependency")
    reference=None if portable else load_ongoing_model(reference_dir,num_threads=threads)
    reviewed_metadata=portable["metadata"] if portable else reference.metadata
    builder=reviewed_metadata.get("feature_builder")
    FEATURES=BASE_FEATURES;model_class=OngoingPredictor
    if builder=="Q05_canonical_ctm_v1":
        from .lane_q import ctm_features
        from .lane_q.ctm_model import CTMModel
        from .lane_q.ctm_poison import assert_fitted_poison
        from .lane_q.windows import online_view
        FEATURES=ctm_features.FEATURES;model_class=CTMModel
    elif builder is not None:
        raise ValueError("Unsupported raw ongoing feature builder")
    raw_pins=cfg.get("raw_training_sha256",{})
    for name,digest in raw_pins.items():
        if sha256(root/name)!=digest:
            raise ValueError(f"Reviewed raw ongoing training input changed: {name}")
    fitted_prior=json.loads(prior.read_text(encoding="utf-8"))
    if ((portable and sha256(prior)!=portable["prior_sha256"]) or
            (not portable and fitted_prior!=reference.prior) or FEATURES!=reviewed_metadata["features"]):
        raise ValueError("Raw ongoing prior/features differ from reviewed candidate")
    folds=pd.read_csv(root/"configs/date_folds.csv")
    fold_map=dict(zip(folds.date,folds.fold))
    release=EvaluationRelease(release_root(root))
    boosters={};metadata={};counts={};tree_hashes={}
    for panel in reviewed_metadata["panels"]:
        sites=static_sites(fitted_prior["panels"][panel])
        frame,labels,windows=extract_panel(release,panel,sites,fold_map,artifact)
        if builder=="Q05_canonical_ctm_v1":
            # Recreate the Q-owned CTM rows from the same raw online windows.
            base_dir=root/cfg["reviewed_base_training_dir"]
            ctm_dir=root/cfg["reviewed_ctm_training_dir"]
            for suffix,actual_frame in [("features",frame),("windows",windows)]:
                path=base_dir/f"{panel}_{suffix}.parquet"
                if str(path.relative_to(root)).replace("\\","/") not in raw_pins:
                    raise ValueError("Raw CTM base comparison is not pinned")
                pd.testing.assert_frame_equal(actual_frame,pd.read_parquet(path),check_exact=True)
            states=[ctm_features.initial_state(online_view(panel,"train",w.date,w.forecast_origin,release=release))
                    for w in windows.itertuples(index=False)]
            parts,substeps=ctm_features.batched_forecast_frames(release.network(panel),states)
            added=pd.concat(parts,ignore_index=True)
            path=ctm_dir/f"{panel}_ctm_features.parquet"
            if str(path.relative_to(root)).replace("\\","/") not in raw_pins:
                raise ValueError("Raw CTM feature comparison is not pinned")
            pd.testing.assert_frame_equal(added,pd.read_parquet(path),check_exact=True)
            if len(added)!=len(frame):raise ValueError("Raw CTM row count changed")
            frame=pd.concat([frame,added],axis=1)
            del states,parts,added
        if sites!=reviewed_metadata["panels"][panel]["sites"]:
            raise ValueError("Raw ongoing site reconstruction changed")
        guard(root,artifact,max(1,len(frame))*500)
        frame.to_parquet(artifact/f"ongoing_fit_{panel}.parquet",index=False)
        fit=frame.loc[frame.target.ge(0)]
        booster,categories=train(fit,FEATURES,num_threads=threads)
        if categories!=reviewed_metadata["panels"][panel]["categories"]:
            raise ValueError("Raw ongoing categories changed")
        if portable:
            tree_hashes[panel]=verify_model(booster,portable["learned_model_sha256"][panel+".txt"])
        else:
            actual=booster.predict(model_frame(fit,FEATURES,categories),num_threads=threads)
            baseline=reference.probabilities(fit,panel)
            if not np.array_equal(actual,baseline):
                raise ValueError(f"Raw ongoing refit probability delta {np.max(np.abs(actual-baseline))}")
            if builder=="Q05_canonical_ctm_v1":
                tree_hashes[panel]=verify_model(booster,learned_model_sha256(reference.boosters[panel].model_to_string()))
        boosters[panel]=booster;metadata[panel]={"categories":categories,"sites":sites}
        counts[panel]={"rows":len(frame),"observed_rows":len(fit),"windows":len(windows),"probability_max_delta":None if portable else 0.}
        if builder=="Q05_canonical_ctm_v1":
            temporary=model_class({panel:booster},{**reviewed_metadata,"panels":{panel:metadata[panel]}},
                                  fitted_prior,num_threads=threads)
            fixture=windows.iloc[0]
            poison=assert_fitted_poison(temporary,release,panel,fixture.date,fixture.forecast_origin)
            if poison["result"]!="PASS":raise ValueError("Raw fitted CTM future poison failed")
            counts[panel].update(raw_base_and_CTM_frames_exact=True,CTM_substeps_per5min=substeps,
                                 learned_model_sha256=tree_hashes[panel],future_poison=poison)
            del temporary
        print("raw ongoing refit",panel,counts[panel],flush=True)
        del frame,labels,windows,fit
        if not portable:
            del actual,baseline
        gc.collect()
    for name,digest in raw_pins.items():
        if sha256(root/name)!=digest:raise ValueError("Raw ongoing training input changed during refit")
    model=model_class(boosters,{**reviewed_metadata,"panels":metadata,"pipeline_raw_refit":True,
                          "num_threads":threads},fitted_prior,num_threads=threads)
    directory=artifact/"ongoing_model"
    guard(root,directory,15*1024**2);model.save(directory)
    receipt={"raw_refit":True,"counts":counts,"reference_sha256":expected,"num_threads":threads,
             "runtime_version":lightgbm.__version__,"sha256":{path.name:sha256(path) for path in directory.iterdir() if path.is_file()}}
    if builder=="Q05_canonical_ctm_v1":
        receipt.update(feature_builder=builder,raw_training_sha256=raw_pins,learned_model_sha256=tree_hashes,
                       verification="Exact raw base/CTM frames, training probabilities and learned model text except num_threads; all8 fitted future poison")
    if portable:
        receipt.update(portable_contract=cfg["ongoing_contract"],portable_contract_sha256=cfg["ongoing_contract_sha256"],
                       learned_model_sha256=tree_hashes,verification="Exact learned model text except num_threads; no saved teacher loaded")
    return model,receipt

def build(config_path):
    root=project_root()
    config_path=Path(config_path).resolve()
    config=yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if config.get("mode") == "board_read_artifact":
        raise ValueError(
            "The Tier 1 incumbent is the retained board-read file "
            f"{config['file']}. Native reproduction is deferred to Tier 2 "
            "under D-0058; Q owns the CNN production port."
        )
    import psutil
    process=psutil.Process()
    available=process.cpu_affinity()
    count=int(config.get("threads",2))
    assigned=available[-4:-2] if count==2 and len(available)>=4 else available[:count]
    process.cpu_affinity(assigned)
    identifier=str(config["id"])
    output=root/"submissions"/f"{identifier}_{config['slug']}.csv"
    if output.exists():
        raise FileExistsError("Artifact already exists; use a new ledger ID/config for a fresh rebuild")
    artifact=root/"artifacts/lead"/identifier
    artifact.mkdir(parents=True,exist_ok=True)
    report_dir=root/"exp"/f"{identifier}_l07_pipeline"
    report_dir.mkdir(parents=True,exist_ok=True)
    job=root/"artifacts/jobs"/f"{identifier}.json"
    job_receipt={"id":identifier,"owner":"L","pid":os.getpid(),"status":"RUNNING",
                 "start_utc":datetime.now(timezone.utc).isoformat(),"command":"tfb.pipeline build",
                 "eta_utc":(datetime.now(timezone.utc)+timedelta(minutes=60)).isoformat(),
                 "checkpoint_path":str(artifact),"threads":int(config.get("threads",2)),
                 "cpu_affinity":assigned,
                 "ram_gb":float(config.get("ram_limit_gb",1.5))}
    job.write_text(json.dumps(job_receipt))
    start=time.monotonic()
    try:
        os.environ["TFB_Q_THREADS"]=str(int(config.get("threads",2)))
        code_start={str(path.relative_to(root)):sha256(path) for path in (root/"src/tfb").rglob("*.py")}
        instrument_hash=verify_instruments(root)
        sources=verify_release_caches(root)
        extra=additional_sources(root)
        sources.update(extra)
        prior,records=prepare_queue_prior(root,config["queue"],artifact/"prior_u1.json")
        lane_cfg={"state":config["state"],"queue":{"prior_path":str(prior),"release_root":str(release_root(root)),
                  "onset_mode":config["queue"].get("onset_mode","site_prior")},"odme":config["odme"]}
        for name,digest in config["state"].get("model_sha256",{}).items():
            if sha256(root/config["state"]["model_dir"]/name)!=digest:
                raise ValueError(f"Reviewed state model changed: {name}")
        onset_receipt=None;ongoing_receipt=None;c03_receipt=None
        if config["queue"].get("c03"):
            if config["queue"].get("ongoing_model_dir") or config["queue"].get("ongoing_contract"):
                raise ValueError("C03 ongoing component conflicts with another ongoing recipe")
            from .c03_queue import prepare
            lane_cfg["queue"]["c03_predictions"],c03_receipt=prepare(root,config["queue"]["c03"])
        if config["queue"].get("onset_model_dir") or config["queue"].get("onset_contract"):
            onset_model,onset_receipt=prepare_onset_model(root,config["queue"],prior,records,artifact,
                                                        int(config.get("threads",2)))
            lane_cfg["queue"]["onset_model"]=onset_model
        if config["queue"].get("ongoing_model_dir") or config["queue"].get("ongoing_contract"):
            ongoing_model,ongoing_receipt=prepare_ongoing_model(root,config["queue"],prior,artifact,
                                                              int(config.get("threads",2)))
            lane_cfg["queue"]["ongoing_model"]=ongoing_model
        lane_cfg["queue"]["num_threads"]=int(config.get("threads",2))
        module_names={"state":"tfb.lane_s.predict","queue":"tfb.lane_q.predict","odme":"tfb.lane_d.predict"}
        if c03_receipt:
            module_names["queue"]="tfb.c03_queue"
        if config["state"].get("module"):
            if config["state"]["module"]!="tfb.lane_s.local_predict":
                raise ValueError("Unsupported reviewed state predictor")
            module_names["state"]=config["state"]["module"]
        function_names={"state":"predict_state","queue":"predict_queue","odme":"predict_odme"}
        prediction_receipts={}
        callbacks={}
        for task in module_names:
            predictor=getattr(importlib.import_module(module_names[task]),function_names[task])
            def callback(panel,split,task=task,predictor=predictor):
                frame=predictor(panel,split,lane_cfg[task])
                path=artifact/f"{task}_{panel}_{split}.csv"
                guard(root,artifact,max(1,len(frame))*180)
                frame.to_csv(path,index=False)
                prediction_receipts[f"{task}/{panel}/{split}"]={"rows":len(frame),"sha256":sha256(path)}
                print("pipeline predicted",task,panel,split,len(frame),flush=True)
                return frame
            callbacks[task]=callback
        output=root/"submissions"/f"{identifier}_{config['slug']}.csv"
        assemble(**callbacks,output=output)
        receipt=validate(output)
        if additional_sources(root)!=extra:
            raise ValueError("Public queue/ODME inputs changed during build")
        for name,digest in config["state"].get("model_sha256",{}).items():
            if sha256(root/config["state"]["model_dir"]/name)!=digest:
                raise ValueError(f"Reviewed state model changed during build: {name}")
        if onset_receipt:
            if config["queue"].get("onset_contract"):
                from .rebuild_contract import load_contract
                load_contract(root,config["queue"],"onset")
            else:
                for name,digest in config["queue"]["onset_model_sha256"].items():
                    if sha256(root/config["queue"]["onset_model_dir"]/name)!=digest:
                        raise ValueError("Reviewed onset artifact changed during build")
        if ongoing_receipt:
            if config["queue"].get("ongoing_contract"):
                from .rebuild_contract import load_contract
                load_contract(root,config["queue"],"ongoing")
            else:
                for name,digest in config["queue"]["ongoing_model_sha256"].items():
                    if sha256(root/config["queue"]["ongoing_model_dir"]/name)!=digest:
                        raise ValueError("Reviewed ongoing artifact changed during build")
        if c03_receipt:
            from .c03_queue import prepare
            prepare(root,config["queue"]["c03"])
        code={str(path.relative_to(root)):sha256(path) for path in (root/"src/tfb").rglob("*.py")}
        if code!=code_start:
            raise ValueError("Pipeline source changed during build; preserve artifact but reject provenance")
        result={"id":identifier,"verdict":"PASS","file":str(output),"sha256":receipt["sha256"],"rows":receipt["rows"],
                "counts":receipt["counts"],"config":str(config_path.relative_to(root)),"config_sha256":sha256(config_path),
                "instrument_sha256":instrument_hash,"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "source_hashes":sources,"code_hashes":code,"predictions":prediction_receipts,
                "prior_sha256":sha256(prior),"onset_model":onset_receipt,"ongoing_model":ongoing_receipt,"c03_model":c03_receipt,
                "pre_read_prediction":config.get("prediction"),
                "runtime_minutes":(time.monotonic()-start)/60,
                "uploaded":False,"public":None}
        (report_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        job_receipt.update(status="COMPLETE",end_utc=datetime.now(timezone.utc).isoformat(),
                           file=str(output),sha256=receipt["sha256"],elapsed_min=result["runtime_minutes"])
        job.write_text(json.dumps(job_receipt))
        print(json.dumps({k:result[k] for k in ["file","sha256","rows","verdict","runtime_minutes"]},indent=2),flush=True)
        return output
    except Exception as error:
        job_receipt.update(status="FAILED",end_utc=datetime.now(timezone.utc).isoformat(),error=str(error))
        job.write_text(json.dumps(job_receipt))
        raise

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="command",required=True)
    parser=sub.add_parser("build")
    parser.add_argument("--config",type=Path,default=Path("configs/incumbent.yaml"))
    parser=sub.add_parser("probe", help="Build a prescribed diagnostic whole-condition ablation")
    parser.add_argument("--config",type=Path,required=True)
    parser=sub.add_parser("validate")
    parser.add_argument("file",type=Path)
    args=cli.parse_args()
    import pyarrow as pa
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    if args.command=="build":
        build(args.config)
    elif args.command=="probe":
        from .probes import build_probe
        build_probe(args.config)
    else:
        print(json.dumps(validate(args.file),indent=2))

if __name__=="__main__":
    main()
