"""Rebuild the frozen OWN population from released observations on the VM."""
from pathlib import Path
from datetime import datetime,timezone
import gc,hashlib,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from tfb.paths import project_root
from tfb.data import load_cache,guard
from .build_lgb_data import sha
from .jobs import owned_job

def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
def logical(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def semantic(item,original_meta_sha256):
    # Compare the scientific item after normalizing only host artifact location.
    canonical={k:v for k,v in item.items() if k!="sha256"}
    canonical["source"]=dict(item["source"],**{"meta.json":original_meta_sha256})
    canonical["path"]=rf"artifacts\lead\0184\{item['panel']}\{item['period']}.npz"
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def main():
    root=project_root();exp=Path(__file__).parents[3]/"exp/0222_s07_vm_continuation"
    reference=json.loads((exp/"OWN_mask_reference.json").read_text())
    cfgpath=root/"exp/0184_l_s_own_instrument/config.yaml"
    assert sha(cfgpath)==reference["config_sha256"]
    cfg=json.loads(cfgpath.read_text())
    dest=root/"artifacts/lane_s/0222/prerequisites/OWN_v2";dest.mkdir(parents=True,exist_ok=True)
    contract=dict(producer_id="0222",producer_sha256=sha(__file__),reference_sha256=sha(exp/"OWN_mask_reference.json"),original_manifest_sha256=reference["original_manifest_sha256"],original_config_sha256=sha(cfgpath),host="vm",metadata_reference_sha256=sha(exp/"OWN_cache_metadata_reference.json"),normalization="Only runtime_seconds cache-build timing and host artifact location; actual cache SHA retained in items",definition="Fresh frozen OWN masks; released finite observations only. Original seeds/strata/blocks/population unchanged. Host artifact path and fresh ZIP serialization are reported separately.")
    cp=dest/"contract.json"
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:write(cp,contract)
    meta_reference=json.loads((exp/"OWN_cache_metadata_reference.json").read_text())
    start=time.monotonic();items=[];proofs={}
    with owned_job("0222_OWN_mask_rebuild",dest,eta_hours=1.5,ram_gib=7),threadpool_limits(1):
        for period in ("jan_feb","validation","private"):
            split="train" if period=="jan_feb" else period
            for panel in cfg["panels"]:
                folder=dest/panel;folder.mkdir(exist_ok=True)
                path=folder/(period+".npz");stamp=folder/(period+".json")
                expected=reference["items"][panel+"|"+period]
                meta_expected=meta_reference[panel+"|"+period]
                if stamp.exists():
                    item=json.loads(stamp.read_text());assert sha(path)==item["sha256"]
                    with np.load(path,allow_pickle=False) as a:actual={k:logical(a[k]) for k in a.files}
                    assert actual==expected["arrays"] and semantic(item,meta_expected["file_sha256"])==expected["semantic_sha256"]
                    items.append(item);proofs[panel+"|"+period]=dict(logical_arrays="EXACT",scientific_item="EXACT",file_sha256=sha(path))
                    continue
                assert not path.exists(),"Partial mask requires inspection before replacement"
                cache=load_cache(panel,split,root=root)
                actual_meta=json.loads((cache.path/"meta.json").read_text())
                actual_fields={k:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest() for k,v in actual_meta.items()}
                assert set(actual_fields)==set(meta_expected["fields"])
                different=[k for k,v in actual_fields.items() if v!=meta_expected["fields"][k]]
                assert set(different)<=set(["runtime_seconds"]),("OWN cache scientific metadata differs",panel,period,different)
                assert isinstance(actual_meta["runtime_seconds"],(int,float)) and actual_meta["runtime_seconds"]>=0
                meta_proof=dict(actual_meta_sha256=sha(cache.path/"meta.json"),original_meta_sha256=meta_expected["file_sha256"],different_fields=different,scientific_metadata="EXACT",runtime_seconds=actual_meta["runtime_seconds"])
                indices=np.asarray([i for i,d in enumerate(cache.dates) if period!="jan_feb" or "2031-01-01"<=d<="2031-02-28"])
                dates=[cache.dates[i] for i in indices]
                source={name:sha(cache.path/name) for name in ("meta.json","speed.npy","flow.npy","elig.npy","target.npy","hist.npy","gap.npy")}
                v=np.array(cache.speed[indices],copy=True);q=np.array(cache.flow[indices],copy=True)
                original_target=np.array(cache.target[indices],copy=True)
                protected=np.array(cache.hist[indices]|cache.gap[indices],copy=True)
                eligible=np.array(cache.elig[indices],copy=True)
                observed=eligible&np.isfinite(v)&np.isfinite(q)&(v>=1)&(q>=0)&~original_target
                candidates=observed&~protected
                seed=cfg["seed"]+int(hashlib.sha256((panel+"|"+period).encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(seed);hidden=np.zeros(v.shape,bool);regimes=np.asarray(cache.regimes)[indices];strata={}
                for regime in ("R1","R2","R3"):
                    rows=np.flatnonzero((candidates&(regimes==regime)[:,None,None]).reshape(-1))
                    number=int(np.floor(cfg["random_fraction"]*len(rows)))
                    selected=rng.choice(rows,size=number,replace=False);hidden.reshape(-1)[selected]=True
                    strata[regime]=dict(observed_unprotected=len(rows),random_selected=number)
                random_count=int(hidden.sum());blocks=[]
                for day,label in enumerate(dates):
                    choices=[t for t in range(12,270) if not protected[day,t+1:t+19].any()]
                    assert choices
                    origin=int(rng.choice(choices));hidden[day,origin+1:origin+19]=True
                    blocks.append(dict(date=label,origin_step=origin,start_step=origin+1,end_step=origin+18))
                target=hidden&candidates
                assert not (target&original_target).any() and not (hidden&protected).any()
                assert observed[target].all() and target.any()
                values=dict(hidden=hidden,target=target,dates=np.asarray(dates))
                actual={k:logical(a) for k,a in values.items()}
                assert actual==expected["arrays"],("OWN logical population differs",panel,period)
                guard(root,folder,expected_bytes=8*1024**2);np.savez_compressed(path,**values)
                item=dict(panel=panel,period=period,split=split,dates=dates,path=path.relative_to(root).as_posix(),sha256=sha(path),source=source,shape=list(v.shape),seed=seed,random_strata=strata,random_hidden=random_count,n_targets=int(target.sum()),n_hidden=int(hidden.sum()),observed_eligible=int(observed.sum()),protected_observed_excluded=int((observed&protected).sum()),blocks=blocks,label_source="Audited float32 released masked observations, never *_true arrays",hidden_source_rule="Hide both fields; constants and all features must exclude hidden cells; published histories/gaps protected")
                assert semantic(item,meta_expected["file_sha256"])==expected["semantic_sha256"],("OWN semantic item differs",panel,period)
                write(stamp,item);items.append(item)
                proofs[panel+"|"+period]=dict(logical_arrays="EXACT",scientific_item="EXACT_AFTER_RUNTIME_AND_LOCATION_NORMALIZATION",file_sha256=sha(path),cache_metadata=meta_proof)
                print(json.dumps(dict(panel=panel,period=period,n_targets=item["n_targets"],comparison="EXACT")),flush=True)
                for a in cache.arrays.values():
                    if isinstance(a,np.memmap):a._mmap.close()
                del cache,v,q,eligible,observed,candidates,hidden,target,protected,original_target,values;gc.collect()
        assert len(items)==30
        write(dest/"manifest.json",dict(id="0222",instrument="I-S-OWN",host="vm",producer_sha256=sha(__file__),source_contract_sha256=sha(cp),original_manifest_sha256=reference["original_manifest_sha256"],frozen_utc=datetime.now(timezone.utc).isoformat(),items=items,read_budget=3,candidate_reads=dict(jan_feb=0,validation=0,private=0),comparison="EXACT_LOGICAL_POPULATION"))
        write(dest/"result.json",dict(id="0222",verdict="PASS",phase="OWN mask regeneration only; no predictions/metrics",panels_periods=proofs,n_targets=sum(x["n_targets"] for x in items),manifest_sha256=sha(dest/"manifest.json"),runtime_min=(time.monotonic()-start)/60,candidate_fits=0,OWN_reads=0,ADV_reads=0,board_reads=0))
if __name__=="__main__":main()
