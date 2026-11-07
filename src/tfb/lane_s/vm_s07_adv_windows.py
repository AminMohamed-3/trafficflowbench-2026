"""Rebuild the unchanged frozen S-ADV windows from released raw train files."""
from pathlib import Path
import gc,hashlib,json,time
import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits
from tfb.paths import project_root,release_root
from tfb.data import guard
from . import q01_training_snapshot as q01
from .build_lgb_data import sha
from .jobs import owned_job
FIRST_SHA="5a183133cb3018f448a31012e8d2e6de230582d1bdaca170beef58a0a8242e97"
REPAIRED_SHA="3bb6edec7dbb003dcf6d0bf7d69935009c867ad4a51413803407104c14e94670"
def write(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False)+"\n")
def exact_csv(frame,path,digest):
    for end in ["\r\n","\n"]:
        value=frame.to_csv(index=False,lineterminator=end).encode()
        if hashlib.sha256(value).hexdigest()==digest:
            guard(project_root(),path,expected_bytes=len(value));path.write_bytes(value);return
    raise AssertionError("Original frozen CSV bytes/population differ: "+path.name)
def repair(original,pool):
    rng=np.random.default_rng(106);pieces=[];changes=[]
    def far(origin,selected):
        t=pd.Timestamp(origin)
        return all(abs(t-pd.Timestamp(value))>=pd.Timedelta(minutes=155) for value in selected)
    for (panel,month),group in original.groupby(["panel",original.date.str[:7]],sort=True):
        onset=group[group.condition.eq("queue_onset")];chosen=onset.to_dict("records")
        origins=list(onset.forecast_origin);ongoing=group[group.condition.eq("queue_ongoing")].sort_values("forecast_origin");retained=[]
        for row in ongoing.to_dict("records"):
            if far(row["forecast_origin"],origins):
                chosen.append(row);origins.append(row["forecast_origin"]);retained.append(row["forecast_origin"])
        candidates=pool[pool.panel.eq(panel)&pool.date.str.startswith(month)&pool.condition.eq("queue_ongoing")].copy()
        times=pd.to_datetime(candidates.forecast_origin,utc=True)
        candidates=candidates[(times.dt.day<=12)&(times.dt.hour*60+times.dt.minute).between(450,1170)&candidates.history_coverage.ge(.7)&candidates.history_last_step_persistence_proxy_iou.le(.9)]
        candidates=candidates.sort_values(["date","forecast_origin"])
        for position in rng.permutation(len(candidates)):
            if len(chosen)==10:break
            row=candidates.iloc[position].to_dict()
            if not far(row["forecast_origin"],origins):continue
            row["family_id"]="_".join(panel.split("_")[:-1])
            row["window_id"]=f"ISADV_{panel}_{row['forecast_origin'].replace('-','').replace(':','')}_queue_ongoing"
            origin=pd.Timestamp(row["forecast_origin"])
            row["gap_start"]=(origin+pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            row["gap_end"]=(origin+pd.Timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
            row["instrument"]="I-S-ADV repaired history/gap-disjoint list"
            chosen.append(row);origins.append(row["forecast_origin"])
        assert len(chosen)==10,(panel,month)
        changes.append(dict(panel=panel,month=month,onsets_unchanged=len(onset),original_ongoing_retained=len(retained),ongoing_replacements=5-len(retained)))
        pieces.extend(chosen)
    result=pd.DataFrame(pieces)[original.columns].sort_values(["panel","date","forecast_origin","condition"]).reset_index(drop=True)
    assert len(result)==160
    conflicts=0
    for panel,group in result.groupby("panel"):
        history=set();gaps=set()
        for origin in pd.to_datetime(group.forecast_origin,utc=True):
            history.update(pd.date_range(origin-pd.Timedelta(minutes=60),periods=12,freq="5min"))
            gaps.update(pd.date_range(origin+pd.Timedelta(minutes=5),periods=18,freq="5min"))
        conflicts+=len(history&gaps)
    assert conflicts==0
    return result,changes
def main():
    root=project_root();dest=root/"artifacts/lane_s/0222/prerequisites/ADV";dest.mkdir(parents=True,exist_ok=True)
    assert sha(q01.__file__)=="a5f707ddd00620090d514dfd2efb28baeb3bba5c6ef500d1d49e79f7428bd958"
    contract=dict(id="0222",host="vm",source_sha256=sha(__file__),Q01_source_sha256=sha(q01.__file__),first_windows_sha256=FIRST_SHA,repaired_windows_sha256=REPAIRED_SHA,selection="Original0106 all8 rawJanFeb candidates and adverse_population, frozen_windows chronological onsets; exact0109 history-gap-disjoint repair, seed106. No selector change.")
    cp=dest/"contract.json"
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:write(cp,contract)
    if (dest/"result.json").exists():
        assert sha(dest/"windows.csv")==REPAIRED_SHA and json.loads((dest/"result.json").read_text())["verdict"]=="PASS";return
    start=time.monotonic();chunks=[];counts={}
    with owned_job("0222_ADV_windows_rebuild",dest,eta_hours=.5,ram_gib=7),threadpool_limits(1):
        release=q01.Release(release_root(root));pa.set_cpu_count(1);pa.set_io_thread_count(1)
        for panel in q01.TASK2_PANELS:
            parts=[]
            for day in release.files(panel,"train","mainline_states"):
                if "2031-01-01"<=day<="2031-02-28":parts.append(q01.candidate_windows(release.train_grid(panel,day),eligible_only=False))
            frame=pd.concat(parts,ignore_index=True);chunks.append(frame);counts[panel]=len(frame)
            print(json.dumps(dict(panel=panel,raw_candidates=len(frame),held_metric_reads=0)),flush=True)
        all_candidates=pd.concat(chunks,ignore_index=True);pool=q01.adverse_population(all_candidates)
        original=q01.frozen_windows(all_candidates,first_onset_per_day=False)
        exact_csv(original,dest/"original160.csv",FIRST_SHA)
        # Reparse the original CSV as the historical0109 script did.
        original=pd.read_csv(dest/"original160.csv")
        result,changes=repair(original,pool)
        exact_csv(result,dest/"windows.csv",REPAIRED_SHA)
        write(dest/"result.json",dict(id="0222",host="vm",verdict="PASS",n_windows=160,onsets_unchanged=80,history_gap_conflicts=0,changes=changes,raw_candidate_counts=counts,source_contract_sha256=sha(cp),windows_sha256=sha(dest/"windows.csv"),runtime_min=(time.monotonic()-start)/60,candidate_fits=0,OWN_reads=0,ADV_reads=0,board_reads=0))
if __name__=="__main__":main()
