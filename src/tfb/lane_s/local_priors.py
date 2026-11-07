"""Map L08 observed-local priors into unchanged F3 dynamic feature slots."""
from datetime import date
import numpy as np
from .features import DYNAMIC,fill_dynamic


def as_f3_summary(constants,structural):
    free=np.asarray(constants["free_speed"],np.float32)
    assert free.ndim==1 and np.isfinite(free).all() and (free>0).all()
    n=len(free);summary={"mean":free}
    for name in ["history_speed","history_flow_lane"]:
        values=np.asarray(constants[name],np.float32)
        assert values.shape==(2,288,n) and not np.isinf(values).any()
        summary[name]=values
    distance=np.asarray(structural["distance_bottleneck_km"],np.float32)
    assert distance.shape==(n,) and not np.isinf(distance).any()
    summary["distance_bottleneck_km"]=distance
    summary["bottleneck_positions_km"]=structural.get("bottleneck_positions_km",[])
    return summary


def fill_scored_dynamic(matrix,columns,station,time,weekend,override,constants,structural):
    return fill_dynamic(matrix,columns,station,time,weekend,override,
                        as_f3_summary(constants,structural))


def fill_monthly_dynamic(matrix,columns,metadata,dates,override,monthly,structural):
    metadata=np.asarray(metadata);override=np.asarray(override,bool)
    assert len(metadata)==len(matrix)==len(override) and metadata.shape[1]>=4
    index={name:i for i,name in enumerate(columns)}
    assert set(DYNAMIC)<=set(index)
    day=metadata[:,0].astype(int);station=metadata[:,2].astype(int)
    time=metadata[:,1].astype(int);weekend=metadata[:,3].astype(int)
    assert ((day>=0)&(day<len(dates))).all()
    assert ((time>=0)&(time<288)&(weekend>=0)&(weekend<=1)).all()
    months=np.asarray([date.fromisoformat(dates[d]).strftime("%Y-%m") for d in day])
    assert set(months)<=set(monthly),"Own-month constants missing; no other-month fallback"
    for month in sorted(set(months)):
        rows=np.flatnonzero(months==month)
        summary=as_f3_summary(monthly[month],structural)
        assert ((station[rows]>=0)&(station[rows]<len(summary["mean"]))).all()
        si,ti,kind=station[rows],time[rows],weekend[rows]
        mean=summary["mean"][si]
        matrix[rows,index["free_flow_mean"]]=mean
        matrix[rows,index["base_B2_speed"]]=np.where(override[rows],mean,matrix[rows,index["base_time_speed"]])
        matrix[rows,index["history_speed"]]=summary["history_speed"][kind,ti,si]
        matrix[rows,index["history_flow_lane"]]=summary["history_flow_lane"][kind,ti,si]
        matrix[rows,index["distance_bottleneck_km"]]=summary["distance_bottleneck_km"][si]
    return matrix
