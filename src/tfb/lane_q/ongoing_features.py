"""Canonical Q-04 cell/step dynamics, derived only from OnlineView."""
from __future__ import annotations
import numpy as np
import pandas as pd

STATIC=['target_hour','weekend','step','link_km','free_speed','capacity','site_distance_km',
        'last_speed_ratio','last_queued','speed_age_minutes']
LOCAL=['last_flow_ratio','minutes_queued','history_queue_count','history_observed_count',
       'upstream_speed_ratio','downstream_speed_ratio','upstream_queued','downstream_queued',
       'tail_distance_km','head_distance_km','tail_distance_stations','head_distance_stations','component_present','component_age_minutes']
DYNAMICS=['tail_velocity15','head_velocity15','tail_velocity30','head_velocity30',
          'upstream_demand_ratio','discharge_ratio','demand_excess_ratio','shock_speed_kmh',
          'projected_tail_distance15','projected_head_distance15']
FEATURES=STATIC+LOCAL+DYNAMICS
DOSES={'F0':STATIC,'Fhalf':STATIC+LOCAL,'Ffull':FEATURES}


def mean_finite(x,axis=None):
    valid=np.isfinite(x)
    count=valid.sum(axis=axis)
    total=np.where(valid,x,0).sum(axis=axis)
    return np.divide(total,count,out=np.full_like(np.asarray(total,dtype=float),np.nan),where=count>0)


def components(queue):
    positions=np.flatnonzero(queue)
    if not len(positions):return []
    groups=np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)
    return [(int(g[0]),int(g[-1])) for g in groups]


def observation_grids(view):
    frame=view.observations()
    n=len(view.network.links);k=view.origin.hour*12+view.origin.minute//5
    steps=(frame.timestamp.dt.hour*12+frame.timestamp.dt.minute//5).to_numpy()
    positions=pd.Index(view.network.links).get_indexer(frame.link_id)
    if (positions<0).any() or (steps>k).any():raise AssertionError('Feature grid crossed online topology/origin')
    rows=np.arange(k+1,dtype=int)[:,None]
    output={}
    for column in ['speed_kmh','flow_vph']:
        grid=np.full((k+1,n),np.nan,dtype=np.float32)
        grid[steps,positions]=frame[column].to_numpy(dtype=np.float32)
        indices=np.maximum.accumulate(np.where(np.isfinite(grid),rows,-1),axis=0)
        carried=np.take_along_axis(grid,np.maximum(indices,0),axis=0)
        carried[indices<0]=np.nan
        output[column]=(grid,carried,indices)
    return output


def static_sites(prior):
    sites=[s for s in prior['sites'] if s['n_windows']>=3]
    return [{'key':f"site_{min(s['positions'])}",'head':max(s['positions'])} for s in sites]


def feature_frame(view,sites):
    net=view.network;fd=net.fd_parameters
    if fd is None:raise ValueError('Released FD parameters required')
    capacity=fd.capacity_vph.to_numpy(dtype=float)
    length=fd.length_km.to_numpy(dtype=float)
    km=np.cumsum(length)-.5*length
    n=len(km);k=view.origin.hour*12+view.origin.minute//5
    grids=observation_grids(view)
    rawspeed,speed,speed_indices=grids['speed_kmh']
    rawflow,flow,_=grids['flow_vph']
    last_speed=speed[-1];last_flow=flow[-1]
    ratio=last_speed/net.free_speed
    queue=np.isfinite(speed)&(speed<=net.cutoffs)
    latest=queue[-1]
    history=view.history
    hpos=pd.Index(net.links).get_indexer(history.link_id)
    valid=np.isfinite(history.speed_kmh)
    q=valid & history.speed_kmh.le(history.link_id.map(net.cutoff_by_link))
    count=np.bincount(hpos[q.to_numpy()],minlength=n)
    observed=np.bincount(hpos[np.asarray(valid)],minlength=n)
    rows=np.arange(k+1)[:,None]
    free=np.isfinite(rawspeed)&(rawspeed>net.cutoffs)
    last_free=np.where(free,rows,-1).max(axis=0)
    fresh_queue=np.isfinite(rawspeed)&(rawspeed<=net.cutoffs)&(rows>last_free[None,:])
    first_queue=np.where(fresh_queue,rows,k+1).min(axis=0)
    duration=np.where(latest,(k-first_queue)*5,0).clip(min=0)
    active=components(latest)
    basis=k
    if not active:
        for basis in range(k-1,-1,-1):
            active=components(queue[basis])
            if active:break
    dynamic={name:np.full(n,np.nan,dtype=float) for name in ['tail_distance_km','head_distance_km','tail_distance_stations',
        'head_distance_stations','tail_velocity15','head_velocity15','tail_velocity30','head_velocity30',
        'upstream_demand_ratio','discharge_ratio','demand_excess_ratio','shock_speed_kmh']}
    if active:
        tails=np.asarray([a for a,b in active]);heads=np.asarray([b for a,b in active])
        distances=np.abs(km[:,None]-np.clip(km[:,None],km[tails],km[heads]))
        assignment=distances.argmin(axis=1)
        avg_flow=mean_finite(rawflow[max(0,k-3):k+1],axis=0)
        avg_speed=mean_finite(rawspeed[max(0,k-3):k+1],axis=0)
        for component,(tail,head) in enumerate(active):
            mask=assignment==component
            positions=np.flatnonzero(mask)
            dynamic['tail_distance_km'][mask]=km[positions]-km[tail]
            dynamic['head_distance_km'][mask]=km[positions]-km[head]
            dynamic['tail_distance_stations'][mask]=positions-tail
            dynamic['head_distance_stations'][mask]=positions-head
            for minutes in [15,30]:
                earlier=k-minutes//5
                old=components(queue[max(0,earlier)])
                if old and basis>earlier:
                    oldtail,oldhead=min(old,key=lambda pair:(abs(km[pair[1]]-km[head]),pair))
                    hours=(basis-max(0,earlier))*5/60
                    dynamic[f'tail_velocity{minutes}'][mask]=(km[tail]-km[oldtail])/hours
                    dynamic[f'head_velocity{minutes}'][mask]=(km[head]-km[oldhead])/hours
            upstream=np.arange(max(0,tail-3),tail)
            if not len(upstream):upstream=np.asarray([tail])
            demand=float(mean_finite(avg_flow[upstream]))
            discharge=float(avg_flow[head])
            v_up=float(mean_finite(avg_speed[upstream]))
            q_down=float(mean_finite(avg_flow[tail:head+1]))
            v_down=float(mean_finite(avg_speed[tail:head+1]))
            dynamic['upstream_demand_ratio'][mask]=demand/capacity[head]
            dynamic['discharge_ratio'][mask]=discharge/capacity[head]
            dynamic['demand_excess_ratio'][mask]=(demand-discharge)/capacity[head]
            if v_up>0 and v_down>0:
                difference=q_down/v_down-demand/v_up
                if abs(difference)>1e-6:
                    dynamic['shock_speed_kmh'][mask]=np.clip((q_down-demand)/difference,-150,150)
    if sites:
        anchors=np.asarray([s['head'] for s in sites])
        chosen=np.abs(km[:,None]-km[anchors]).argmin(axis=1)
        site_uid=[sites[i]['key'] for i in chosen]
        site_distance=km-km[anchors[chosen]]
    else:
        site_uid=['other']*n;site_distance=np.full(n,np.nan)
    before=np.r_[np.nan,ratio[:-1]];after=np.r_[ratio[1:],np.nan]
    base={'link_km':km,'free_speed':net.free_speed,'capacity':capacity,'site_distance_km':site_distance,
        'last_speed_ratio':ratio,'last_queued':latest.astype(float),'speed_age_minutes':(k-speed_indices[-1])*5,
        'last_flow_ratio':last_flow/capacity,'minutes_queued':duration,'history_queue_count':count,'history_observed_count':observed,
        'upstream_speed_ratio':before,'downstream_speed_ratio':after,'upstream_queued':np.r_[0,latest[:-1]],
        'downstream_queued':np.r_[latest[1:],0],'component_present':np.full(n,latest.any()),
        'component_age_minutes':np.full(n,(k-basis)*5 if active else np.nan),**dynamic}
    result=pd.DataFrame({name:np.tile(value,6) for name,value in base.items()})
    steps=np.repeat(np.arange(1,7),n)
    result['step']=steps
    result['target_hour']=view.origin.hour+view.origin.minute/60+steps*5/60
    result['weekend']=int(view.origin.weekday()>=5)
    result['site_uid']=np.tile(site_uid,6)
    result['projected_tail_distance15']=result.tail_distance_km-result.tail_velocity15*steps*5/60
    result['projected_head_distance15']=result.head_distance_km-result.head_velocity15*steps*5/60
    result=result[['site_uid',*FEATURES]]
    result[FEATURES]=result[FEATURES].astype(np.float32)
    if np.isinf(result[FEATURES].to_numpy()).any():raise AssertionError('Nonfinite online dynamics')
    return result
