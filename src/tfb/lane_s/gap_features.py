"""S-03 F2: gap/queue-edge and ramp information integrated into the base."""
import numpy as np,pandas as pd
from .features import day_features as regular_features,geometry
from .reference_data import interp_space

EDGE=('speed_before','speed_after','flow_lane_before','flow_lane_after','density_before','density_after',
      'queued_before','queued_after','tail_distance_before_km','tail_distance_after_km',
      'head_distance_before_km','head_distance_after_km','tail_distance_linear_km','head_distance_linear_km',
      'queue_linear','speed_linear','flow_lane_linear','density_linear','progress','onset_minutes_from_T30',
      'edge_observed_fraction_before','edge_observed_fraction_after')
RAMP=tuple(f'ramp_{kind}_t{offset}' for kind in ('on_near','off_near','on_corridor','off_corridor') for offset in (-1,0,1))

def edge_values(values,step,side):
    nt,ns=values.shape
    times=np.arange(nt)[:,None]
    observed=np.isfinite(values)&(times<=step if side=='before' else times>=step)
    selected=np.max(np.where(observed,times,-1),axis=0) if side=='before' else np.min(np.where(observed,times,nt),axis=0)
    exists=(selected>=0)&(selected<nt)
    return np.where(exists,values[np.clip(selected,0,nt-1),np.arange(ns)],np.nan)

def extent_distance(speed,g):
    queued=np.isfinite(speed)&(speed<=g['v_cut'])
    groups=np.flatnonzero(np.diff(np.r_[False,queued,False].astype(np.int8)))
    if not len(groups):return np.full(len(speed),np.nan),np.full(len(speed),np.nan),queued
    # Select the closest observed queue component separately at each location.
    low=groups[::2];high=groups[1::2]-1
    tail=g['position'][low]-g['length'][low]/2;head=g['position'][high]+g['length'][high]/2
    distance=np.maximum(tail[None,:]-g['position'][:,None],0)+np.maximum(g['position'][:,None]-head[None,:],0)
    choice=distance.argmin(axis=1)
    return g['position']-tail[choice],head[choice]-g['position'],queued

def context_for_day(cache,day,windows):
    g=geometry(cache);nt,ns=cache.speed.shape[1:];shape=(nt,ns)
    out={name:np.full(shape,np.nan,np.float32) for name in EDGE}
    out.update({name:np.full(shape,np.nan,np.float32) for name in ('minutes_since_gap_start','minutes_until_gap_end')})
    out['window_condition']=np.zeros(shape,np.float32)
    own=windows[(windows.panel==cache.meta['panel'])&(windows.date==cache.dates[day])] if windows is not None and len(windows) else []
    for row in own.itertuples(index=False) if hasattr(own,'itertuples') else []:
        origin=pd.Timestamp(row.forecast_origin);t=origin.hour*12+origin.minute//5
        assert 12<=t and t+19<nt
        sl=slice(t+1,t+19);assert cache.gap[day,sl].all(),'Window/gap population mismatch'
        before_v=edge_values(cache.speed[day],t,'before');after_v=edge_values(cache.speed[day],t+19,'after')
        before_q=edge_values(cache.flow[day],t,'before')/g['lanes'];after_q=edge_values(cache.flow[day],t+19,'after')/g['lanes']
        before_k=before_q/np.maximum(before_v,1);after_k=after_q/np.maximum(after_v,1)
        tail0,head0,queue0=extent_distance(before_v,g);tail1,head1,queue1=extent_distance(after_v,g)
        alpha=np.arange(1,19,dtype=float)[:,None]/19.
        out['minutes_since_gap_start'][sl]=(np.arange(18)*5)[:,None]
        out['minutes_until_gap_end'][sl]=(np.arange(17,-1,-1)*5)[:,None]
        condition=1 if row.condition=='queue_onset' else 2
        out['window_condition'][sl]=condition
        def put(name,value):out[name][sl]=value
        for name,value in [('speed_before',before_v),('speed_after',after_v),('flow_lane_before',before_q),('flow_lane_after',after_q),
                           ('density_before',before_k),('density_after',after_k),('queued_before',queue0),('queued_after',queue1),
                           ('tail_distance_before_km',tail0),('tail_distance_after_km',tail1),('head_distance_before_km',head0),('head_distance_after_km',head1)]:
            put(name,value)
        tail_linear=(1-alpha)*tail0+alpha*tail1;head_linear=(1-alpha)*head0+alpha*head1
        # Onset has no visible queue before T; use the visible later location and
        # a contraction toward its head at the documented T+30 emergence time.
        if condition==1:
            active=np.maximum((np.arange(1,19)[:,None]*5-30)/65.,0)
            length1=tail1+head1
            tail_linear=tail1-(1-active)*length1;head_linear=np.broadcast_to(head1,(18,ns)).copy()
            put('onset_minutes_from_T30',(np.arange(1,19)*5-30)[:,None])
        put('tail_distance_linear_km',tail_linear);put('head_distance_linear_km',head_linear)
        queued=(tail_linear>=0)&(head_linear>=0)
        if condition==1:queued&=(np.arange(1,19)[:,None]*5>=30)
        put('queue_linear',np.where(np.isfinite(tail_linear)&np.isfinite(head_linear),queued,np.nan))
        for name,a,b in [('speed_linear',before_v,after_v),('flow_lane_linear',before_q,after_q),('density_linear',before_k,after_k)]:
            put(name,(1-alpha)*a+alpha*b)
        put('progress',alpha);put('edge_observed_fraction_before',np.isfinite(cache.speed[day,t]).mean())
        put('edge_observed_fraction_after',np.isfinite(cache.speed[day,t+19]).mean())
    return out

def ramp_features(cache,day,g):
    nt,ns=cache.speed.shape[1:];result={}
    attachments=cache.ramps
    for offset in (-1,0,1):
        t=np.arange(nt)+offset;usable=(t>=0)&(t<nt)
        flows=cache.ramp_flow[day,np.clip(t,0,nt-1)].astype(float)
        for kind,label in [('OR','on'),('FR','off')]:
            chosen=attachments.ramp_type.eq(kind).to_numpy()
            values=flows[:,chosen];valid=np.isfinite(values)
            total=np.where(valid,values,0).sum(axis=1)
            total=np.where(valid.any(axis=1)&usable,total,np.nan)
            result[f'ramp_{label}_corridor_t{offset}']=np.broadcast_to(total[:,None],(nt,ns)).astype(np.float32)
            near=np.full((nt,ns),np.nan,np.float32)
            locations=g['position'][attachments.loc[chosen,'station_idx'].to_numpy(int)]
            for station,x in enumerate(g['position']):
                source=abs(locations-x)<=2
                if source.any():
                    a=values[:,source];ok=np.isfinite(a)
                    near[:,station]=np.where(ok.any(axis=1)&usable,np.where(ok,a,0).sum(axis=1),np.nan)
            result[f'ramp_{label}_near_t{offset}']=near
    return {name:result[name] for name in RAMP}

def day_features(cache,day,bases,panel_code,gap_context=None):
    context=gap_context if gap_context is not None else context_for_day(cache,day,getattr(cache,'s_windows',None))
    result,override=regular_features(cache,day,bases,panel_code,context)
    for name in EDGE:result['gap_'+name]=context[name]
    result.update(ramp_features(cache,day,geometry(cache)))
    return result,override

def feature_groups(columns):
    groups={}
    for name in columns:
        if name.startswith('gap_') or name in ('inside_gap','minutes_since_gap_start','minutes_until_gap_end','window_condition'):group='gap_context'
        elif name.startswith('ramp_'):group='ramps'
        elif 'asm' in name:group='asm'
        elif name.startswith('time_') and name not in ('time_sin','time_cos'):group='same_station'
        elif name.startswith('spatial_') or name.startswith('characteristic_'):group='spatial_characteristic'
        elif name in ('history_speed','history_flow_lane','distance_bottleneck_km'):group='history'
        elif name.startswith('fd_') or name.startswith('density_') or name in ('lanes','vf','capacity_lane','critical_density_lane','jam_density_lane','v_cut','length_km','position_km'):group='physics'
        elif name.startswith('base_') or name=='free_flow_mean':group='bases'
        else:group='context'
        groups.setdefault(group,[]).append(name)
    return groups

