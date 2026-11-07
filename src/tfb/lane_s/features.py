"""Canonical S-03 features from released observations and physical geometry.

Truth is used only for fit-date history summaries and training row sampling.
Neither station identifiers nor file/row order enter model features.
"""
from datetime import date
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter1d
from scipy.signal import find_peaks
from .predict import free_flow_means

DYNAMIC=('base_B2_speed','free_flow_mean','history_speed','history_flow_lane','distance_bottleneck_km')
CATEGORICAL=('panel','regime','weekday')


def geometry(cache):
    s=cache.stations
    length=s.length_km.to_numpy(float);position=length.cumsum()-length/2
    vf=s.free_speed_kmh.to_numpy(float);cap=s.capacity_vph.to_numpy(float)
    lanes=s.lanes.to_numpy(float);kc=s.critical_density.to_numpy(float);kj=s.k_jam.to_numpy(float)
    return dict(position=position,length=length,vf=vf,cap=cap,lanes=lanes,kc=kc,kj=kj,w=cap/(kj-kc),v_cut=s.v_cut.to_numpy(float))


def temporal(values,count=3):
    nt,ns=values.shape; t=np.broadcast_to(np.arange(nt)[:,None],values.shape)
    valid=np.isfinite(values)
    prev=np.maximum.accumulate(np.where(valid,t,-1),axis=0)
    nxt=np.minimum.accumulate(np.where(valid,t,nt)[::-1],axis=0)[::-1]
    out={}
    columns=np.broadcast_to(np.arange(ns),values.shape)
    for label,current in [('before',prev.copy()),('after',nxt.copy())]:
        for k in range(count):
            exists=(current>=0)&(current<nt); ci=np.clip(current,0,nt-1)
            out[f'{label}{k+1}_value']=np.where(exists,values[ci,columns],np.nan)
            out[f'{label}{k+1}_minutes']=np.where(exists,abs(t-current)*5.,np.nan)
            if label=='before':
                query=current-1;current=np.where(query>=0,prev[np.clip(query,0,nt-1),columns],-1)
            else:
                query=current+1;current=np.where(query<nt,nxt[np.clip(query,0,nt-1),columns],nt)
    return out


def spatial_neighbours(speed,flow,occ,positions,lanes,offset):
    nt,ns=speed.shape;t=np.arange(nt); source_t=t+offset
    usable=(source_t>=0)&(source_t<nt); tt=np.clip(source_t,0,nt-1)
    v,q,o=speed[tt],flow[tt],occ[tt]
    observed=np.isfinite(v)&usable[:,None]
    indices=np.broadcast_to(np.arange(ns),v.shape)
    up=np.maximum.accumulate(np.where(observed,indices,-1),axis=1)
    down=np.minimum.accumulate(np.where(observed,indices,ns)[:,::-1],axis=1)[:,::-1]
    up=np.concatenate([np.full((nt,1),-1),up[:,:-1]],axis=1)
    down=np.concatenate([down[:,1:],np.full((nt,1),ns)],axis=1)
    rows=np.broadcast_to(np.arange(nt)[:,None],v.shape);out={}
    for label,idx in [('up',up),('down',down)]:
        exists=(idx>=0)&(idx<ns)&usable[:,None];si=np.clip(idx,0,ns-1)
        out[label+'_speed']=np.where(exists,v[rows,si],np.nan)
        out[label+'_flow_lane']=np.where(exists,q[rows,si]/lanes[si],np.nan)
        out[label+'_density_occ']=np.where(exists,285*o[rows,si],np.nan)
        out[label+'_distance_km']=np.where(exists,abs(positions[si]-positions[None,:]),np.nan)
    return out


def shifted_neighbours(speed,flow,positions,lanes,characteristic,distance=2.):
    nt,ns=speed.shape;out={}
    for label,desired in [('up',positions-distance),('down',positions+distance)]:
        source=np.searchsorted(positions,desired,side='right')-1 if label=='up' else np.searchsorted(positions,desired)
        valid_source=(source>=0)&(source<ns);si=np.clip(source,0,ns-1)
        lag=np.rint((positions[si]-positions)/characteristic*60/5).astype(int)
        ti=np.arange(nt)[:,None]+lag[None,:]
        valid=valid_source[None,:]&(ti>=0)&(ti<nt);tt=np.clip(ti,0,nt-1)
        out[label+'_speed']=np.where(valid,speed[tt,si],np.nan)
        out[label+'_flow_lane']=np.where(valid,flow[tt,si]/lanes[si],np.nan)
    return out


def fit_summaries(cache,fit_dates):
    selected=[cache.day_index(d) for d in fit_dates]
    g=geometry(cache);v=np.asarray(cache.speed_true[selected]);q=np.asarray(cache.flow_true[selected])/g['lanes']
    weekend=np.asarray([date.fromisoformat(d).weekday()>=5 for d in fit_dates])
    hs=np.full((2,288,len(g['lanes'])),np.nan,np.float32);hq=hs.copy()
    for kind in [0,1]:
        rows=weekend==kind
        if rows.any():
            hs[kind]=np.nanmedian(v[rows],axis=0);hq[kind]=np.nanmedian(q[rows],axis=0)
    valid=np.isfinite(v);count=valid.sum(axis=(0,1));queued=(valid&(v<=g['v_cut'])).sum(axis=(0,1))
    rate=np.divide(queued,count,out=np.zeros(len(count),float),where=count>0)
    # Recurrent bottlenecks: at most three spatially separated local maxima.
    peaks=list(find_peaks(rate)[0]);peaks.extend([0,len(rate)-1]);peaks=sorted(set(peaks),key=lambda x:rate[x],reverse=True)
    chosen=[]
    for s in peaks:
        if rate[s]>0 and all(abs(g['position'][s]-g['position'][j])>=2 for j in chosen):chosen.append(s)
        if len(chosen)==3:break
    distance=np.min(abs(g['position'][:,None]-g['position'][chosen][None,:]),axis=1) if chosen else np.full(len(rate),np.nan)
    return dict(mean=free_flow_means(cache.meta['panel'],tuple(fit_dates)),history_speed=hs,history_flow_lane=hq,distance_bottleneck_km=distance,bottleneck_positions_km=g['position'][chosen].tolist())


def day_features(cache,day,bases,panel_code,gap_context=None):
    g=geometry(cache);nt,ns=cache.speed.shape[1:];shape=(nt,ns);features={}
    def put(name,value):
        a=np.asarray(value);a=a[:,None] if a.ndim==1 and len(a)==nt else a
        features[name]=np.broadcast_to(a,shape).astype(np.float32,copy=False)
    v,q,o=cache.speed[day],cache.flow[day],cache.occ[day]
    for source in ['time','space','asm']:
        put('base_'+source+'_speed',bases[source][0][day]);put('base_'+source+'_flow_lane',bases[source][1][day]/g['lanes'])
    for name in DYNAMIC:put(name,np.nan)
    neighbours=temporal(v)
    for name,a in neighbours.items():put('time_speed_'+name,a)
    for name,a in temporal(q).items():put('time_flow_'+name,a/g['lanes'] if name.endswith('value') else a)
    for name,a in temporal(o,count=1).items():
        if name.endswith('value'):put('time_density_occ_'+name,285*a)
    for offset in [-1,0,1]:
        for name,a in spatial_neighbours(v,q,o,g['position'],g['lanes'],offset).items():put(f'spatial_t{offset}_'+name,a)
    for label,c in [('free',g['vf']),('congested',-g['w'])]:
        for name,a in shifted_neighbours(v,q,g['position'],g['lanes'],c).items():put('characteristic_'+label+'_'+name,a)
    for source in ['time','space','asm']:
        bq=bases[source][1][day];bv=bases[source][0][day]
        put('fd_congested_speed_'+source,bq/np.maximum(g['kj']-bq/g['w'],1))
        put('density_'+source,bq/np.maximum(bv,1))
    for name,key in [('lanes','lanes'),('vf','vf'),('capacity_lane','cap'),('critical_density_lane','kc'),('jam_density_lane','kj'),('v_cut','v_cut'),('length_km','length'),('position_km','position')]:
        put(name,g[key]/g['lanes'] if name in ('capacity_lane','critical_density_lane','jam_density_lane') else g[key])
    dt=date.fromisoformat(cache.dates[day]);t=np.arange(nt)[:,None]
    put('panel',panel_code);put('regime',{'R1':0,'R2':1,'R3':2}[cache.regimes[day]])
    put('time_sin',np.sin(2*np.pi*t/nt));put('time_cos',np.cos(2*np.pi*t/nt));put('weekday',dt.weekday());put('weekend',int(dt.weekday()>=5))
    observed=np.isfinite(v);support=uniform_filter(np.ones(shape),size=(7,9),mode='constant')
    obs_count=uniform_filter(observed.astype(float),size=(7,9),mode='constant')
    put('observed_fraction_nearby',obs_count/support)
    queued=observed&(v<=g['v_cut']);queue_count=uniform_filter(queued.astype(float),size=(7,9),mode='constant')
    put('queued_fraction_nearby',np.divide(queue_count,obs_count,out=np.full(shape,np.nan),where=obs_count>0))
    put('inside_gap',cache.gap[day])
    for name in ['minutes_since_gap_start','minutes_until_gap_end','window_condition']:put(name,(gap_context or {}).get(name,np.nan if name!='window_condition' else 0))
    free_override=(neighbours['before1_value']>.85*g['vf'])&(neighbours['after1_value']>.85*g['vf'])
    return features,free_override


def fill_dynamic(matrix,columns,station_index,time_index,weekend,override,summaries):
    index={name:i for i,name in enumerate(columns)}
    means=summaries['mean'][station_index]
    matrix[:,index['free_flow_mean']]=means
    matrix[:,index['base_B2_speed']]=np.where(override,means,matrix[:,index['base_time_speed']])
    matrix[:,index['history_speed']]=summaries['history_speed'][weekend,time_index,station_index]
    matrix[:,index['history_flow_lane']]=summaries['history_flow_lane'][weekend,time_index,station_index]
    matrix[:,index['distance_bottleneck_km']]=summaries['distance_bottleneck_km'][station_index]
    return matrix


def training_sample(cache,day,true_speed,rng):
    g=geometry(cache);target=cache.target[day]
    congested=np.isfinite(true_speed)&(true_speed<=g['v_cut'])
    in_time=maximum_filter1d(congested,size=7,axis=0,mode='constant')
    near=np.zeros_like(congested)
    for s,x in enumerate(g['position']):near[:,s]=in_time[:,abs(g['position']-x)<=1].any(axis=1)
    free=true_speed>.85*g['vf']; draw=rng.random(target.shape)<.1
    selected=target&(~free|near|draw)
    weight=np.where(free&~near,10.,1.)
    return selected,weight
