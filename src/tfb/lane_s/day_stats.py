"""Additive day statistics for exact paired bootstrap of the Lane S objective."""
import numpy as np
import pandas as pd
from tfb import metric


def day_statistics(cache, pv, pq, tv, tq, root=None, mode='A'):
    """Assign forward transitions, including midnight, to their start date.

    Model arrays remain float32. Small daily scoring fields and reductions use
    float64, matching the verified Lead metric on exact released truth.
    """
    if not cache.meta['stations_links_one_to_one']:
        raise ValueError('Day statistics require one station per scored link')
    if mode not in ('A','B','C'):
        raise ValueError('Unknown physics mode')
    params, topo, on, off = metric._parameters(cache, root or metric.project_root())
    links = params.link_id.to_numpy()
    lane=params.lanes.to_numpy(float)
    vf=params.free_speed_kmh.to_numpy(float)
    cap=params.capacity_vph.to_numpy(float)
    crit=params.critical_density.to_numpy(float)
    jam=params.k_jam.to_numpy(float)
    length=params.length_km.to_numpy(float)
    shape=tuple(cache.meta['shape'])
    if any(x.shape!=shape for x in (pv,pq,tv,tq)):
        raise ValueError('Day-statistic arrays are not aligned')

    def field(d):
        target=cache.target[d]
        v=np.where(target,pv[d],tv[d]).astype(float)
        q=np.where(target,pq[d],tq[d]).astype(float)
        valid=cache.elig[d]&np.isfinite(v)&np.isfinite(q)
        return v,q,valid,q/np.maximum(v,1)*length,np.asarray(tq[d],float)/np.maximum(tv[d],1)*length

    available={r:np.zeros(len(links),bool) for r in metric.REGIMES}
    for d,r in enumerate(cache.regimes):
        available[r]|=field(d)[2].any(axis=0)
    topology_ids=set(topo.index)
    observable={}
    for r,seen in available.items():
        if mode=='C': seen=seen&~np.isin(links,list(on|off))
        seen_links=set(links[seen])
        keep=[]
        for link in links:
            if link not in topo.index:
                keep.append(False); continue
            incoming={s.strip() for s in str(topo.loc[link,'incoming_link_ids']).split(';')}
            outgoing={s.strip() for s in str(topo.loc[link,'outgoing_link_ids']).split(';')}
            keep.append(((incoming|outgoing)&topology_ids).issubset(seen_links))
        observable[r]=seen&np.asarray(keep,bool)

    def ramp_valid(d):
        valid=np.ones((shape[1],shape[2]),bool)
        if mode=='C': return valid
        for kind,attached in [('OR',on),('FR',off)]:
            values=np.ones_like(valid)
            values[:,np.isin(links,list(attached))]=False
            for ramp in cache.ramps[cache.ramps.ramp_type==kind].itertuples(index=False):
                values[:,ramp.station_idx]|=cache.ramp_elig[d,:,ramp.idx]&np.isfinite(cache.ramp_flow[d,:,ramp.idx])
            valid&=values
        return valid

    rows=[]
    for d,r in enumerate(cache.regimes):
        v,q,valid,n,nt=field(d)
        target=cache.target[d]
        usable=target&np.isfinite(tv[d])&np.isfinite(tq[d])
        pred_v=np.asarray(pv[d],float)
        pred_q=np.asarray(pq[d],float)
        dv=np.where(np.isfinite(pred_v),pred_v,0)-tv[d]
        dq=(np.where(np.isfinite(pred_q),pred_q,0)-tq[d])/lane
        density=q/np.maximum(v,1)/lane
        qfd=np.where(density<=crit/lane,vf*density,
            cap/lane/np.maximum(jam/lane-crit/lane,1e-9)*np.maximum(jam/lane-density,0))
        supplied=target&valid
        row={'panel':cache.meta['panel'],'date':str(cache.dates[d]),'regime':str(r),
             'n_target':int(usable.sum()),'sq_v':float((dv[usable]**2).sum()),'sq_q_lane':float((dq[usable]**2).sum()),
             'fd_error':float(np.abs((q/lane)[valid]-qfd[valid]).sum()),'fd_flow':float(np.abs((q/lane)[valid]).sum()),
             'n_fd_rows':int(valid.sum()),'n_submitted':int(supplied.sum()),'n_low_flow':int((supplied&(q<50)).sum())}
        ramps=ramp_valid(d)
        transitions=valid[:-1]&valid[1:]&observable[r][None,:]&ramps[:-1]
        dn,dnt=np.diff(n,axis=0),np.diff(nt,axis=0)
        error=float(np.abs(dn-dnt)[transitions].sum())
        scale=float(np.abs(dnt)[transitions].sum())
        count=int(transitions.sum())
        if d+1<len(cache.dates) and cache.regimes[d+1]==r and shape[1]==288:
            if (pd.Timestamp(str(cache.dates[d+1]))-pd.Timestamp(str(cache.dates[d]))).days==1:
                _,_,next_valid,next_n,next_nt=field(d+1)
                valid_cross=valid[-1]&next_valid[0]&observable[r]&ramps[-1]
                cross=(next_n[0]-n[-1])-(next_nt[0]-nt[-1])
                error+=float(np.abs(cross[valid_cross]).sum())
                scale+=float(np.abs((next_nt[0]-nt[-1])[valid_cross]).sum())
                count+=int(valid_cross.sum())
        row.update(lwr_error=error,lwr_rhs=scale,n_transition=count)
        rows.append(row)
    return pd.DataFrame(rows)
