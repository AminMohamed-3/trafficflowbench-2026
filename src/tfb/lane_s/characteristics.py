"""Observed values on +/-5,10,15 minute physical traffic characteristics."""
import numpy as np
from .features import geometry

def interpolate_observed(speed,flow,occ,coordinate,query,position,source_step,target_lanes):
    nt,ns=speed.shape;shape=(nt,ns)
    rows=np.arange(nt)+source_step;time_ok=(rows>=0)&(rows<nt);tt=np.clip(rows,0,nt-1)
    v,q,o=speed[tt],flow[tt],occ[tt]
    valid=np.isfinite(v)&np.isfinite(q)&time_ok[:,None]
    index=np.broadcast_to(np.arange(ns),shape)
    before=np.maximum.accumulate(np.where(valid,index,-1),axis=1)
    after=np.minimum.accumulate(np.where(valid,index,ns)[:,::-1],axis=1)[:,::-1]
    left=np.searchsorted(coordinate,query,side='right')-1;right=left+1
    up=before[:,np.clip(left,0,ns-1)];down=after[:,np.clip(right,0,ns-1)]
    up=np.where(left[None,:]>=0,up,-1);down=np.where(right[None,:]<ns,down,ns)
    has_up=up>=0;has_down=down<ns;ui=np.clip(up,0,ns-1);di=np.clip(down,0,ns-1)
    row=np.broadcast_to(np.arange(nt)[:,None],shape)
    cu,cd=coordinate[ui],coordinate[di]
    denominator=cd-cu
    fraction=np.divide(query[None,:]-cu,denominator,out=np.zeros(shape),where=denominator>0)
    fraction=np.clip(fraction,0,1)
    wu=np.where(has_up&has_down,1-fraction,has_up.astype(float))
    wd=np.where(has_up&has_down,fraction,has_down.astype(float))
    domain=(query>=coordinate[0])&(query<=coordinate[-1])
    support=(has_up|has_down)&time_ok[:,None]&domain[None,:]
    result={}
    for name,a in [('speed',v),('flow_per_target_lane',q),('density_occ',285*o)]:
        av,bv=a[row,ui],a[row,di]
        good_u=np.isfinite(av)&has_up;good_d=np.isfinite(bv)&has_down
        weight_u=wu*good_u;weight_d=wd*good_d;weight=weight_u+weight_d
        out=np.divide(np.where(good_u,av,0)*weight_u+np.where(good_d,bv,0)*weight_d,weight,out=np.full(shape,np.nan),where=weight>0)
        out=np.where(support,out,np.nan)
        if name=='flow_per_target_lane':out=out/target_lanes[None,:]
        result[name]=out.astype(np.float32)
    # Physical uncertainty of the interpolation, not station identifiers.
    desired_position=np.interp(query,coordinate,position)
    distance=wu*abs(position[ui]-desired_position[None,:])+wd*abs(position[di]-desired_position[None,:])
    result['support_distance_km']=np.where(support,distance,np.nan).astype(np.float32)
    return result

def day_characteristics(cache,day):
    g=geometry(cache);result={}
    for label,velocity,sign in [('free',g['vf'],1),('congested',g['w'],-1)]:
        increments=g['length']/velocity*60
        coordinate=increments.cumsum()-increments/2
        for step in [-3,-2,-1,1,2,3]:
            query=coordinate+sign*step*5
            fields=interpolate_observed(cache.speed[day],cache.flow[day],cache.occ[day],coordinate,query,g['position'],step,g['lanes'])
            for name,a in fields.items():result[f'travel_{label}_step{step}_{name}']=a
    assert len(result)==48
    return result

def known_answers():
    from types import SimpleNamespace
    import pandas as pd
    ns=400;nt=72;length=np.full(ns,.2);lanes=2+np.arange(ns)%3
    vf=np.full(ns,120.);capacity=10000*lanes;kc=capacity/vf;kj=kc+capacity/30.
    stations=pd.DataFrame({'length_km':length,'lanes':lanes,'free_speed_kmh':vf,'capacity_vph':capacity,'critical_density':kc,'k_jam':kj,'v_cut':.6*vf})
    rng=np.random.default_rng(4701);max_error=0.;checked=0
    for label,velocity,sign in [('free',vf,1),('congested',np.full(ns,30.),-1)]:
        increment=length/velocity*60;coordinate=increment.cumsum()-increment/2
        invariant=coordinate[None,:]-sign*np.arange(nt)[:,None]*5
        q=8000+2*invariant;v=80+.01*invariant;o=q/25000
        missing=rng.random((nt,ns))<.3
        cache=SimpleNamespace(stations=stations,speed=np.where(missing,np.nan,v)[None],flow=np.where(missing,np.nan,q)[None],occ=np.where(missing,np.nan,o)[None])
        fields=day_characteristics(cache,0)
        for step in [-3,-2,-1,1,2,3]:
            name=f'travel_{label}_step{step}_flow_per_target_lane';pred=fields[name]
            # Ignore physical-domain edges and source intervals without two
            # observed bracketing cells, where flat endpoint fallback is explicit.
            desired=coordinate+sign*step*5;inside=(desired>coordinate[10])&(desired<coordinate[-11])
            usable=np.isfinite(pred)&inside[None,:]
            error=abs(pred-q/lanes)[usable]
            assert error.size and error.max()<.0003
            max_error=max(max_error,float(error.max()));checked+=int(usable.sum())
        constant=np.full((nt,ns),2000.)
        field=interpolate_observed(np.full_like(constant,100),constant,np.ones_like(constant)*.1,coordinate,coordinate+5,g_position:=length.cumsum()-length/2,1,lanes)['flow_per_target_lane']
        valid=np.isfinite(field);assert np.max(abs(field[valid]-(constant/lanes)[valid]))<.0001
    return {'verdict':'PASS','traveling_wave_cells':checked,'max_float32_flow_lane_error':max_error,'varying_lane_totalflow_preservation':'PASS','n_characteristic_features':48}

if __name__=='__main__':
    import json
    print(json.dumps(known_answers()),flush=True)
