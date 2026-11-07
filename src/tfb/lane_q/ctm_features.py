"""Q05 canonical triangular-FD CTM; inputs come only from OnlineView."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .ongoing_features import feature_frame as base_frame,observation_grids,mean_finite,FEATURES as BASE

CORE=['ctm_density_ratio','ctm_speed_ratio','ctm_queued']
EXTRA=['ctm_flow_ratio','ctm_inflow_ratio','ctm_outflow_ratio','ctm_density_change',
       'ctm_queue_tail_distance_km','ctm_queue_head_distance_km','ctm_initial_density_ratio',
       'ctm_initial_observed','ctm_initial_age_minutes','ctm_initial_valid_share',
       'ctm_boundary_demand_ratio','ctm_boundary_observed_share','ctm_boundary_fallback']
FEATURES=BASE+CORE+EXTRA
DOSES={'F0':BASE,'Fhalf':BASE+CORE,'Ffull':FEATURES}
BUILDER='Q05_canonical_ctm_v1'


def parameters(network):
    fd=network.fd_parameters
    names=['length_km','free_speed_kmh','capacity_vph','critical_density','k_jam']
    if fd is None or any(c not in fd for c in names):raise ValueError('CTM requires released triangular FD')
    length,vf,capacity,kcrit,kjam=[fd[c].to_numpy(dtype=np.float32) for c in names]
    if not all(np.isfinite(x).all() for x in [length,vf,capacity,kcrit,kjam]):raise ValueError('Invalid FD')
    if ((length<=0)|(vf<=0)|(capacity<=0)|(kcrit<=0)|(kjam<=kcrit)).any():raise ValueError('Invalid FD bounds')
    if not np.allclose(vf*kcrit,capacity,rtol=1e-5):raise ValueError('FD critical density units disagree')
    wave=capacity/(kjam-kcrit)
    return length,vf,capacity,kcrit,kjam,wave


def initial_state(view):
    """Carry observations forward to T, then interpolate spatial density gaps.

    No future time interpolation. A completely unobserved chain uses zero
    density and an explicit missing-quality flag; missing boundary demand uses
    initial FD flows and is explicitly flagged, never an invented measurement.
    """
    length,vf,capacity,kcrit,kjam,wave=parameters(view.network)
    grids=observation_grids(view);rawspeed,speed,si=grids['speed_kmh'];rawflow,flow,fi=grids['flow_vph']
    v=speed[-1];q=flow[-1];valid=np.isfinite(v)&np.isfinite(q)&(v>0)&(q>=0)
    km=np.cumsum(length,dtype=np.float64)-.5*length
    density=np.zeros(len(length),dtype=np.float32)
    if valid.any():density[:]=np.interp(km,km[valid],q[valid]/v[valid]).astype(np.float32)
    density=np.clip(density,0,kjam).astype(np.float32)
    # Four 5-minute observations span T-15 through T, including observed T.
    boundary=rawflow[max(0,len(rawflow)-4):,:min(3,len(length))]
    observed=np.isfinite(boundary)&(boundary>=0)
    fallback=not observed.any()
    if fallback:
        demand=float(mean_finite(np.minimum(vf*density,wave*(kjam-density))[:min(3,len(length))]))
    else:demand=float(boundary[observed].mean(dtype=np.float64))
    age=np.maximum(len(rawspeed)-1-si[-1],len(rawflow)-1-fi[-1])*5
    age=np.where(valid,age,np.nan).astype(np.float32)
    return {'density':density,'demand':np.float32(max(0,demand)),
        'valid':valid.astype(np.float32),'age':age,'valid_share':np.float32(valid.mean()),
        'boundary_share':np.float32(observed.mean()),'boundary_fallback':np.float32(fallback)}


def simulate(network,density,demand):
    """Godunov sending/receiving fluxes with CFL substeps; open downstream.

    Density is vehicles/km across all lanes and flux is vehicles/hour. Supports
    arbitrary batch axes; each window follows exactly the single-window math.
    Returns each 5-minute state and mean incoming/outgoing flux in that interval.
    """
    length,vf,capacity,kcrit,kjam,wave=parameters(network)
    k=np.asarray(density,dtype=np.float32).copy()
    if k.shape[-1]!=len(length) or not np.isfinite(k).all() or ((k<0)|(k>kjam)).any():raise ValueError('Invalid initial density')
    boundary=np.broadcast_to(np.asarray(demand,dtype=np.float32),k.shape[:-1])
    if not np.isfinite(boundary).all() or (boundary<0).any():raise ValueError('Invalid boundary demand')
    cfl=float(np.min(length/np.maximum(vf,wave)))
    subdivisions=max(1,int(np.ceil((5/60)/(.95*cfl))))
    dt=np.float32((5/60)/subdivisions);scale=dt/length
    densities=[];inflows=[];outflows=[]
    faces=np.empty((*k.shape[:-1],k.shape[-1]+1),dtype=np.float32)
    for step in range(6):
        incoming=np.zeros_like(k);outgoing=np.zeros_like(k)
        for _ in range(subdivisions):
            sending=np.minimum(vf*k,capacity)
            receiving=np.minimum(capacity,wave*(kjam-k))
            faces[...,0]=np.minimum(boundary,receiving[...,0])
            faces[...,1:-1]=np.minimum(sending[...,:-1],receiving[...,1:])
            faces[...,-1]=sending[...,-1]
            k+=scale*(faces[...,:-1]-faces[...,1:])
            # Under the CFL bound these are invariant without mass-changing clipping.
            incoming+=faces[...,:-1]/np.float32(subdivisions)
            outgoing+=faces[...,1:]/np.float32(subdivisions)
        if not np.isfinite(k).all() or ((k<-.001)|(k>kjam+.001)).any():raise AssertionError('CTM invariant domain failed')
        densities.append(k.copy());inflows.append(incoming);outflows.append(outgoing)
    density=np.stack(densities)
    fdflow=np.minimum(np.minimum(vf*density,capacity),wave*(kjam-density))
    speed=np.divide(fdflow,density,out=np.broadcast_to(vf,density.shape).copy(),where=density>1e-6)
    return {'density':density,'flow':fdflow,'speed':speed,'inflow':np.stack(inflows),'outflow':np.stack(outflows),'substeps_per_5min':subdivisions}


def forecast_frame(network,state,result):
    length,vf,capacity,kcrit,kjam,wave=parameters(network);n=len(length)
    density=result['density'];speed=result['speed'];queued=speed<=network.cutoffs
    km=np.cumsum(length,dtype=np.float64)-.5*length
    tails=np.full((6,n),np.nan,dtype=np.float32);heads=tails.copy()
    # Nearest simulated component preserves multiple spatial queue sites.
    from .ongoing_features import components
    for step in range(6):
        pairs=components(queued[step])
        if pairs:
            a=np.asarray([p[0] for p in pairs]);b=np.asarray([p[1] for p in pairs])
            chosen=np.abs(km[:,None]-np.clip(km[:,None],km[a],km[b])).argmin(axis=1)
            tails[step]=km-km[a[chosen]];heads[step]=km-km[b[chosen]]
    values={'ctm_density_ratio':density/kcrit,'ctm_speed_ratio':speed/vf,'ctm_queued':queued,
        'ctm_flow_ratio':result['flow']/capacity,'ctm_inflow_ratio':result['inflow']/capacity,
        'ctm_outflow_ratio':result['outflow']/capacity,'ctm_density_change':(density-state['density'])/kcrit,
        'ctm_queue_tail_distance_km':tails,'ctm_queue_head_distance_km':heads,
        'ctm_initial_density_ratio':np.tile(state['density']/kcrit,(6,1)),
        'ctm_initial_observed':np.tile(state['valid'],(6,1)),
        'ctm_initial_age_minutes':np.tile(state['age'],(6,1)),
        'ctm_initial_valid_share':np.full((6,n),state['valid_share']),
        'ctm_boundary_demand_ratio':np.full((6,n),state['demand']/capacity[0]),
        'ctm_boundary_observed_share':np.full((6,n),state['boundary_share']),
        'ctm_boundary_fallback':np.full((6,n),state['boundary_fallback'])}
    frame=pd.DataFrame({name:np.asarray(values[name],dtype=np.float32).reshape(-1) for name in CORE+EXTRA})
    if np.isinf(frame.to_numpy()).any():raise AssertionError('Infinite CTM feature')
    return frame


def feature_frame(view,sites):
    base=base_frame(view,sites);state=initial_state(view)
    result=simulate(view.network,state['density'],state['demand'])
    return pd.concat([base,forecast_frame(view.network,state,result)],axis=1)


def batched_forecast_frames(network,states):
    result=simulate(network,np.stack([s['density'] for s in states]),np.asarray([s['demand'] for s in states]))
    frames=[]
    for i,state in enumerate(states):
        one={name:value[:,i] for name,value in result.items() if name!='substeps_per_5min'}
        frames.append(forecast_frame(network,state,one))
    return frames,result['substeps_per_5min']
