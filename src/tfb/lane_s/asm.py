"""Canonical Treiber-Helbing ASM with exact discrete exponential kernels."""
import numpy as np
from scipy.signal import lfilter
from .predict import predict_arrays


def characteristic_kernel(values, positions, speeds, sigma_km, tau_min, step_min=5.):
    """Normalised kernel exp(-|dx|/sigma-|dt-dx/c|/tau), within each day.

    Recursions sum past/future exponentials exactly. At a fractional shifted
    index, the two sums receive their exact exponential factors, not a linear
    interpolation. All stations participate; no spatial or temporal cutoff.
    """
    if sigma_km <= 0 or tau_min <= 0 or np.any(np.asarray(speeds) == 0):
        raise ValueError('Positive widths and nonzero characteristic speeds required')
    values=np.asarray(values)
    valid=np.isfinite(values)
    fields=np.stack([np.where(valid,values,0.),valid.astype(float)],axis=0)
    r=np.exp(-step_min/tau_min)
    left=lfilter([1.],[1.,-r],fields,axis=2)
    right=lfilter([1.],[1.,-r],fields[:,:,::-1,:],axis=2)[:,:,::-1,:]
    nd,nt,ns=values.shape
    from scipy.sparse import csr_matrix
    result=np.zeros((2,nd,nt,ns),dtype=float)
    dx=np.asarray(positions)[None,:]-np.asarray(positions)[:,None]
    space=np.exp(-abs(dx)/sigma_km)
    shift=dx/np.asarray(speeds)[:,None]*60./step_min
    whole=np.floor(shift).astype(int); fraction=shift-whole
    times=np.arange(nt)
    # Group equal integer shifts; CSR multiplication handles all station pairs
    # in compiled code while preserving their fractional exponential factors.
    flat=result.reshape(-1,ns)
    for m in np.unique(whole):
        target,source=np.nonzero(whole==m)
        wl=space[target,source]*np.exp(-step_min/tau_min*fraction[target,source])
        wr=space[target,source]*np.exp(-step_min/tau_min*(1-fraction[target,source]))
        L=csr_matrix((wl,(target,source)),shape=(ns,ns))
        R=csr_matrix((wr,(target,source)),shape=(ns,ns))
        li=times+m; ri=li+1
        lf=np.where(li<0,0.,np.exp(-step_min/tau_min*np.maximum(li-(nt-1),0)))
        rf=np.where(ri>=nt,0.,np.exp(step_min/tau_min*np.minimum(ri,0)))
        lv=left[:,:,np.clip(li,0,nt-1),:]*lf[None,None,:,None]
        rv=right[:,:,np.clip(ri,0,nt-1),:]*rf[None,None,:,None]
        flat += L.dot(lv.reshape(-1,ns).T).T
        flat += R.dot(rv.reshape(-1,ns).T).T
    return np.divide(result[0],result[1],out=np.full_like(result[0],np.nan),where=result[1]>0).astype(np.float32)


def predict_asm(cache,cfg):
    """Speed mixes congested/free fields; total flow uses free characteristic."""
    base_v,base_q=predict_arrays(cache,{'method':'B1'})
    st=cache.stations
    vf=st.free_speed_kmh.to_numpy(float)
    cap=st.capacity_vph.to_numpy(float)
    # The cache uses total-lane FD densities, matching total capacity.
    w=cap/(st.k_jam.to_numpy(float)-st.critical_density.to_numpy(float))
    length=st.length_km.to_numpy(float)
    positions=length.cumsum()-length/2
    sigma,tau=float(cfg['sigma_km']),float(cfg['tau_min'])
    vf_field=characteristic_kernel(cache.speed,positions,vf,sigma,tau)
    vc_field=characteristic_kernel(cache.speed,positions,-w,sigma,tau)
    weight=.5*(1+np.tanh((float(cfg.get('v_c_kmh',70))-np.minimum(vf_field,vc_field))/float(cfg.get('delta_v_kmh',20))))
    speed=weight*vc_field+(1-weight)*vf_field
    flow=characteristic_kernel(cache.flow,positions,vf,sigma,tau)
    speed=np.where(np.isfinite(speed),speed,base_v).astype(np.float32)
    flow=np.where(np.isfinite(flow),flow,base_q).astype(np.float32)
    np.clip(speed,3.,120.75,out=speed)
    upper=1.3*cap; upper32=upper.astype(np.float32)
    upper32=np.where(upper32.astype(float)>upper,np.nextafter(upper32,np.float32(-np.inf)),upper32)
    np.maximum(flow,50.,out=flow); np.minimum(flow,upper32,out=flow)
    return speed,flow


def known_answers():
    rng=np.random.default_rng(123)
    data=rng.uniform(10,100,(2,8,4)); data[rng.random(data.shape)<.45]=np.nan
    positions=np.array([0.,.4,1.7,3.1]); max_delta=0.
    for speeds in [np.array([100.,110.,90.,113.]),np.array([-23.,-25.,-27.,-26.])]:
        actual=characteristic_kernel(data,positions,speeds,.6,3.)
        expected=np.empty_like(actual)
        for d in range(2):
            for t in range(8):
                for s in range(4):
                    dx=positions[None,:]-positions[s]
                    dt=5*(np.arange(8)[:,None]-t)
                    weights=np.exp(-abs(dx)/.6-abs(dt-dx/speeds[s]*60)/3.)
                    valid=np.isfinite(data[d]); expected[d,t,s]=(weights*np.where(valid,data[d],0)).sum()/(weights*valid).sum()
        max_delta=max(max_delta,float(np.max(abs(actual-expected))))
    constant=np.full((1,8,4),65.); constant[:,2:6,1:3]=np.nan
    constant_error=float(np.max(abs(characteristic_kernel(constant,positions,np.ones(4)*100,.6,5)-65)))
    assert max_delta<1e-5 and constant_error<1e-5
    return {'brute_force_max_abs_delta':max_delta,'constant_error':constant_error,'characteristic_sign':'dx=observed-target; upstream observations are earlier for +vf, later for -w','verdict':'PASS'}
