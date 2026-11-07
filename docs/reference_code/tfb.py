"""Scratch helpers: load one panel's train days as dense arrays [day, t, link] ordered by topology."""
import pandas as pd, numpy as np, glob, os
ROOT='data/raw/kaggle_public'
def net(panel):
    b=f'{ROOT}/corridors/{panel}/network'
    fd=pd.read_csv(f'{b}/fd_parameters.csv'); links=pd.read_csv(f'{b}/links.csv'); topo=pd.read_csv(f'{b}/lwr_mainline_topology.csv',dtype=str)
    return fd,links,topo
def load_days(panel, split='train', months=None, masked_only=False):
    base=f'{ROOT}/corridors/{panel}/{split}'
    tf=sorted(glob.glob(f'{base}/mainline_states/*/*.parquet'))
    mf={os.path.basename(p):p for p in glob.glob(f'{base}/mainline_states_masked/*/*.parquet')}
    cols=['date','timestamp','station_id','link_id','milepost','speed_kmh','flow_vph','pct_observed','is_score_eligible']
    out=[]
    for p in (tf if not masked_only else sorted(mf.values())):
        n=os.path.basename(p)
        if n not in mf: continue
        t=pd.read_parquet(p,columns=cols) if not masked_only else None
        m=pd.read_parquet(mf[n],columns=cols+['mask_regime'])
        out.append((n,t,m))
    return out
def to_arrays(frames, link_order):
    """frames: list of (name,truth_df,masked_df). returns dict of arrays [D,288,L]."""
    L=len(link_order); idx={l:i for i,l in enumerate(link_order)}
    D=len(frames)
    V=np.full((D,288,L),np.nan); Q=np.full((D,288,L),np.nan); VM=np.full((D,288,L),np.nan); QM=np.full((D,288,L),np.nan)
    E=np.zeros((D,288,L),bool); reg=[]; dates=[]
    for d,(n,t,m) in enumerate(frames):
        ts=pd.to_datetime(m.timestamp); ti=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(); li=m.link_id.map(idx).to_numpy()
        ok=~np.isnan(li.astype(float)); li=li[ok].astype(int); ti2=ti[ok]
        VM[d,ti2,li]=m.speed_kmh.to_numpy()[ok]; QM[d,ti2,li]=m.flow_vph.to_numpy()[ok]; E[d,ti2,li]=m.is_score_eligible.to_numpy()[ok]==1
        if t is not None:
            ts=pd.to_datetime(t.timestamp); ti=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(); li=t.link_id.map(idx).to_numpy(); ok=~np.isnan(li.astype(float)); li=li[ok].astype(int)
            V[d,ti[ok],li]=t.speed_kmh.to_numpy()[ok]; Q[d,ti[ok],li]=t.flow_vph.to_numpy()[ok]
        reg.append(m.mask_regime.iloc[0]); dates.append(m.date.iloc[0])
    return dict(V=V,Q=Q,VM=VM,QM=QM,E=E,reg=np.array(reg),dates=np.array(dates))

def interp_time(X):
    """linear interpolation along axis 1 (time) per (day, link); edges filled by nearest."""
    D,T,L=X.shape; out=X.copy(); t=np.arange(T)
    for d in range(D):
        for l in range(L):
            y=X[d,:,l]; ok=~np.isnan(y)
            if ok.sum()==0: continue
            out[d,:,l]=np.interp(t,t[ok],y[ok])
    return out
def interp_space(X, pos):
    """linear interpolation along axis 2 (links, positions pos) per (day, t)."""
    D,T,L=X.shape; out=X.copy()
    for d in range(D):
        for i in range(T):
            y=X[d,i]; ok=~np.isnan(y)
            if ok.sum()==0: continue
            out[d,i]=np.interp(pos,pos[ok],y[ok])
    return out
def score_state(P_v,P_q,A,lanes):
    """A: arrays dict; targets = E & isnan(VM) & truth present. returns per regime dict."""
    tgt=A['E']&np.isnan(A['VM'])&~np.isnan(A['V'])
    res={}
    for r in ['R1','R2','R3']:
        dm=A['reg']==r
        if dm.sum()==0: continue
        t=tgt[dm]; ev=(P_v[dm]-A['V'][dm])[t]; eq=((P_q[dm]-A['Q'][dm])/lanes[None,None,:])[t]
        rv=np.sqrt(np.mean(ev**2)); rq=np.sqrt(np.mean(eq**2))
        res[r]=dict(n=int(t.sum()),rmse_v=rv,rmse_q=rq,S=0.54*max(0,1-rv/25)+0.46*max(0,1-rq/600))
    res['S_state']=np.mean([res[r]['S'] for r in ['R1','R2','R3'] if r in res])
    return res
def score_physics(P_v,P_q,A,fdp,length):
    """proxy: rhs = dN_true. fdp: dict of arrays per link (lanes, vf, cap, kjam). No ramp-validity filter."""
    tgt=A['E']&np.isnan(A['VM'])&~np.isnan(A['V'])
    res={}
    lanes=fdp['lanes']; vf=fdp['vf']; cap=fdp['cap']; kc=cap/np.maximum(vf,1); kj=np.maximum(fdp['kjam'],kc*1.05)
    for r in ['R1','R2','R3']:
        dm=np.where(A['reg']==r)[0]
        if len(dm)==0: continue
        E=A['E'][dm]; t=tgt[dm]
        v=np.where(t,P_v[dm],A['V'][dm]); q=np.where(t,P_q[dm],A['Q'][dm])
        v=np.where(E,v,np.nan); q=np.where(E,q,np.nan)
        vt=np.where(E,A['V'][dm],np.nan); qt=np.where(E,A['Q'][dm],np.nan)
        # FD
        k=q/np.maximum(v,1); ql=q/lanes; kl=k/lanes; capl=cap/lanes; kcl=kc/lanes; kjl=kj/lanes
        qfd=np.where(kl<=kcl, vf*kl, capl/np.maximum(kjl-kcl,1e-9)*np.maximum(kjl-kl,0))
        ok=E
        s_fd=1-np.nansum(np.abs(ql-qfd)[ok])/np.nansum(np.abs(ql)[ok])
        kt=qt/np.maximum(vt,1)
        N=k*length; Nt=kt*length
        # flatten days in chronological order: [D*288, L]; consecutive days are not necessarily consecutive dates -> only within day + across adjacent dates
        Nf=N.reshape(-1,N.shape[2]); Ntf=Nt.reshape(-1,N.shape[2])
        dN=Nf[1:]-Nf[:-1]; dNt=Ntf[1:]-Ntf[:-1]
        # break transitions at day boundaries when dates not consecutive
        dates=pd.to_datetime(A['dates'][dm]); valid=np.ones(dN.shape[0],bool)
        for i in range(len(dm)-1):
            if (dates[i+1]-dates[i]).days!=1: valid[(i+1)*288-1]=False
        okT=valid[:,None]&~np.isnan(dN)&~np.isnan(dNt)
        num=np.abs(dN-dNt)[okT].sum(); den=np.abs(dNt)[okT].sum()
        res[r]=dict(S_FD=s_fd,S_LWR=max(0,1-num/den),E=num/den)
    res['S_physics']=np.mean([(res[r]['S_FD']+2*res[r]['S_LWR'])/3 for r in ['R1','R2','R3'] if r in res])
    return res
