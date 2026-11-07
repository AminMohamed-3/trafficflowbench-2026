"""Q07 causal, transport-aligned precursors, added to the full Q03 base.

A short corridor has partial travel coverage, not an unobserved station invented
outside its boundary. Interpolation brackets are both observations at or before T.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .onset_features import feature_frame as base_features, NUMERIC_FEATURES as BASE_FEATURES

REPAIR_FEATURES = ["upstream_flow_ratio", "upstream_speed_ratio", "upstream_flow_age_min",
    "upstream_speed_age_min", "upstream_distance_km", "upstream_travel_min",
    "upstream_observed_fraction", "upstream_exists", "transport_path_coverage_30",
    "transport_unobserved_lead_30"]
TRANSPORT_FEATURES = [f"transport_{name}_{h}" for h in (5,15,30) for name in
    ("flow_ratio", "speed_ratio", "flow_mean_ratio", "flow_slope_ratio", "speed_min_ratio",
     "flow_age_min", "speed_age_min", "travel_min", "unobserved_lead_min",
     "observed_fraction", "source_count", "interpolation_span_min")]
CONTRAST_FEATURES = ["upstream_vs_site_flow", "upstream_vs_site_speed", "transport30_vs_site_flow",
    "transport30_vs_upstream_flow", "transport5_vs_30_flow", "transport15_vs_30_flow"]
NUMERIC_FEATURES = BASE_FEATURES + REPAIR_FEATURES + TRANSPORT_FEATURES + CONTRAST_FEATURES
DOSES = {"F0": BASE_FEATURES, "Fhalf": BASE_FEATURES+REPAIR_FEATURES,
         "Ffull": NUMERIC_FEATURES}


def causal_sample(times, values, wanted):
    """Return value, age and interpolation span; no right bracket after T.

    All times are minutes relative to T. Future queries use a last-known proxy,
    and the caller exposes the unsampled lead separately. No backward fill.
    """
    times,values=np.asarray(times,float),np.asarray(values,float)
    if np.any(times>0):raise AssertionError("Future observation in causal sampler")
    keep=np.isfinite(values)&np.isfinite(times)
    times,values=times[keep],values[keep]
    if not len(times):return np.nan,np.nan,np.nan
    order=np.argsort(times,kind="stable");times,values=times[order],values[order]
    query=min(float(wanted),0.)
    left=int(np.searchsorted(times,query,side="right"))-1
    if left<0:return np.nan,np.nan,np.nan
    if times[left]==query:return float(values[left]),0.,0.
    if left+1<len(times):
        right=left+1
        gap=times[right]-times[left]
        if gap<=0:raise AssertionError("Duplicate causal sample timestamp")
        value=values[left]+(values[right]-values[left])*(query-times[left])/gap
        return float(value),float(min(query-times[left],times[right]-query)),float(gap)
    return float(values[left]),float(query-times[left]),0.


class TransportContext:
    def __init__(self,view):
        self.network=view.network
        fd=view.network.fd_parameters
        self.capacity=fd.capacity_vph.to_numpy(float)
        length=fd.length_km.to_numpy(float)
        # Detector centres: an adjacent station's travel distance is nonzero.
        self.km=np.cumsum(length)-length/2
        self.travel=np.cumsum(length/view.network.free_speed*60)-length/view.network.free_speed*30
        if not (np.isfinite(self.capacity).all() and (self.capacity>0).all()
                and np.isfinite(self.travel).all() and np.all(np.diff(self.travel)>0)):
            raise ValueError("Invalid released transport geometry")
        obs=view.observations()
        if len(obs) and obs.timestamp.max()>view.origin:raise AssertionError("Online bound failed")
        self.series={}
        for link,g in obs.groupby("link_id",sort=False):
            t=(g.timestamp-view.origin).dt.total_seconds().to_numpy()/60
            if len(np.unique(t))!=len(t):raise AssertionError("Duplicate semantic observation")
            self.series[link]=(t,g.flow_vph.to_numpy(float),g.speed_kmh.to_numpy(float))

    def sample(self,pos,wanted,column):
        series=self.series.get(self.network.links[pos])
        if series is None:return np.nan,np.nan,np.nan
        return causal_sample(series[0],series[1 if column=="flow" else 2],wanted)

    def fraction(self,pos):
        series=self.series.get(self.network.links[pos])
        if series is None:return 0.
        keep=(series[0]>=-30)&(series[0]<=0)
        return float(min(1.,np.count_nonzero(keep&np.isfinite(series[1]))/7))

    def upstream(self,anchor):
        # Truly upstream if possible; a boundary site has no invented upstream.
        available=[]
        for pos in range(anchor):
            q,age,_=self.sample(pos,0.,"flow")
            if np.isfinite(q):available.append((pos,q,age))
        if not available:
            return {"upstream_flow_ratio":np.nan,"upstream_speed_ratio":np.nan,
                "upstream_flow_age_min":np.nan,"upstream_speed_age_min":np.nan,
                "upstream_distance_km":np.nan,"upstream_travel_min":np.nan,
                "upstream_observed_fraction":0.,"upstream_exists":0.}
        pos,q,age=max(available,key=lambda x:x[0])
        speed,speed_age,_=self.sample(pos,0.,"speed")
        return {"upstream_flow_ratio":q/self.capacity[anchor],
            "upstream_speed_ratio":speed/self.network.free_speed[pos],
            "upstream_flow_age_min":age,"upstream_speed_age_min":speed_age,
            "upstream_distance_km":self.km[anchor]-self.km[pos],
            "upstream_travel_min":self.travel[anchor]-self.travel[pos],
            "upstream_observed_fraction":self.fraction(pos),"upstream_exists":1.}

    def aligned(self,anchor,horizon):
        order=sorted(range(anchor),key=lambda j:(abs(self.travel[anchor]-self.travel[j]-horizon),-j))
        sources=[]
        for pos in order:
            travel=self.travel[anchor]-self.travel[pos]
            q,qage,span=self.sample(pos,horizon-travel,"flow")
            if np.isfinite(q):
                sources.append((pos,travel,q,qage,span))
                if len(sources)==3:break
        names=[name.rsplit("_",1)[0].removeprefix("transport_") for name in TRANSPORT_FEATURES if name.endswith(f"_{horizon}")]
        if not sources:
            result={f"transport_{name}_{horizon}":np.nan for name in names}
            result[f"transport_source_count_{horizon}"]=0.
            result[f"transport_observed_fraction_{horizon}"]=0.
            return result
        # Nearest three physical observations, with stable geometry-only weights.
        weights=np.array([1/(1+abs(travel-horizon)) for _,travel,*_ in sources]);weights/=weights.sum()
        def average(values):
            values=np.asarray(values,float);valid=np.isfinite(values)
            return float(np.dot(values[valid],weights[valid])/weights[valid].sum()) if valid.any() else np.nan
        speeds=[];speed_ages=[];means=[];slopes=[];mins=[]
        for pos,travel,q,qage,span in sources:
            speed,speed_age,_=self.sample(pos,horizon-travel,"speed")
            speeds.append(speed/self.network.free_speed[pos]);speed_ages.append(speed_age)
            qs=[];vs=[]
            for offset in (-20,-10,0):
                qq,_,_=self.sample(pos,horizon+offset-travel,"flow")
                vv,_,_=self.sample(pos,horizon+offset-travel,"speed")
                qs.append(qq/self.capacity[anchor]);vs.append(vv/self.network.free_speed[pos])
            valid=np.isfinite(qs)
            means.append(float(np.mean(np.array(qs)[valid])) if valid.any() else np.nan)
            slopes.append(float(np.polyfit(np.array([-20,-10,0])[valid],np.array(qs)[valid],1)[0]) if valid.sum()>=2 else np.nan)
            validv=np.isfinite(vs);mins.append(float(np.min(np.array(vs)[validv])) if validv.any() else np.nan)
        data={"flow_ratio":average([q/self.capacity[anchor] for _,_,q,_,_ in sources]),
            "speed_ratio":average(speeds),"flow_mean_ratio":average(means),"flow_slope_ratio":average(slopes),
            "speed_min_ratio":average(mins),"flow_age_min":average([x[3] for x in sources]),
            "speed_age_min":average(speed_ages),"travel_min":average([x[1] for x in sources]),
            "unobserved_lead_min":average([max(0,horizon-x[1]) for x in sources]),
            "observed_fraction":average([self.fraction(x[0]) for x in sources]),
            "source_count":float(len(sources)),"interpolation_span_min":average([x[4] for x in sources])}
        return {f"transport_{k}_{horizon}":v for k,v in data.items()}


def feature_frame(view,sites,*,prior_context=None,base_frame=None):
    base=base_features(view,sites,prior_context=prior_context) if base_frame is None else base_frame.copy()
    expected=[f"{view.panel}/{site['key']}" for site in sites]
    if base.site_uid.tolist()!=expected:raise AssertionError("Base/site geometry does not match")
    context=TransportContext(view)
    rows=[]
    for idx,site in enumerate(sites):
        if site['other']:
            last=[context.sample(pos,0.,"speed")[0]/view.network.free_speed[pos] for pos in site['positions']]
            anchor=site['positions'][int(np.nanargmin(last))] if np.isfinite(last).any() else site['positions'][0]
        else:anchor=max(site['positions'])
        row=context.upstream(anchor)
        row['transport_path_coverage_30']=min(1.,max(0.,context.travel[anchor]-context.travel[0])/30)
        # Boundary-limited source lead, independent of sensor missingness.
        row['transport_unobserved_lead_30']=max(0.,30-(context.travel[anchor]-context.travel[0]))
        for h in (5,15,30):row.update(context.aligned(anchor,h))
        row['upstream_vs_site_flow']=row['upstream_flow_ratio']-float(base.iloc[idx].flow_ratio_mean_0)
        row['upstream_vs_site_speed']=row['upstream_speed_ratio']-float(base.iloc[idx].site_last_speed_ratio)
        row['transport30_vs_site_flow']=row['transport_flow_ratio_30']-float(base.iloc[idx].flow_ratio_mean_0)
        row['transport30_vs_upstream_flow']=row['transport_flow_ratio_30']-row['upstream_flow_ratio']
        row['transport5_vs_30_flow']=row['transport_flow_ratio_5']-row['transport_flow_ratio_30']
        row['transport15_vs_30_flow']=row['transport_flow_ratio_15']-row['transport_flow_ratio_30']
        rows.append(row)
    extra=pd.DataFrame(rows,columns=REPAIR_FEATURES+TRANSPORT_FEATURES+CONTRAST_FEATURES).astype(np.float32)
    result=pd.concat([base.reset_index(drop=True),extra],axis=1)
    return result[['site_uid',*NUMERIC_FEATURES]]
