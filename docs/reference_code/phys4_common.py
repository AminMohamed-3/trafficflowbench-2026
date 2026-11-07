"""Claude helper phys4 (2026-10-04): shared loaders. C13 on OPT (cross-fit chain per day parity, exactly polish_eval combo)
and on I-S-OWN March (all-days chain). LWR decomposition helpers."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import polish_eval as PE
from tfb import metric

OUT = P.ROOT / "artifacts/claude_ref/phys4"
PANELS = P.P["panels"]
CHAIN = json.loads((PE.OUT / "eval_combo.json").read_text())["base"]
D_OPT = len(P.P["OPT_held_dates"])
PAR = np.arange(D_OPT) % 2


def c13_opt(pn):
    """Returns held, tv, tq, ctx, v13, q13 (C13 cross-fit by day parity)."""
    held, tv, tq, ctx, v0, q0 = PE.opt_load(pn)
    nv = np.zeros_like(v0); nq = np.zeros_like(q0)
    for f in (0, 1):
        a, b = PE.apply_chain(CHAIN["_fold"][str(f)], ctx, v0, q0)
        nv[PAR == f] = a[PAR == f]; nq[PAR == f] = b[PAR == f]
    return held, tv, tq, ctx, nv, nq


def consts(held):
    p, topo, on, off = metric._parameters(held, P.ROOT)
    return dict(length=p.length_km.to_numpy(float), lanes=p.lanes.to_numpy(float), vf=p.free_speed_kmh.to_numpy(float),
                cap=p.capacity_vph.to_numpy(float), crit=p.critical_density.to_numpy(float), jam=p.k_jam.to_numpy(float),
                links=p.link_id.to_numpy(), topo=topo, on=on, off=off)


def lwr_mask(held, tv, tq, pv, pq):
    """Per (day, step t -> t+1, station) transition mask used by S_LWR (mode A, ramp filter), within-day only."""
    import flow_screen as FS
    c = consts(held); links = c["links"]; topo = c["topo"]; topo_ids = set(topo.index)
    T = np.asarray(held.target, bool)
    v = np.where(T, pv, tv); q = np.where(T, pq, tq)
    valid = np.asarray(held.elig, bool) & np.isfinite(v) & np.isfinite(q)
    regs = list(held.regimes)
    avail = {r: np.zeros(len(links), bool) for r in metric.REGIMES}
    for d, r in enumerate(regs):
        avail[r] |= valid[d].any(0)
    obs = {r: np.asarray([l in topo.index and
                          ({x.strip() for x in str(topo.loc[l, "incoming_link_ids"]).split(";")} & topo_ids).issubset(set(links[a]))
                          and ({x.strip() for x in str(topo.loc[l, "outgoing_link_ids"]).split(";")} & topo_ids).issubset(set(links[a]))
                          for l in links], bool) & a for r, a in avail.items()}
    out = np.zeros((T.shape[0], T.shape[1] - 1, T.shape[2]), bool)
    for d, r in enumerate(regs):
        rv = np.ones((T.shape[1], len(links)), bool)
        for kind, att in (("OR", c["on"]), ("FR", c["off"])):
            keep = np.ones_like(rv); keep[:, np.isin(links, list(att))] = False
            for rp in held.ramps.loc[held.ramps.ramp_type == kind].itertuples(index=False):
                keep[:, rp.station_idx] |= held.ramp_elig[d, :, rp.idx] & np.isfinite(held.ramp_flow[d, :, rp.idx])
            rv &= keep
        out[d] = valid[d, :-1] & valid[d, 1:] & obs[r][None, :] & rv[:-1]
    return out, c
