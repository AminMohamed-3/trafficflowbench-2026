"""Claude helper 2026-10-04: OPT screen of flow_krige (gated kriging blend) vs F4, all 10 panels, J_S = .35 S_state + .15 S_phys.
Band weights are fitted leave-one-panel-out and on the other date fold (even/odd held-day index), pooled least squares on
linear per-lane flow error over target cells (truth of held dates of OTHER panels / other fold only).
Per-day score components replicate tfb.metric (checked against metric totals) for a paired date bootstrap.
Usage: python docs/reference_code/flow_screen.py stage1   (kriging + sufficient stats per panel)
       python docs/reference_code/flow_screen.py stage2   (fit weights, score, bootstrap)"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_krige as FK
from tfb import metric

OUT = P.ROOT / "artifacts/claude_ref/flow/opt"
PANELS = P.P["panels"]
NB = len(FK.BANDS) - 1


def stage1():
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        st = held.stations.sort_values("idx")
        N, den = FK.krige_log_flow(np.asarray(held.flow, float), FK.positions(held.stations))
        np.save(OUT / f"{panel}_N.npy", N.astype(np.float32)); np.save(OUT / f"{panel}_den.npy", den.astype(np.float32))
        lanes = st.lanes.to_numpy(float); vcut = st.v_cut.to_numpy(float)
        T = np.asarray(held.target, bool) & np.isfinite(tq) & (pv >= FK.GATE * vcut)
        b = FK.band_index(pq, lanes)
        eF = (pq - tq) / lanes; d = den / (den + FK.SHRINK) * (np.exp(N) - pq) / lanes
        D = len(held.dates)
        sdd = np.zeros((D, NB)); sfd = np.zeros((D, NB))
        for day in range(D):
            m = T[day]
            np.add.at(sdd[day], b[day][m], d[day][m] ** 2)
            np.add.at(sfd[day], b[day][m], eF[day][m] * d[day][m])
        np.savez(OUT / f"{panel}_stats.npz", sdd=sdd, sfd=sfd)
        print(panel, "done", flush=True)


def day_components(held, tv, tq, pv, pq):
    """Per-day components of state_score and physics_score (mode A, ramp filter on), observable from the full set."""
    root = P.ROOT
    p, topo, on_att, off_att = metric._parameters(held, root)
    lanes = p.lanes.to_numpy(float); vf = p.free_speed_kmh.to_numpy(float); cap = p.capacity_vph.to_numpy(float)
    crit = p.critical_density.to_numpy(float); jam = p.k_jam.to_numpy(float); length = p.length_km.to_numpy(float)
    links = p.link_id.to_numpy(); topo_ids = set(topo.index)
    slanes = np.maximum(held.stations.sort_values("idx").lanes.to_numpy(float), 1.)
    D = len(held.dates); regimes = list(held.regimes)
    cols = ["sv", "sq", "n", "fe", "fs", "lwe", "lws"]
    C = np.zeros((D, len(cols)))

    def field(day):
        target = np.asarray(held.target[day], bool)
        tvd, tqd = np.asarray(tv[day], float), np.asarray(tq[day], float)
        v = np.where(target, pv[day], tvd); q = np.where(target, pq[day], tqd)
        valid = np.asarray(held.elig[day], bool) & np.isfinite(v) & np.isfinite(q)
        return v, q, valid, q / np.maximum(v, 1.) * length, tqd / np.maximum(tvd, 1.) * length, target

    avail = {r: np.zeros(len(links), bool) for r in metric.REGIMES}
    for day, r in enumerate(regimes):
        avail[r] |= field(day)[2].any(axis=0)
    obs = {r: np.asarray([l in topo.index and
                          ({x.strip() for x in str(topo.loc[l, "incoming_link_ids"]).split(";")} & topo_ids).issubset(set(links[a]))
                          and ({x.strip() for x in str(topo.loc[l, "outgoing_link_ids"]).split(";")} & topo_ids).issubset(set(links[a]))
                          for l in links], bool) & a for r, a in avail.items()}

    def ramp_valid(day):
        res = np.ones((tv.shape[1], len(links)), bool)
        for kind, att in (("OR", on_att), ("FR", off_att)):
            keep = np.ones_like(res); keep[:, np.isin(links, list(att))] = False
            for rp in held.ramps.loc[held.ramps.ramp_type == kind].itertuples(index=False):
                keep[:, rp.station_idx] |= held.ramp_elig[day, :, rp.idx] & np.isfinite(held.ramp_flow[day, :, rp.idx])
            res &= keep
        return res

    for day, r in enumerate(regimes):
        mask = np.asarray(held.target[day], bool)
        usable = mask & np.isfinite(tv[day]) & np.isfinite(tq[day])
        C[day, 0] = np.sum((pv[day] - tv[day])[usable] ** 2)
        C[day, 1] = np.sum(((pq[day] - tq[day]) / slanes)[usable] ** 2)
        C[day, 2] = usable.sum()
        v, q, valid, n, tn, target = field(day)
        k_lane = q / np.maximum(v, 1.) / lanes; q_lane = q / lanes
        q_fd = np.where(k_lane <= crit / lanes, vf * k_lane, (cap / lanes / np.maximum(jam / lanes - crit / lanes, 1e-9)) * np.maximum(jam / lanes - k_lane, 0))
        C[day, 3] = np.abs(q_lane[valid] - q_fd[valid]).sum(); C[day, 4] = np.abs(q_lane[valid]).sum()
        ok = obs[r][None, :]; rv = ramp_valid(day)
        tr = valid[:-1] & valid[1:] & ok & rv[:-1]
        C[day, 5] += np.abs((n[1:] - n[:-1])[tr] - (tn[1:] - tn[:-1])[tr]).sum(); C[day, 6] += np.abs((tn[1:] - tn[:-1])[tr]).sum()
        if day + 1 < D and regimes[day + 1] == r and (pd.Timestamp(held.dates[day + 1]) - pd.Timestamp(held.dates[day])).days == 1:
            _, _, nv, nn, nt, _ = field(day + 1)
            cr = valid[-1] & nv[0] & obs[r] & rv[-1]
            C[day, 5] += np.abs((nn[0] - n[-1])[cr] - (nt[0] - tn[-1])[cr]).sum(); C[day, 6] += np.abs((nt[0] - tn[-1])[cr]).sum()
    return C, regimes


def aggregate(C, regimes, wts):
    out = {}
    for r in metric.REGIMES:
        m = np.array([x == r for x in regimes]); s = (C[m] * wts[m, None]).sum(0)
        sv, sq, n, fe, fs, lwe, lws = s
        S_state = .54 * max(0, 1 - np.sqrt(sv / n) / 25) + .46 * max(0, 1 - np.sqrt(sq / n) / 600)
        S_FD = max(0, 1 - fe / (fs + 1e-9)); S_LWR = max(0, 1 - min(1, lwe / (lws + 1e-9)))
        out[r] = (S_state, S_FD, S_LWR, (S_FD + 2 * S_LWR) / 3)
    a = np.mean(list(out.values()), axis=0)
    return dict(S_state=a[0], S_FD=a[1], S_LWR=a[2], S_phys=a[3], J=.35 * a[0] + .15 * a[3])


def fit_weights(stats, exclude_panel, fold, D):
    days = np.arange(D) % 2 != fold
    sdd = sum(stats[p]["sdd"][days].sum(0) for p in stats if p != exclude_panel)
    sfd = sum(stats[p]["sfd"][days].sum(0) for p in stats if p != exclude_panel)
    return np.clip(-sfd / np.maximum(sdd, 1e-9), 0, 1)


def stage2(nboot=1000):
    stats = {p: dict(np.load(OUT / f"{p}_stats.npz")) for p in PANELS}
    D = len(P.P["OPT_held_dates"])
    allw = fit_weights(stats, None, -1, D)
    print("pooled weights (all panels, all days, in-sample, info only)", allw.round(3), flush=True)
    rows = []; comps = {}
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        N = np.load(OUT / f"{panel}_N.npy").astype(float); den = np.load(OUT / f"{panel}_den.npy").astype(float)
        T = np.asarray(held.target, bool)
        nq = pq.copy()
        for fold in (0, 1):
            w = fit_weights(stats, panel, fold, D)
            fq = FK.blend(pv, pq, N, T, held.stations, w, den=den)
            sel = (np.arange(D) % 2 == fold)
            nq[sel] = fq[sel]
        ref = P.score(held, tv, tq, pv, pq); new = P.score(held, tv, tq, pv, nq)
        Cf, reg = day_components(held, tv, tq, pv, pq); Cn, _ = day_components(held, tv, tq, pv, nq)
        chk = aggregate(Cf, reg, np.ones(D))
        assert abs(chk["J"] - ref["J"]) < 2e-4 and abs(chk["S_LWR"] - ref["S_LWR"]) < 2e-4, (chk, ref)
        comps[panel] = (Cf, Cn, reg)
        la = held.stations.sort_values("idx").lanes.to_numpy(float); m = T & np.isfinite(tq)
        rf = np.sqrt(np.mean(((pq - tq) / la)[m] ** 2)); rn = np.sqrt(np.mean(((nq - tq) / la)[m] ** 2))
        low = float(((nq < 50) & m).mean() / m.mean())
        row = dict(panel=panel, **{f"F4_{k}": float(v) for k, v in ref.items()}, **{f"new_{k}": float(v) for k, v in new.items()},
                   rmse_q_F4=float(rf), rmse_q_new=float(rn), share_q_below50=low)
        rows.append(row)
        print(panel, "F4", ref, "new", new, "rmse_q", round(rf, 2), round(rn, 2), flush=True)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, D, D); wts = np.bincount(idx, minlength=D).astype(float)
        dl = {k: [] for k in ("J", "S_state", "S_phys", "S_LWR")}
        for panel, (Cf, Cn, reg) in comps.items():
            a, b = aggregate(Cf, reg, wts), aggregate(Cn, reg, wts)
            for k in dl: dl[k].append(b[k] - a[k])
        boots.append({k: np.mean(v) for k, v in dl.items()})
    B = pd.DataFrame(boots)
    df = pd.DataFrame(rows)
    summary = {k: dict(mean_delta=float((df[f"new_{k}"] - df[f"F4_{k}"]).mean()), ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))])
               for k in ("J", "S_state", "S_phys", "S_LWR")}
    print(json.dumps(summary, indent=1))
    df.to_csv(OUT / "opt_screen.csv", index=False)
    (OUT / "opt_summary.json").write_text(json.dumps(dict(summary=summary, pooled_weights=allw.tolist(),
                                                           params=dict(ell=FK.ELL, lam=FK.LAM, K=FK.K, bin=FK.BIN, bands=FK.BANDS.tolist(), gate=FK.GATE, shrink=FK.SHRINK)), indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]]()
