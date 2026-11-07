"""Claude helper (state2, ledger 0360) 2026-10-04: relative common-mode speed correction on top of F4 (idea 4, strengthened).
Generator fact (measured on OPT): observed speed noise has a corridor-wide multiplicative shock per (panel, day, step), present
on free and congested cells, white in time, no distance decay. Estimate from OBSERVED cells of the same step only:
  c_s   = median observed speed of station s over free cells (v >= v_cut + 5) of the scored days (split-local constant),
  gr    = mean over observed free cells with |v - c_s| < 4 of (v - c_s) / c_s   (needs >= 5 such cells, else 0),
  v_new = v_F4 * (1 + b * gr),  b = w b_free + (1 - w) b_cong,  w = clip((v_F4 - v_cut) / 10, 0, 1).
Flow unchanged. (b_free, b_cong) fitted by pooled least squares on F4 residuals, leave-one-panel-out and other day-parity fold
(OPT screen); for OWN the pair is fitted on all OPT panels and days (train split, then applied to March: cross-split transfer).
Usage: python docs/reference_code/state2_cm.py opt     (OPT screen, writes artifacts/claude_ref/state2/opt_*)
       python docs/reference_code/state2_cm.py own     (I-S-OWN March confirmation, needs opt_coef.json)"""
import sys, json, importlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
from tfb import metric

OUT = P.ROOT / "artifacts/claude_ref/state2"
PANELS = P.P["panels"]
MINN = 5
CLIP = 4.


def common_factor(speed, observed, stations):
    """gr (D, T) relative common shock and n (D, T) cells used; observed cells only."""
    st = stations.sort_values("idx")
    vcut = st.v_cut.to_numpy(float)
    v = np.where(observed, speed, np.nan)
    free = observed & (v >= vcut + 5)
    c = np.array([np.median(v[:, :, l][free[:, :, l]]) if free[:, :, l].sum() > 20 else np.nan for l in range(v.shape[2])])
    dev = v - c
    ok = free & np.isfinite(c) & (np.abs(dev) < CLIP)
    n = ok.sum(2)
    gr = np.where(n >= MINN, np.where(ok, dev / np.where(np.isfinite(c), c, 1.), 0.).sum(2) / np.maximum(n, 1), 0.)
    return gr, n


def design(pv, gr, vcut):
    w = np.clip((pv - vcut) / 10., 0, 1)
    base = gr[:, :, None] * pv
    return w * base, (1 - w) * base


def apply(pv, gr, vcut, coef):
    a, b = design(pv, gr, vcut)
    return pv + coef[0] * a + coef[1] * b


def suff_stats(pv, tv, gr, vcut, mask):
    """Per day X'X (2x2) and X'y for the residual tv - pv on target cells."""
    a, b = design(pv, gr, vcut); y = tv - pv
    D = pv.shape[0]
    XX = np.zeros((D, 2, 2)); Xy = np.zeros((D, 2))
    for d in range(D):
        m = mask[d]
        X = np.c_[a[d][m], b[d][m]]
        XX[d] = X.T @ X; Xy[d] = X.T @ y[d][m]
    return XX, Xy


def solve(XX, Xy):
    return np.linalg.solve(XX + 1e-6 * np.eye(2), Xy)


# ---- per-day score components (replicates tfb.metric, mode A, ramp filter on; same construction as the flow helper's screen)
def day_components(held, tv, tq, pv, pq):
    root = P.ROOT
    p, topo, on_att, off_att = metric._parameters(held, root)
    lanes = p.lanes.to_numpy(float); vf = p.free_speed_kmh.to_numpy(float); cap = p.capacity_vph.to_numpy(float)
    crit = p.critical_density.to_numpy(float); jam = p.k_jam.to_numpy(float); length = p.length_km.to_numpy(float)
    links = p.link_id.to_numpy(); topo_ids = set(topo.index)
    slanes = np.maximum(held.stations.sort_values("idx").lanes.to_numpy(float), 1.)
    D = len(held.dates); regimes = list(held.regimes)
    C = np.zeros((D, 7))

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


def classes_rmse(held, tv, pv, nv):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import state2_diag as SD
    tgt = np.asarray(held.target, bool); vobs = np.asarray(held.speed, float)
    obs = np.asarray(held.elig, bool) & ~tgt & np.isfinite(vobs)
    cls = SD.classes(held, vobs, obs)
    m = tgt & np.isfinite(tv)
    out = {}
    for c in ("free", "cong", "front", "nob"):
        mc = m & (cls == c)
        out[c] = dict(n=int(mc.sum()), F4=float(np.sqrt(((pv - tv)[mc] ** 2).mean())), new=float(np.sqrt(((nv - tv)[mc] ** 2).mean())))
    return out


def opt(nboot=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        tgt = np.asarray(held.target, bool)
        observed = np.asarray(held.elig, bool) & ~tgt & np.isfinite(np.asarray(held.speed, float))
        gr, n = common_factor(np.asarray(held.speed, float), observed, held.stations)
        vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
        XX, Xy = suff_stats(pv, tv, gr, vcut, tgt & np.isfinite(tv))
        np.savez(OUT / f"opt_{panel}_cm.npz", gr=gr.astype(np.float32), n=n.astype(np.int16), XX=XX, Xy=Xy)
        cache[panel] = (XX, Xy)
        print(panel, "stats done; in-panel coef", solve(XX.sum(0), Xy.sum(0)).round(3), "sd gr", round(float(gr[n >= MINN].std()), 5), flush=True)
    D = len(P.P["OPT_held_dates"]); par = np.arange(D) % 2
    allc = solve(sum(c[0].sum(0) for c in cache.values()), sum(c[1].sum(0) for c in cache.values()))
    (OUT / "opt_coef.json").write_text(json.dumps(dict(b_free=float(allc[0]), b_cong=float(allc[1]),
                                                        fitted_on="all 10 panels x 43 OPT held dates (train split)"), indent=1))
    print("pooled coef (all OPT, used for OWN)", allc.round(4), flush=True)
    rows = []; comps = {}; classes = {}
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        z = np.load(OUT / f"opt_{panel}_cm.npz"); gr = z["gr"].astype(float)
        vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
        nv = pv.copy(); used = []
        for f in (0, 1):
            tr = par != f
            XX = sum(cache[p][0][tr].sum(0) for p in PANELS if p != panel)
            Xy = sum(cache[p][1][tr].sum(0) for p in PANELS if p != panel)
            coef = solve(XX, Xy); used.append(coef.round(4).tolist())
            sel = par == f
            nv[sel] = apply(pv[sel], gr[sel], vcut, coef)
        ref = P.score(held, tv, tq, pv, pq); new = P.score(held, tv, tq, nv, pq)
        Cf, reg = day_components(held, tv, tq, pv, pq); Cn, _ = day_components(held, tv, tq, nv, pq)
        chk = aggregate(Cf, reg, np.ones(D))
        assert abs(chk["J"] - ref["J"]) < 2e-4 and abs(chk["S_state"] - ref["S_state"]) < 2e-4, (chk, ref)
        comps[panel] = (Cf, Cn, reg)
        classes[panel] = classes_rmse(held, tv, pv, nv)
        rows.append(dict(panel=panel, coef_fold0=used[0], coef_fold1=used[1], **{f"F4_{k}": v for k, v in ref.items()},
                         **{f"new_{k}": v for k, v in new.items()}))
        print(panel, "F4", ref, "new", new, "coef", used, flush=True)
    rng = np.random.default_rng(0); boots = []
    for _ in range(nboot):
        wts = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        dl = {k: [] for k in ("J", "S_state", "S_phys", "S_LWR", "S_FD")}
        for panel, (Cf, Cn, reg) in comps.items():
            a, b = aggregate(Cf, reg, wts), aggregate(Cn, reg, wts)
            for k in dl: dl[k].append(b[k] - a[k])
        boots.append({k: np.mean(v) for k, v in dl.items()})
    B = pd.DataFrame(boots); df = pd.DataFrame(rows)
    keymap = {"J": "J", "S_state": "S_state", "S_phys": "S_phys", "S_LWR": "S_LWR", "S_FD": "S_FD"}
    summary = {k: dict(mean_delta=float((df[f"new_{k}"] - df[f"F4_{k}"]).mean()), ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))])
               for k in keymap}
    agg = {}
    for c in ("free", "cong", "front", "nob"):
        n = sum(classes[p][c]["n"] for p in PANELS)
        agg[c] = dict(n=n, F4=float(np.sqrt(sum(classes[p][c]["F4"] ** 2 * classes[p][c]["n"] for p in PANELS) / n)),
                      new=float(np.sqrt(sum(classes[p][c]["new"] ** 2 * classes[p][c]["n"] for p in PANELS) / n)))
    print(json.dumps(summary, indent=1)); print(json.dumps(agg, indent=1))
    df.to_csv(OUT / "opt_screen.csv", index=False)
    (OUT / "opt_summary.json").write_text(json.dumps(dict(summary=summary, pooled_coef=allc.tolist(), class_rmse=agg,
                                                           class_rmse_by_panel=classes, minn=MINN, clip=CLIP), indent=1))


# ---- I-S-OWN March
def own_modules():
    import tfb
    s = str(Path(P.P["S_source_root"]) / "src" / "tfb")
    if s not in tfb.__path__: tfb.__path__.insert(0, s)
    sys.path.insert(0, str(Path(P.P["D_worktree"]) / "src"))
    tfb.__path__.insert(0, str(Path(P.P["D_worktree"]) / "src" / "tfb"))
    own = importlib.import_module("tfb.lane_s.own_instrument")
    bind = importlib.import_module("tfb.lane_d.f4_own_metadata_binding")
    return own, bind


def own():
    own_m, bind = own_modules()
    coef = json.loads((OUT / "opt_coef.json").read_text()); cf = (coef["b_free"], coef["b_cong"])
    items = [i for i in json.loads(Path(P.P["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    ref_rows, new_rows, per, ref_nb, new_nb = [], [], [], [], []
    for item in items:
        item = bind.bound_own_item(P.P, item)
        view, tv, tq, score_target, hidden = own_m.load_item(P.ROOT, item)
        f = P.D / "F4_predictions/OWN" / f"{item['panel']}_validation"
        pv, pq = (np.load(f / f"{n}.npy").astype(float) for n in ("speed", "flow"))
        speed = np.asarray(view.speed, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~np.asarray(view.gap, bool) & np.isfinite(speed)
        gr, n = common_factor(speed, observed, view.stations)
        vcut = view.stations.sort_values("idx").v_cut.to_numpy(float)
        nv = np.where(score_target, apply(pv, gr, vcut, cf), pv)
        a, _ = own_m.state_rows(view, tv, tq, pv, pq, score_target)
        b, _ = own_m.state_rows(view, tv, tq, nv, pq, score_target)
        ref_rows += a; new_rows += b
        nbt = score_target & ~np.asarray(view.gap, bool)
        ref_nb += own_m.state_rows(view, tv, tq, pv, pq, nbt)[0]; new_nb += own_m.state_rows(view, tv, tq, nv, pq, nbt)[0]
        m = score_target & np.isfinite(tv)
        blk = m & (np.asarray(view.gap, bool))
        nb = m & ~blk
        r = lambda x, mm: float(np.sqrt(((x - tv)[mm] ** 2).mean())) if mm.any() else None
        per.append(dict(panel=item["panel"], n=int(m.sum()), block_share=float(blk.sum() / m.sum()),
                        rmse_v_F4=r(pv, m), rmse_v_new=r(nv, m), rmse_v_F4_nonblock=r(pv, nb), rmse_v_new_nonblock=r(nv, nb),
                        rmse_v_F4_block=r(pv, blk), rmse_v_new_block=r(nv, blk), sd_gr=float(gr[n >= MINN].std()),
                        share_steps_with_cm=float((n >= MINN).mean())))
        print(json.dumps(per[-1]), flush=True)
    panels = PANELS; fam = P.P["families"]
    groups = [[i for i, p in enumerate(panels) if fam[p] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in ref_rows})
    ci = own_m.paired_state(ref_rows, new_rows, panels, dates, groups)
    ci_nb = own_m.paired_state(ref_nb, new_nb, panels, dates, groups)
    print("all OWN targets", json.dumps(ci, indent=1)); print("non-block OWN targets", json.dumps(ci_nb, indent=1))
    (OUT / "own_march.json").write_text(json.dumps(dict(coef=coef, paired_state=ci, paired_state_nonblock=ci_nb, per_panel=per), indent=1))


if __name__ == "__main__":
    {"opt": opt, "own": own}[sys.argv[1]]()
