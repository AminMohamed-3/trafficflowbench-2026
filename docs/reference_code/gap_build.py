"""Claude gap helper 2026-10-04: build specialist training rows (simulated blocks in the layer's visible data) and eval
rows (OWN artificial blocks or official gap windows) per panel.
Usage: python docs/reference_code/gap_build.py own validation|private [panels]
       python docs/reference_code/gap_build.py off validation|private [panels]
Output: artifacts/claude_ref/gap/rows_<kind>_<period>_<panel>.npz"""
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import gap_lib as G
import b6full_eval as EV

NCELL = 300


def build(kind, period, panel):
    t0 = time.time()
    L = G.own_layer(panel, period) if kind == "own" else G.off_layer(panel, period)
    rng = np.random.default_rng(__import__("zlib").crc32(f"{kind}{period}{panel}".encode()))
    S = L.sp.shape[2]
    orig = G.sim_origins(L, rng)
    Xs, Y, M = [], [], []
    profs = {}
    names = None
    for bi, (d, t, c, k, pers) in enumerate(orig):
        if d not in profs:
            profs = {d: G.loo_prof(L, d)}
        F = G.block_features(L, d, t, c, k, profs[d])
        names = list(F)
        tv = L.sp[d, t + 1:t + 19]; tq = L.fl[d, t + 1:t + 19]
        ok = np.flatnonzero((np.isfinite(tv) & np.isfinite(tq)).ravel())
        if len(ok) == 0:
            continue
        sel = rng.choice(ok, min(NCELL, len(ok)), replace=False)
        Xs.append(np.stack([F[n].ravel()[sel] for n in names], 1))
        Y.append(np.stack([(tv / L.vf).ravel()[sel], (tq / L.cap).ravel()[sel]], 1))
        M.append(np.stack([np.full(len(sel), bi), np.full(len(sel), d), np.full(len(sel), t), np.full(len(sel), G.CLS[c]), np.full(len(sel), k),
                           sel // S + 1, sel % S, np.full(len(sel), -1 if not np.isfinite(pers) else pers * 1000)], 1))
    out = dict(Xtr=np.concatenate(Xs).astype(np.float32), Ytr=np.concatenate(Y).astype(np.float32), Mtr=np.concatenate(M).astype(np.int32),
               names=np.array(names))
    # eval blocks
    Xe, Me, E = [], [], []
    if kind == "own":
        pn = panel
        z = np.load(G.ROOT / "artifacts/claude_ref/polish" / f"c12_own_{pn}.npz") if period == "validation" else None
        evb = []
        for (d, t) in L.blocks:
            c, k, pers = G.classify(L.kv_truth[d], L.el[d], L.vcut, t)
            evb.append((d, t, c, k, pers))
        if z is not None:
            c12v, c12q = z["v"], z["q"]
            r = np.load(EV.OUT / f"ref_own_{pn}.npz"); c11v, c11q = r["c11_v"], r["c11_q"]
            iv, iq, _ = EV.imp("own", pn, [0, 1, 2])
        else:   # April OWN: state5 rebuild of C13 (state5_c13apr) and its imputer
            z = np.load(G.ROOT / "artifacts/claude_ref/state5" / f"own_apr_{pn}.npz")
            assert (z["target"] == L.score_target).all() and (z["gap"] == L.gapmask).all()
            c12v, c12q, c11v, c11q = z["v13"], z["q13"], z["c11_v"], z["c11_q"]
            iv, iq = z["bl_v"], z["bl_q"]   # pre-projection blend stands in for the imputer column (not used in scoring)
        for bi, (d, t, c, k, pers) in enumerate(evb):
            F = G.block_features(L, d, t, c if c in ("onset", "ongoing") else c, k, G.loo_prof(L, d))
            m = L.score_target[d, t + 1:t + 19]
            ix = np.flatnonzero(m.ravel())
            Xe.append(np.stack([F[n].ravel()[ix] for n in names], 1))
            jj = ix // S + 1; ss = ix % S
            Me.append(np.stack([np.full(len(ix), bi), np.full(len(ix), d), np.full(len(ix), t), np.full(len(ix), G.CLS[c]), np.full(len(ix), k),
                                jj, ss, np.full(len(ix), -1 if not np.isfinite(pers) else pers * 1000)], 1))
            cols = [L.tv[d, t + jj, ss], L.tq[d, t + jj, ss]]
            if z is not None:
                cols += [c12v[d, t + jj, ss], c12q[d, t + jj, ss], c11v[d, t + jj, ss], c11q[d, t + jj, ss], iv[d, t + jj, ss], iq[d, t + jj, ss]]
            E.append(np.stack(cols, 1))
    else:
        for bi, (d, t, c) in enumerate(L.windows):
            k = 6 if c == "onset" else 0
            F = G.block_features(L, d, t, c, k, G.loo_prof(L, d))
            m = L.target[d, t + 1:t + 19]
            ix = np.flatnonzero(m.ravel())
            Xe.append(np.stack([F[n].ravel()[ix] for n in names], 1))
            jj = ix // S + 1; ss = ix % S
            Me.append(np.stack([np.full(len(ix), bi), np.full(len(ix), d), np.full(len(ix), t), np.full(len(ix), G.CLS[c]), np.full(len(ix), k),
                                jj, ss, np.full(len(ix), -1)], 1))
            E.append(np.zeros((len(ix), 0)))
    if not Xe:
        Xe, Me, E = [np.zeros((0, len(names)))], [np.zeros((0, 8))], [np.zeros((0, 0))]
    out.update(Xev=np.concatenate(Xe).astype(np.float32), Mev=np.concatenate(Me).astype(np.int32), Eev=np.concatenate(E).astype(np.float64),
               vf=L.vf, cap=L.cap, lanes=L.lanes, vcut=L.vcut)
    np.savez_compressed(G.OUT / f"rows_{kind}_{period}_{panel}.npz", **out)
    import collections
    print(kind, period, panel, "train rows", len(out["Ytr"]), "blocks", len(orig), dict(collections.Counter(o[2] for o in orig)),
          "eval rows", len(out["Mev"]), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    kind, period = sys.argv[1], sys.argv[2]
    panels = sys.argv[3].split(",") if len(sys.argv) > 3 else G.PANELS
    for pn in panels:
        build(kind, period, pn)
