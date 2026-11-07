"""Claude helper asm (2026-10-04, ledger 0411): state5 (S08 as stack input + split-local stage 3, ledger 0407) x phys4
(LSA nudge + stage 3, small or big 'L' models, ledger 0405) interaction on I-S-OWN March AND April.

Per split and panel (state5 arrays, artifacts/claude_ref/state5/own_{mar,apr}_s08_<pn>.npz and own[_apr]_<pn>.npz):
  F13  = C13 stack with F4 input (own_<pn> / own_apr_<pn> v13)
  S13  = C13 stack with S08 input (own_{mar,apr}_s08 v13)
  F15  = F13 -> LSA -> phys4 S3 (p = 0368 prob of the F4 input)          + clip
  S15  = S13 -> LSA -> phys4 S3 (p = 0368 prob of the S08 input)         + clip
  S17  = S13 -> split-local stage 3 (state5 s3loc cross-fit preds, day parity) -> soft phys3 + TV  + clip   (= state5 final)
  S17L = S17 (before clip) -> LSA -> phys4 S3                             + clip   (both stage 3s)
phys4 models are fitted on OPT (train months): out of sample on both OWN months. s3loc preds are cross-fitted by day parity.
Scoring: state5_lib.compare (nb / mix / all, date bootstrap 1000). p: shipped 0368 models on the base input, full-grid
convention as polish_ref.own (rv = 0360 speed on targets, input elsewhere).
Usage: python docs/reference_code/asm_inter2.py own [L|small]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
TAG = sys.argv[2] if len(sys.argv) > 2 else "L"
os.environ["PHYS4_S3"] = "" if TAG == "small" else "L"
import state5_lib as L
import phys4_s3 as S3
import polish_eval as E
import research2_tv as TVM
import asm_inter as AI

OUT = L.ROOT / "artifacts/claude_ref/asm"


def p_of(vobs, target, gap, stations, pv, pq, m368, cf):
    import state2_cm as CM, s2stage as S2
    observed = np.isfinite(vobs) & ~target & ~gap
    vcut = stations.sort_values("idx").v_cut.to_numpy(float)
    gr, _ = CM.common_factor(vobs, observed, stations)
    rv = np.where(target, CM.apply(pv, gr, vcut, cf), pv)
    return S2.hedge_368(m368, vobs, observed, stations, rv, pq, target)[1]


def own():
    import phys3_stack as ST
    from tfb import metric
    from tfb.data import load_cache
    m368 = ST.load_368(); mq, mv = S3.models()
    cj = json.loads((L.ROOT / "artifacts/claude_ref/state2/opt_coef.json").read_text()); cf = (cj["b_free"], cj["b_cong"])
    res = {"phys4_models": TAG}
    for split, tag, grp, fF in (("validation", "mar", "own_validation", "own_{pn}.npz"), ("private", "apr", "own_private", "own_apr_{pn}.npz")):
        sh = L.share_array(split); st = {k: {} for k in ("F13", "S13", "F15", "S15", "S17", "S17L")}; info = {}
        pr = np.load(L.OUT / f"s3loc_pred_{split}_own_{tag}_s08.npz"); t0 = time.time()
        for pn in L.PANELS:
            zS = dict(np.load(L.OUT / f"own_{tag}_s08_{pn}.npz")); zF = dict(np.load(L.OUT / fF.format(pn=pn)))
            la, vc, cap, stn = L.lanes_of(pn, split)
            T = zS["target"].astype(bool); gap = zS["gap"].astype(bool)
            assert (zF["target"] == T).all() and np.allclose(zF["tv"], zS["tv"], equal_nan=True)
            vo, qo = zS["vobs"].astype(float), zS["qobs"].astype(float)
            so = np.where(np.isfinite(vo) & np.isfinite(qo), vo, np.nan); fo = np.where(np.isfinite(vo) & np.isfinite(qo), qo, np.nan)
            prm = metric._parameters(load_cache(pn, split), L.ROOT)[0]; length = prm.length_km.to_numpy(float); vf = prm.free_speed_kmh.to_numpy(float)
            cl = AI.clipvec(pn, stn)
            s08 = L.load_s08(grp, pn)
            pF = p_of(vo, T, gap, stn, zS["f4v"].astype(float), zS["f4q"].astype(float), m368, cf)
            pS = p_of(vo, T, gap, stn, s08["speed"].astype(float), s08["flow"].astype(float), m368, cf)
            F13 = (zF["v13"].astype(float), zF["q13"].astype(float)); S13 = (zS["v13"].astype(float), zS["q13"].astype(float))

            def p4(v, q, p):
                (_, _), (v15, q15) = S3.stage3(mq, mv, v, q, T, gap, so, fo, p, stn, length, vf)
                return AI.clipv(v15, T, cl), q15
            # state5 split-local stage 3 (cross-fit preds) + soft phys3 + TV
            ix = pr[f"{pn}_ix"].astype(int); I = (ix[:, 0], ix[:, 1], ix[:, 2])
            nv, nq = S13[0].copy(), S13[1].copy()
            nv[I] = np.maximum(nv[I] + pr[f"{pn}_pv"].astype(float), 3.)
            nq[I] = np.clip(nq[I] + la[ix[:, 2]] * pr[f"{pn}_pq"].astype(float), 50., 1.3 * cap[ix[:, 2]])
            nv, nq = E.project_soft(E.Ctx(stn, T, vo, None, nv, nq, gap), dict(r=3., a=.9, g=1., h=.1))
            nv, nq = TVM.smooth(nv, nq, T & ~gap, stn, .004, 0.)
            arr = dict(F13=F13, S13=S13, F15=p4(*F13, pF), S15=p4(*S13, pS), S17=(AI.clipv(nv, T, cl), nq), S17L=p4(nv, nq, pS))
            for k, (v, q) in arr.items():
                st[k][pn] = L.stats(pn, zS, v, q, la)
            info[pn] = dict(p_cand_F=int(np.isfinite(pF[T]).sum()), p_cand_S=int(np.isfinite(pS[T]).sum()), s3l_cells=int(len(ix)))
            print(split, pn, info[pn], round(time.time() - t0), "s", flush=True)
        r = {}
        for name, (new, ref) in dict(phys4_on_F4base=("F15", "F13"), state5_vs_F13=("S17", "F13"), S08input_vs_F13=("S13", "F13"),
                                     phys4_on_S08base=("S15", "S13"), s3l_on_S08base=("S17", "S13"), phys4_on_state5=("S17L", "S17"),
                                     s3l_on_phys4_S08=("S17L", "S15"), both_vs_F13=("S17L", "F13"), phys4only_S08_vs_F13=("S15", "F13"),
                                     both_vs_state5only=("S17L", "S17"), both_vs_phys4only_F4=("S17L", "F15")).items():
            r[name] = L.compare(st[ref], st[new], sh)
            print("OWN", tag, name, {k: (v["delta"], v["ci"], v["n_pos"]) for k, v in r[name].items()}, flush=True)
        res[split] = dict(contrasts=r, per_panel=info)
    (OUT / f"inter2_own_{TAG}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"own": own}[sys.argv[1]]()
