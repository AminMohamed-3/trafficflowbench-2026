"""asm2 build step: S 0631 TabPFN nongap switch-speed piece (user-approved 2026-10-08; S ledger 0631/0763).

Recipe (frozen by S, exp/S_task1/APRIL_0631_CONFIRMATION_PROTOCOL.md; reproduced here without change):
  context  = S's per-split I-S-OWN context rows (OWN pseudo-target nongap switch cells on date index 2 mod 4, weights W),
             shipped as artifacts/claude_ref/asm2/s0631/context_{validation,private}.npz (exact copies of S's source caches)
  heads    = for seed 0, 1: sample min(2048, n) positive-weight context rows without replacement, p ~ W*VF^2,
             RNG 48080+100*seed; ordinary ridge (s_task1_crossing.ridge) and TabPFNRegressor(n_estimators=2,
             random_state=seed, device=cuda, inference_precision=float32, fit_mode=fit_preprocessors, n_jobs=3) on
             float32 X and VF-normalised speed residuals; query chunks of 128
  decode   = mean(ridge) + .5*(mean(TabPFN) - mean(ridge)), then speed = clip(parent + decode, 3, 1.15*topology free-flow speed)
  rows     = official target nongap switch-bracket cells (fields() below = S's s_task1_tabpfn_switch.fields), parent = the
             final A state (Task 1 chain + S lane + phys5), probability = the chain's 0368 hedge p. Flow unchanged.
Original CUDA inference only (CPU drifts up to .5 km/h); pinned public checkpoint Prior-Labs/TabPFN-v2-reg (RULES.md).
The TabPFN runtime (tabpfn 2.0.9 + its pinned wheels) and checkpoint are installed once by s_task1_tabpfn_runtime.prepare()
into artifacts/lane_s/S_task1/tabpfn_v2 (official PyPI/HF downloads, sha-verified) and prepended to sys.path only in
the `predict` process.

  python asm2_s0631.py predict --query Q.npz --out ROWS.npz [--parity P.json]   (GPU)
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "artifacts/claude_ref/asm2/s0631"
SPLITS = ("validation", "private")
CONTEXT_ROWS = {"validation": 1214, "private": 1110}   # positive-weight context rows (S receipts)
TOL = .005   # S's preregistered same-CUDA replay tolerance (km/h)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()


def digest(x):   # = s0416_density.digest
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def fields(h, parent, probability):
    """Exact copy of s_task1_tabpfn_switch.fields (S 0628/0631/0763): nine bracket fields on target nongap cells whose
    nearest observed speeds before/after (same day, station) fall on opposite sides of v_cut."""
    from state3_feat import nearest_axis
    st = h.stations.sort_values('idx')
    vf = st.free_speed_kmh.to_numpy(float)
    cut = st.v_cut.to_numpy(float)
    speed = np.asarray(h.speed, float)
    observed = np.asarray(h.visible, bool) & np.asarray(h.elig, bool) & ~h.target & ~h.gap & np.isfinite(speed)
    before, after = nearest_axis(observed, 1)
    di, ti, si = np.indices(speed.shape)
    pre = speed[di, np.maximum(before, 0), si]
    post = speed[di, np.maximum(after, 0), si]
    mask = h.target & ~h.gap & (before >= 0) & (after >= 0) & ((pre < cut) != (post < cut)) & np.isfinite(parent)
    ix = np.argwhere(mask); I = tuple(ix.T)
    a, b = before[I], after[I]
    fraction = (ix[:, 1] - a) / (b - a)
    scale = vf[ix[:, 2]]; pv = parent[I]; pre, post = pre[I], post[I]
    interpolation = pre + fraction * (post - pre)
    x = np.column_stack((np.ones(len(ix)), pv / scale, pre / scale, post / scale,
                         fraction, np.log1p(b - a), (pre >= cut[ix[:, 2]]).astype(float),
                         np.nan_to_num(probability[I], nan=.5), (interpolation - pv) / scale))
    assert x.shape == (len(ix), 9) and np.isfinite(x).all()
    return x, ix, scale


def ridge(X, y, weights):   # = s_task1_crossing.ridge
    mean = np.average(X, axis=0, weights=weights); sd = np.sqrt(np.average((X - mean)**2, axis=0, weights=weights))
    mean[0] = 0.; sd[sd < 1e-6] = 1.
    xx = (X - mean) / sd; penalty = .001 * weights.sum() * np.eye(X.shape[1]); penalty[0, 0] = 0.
    beta = np.linalg.solve(xx.T @ (weights[:, None] * xx) + penalty, xx.T @ (weights * y))
    return mean, sd, beta


def sampled(r, seed):
    cand = np.flatnonzero(r["fit"] & (r["W"] > 0)); w = r["W"][cand] * r["scale"][cand]**2
    ix = np.random.default_rng(48080 + 100 * seed).choice(cand, min(2048, len(cand)), replace=False, p=w / w.sum())
    return ix, r["X"][ix].astype(np.float32), r["Y"][ix] / r["scale"][ix]


def verify_inputs():
    """Check the shipped s0631 inputs against their sha list (context caches + reference-only parity files)."""
    n = 0
    for line in (D / "inputs_sha256.txt").read_text().splitlines():
        if not line.strip(): continue
        digest_, size, name = line.split(maxsplit=2)
        p = ROOT / name
        assert p.is_file() and p.stat().st_size == int(size) and sha(p) == digest_, f"s0631 input mismatch: {name}"
        n += 1
    return n


def runtime():
    """Pinned TabPFN runtime + checkpoint (downloads once from official PyPI/HF if absent, sha-verified)."""
    import s_task1_tabpfn_runtime as R
    man = R.prepare()
    sys.path.insert(0, str(R.RUNTIME))
    import importlib.metadata as md
    import torch
    from tabpfn import TabPFNRegressor
    import tabpfn.model.loading as loading

    def forbidden(*args, **kwargs):
        raise RuntimeError("downloads forbidden at inference; use the pinned local checkpoint")
    loading.download_model = forbidden
    mp = R.OUT / "tabpfn-v2-regressor.ckpt"
    assert sha(mp) == R.MODEL_SHA, "TabPFN checkpoint sha mismatch"
    assert torch.cuda.is_available(), "S 0631 requires original CUDA inference (CPU drifts up to .5 km/h)"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"   # published checkpoint carries the v2 criterion object (S 0600)
    versions = {p: md.version(p) for p in R.PACKAGES}
    assert versions == R.PACKAGES, versions

    def make(x, y, seed):
        m = TabPFNRegressor(n_estimators=2, random_state=seed, device="cuda", model_path=mp,
                            inference_precision=torch.float32, fit_mode="fit_preprocessors", n_jobs=3)
        m.fit(x, y); return m
    env = dict(packages=versions, torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
               model_sha256=R.MODEL_SHA, revision=R.REVISION, download_manifest_sha256=sha(R.OUT / R.MANIFEST_NAME))
    return make, torch, env


def predict_rows(model, hx, scale):
    out = np.zeros(len(hx))
    for a in range(0, len(hx), 128):
        out[a:a + 128] = model.predict(hx[a:a + 128]) * scale[a:a + 128]
    assert np.isfinite(out).all(); return out


def reference(split):
    """S's retained confirmation outputs on its I-S-OWN held rows (reference only, never served)."""
    R = D / "reference"
    if split == "validation":
        rec = [json.loads((R / f"validation_seed{s}.json").read_text()) for s in (0, 1)]
        npz = [dict(np.load(R / f"validation_seed{s}.npz")) for s in (0, 1)]
        return [dict(sampled_index_sha256=r["sampled_index_sha256"], label_sha256=r["label_sha256"], ridge_mean=r["ridge_mean"],
                     ridge_sd=r["ridge_sd"], ridge_beta=r["ridge_beta"], ridge=z["ridge"], tabpfn=z["tabpfn"], ix=z["ix"]) for r, z in zip(rec, npz)]
    scr = json.loads((R / "private_screen.json").read_text()); z = dict(np.load(R / "private_predictions.npz"))
    return [dict(sampled_index_sha256=h["sampled_index_sha256"], label_sha256=h["label_sha256"], ridge_mean=h["ridge_mean"],
                 ridge_sd=h["ridge_sd"], ridge_beta=h["ridge_beta"], ridge=z[f"ridge{s}"], tabpfn=z[f"pfn{s}"], ix=z["ix"],
                 direction=z["direction"]) for s, h in zip((0, 1), scr["head_receipts"])]


def predict(a):
    t0 = time.time()
    n_inputs = verify_inputs()
    make, torch, env = runtime()
    torch.set_num_threads(3); torch.cuda.reset_peak_memory_stats()
    q = dict(np.load(a.query))
    out_sid, out_v, log = [], [], dict(inputs_verified=n_inputs, env=env, splits={})
    for split in SPLITS:
        r = dict(np.load(D / f"context_{split}.npz"))
        cand = np.flatnonzero(r["fit"] & (r["W"] > 0)); assert len(cand) == CONTEXT_ROWS[split], (split, len(cand))
        use = q["split"] == split
        qx = q["X"][use].astype(np.float32); qs = q["scale"][use]; qv = q["parent_v"][use]; cap = q["cap"][use]
        held = r["held"]; hx = r["X"][held].astype(np.float32); hs = r["scale"][held]
        refs = reference(split) if a.parity else None
        heads, sl = [], dict(query_rows=int(use.sum()), context_rows=len(cand), seeds={})
        for seed in (0, 1):
            ix, x, y = sampled(r, seed)
            mean, sd, beta = ridge(x, y, np.ones(len(x)))
            model = make(x, y, seed)
            # S's held rows first, always (S's own call order on a fresh fit), so served rows never depend on --parity
            ph = predict_rows(model, hx, hs); ch = (((hx - mean) / sd) @ beta) * hs
            pq = predict_rows(model, qx, qs)
            cq = (((qx - mean) / sd) @ beta) * qs
            heads.append((cq, pq))
            rec = dict(sampled_rows=len(ix), sampled_index_sha256=digest(ix), label_sha256=digest(y))
            if refs is not None:   # same fitted heads must reproduce S's retained held-row outputs
                ref = refs[seed]
                assert np.array_equal(ref["ix"], r["ix"][held]), "reference held rows differ from the context cache"
                rec.update(sampled_index_equal=rec["sampled_index_sha256"] == ref["sampled_index_sha256"],
                           label_equal=rec["label_sha256"] == ref["label_sha256"],
                           ridge_params_equal=mean.tolist() == ref["ridge_mean"] and sd.tolist() == ref["ridge_sd"] and beta.tolist() == ref["ridge_beta"],
                           held_rows=int(held.sum()), held_ridge_max_abs=float(np.max(np.abs(ch - ref["ridge"]))),
                           held_tabpfn_max_abs=float(np.max(np.abs(ph - ref["tabpfn"]))),
                           held_tabpfn_n_diff=int((ph != ref["tabpfn"]).sum()))
                rec["_held"] = (ch, ph)
            sl["seeds"][seed] = rec
            del model; torch.cuda.empty_cache()
            print("S0631", split, "seed", seed, {k: v for k, v in rec.items() if k != "_held"}, flush=True)
        cm = np.mean([h[0] for h in heads], 0); pm = np.mean([h[1] for h in heads], 0)
        direction = cm + .5 * (pm - cm)
        new = np.clip(qv + direction, 3., cap)
        assert np.isfinite(new).all() and (new >= 3.).all() and (new <= cap).all()
        if refs is not None:
            hc = np.mean([sl["seeds"][s]["_held"][0] for s in (0, 1)], 0); hp_ = np.mean([sl["seeds"][s]["_held"][1] for s in (0, 1)], 0)
            hd = hc + .5 * (hp_ - hc)
            rc = np.mean([x["ridge"] for x in refs], 0); rp = np.mean([x["tabpfn"] for x in refs], 0)
            rd = refs[0]["direction"] if "direction" in refs[0] else rc + .5 * (rp - rc)
            sl["held_direction_max_abs"] = float(np.max(np.abs(hd - rd))); sl["held_direction_n_diff"] = int((hd != rd).sum())
            seeds_ok = all(sl["seeds"][s]["sampled_index_equal"] and sl["seeds"][s]["label_equal"] and sl["seeds"][s]["ridge_params_equal"]
                           and sl["seeds"][s]["held_ridge_max_abs"] == 0 and sl["seeds"][s]["held_tabpfn_max_abs"] <= TOL for s in (0, 1))
            sl["parity_passed"] = bool(seeds_ok and sl["held_direction_max_abs"] <= TOL)
            for s in (0, 1): sl["seeds"][s].pop("_held")
        sl.update(changed=int((new != qv).sum()), mean_abs_dv=float(np.abs(new - qv).mean()) if len(new) else 0.,
                  max_abs_dv=float(np.abs(new - qv).max()) if len(new) else 0.,
                  clipped_low=int((qv + direction < 3.).sum()), clipped_high=int((qv + direction > cap).sum()))
        log["splits"][split] = sl
        out_sid.append(q["submission_id"][use]); out_v.append(new)
        print("S0631", split, {k: v for k, v in sl.items() if k != "seeds"}, flush=True)
    log["peak_cuda_GiB"] = torch.cuda.max_memory_allocated() / 2**30; log["seconds"] = round(time.time() - t0, 1)
    sid = np.concatenate(out_sid); assert len(np.unique(sid)) == len(sid)
    np.savez_compressed(a.out, submission_id=sid, speed_kmh=np.concatenate(out_v))
    log.update(query=str(a.query), query_sha256=sha(a.query), out=str(a.out), out_sha256=sha(a.out), source_sha256=sha(__file__))
    Path(str(a.out) + ".json").write_text(json.dumps(log, indent=1, default=str))
    if a.parity:
        Path(a.parity).write_text(json.dumps(log, indent=1, default=str))
        assert all(log["splits"][s]["parity_passed"] for s in SPLITS), "S0631 parity vs S's confirmation outputs failed"
    print("S0631 done", a.out, "rows", len(sid), "seconds", log["seconds"], flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--query", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parity", default=None, help="also predict S's held rows and assert equality with S's outputs; write JSON")
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a)
    else:
        print("verified s0631 inputs", verify_inputs())


if __name__ == "__main__":
    main()
