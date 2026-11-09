"""gap2w serve (C build helper 2026-10-13): CPU serving of the gap2 members from their fitted weights.
Weights in (gap2w_cnn.py: weights/<tag>_off_<split>_<panel>.pt + receipt_<tag>_off_<split>.json), official-row
predictions out, in the file format of artifacts/claude_ref/gap2/pred_<tag>_off_<split>.npz:
  <out>/pred_<tag>_off_<split>.npz with <panel>_v, <panel>_q (float32, km/h and vph, raw = not smoothed, rows in the order of
  the gap rows table M; zero-length arrays for panels without official gap windows)
  <out>/serve_<split>.json: receipt (weights and receipt hashes checked, input hashes vs the training receipt, output
  hashes, torch build, seconds, peak RAM, and with --check-gpu the largest difference to the training device's file).
Inputs are rebuilt from the release exactly as in training (gap2w_cnn.build_inputs: gap_lib.off_layer, b6full_imp.inputs,
gap2_ramps.calib_fields on-ramp channels). Gap rows table M = artifacts/claude_ref/gap/offpred_v3.npz '<split>|<panel>|M'
(already a fitted input of the gap stage of asm2_build.py; equal to rows_off_<split>_<panel>.npz Mev).
No GPU is used (device cpu, 1 torch thread). A weights file whose sha256 differs from its receipt stops the run.
Usage: python docs/reference_code/gap2w_serve.py [--splits validation,private] [--tags cnr0,cnr1,cnr2]
           [--weights artifacts/claude_ref/c_gap2c47/weights] [--out artifacts/claude_ref/c_gap2c47/pred_cpu] [--check-gpu DIR]"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU serving, also on a box with a GPU
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import sys, json, time, argparse, platform
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
import asm2_phys_lwr_diag as _P; sys.modules["phys_lwr_diag"] = _P   # portable shim, as asm2_build.py
import torch
import gap2w_cnn as W
from gap2w_cnn import G, BI
torch.set_num_threads(1)
GAP_PRED = ROOT / "artifacts/claude_ref/gap/offpred_v3.npz"
DEF = ROOT / "artifacts/claude_ref/c_gap2c47"


def peak_ram_mb():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return round(getattr(m, "peak_wset", m.rss) / 2 ** 20, 1)
    except ImportError:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def serve_panel(split, panel, tags, wdir, receipts, M, log):
    """Returns {tag: (v, q)} float32 rows in M order, and the input check."""
    L, Xn, vf, cap, lanes = W.build_inputs(panel, split, "off")
    D, T, S = L.sp.shape
    assert len(L.windows) == len(np.unique(M[:, 0])), (split, panel, "windows vs gap rows table")
    xsha = W.sha_arr(Xn); X = torch.tensor(Xn)
    chk = dict(Xn_float32_sha256=xsha, rows_Mev_sha256=W.sha_arr(M))
    out = {}
    for tag in tags:
        r = receipts[tag]["panels"][panel]
        wf = wdir / r["weights_file"]
        assert W.sha_file(wf) == r["weights_file_sha256"], f"{wf.name}: sha256 differs from the receipt"
        sd = torch.load(wf, map_location="cpu", weights_only=True)
        assert W.sha_state(sd) == r["weights_tensors_sha256"], f"{wf.name}: tensors differ from the receipt"
        model = BI.UNet(cin=Xn.shape[1] + 5); model.load_state_dict(sd); model.eval()
        for k_ in ("Xn_float32_sha256", "rows_Mev_sha256"):
            if r["inputs"][k_] != chk[k_]:
                log.setdefault("input_mismatch", []).append(f"{tag}/{split}/{panel}/{k_}")
        day = {}
        ov = np.zeros(len(M), np.float32); oq = np.zeros(len(M), np.float32); seen = np.zeros(len(M), bool)
        for bi, (d, t, c) in enumerate(L.windows):
            if d not in day:
                same = [(tt, cc, 6 if cc == "onset" else 0) for (dd, tt, cc) in L.windows if dd == d]
                C = torch.tensor(W.cond_maps(1, T, S, [(0, t_, c_, k) for (t_, c_, k) in same]))
                with torch.no_grad():
                    Y = model(torch.cat([X[d:d + 1], C], 1)).numpy().astype(float)[0]
                day[d] = (np.clip(Y[0] * vf, 3., 1.3 * vf), np.clip(Y[1] * cap, 0., 1.3 * cap))
            v, q = day[d]
            m = M[:, 0] == bi
            assert (M[m, 1] == d).all() and (M[m, 2] == t).all(), (split, panel, bi)
            jj, ss = M[m, 5], M[m, 6]
            ov[m] = v[t + jj, ss]; oq[m] = q[t + jj, ss]; seen |= m
        assert seen.all()
        out[tag] = (ov, oq)
        del model
    return out, xsha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="validation,private"); ap.add_argument("--tags", default="cnr0,cnr1,cnr2")
    ap.add_argument("--weights", default=str(DEF / "weights")); ap.add_argument("--out", default=str(DEF / "pred_cpu"))
    ap.add_argument("--check-gpu", default=None, help="directory with the training device's pred_<tag>_off_<split>.npz: report the largest difference")
    a = ap.parse_args()
    wdir = Path(a.weights); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tags = a.tags.split(","); z3 = np.load(GAP_PRED)
    for split in a.splits.split(","):
        t0 = time.time(); c0 = time.process_time()
        receipts = {}
        for tag in tags:
            rf = wdir / f"receipt_{tag}_off_{split}.json"; receipts[tag] = json.loads(rf.read_text())
            assert receipts[tag].get("complete"), f"{rf.name}: training run not complete"
        log = dict(split=split, tags=tags, torch=torch.__version__, device="cpu", torch_threads=torch.get_num_threads(), platform=platform.platform(),
                   python=platform.python_version(), numpy=np.__version__, gap_rows_table=dict(file=str(GAP_PRED.relative_to(ROOT)).replace("\\", "/"), sha256=W.sha_file(GAP_PRED)),
                   receipts={t: W.sha_file(wdir / f"receipt_{t}_off_{split}.json") for t in tags}, panels={})
        res = {t: {} for t in tags}
        for pn in G.PANELS:
            key = f"{split}|{pn}|M"
            if key not in z3.files:
                for t in tags:
                    assert "skipped" in receipts[t]["panels"][pn], (t, split, pn)
                    res[t][f"{pn}_v"] = np.zeros(0, np.float32); res[t][f"{pn}_q"] = np.zeros(0, np.float32)
                continue
            tp = time.time()
            o, xsha = serve_panel(split, pn, tags, wdir, receipts, z3[key], log)
            for t in tags:
                res[t][f"{pn}_v"], res[t][f"{pn}_q"] = o[t]
            log["panels"][pn] = dict(rows=int(len(z3[key])), seconds=round(time.time() - tp, 1), Xn_float32_sha256=xsha,
                                     Xn_equals_training_receipt=all(receipts[t]["panels"][pn]["inputs"]["Xn_float32_sha256"] == xsha for t in tags))
            print(split, pn, log["panels"][pn]["rows"], "rows", log["panels"][pn]["seconds"], "s", "inputs equal to training:", log["panels"][pn]["Xn_equals_training_receipt"], flush=True)
        log["outputs"] = {}
        for t in tags:
            f = out / f"pred_{t}_off_{split}.npz"; np.savez_compressed(f, **res[t])
            log["outputs"][f.name] = dict(sha256=W.sha_file(f), rows=int(sum(len(v) for k, v in res[t].items() if k.endswith("_v"))),
                                          values_sha256=W.sha_arr(np.concatenate([res[t][f"{pn}_{c}"] for pn in G.PANELS for c in "vq"])))
        if a.check_gpu:
            worst = dict(v=0., q=0.); eq = True
            for t in tags:
                zg = np.load(Path(a.check_gpu) / f"pred_{t}_off_{split}.npz")
                for pn in G.PANELS:
                    for c in "vq":
                        g = zg[f"{pn}_{c}"].astype(float); s = res[t][f"{pn}_{c}"].astype(float)
                        assert g.shape == s.shape, (t, pn, c)
                        if len(g):
                            worst[c] = max(worst[c], float(np.abs(g - s).max())); eq &= bool(np.array_equal(g, s))
            log["vs_training_device"] = dict(dir=str(a.check_gpu), max_abs_diff_speed_kmh=worst["v"], max_abs_diff_flow_vph=worst["q"], bitwise_equal=eq)
            print(split, "CPU serve vs training-device predictions: max |dv| %.3g km/h, max |dq| %.3g vph, bitwise equal %s" % (worst["v"], worst["q"], eq), flush=True)
        log.update(seconds=round(time.time() - t0, 1), cpu_seconds=round(time.process_time() - c0, 1), peak_ram_mb=peak_ram_mb(),
                   inputs_equal_to_training=not log.get("input_mismatch"))
        (out / f"serve_{split}.json").write_text(json.dumps(log, indent=1))
        print("SERVE DONE", split, "wall %.1f s cpu %.1f s peak RAM %.0f MB" % (log["seconds"], log["cpu_seconds"], log["peak_ram_mb"]),
              "| inputs equal to the training receipt:", log["inputs_equal_to_training"], flush=True)


if __name__ == "__main__":
    main()
