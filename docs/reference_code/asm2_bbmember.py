"""asm2 build step: pretrained-backbone ongoing member (C28 board read .89450: ConvNeXt-Tiny up2 e8, drop-in a = .2).

Per official ongoing window the shipped chain probability f (type A: selection-conditioned sccnn probabilities; types B/C:
the base CNN ensemble) is blended with the member's probabilities, p = (1-a) f + a member, and the unchanged C26 decoder
runs (c_bb_board.decode = ong2_probe build --sc --noT):
  A  : k, o = eiou_order(p); m = constrained(p, o, k, pm, .92); band raise to IoU(m, pm) >= .8
  B/C: k, o = eiou_order(p); m = constrained(p, o, k, pm, .92)
  pm = last observed queue state over t-12..t-1 per link from the released masked cube, tiled over the 6 horizons.
Gate: with a = 0 the same code must reproduce the build's own ongoing rows (ong2_probe output) exactly.

The member is a config (JSON) so the tuning program's final choice drops in without code changes:
  {"name": str, "a": float, "spec": {...c_bb_arch spec...}, "drop": [...], "source": "weights" | "probs",
   "weights": "artifacts/.../model.pt", "weights_sha256": str, "device": "cuda"|"cpu", "amp_bf16": bool,
   "probs": "artifacts/.../member.parquet", "probs_sha256": str}
source=weights: serve the shipped fine-tuned weights (architecture built with pretrained=False: no ImageNet download at
  serving; timm/torchvision pins in requirements-final.txt), inputs qcnn_predict_adapt.official_x + selection flags
  (c_bb_serve.predict), no per-date adaptation.
source=probs: use stored member probabilities (sha-pinned) - an audit/interim option when the weights are not shipped.

  python asm2_bbmember.py serve  --cfg CFG --out MEMBER.parquet
  python asm2_bbmember.py decode --cfg CFG --member MEMBER.parquet --base BASE.parquet --sc SC.parquet \
         --parent-ongoing ONGOING.parquet --out NEW_ONGOING.parquet
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time, hashlib, argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
K = ["window_id", "timestamp", "link_id"]


def rp(p):
    """Single-threaded parquet read (values identical): lowers Arrow thread-pool use; see the exit note in main()."""
    return pd.read_parquet(p, use_threads=False)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def load_cfg(p):
    cfg = json.loads(Path(p).read_text()); assert 0. <= float(cfg["a"]) <= 1. and cfg["source"] in ("weights", "probs"), cfg
    return cfg


def serve(a):
    cfg = load_cfg(a.cfg); t0 = time.time(); out = Path(a.out); rec = dict(cfg=cfg, cfg_sha256=sha(a.cfg))
    if cfg["source"] == "probs":
        src = ROOT / cfg["probs"]; assert sha(src) == cfg["probs_sha256"], "stored member probabilities sha mismatch"
        shutil.copyfile(src, out); rec.update(mode="stored-probabilities")
    else:
        import torch, c_bb_arch, qcnn_predict_adapt as qpa, ong2_sccnn as S, c_bb_serve as CS
        w = ROOT / cfg["weights"]; assert sha(w) == cfg["weights_sha256"], "member weights sha mismatch"
        dev = cfg.get("device", "cuda"); assert dev == "cpu" or torch.cuda.is_available(), "member config asks for CUDA"
        amp = bool(cfg.get("amp_bf16", False)) and dev == "cuda"
        model = c_bb_arch.make(dict(cfg["spec"], pretrained=False)).to(dev)   # weights come from the shipped state dict
        model.load_state_dict(torch.load(w, map_location=dev)); model.eval()
        S.enable()
        types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv").set_index("window_id").typ.to_dict()
        rows = []
        for split in ("validation", "private"):
            data = qpa.load_split(split, False)
            for d_str in sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin}):
                got = CS.predict(model, data, d_str, cfg["drop"], dev, types, amp)
                if got is not None: rows.append(got)
        df = pd.concat(rows, ignore_index=True); df.to_parquet(out)
        rec.update(mode="weights", device=dev, amp_bf16=amp, torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None)
    df = rp(out); assert df.set_index(K).index.is_unique and np.isfinite(df.prob).all()
    rec.update(rows=len(df), windows=int(df.window_id.nunique()), out=str(out), out_sha256=sha(out), seconds=round(time.time() - t0, 1),
               source_sha256=sha(__file__))
    Path(str(out) + ".json").write_text(json.dumps(rec, indent=1)); print("BBMEMBER serve", json.dumps({k: v for k, v in rec.items() if k != "cfg"}), flush=True)


def decode(a):
    from tfb.data import load_cache
    from qcnn_data import cube
    from gapres_selcond_rule import constrained, iou, eiou_order
    from ong2_band_rule import raise_to
    import asm_build as AB
    cfg = load_cfg(a.cfg); w8 = float(cfg["a"]); t0 = time.time()

    def dec(f, pm, typ):
        k, o = eiou_order(f)
        m, _ = constrained(f, o, k, pm, .92)
        if typ == "A" and iou(m, pm) < .8:
            m = raise_to(m, f, pm, .8)
        return m
    ctl = rp(a.base); sc = rp(a.sc).set_index(K).prob
    par = rp(a.parent_ongoing).set_index("submission_id").queue_pred
    mem = rp(a.member).set_index(K).prob
    types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv"); types = types[types.split != "train"].set_index("window_id")
    windows = set(ctl.window_id); assert set(mem.index.get_level_values(0)) >= windows
    key = pd.concat([x[x.window_id.isin(windows)] for x in pd.read_csv(
        AB.KEY, usecols=["submission_id", "split", "window_id", "timestamp", "link_id"], dtype=str, chunksize=250000)], ignore_index=True)
    km = key.assign(submission_id=key.submission_id.astype(np.int64)).set_index(K)
    out, wins = [], []
    for split in ("validation", "private"):
        for p in sorted(types[types.split == split].panel.unique()):
            c = load_cache(p, split); st = c.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
            c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
            obs, q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
            for wid, w in types[(types.split == split) & (types.panel == p)].iterrows():
                g = ctl[ctl.window_id == wid].sort_values(["timestamp", "link_id"])
                ts = sorted(g.timestamp.unique()); H = len(ts); L = len(g) // H
                idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
                d = c.day_index(str(w.date)); t = int(w.t)
                last = np.zeros(c3.shape[-1], bool)
                for k_ in range(t - 12, t):
                    last = np.where(obs[d, k_], q[d, k_], last)
                pm = np.broadcast_to(last[idx], (H, L)).ravel()
                ix = list(zip(g.window_id, g.timestamp, g.link_id))
                f = (sc.loc[ix] if w.typ == "A" else g.prob).to_numpy(np.float64)
                if w.typ != "A":
                    assert eiou_order(f)[0] == int(g.queue_pred.sum()), wid
                fe = mem.loc[ix].to_numpy(np.float64); assert np.isfinite(fe).all()
                sid = km.loc[ix].submission_id.to_numpy(); assert (km.loc[ix].split == split).all()
                m0 = dec(f, pm, w.typ)
                assert np.array_equal(m0.astype(int), par.loc[sid].to_numpy().astype(int)), f"a=0 does not reproduce the build's ongoing rows {wid}"
                m1 = dec((1 - w8) * f + w8 * fe, pm, w.typ)
                out.append(pd.DataFrame(dict(submission_id=sid, queue_pred=m1.astype(float), parent=m0.astype(float))))
                wins.append(dict(window_id=wid, split=split, typ=w.typ, n_parent=int(m0.sum()), n_new=int(m1.sum()), changed=int((m0 != m1).sum())))
    rep = pd.concat(out, ignore_index=True).sort_values("submission_id").reset_index(drop=True); wd = pd.DataFrame(wins)
    assert len(rep) == len(par) == 87000 and rep.submission_id.is_unique and set(rep.submission_id) == set(par.index)
    rep[["submission_id", "queue_pred"]].to_parquet(a.out, index=False); wd.to_csv(str(a.out) + ".windows.csv", index=False)
    ch = wd[wd.changed > 0]
    rec = dict(cfg=cfg, cfg_sha256=sha(a.cfg), a=w8, member=str(a.member), member_sha256=sha(a.member), base_sha256=sha(a.base), sc_sha256=sha(a.sc),
               parent_ongoing_sha256=sha(a.parent_ongoing), parent_reproduced_at_a0=int(len(rep)), changed_cells=int(wd.changed.sum()),
               changed_by_split=wd.groupby("split").changed.sum().to_dict(), changed_windows_by_split=ch.groupby("split").size().to_dict(),
               changed_by_type=wd.groupby("typ").changed.sum().to_dict(), out=str(a.out), out_sha256=sha(a.out),
               seconds=round(time.time() - t0, 1), source_sha256=sha(__file__))
    Path(str(a.out) + ".json").write_text(json.dumps(rec, indent=1, default=int))
    print("BBMEMBER decode", json.dumps({k: v for k, v in rec.items() if k != "cfg"}, default=int), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve"); p.add_argument("--cfg", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("decode"); p.add_argument("--cfg", required=True); p.add_argument("--member", required=True)
    p.add_argument("--base", required=True); p.add_argument("--sc", required=True); p.add_argument("--parent-ongoing", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(); serve(a) if a.cmd == "serve" else decode(a)
    # Outputs and receipts are written and closed. pyarrow 23.0.1 (pandas 3 Arrow-backed strings) can abort at interpreter
    # teardown ("terminate called without an active exception", 27/40 on VM1) after the work is done; skip that teardown.
    # Failures above still raise with a traceback and a nonzero exit.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
