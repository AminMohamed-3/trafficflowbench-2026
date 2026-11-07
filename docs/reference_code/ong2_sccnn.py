"""ong2 2026-10-05: selection-conditioned ongoing CNN (sccnn). The organizer's ongoing selector conditions on the
future: every official ongoing window has persistence IoU <= .9 (selector definition: last observed state per link
over t-12..t-1 vs observed horizon cells), and type A windows (ong2_official_types.py) also had persistence > .9 at
origin t-1. The CNN learns P(Y | x); we want P(Y | x, selection). Feed the selection facts as inputs, using the
zeroed prior channel slots of the npt recipe (127..132, dropped and zero in every C03 input):
  127 f_hard   = 1[pers(t) <= .9]                                   (official: always 1)
  128 f_prevA  = 1[t-1 passes the pre-T gate and pers(t-1) > .9]    (official: type A)
  129 f_unk    = 1 when the t-1 status is unknown                   (official: type B, blocked by the 72-step spacing)
Training/fine-tuning flags come from labels (train raw layer / split masked layer); with probability P_UNK an
ongoing window's t-1 status is hidden (f_prevA 0, f_unk 1) so the model also learns the marginal for type B.
Onset windows: all flags 0. Flags are computed from the same arrays the labels come from (no other future data).
enable() patches qcnn_train.inputs (training) and qcnn_adapt/qcnn_predict_adapt.make_x (fine-tune and pseudo scoring).
"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))

P_UNK = .25
MODE = "train"            # train: flags from labels with P_UNK dropout; truth: no dropout; unknown: f_unk for all
_rng = np.random.default_rng(12345)


def flags_day(cd, t, cov):
    """cd: (288, 5, L) cube of one day (ch 3 observed, ch 4 queued). Returns f_hard, f_prevA."""
    obs, q = cd[:, 3] > .5, cd[:, 4] > .5

    def pers(o):
        last = np.zeros(q.shape[1], bool)
        for s in range(o - 12, o):
            last = np.where(obs[s], q[s], last)
        val = obs[o + 1:o + 7]; fut = q[o + 1:o + 7] & val
        p = np.broadcast_to(last, fut.shape) & val; u = (p | fut).sum()
        return (p & fut).sum() / u if u else 1., bool(fut.any())
    pt, _ = pers(t)
    gate = obs[t - 13:t - 1].mean() >= cov and bool(((q[t - 13:t - 1] & obs[t - 13:t - 1]).sum(0) >= 2).any())
    pa = False
    if gate:
        p1, f1 = pers(t - 1); pa = f1 and p1 > .9
    return float(pt <= .9), float(pa)


def fill(x, fl):
    """x: (B, C, L) tensor after apply_drop; fl: (B, 3) flags."""
    f = torch.tensor(np.asarray(fl, np.float32), dtype=x.dtype)
    x[:, 127:130] = f[:, :, None].expand(-1, -1, x.shape[-1]).to(x.device)
    return x


def make_flags(cube_days, B, ongoing, cov):
    out = []
    for (d, t), og in zip(B, ongoing):
        if not og:
            out.append((0., 0., 0.)); continue
        fh, fa = flags_day(cube_days[d], t, cov)
        if MODE == "unknown" or (MODE == "train" and _rng.random() < P_UNK):
            out.append((fh, 0., 1.))
        else:
            out.append((fh, fa, 0.))
    return out


def enable():
    import qcnn_train, qcnn_adapt, qcnn_predict_adapt
    if getattr(qcnn_train, "_sccnn", False):
        return
    oi, om = qcnn_train.inputs, qcnn_adapt.make_x

    def inputs(panel, W, dev, drop=()):
        x = oi(panel, W, dev, drop)
        fl = make_flags(panel.raw, list(zip(W.day, W.t)), (W.condition == "queue_ongoing").to_numpy(), .7)
        return fill(x, fl)

    def make_x(c3, rc, stat, B, drop):
        x = om(c3, rc, stat, B, drop)
        return fill(x, make_flags(c3, B, [True] * len(B), .5))
    qcnn_train.inputs = inputs
    qcnn_adapt.make_x = make_x; qcnn_predict_adapt.make_x = make_x
    qcnn_train._sccnn = True
    print("sccnn: selection flags enabled (P_UNK %.2f)" % P_UNK, flush=True)
