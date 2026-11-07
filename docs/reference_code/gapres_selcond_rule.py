"""gapres: selection-conditioned decoding rule (shared by gapres_selcond.py tests and gapres_selcond_official.py)."""
import numpy as np


def iou(a, b):
    u = (a | b).sum()
    return 1. if u == 0 else (a & b).sum() / u


def eiou_order(f):
    o = np.argsort(-f); c = np.cumsum(f[o]); m = np.arange(1, len(f) + 1)
    return int(np.argmax(c / (m + f.sum() - c))) + 1, o


def exp_iou(mask, f):
    # E[IoU] ~ E[inter] / E[union] under independent marginals
    inter = f[mask].sum(); union = mask.sum() + f.sum() - inter
    return inter / max(union, 1e-9)


def constrained(f, o, k, pers, c):
    base = np.zeros(len(f), bool); base[o[:k]] = True
    if iou(base, pers) <= c:
        return base, "none"
    best, tag = None, None
    for kind in ("grow", "shrink"):
        m = base.copy()
        if kind == "grow":
            cand = [j for j in o if not m[j] and not pers[j]]
            for j in cand:
                m[j] = True
                if iou(m, pers) <= c:
                    break
        else:
            cand = [j for j in o[:k][::-1] if pers[j]]
            for j in cand:
                m[j] = False
                if iou(m, pers) <= c:
                    break
        if iou(m, pers) <= c and (best is None or exp_iou(m, f) > exp_iou(best, f)):
            best, tag = m, kind
    return (base, "none") if best is None else (best, tag)
