"""ong2 2026-10-05: persistence-band decode for 'crossing' (type A) ongoing windows.
Type A official windows (previous origin allowed and pre-T eligible, so its persistence IoU was > .9) have truth
persistence IoU in a narrow band (train official: .76-.895, median .87). Rule on top of selcond (cap hi):
if IoU(mask, pers) < lo, move the mask toward persistence (add the most probable persistent cells or drop the least
probable non-persistent cells, greedy by expected IoU) until IoU(mask, pers) >= lo. Decode on all horizon cells."""
import numpy as np
from gapres_selcond_rule import iou, eiou_order, exp_iou, constrained


def raise_to(m, f, pers, lo, sub=None):
    """Greedy: each step take the best of (add highest-p persistent cell not in mask) / (drop lowest-p non-persistent cell
    in mask) by expected IoU, until IoU(m, pers) >= lo. sub: optional bool mask of cells the IoU and moves are
    restricted to (e.g. horizon steps 1..5)."""
    m = m.copy(); sub = np.ones(len(f), bool) if sub is None else sub
    add = [j for j in np.argsort(-f) if pers[j] and not m[j] and sub[j]]
    drop = [j for j in np.argsort(f) if m[j] and not pers[j] and sub[j]]
    ia = idr = 0
    while iou(m[sub], pers[sub]) < lo and (ia < len(add) or idr < len(drop)):
        best = None
        for kind in ("add", "drop"):
            if kind == "add" and ia < len(add):
                mm = m.copy(); mm[add[ia]] = True
            elif kind == "drop" and idr < len(drop):
                mm = m.copy(); mm[drop[idr]] = False
            else:
                continue
            e = exp_iou(mm, f)
            if best is None or e > best[0]:
                best = (e, kind, mm)
        _, kind, m = best
        if kind == "add":
            ia += 1
        else:
            idr += 1
    return m


def band_decode(f, pers, hi=.92, lo=None):
    """f, pers: flat arrays over all horizon cells. Returns (mask, tag)."""
    k, o = eiou_order(f)
    m, tag = constrained(f, o, k, pers, hi)
    if lo is not None and iou(m, pers) < lo:
        m = raise_to(m, f, pers, lo); tag = "raise"
    return m, tag
