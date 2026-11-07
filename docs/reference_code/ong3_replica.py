"""ong3 2026-10-05: organizer ongoing-selector replica on the released (masked) layer of a test split, every day.
qsel_replica rules (gate: eligible share of t-12..t-1 >= .7; some link with >= 2 queued eligible steps; a queued observed
cell in t+1..t+6; persistence IoU <= .9 with last observed state over t-12..t-1; earliest first per day; >= 72 steps after
the day's onset-selector picks and earlier ongoing picks), applied to the masked layer (pseudo labels), all days (not only
the first 5 of the split). Each pick typed like ong2_official_types: B if t-1 is blocked by the 72-step spacing,
A if t-1 passes the pre-T gate (so pers(t-1) > .9 or no queue in its horizon), else C.
These picks are the most official-like pseudo windows (selection includes the 'earliest' rule ong2's crossing analog lacks).
"""
import numpy as np
from qsel_replica import onset_select, day_candidates, picks, SP


def gate(q, el, t):
    return el[t - 12:t].mean() >= .7 and bool((((q & el)[t - 12:t]).sum(0) >= 2).any())


def replica_day(q, obs, el):
    """q, obs, el: (288, L) bool. Returns [(t, typ)]."""
    ons = [t for _, t in onset_select(q[None], el[None])]
    cands = day_candidates(q, obs, el)
    pk = picks(cands, ons)
    out = []
    for i, t in enumerate(pk):
        blk = ons + pk[:i]
        if any(abs(t - 1 - s) < SP for s in blk) or t - 1 < 12:
            ty = "B"
        else:
            ty = "A" if gate(q, el, t - 1) else "C"
        out.append((t, ty))
    return out
