"""Canonical target-only-parent occupancy features, S0424.
Reuses S0419's frozen recipe and artifact readers, writes to separate result/model dirs.
fit: seed from S0419_SEED. own validation/private. xbuild/xscore for other-date S3L.
"""
import os,sys,types
import numpy as np
import s0419_occ as O
import s0416_density as B
old_features=O.features

def features(h,v,q,pf):
    # Canonical full field: parent only on targets, released observations everywhere else.
    # Missing non-targets never enter the neighbor residual sums.
    target=np.asarray(h.target,bool)
    cv=np.where(target,v,np.asarray(h.speed,float))
    cq=np.where(target,q,np.asarray(h.flow,float))
    h64=types.SimpleNamespace(target=target,gap=h.gap,stations=h.stations,
        speed=np.asarray(h.speed,float),flow=np.asarray(h.flow,float),occ=np.asarray(h.occ,float))
    X,ix,names=old_features(h64,cv,cq,pf)
    return X,ix,names

O.features=features
O.BASE_OUT=B.ROOT/'artifacts/lane_s/0424_occ'
O.OUT=O.BASE_OUT if O.SEED==4701 else O.BASE_OUT/f'seed{O.SEED}'

def main():
    cmd=sys.argv[1]
    if cmd=='fit': return O.main()
    if cmd=='own':
        import s0419_own as W
        W.OUT=B.ROOT/'artifacts/lane_s/0424_own'
        return W.evaluate(sys.argv[2])
    if cmd in ('xbuild','xscore'):
        import s0419_xparent as X
        X.OUT=B.ROOT/'artifacts/lane_s/0424_xparent'
        return X.build() if cmd=='xbuild' else X.score()
    raise ValueError(cmd)

if __name__=='__main__': main()
