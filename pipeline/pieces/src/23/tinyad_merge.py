"""Row 23 member: the per-date adapted Tiny probabilities -> the A1 member (R2 piece worker, 2026-10-21).

  python tinyad_merge.py SHIPPED_TINY.parquet OUT.parquet [--ref STORED_MEMBER.parquet]

Logic copied from exp/L_tinyad/tinyad.py merge() (sha 23071a3061696efc39c652fa7e2855cf414553ac0bc15fdb71a18ec68d19c477),
unchanged except the paths: the 24 per-date files (window_id, timestamp, link_id, p_un, p_ad, kind, H; stored adapted
predictions of the A1 VM run, weights rows 23_tinyad/dates/*) are resolved through lib/weights.py (sha checked against the
weights row AND, as in the original, against each file's own .json receipt), and the shipped VM Tiny member (row 21 member,
sha c02523d2) is passed in. Output:
  p = clip(shipped + .75 (p_ad - p_un), 0, 1)   in float64, cast to the shipped dtype, rows in the shipped order.
Checks (as the original): 87,000 unique keys; with mix 0 the member equals the shipped one. --ref: the result must equal the
stored member (sha a7c7c2a9) bytewise, or else value for value (keys and prob bit-equal; parquet bytes may differ with
another pandas / pyarrow)."""
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import weights as WT

SHIP_SHA = "c02523d2e9464f9e4bda710ad889375c21c44dab1df9ee30d52b41d4cb7ddf6a"
REF_SHA = "a7c7c2a94712c8cc0ad1d5bf32fac292f5df92a501faa8d2bbe49e8e54de3895"
MIX = .75                                                           # tinyad.RECIPE["mix"]
K = ["window_id", "timestamp", "link_id"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ship"); ap.add_argument("out"); ap.add_argument("--ref")
    a = ap.parse_args()
    dates = sorted(r["repo_path"] for r in WT.rows("23") if r["repo_path"].startswith("23_tinyad/dates/") and r["repo_path"].endswith(".parquet"))
    assert len(dates) == 24, len(dates)
    parts = []
    for rp in dates:                                                # sorted by name = sorted glob of the original
        f = WT.resolve(rp); j = json.loads(Path(str(WT.resolve(rp + ".json"))).read_text()); assert sha(f) == j["sha256"], f
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    ship = pd.read_parquet(a.ship); assert sha(a.ship) == SHIP_SHA, "shipped VM Tiny member sha"
    s = ship.set_index(K).prob; assert len(df) == len(ship) == 87000 and df.set_index(K).index.is_unique
    shp = s.loc[list(zip(df.window_id, df.timestamp, df.link_id))].to_numpy()
    p = np.clip(shp.astype(np.float64) + MIX * (df.p_ad.to_numpy(np.float64) - df.p_un.to_numpy(np.float64)), 0., 1.).astype(shp.dtype)
    p0 = np.clip(shp.astype(np.float64) + 0. * (df.p_ad.to_numpy(np.float64) - df.p_un.to_numpy(np.float64)), 0., 1.).astype(shp.dtype)
    assert int((p0 != shp).sum()) == 0
    out = df[K].copy(); out["prob"] = p
    out = out.set_index(K).loc[list(zip(ship.window_id, ship.timestamp, ship.link_id))].reset_index()     # shipped row order
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out, index=False)
    rec = dict(rows=len(out), cells_moved=int((p != shp).sum()), un_vs_ship_cells=int((df.p_un.to_numpy() != shp).sum()), sha256=sha(a.out))
    if a.ref:
        ref = pd.read_parquet(a.ref); assert sha(a.ref) == REF_SHA, "stored member sha"
        rec["ref_sha_equal"] = rec["sha256"] == REF_SHA
        rec["ref_keys_equal"] = bool((ref[K].to_numpy() == out[K].to_numpy()).all())
        rec["ref_prob_cells_differ"] = int((ref.prob.to_numpy() != out.prob.to_numpy()).sum()) if rec["ref_keys_equal"] else None
        rec["ref_dtype"] = f"{ref.prob.dtype}|{out.prob.dtype}"
        assert rec["ref_sha_equal"] or (rec["ref_keys_equal"] and rec["ref_prob_cells_differ"] == 0 and ref.prob.dtype == out.prob.dtype), rec
    Path(str(a.out) + ".json").write_text(json.dumps(rec, indent=1))
    print("TINYAD_MERGE", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
