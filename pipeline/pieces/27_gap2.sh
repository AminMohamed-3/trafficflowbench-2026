#!/usr/bin/env bash
# Row 27 (ADOPTED.md row 27): L_gap2 gap rows, C66 -> C69 (artifacts/claude_ref/l_gap2/c_gap_on_c66.csv, sha d1ada3d6).
# Only the 128,465 gap rows change: gap stage W3 + gap2 m (March rows: GAPW W3 + m3; April rows: G2 April-own J fit,
# weights_g2.json), gap2 members = G1 off8k_s0-2 (shipped cnr0-2 warm-restarted +4000 it, saved weights, CPU serve), then
# G3 v6r specialist blended after the gap2 step (a per split: March all-dates fit, April own-view fit). Other lines byte for byte.
# Usage: 27_gap2.sh PARENT.csv OUT.csv      PARENT = C66 c_bundle18_s2b_s7b.csv (sha f1a5278f; SKIP_PARENT_SHA=1 skips)
#   env STRICT_SERVE=1: a G1 served-value difference to the 2026-10-18 laptop serve is fatal (else reported only).
# Steps (original recipe artifacts/claude_ref/l_gap2/RECEIPT.json "recipe" 1-7; steps 2-5 there are FITS, here read from saved files):
#   1. G1 serve: docs/reference_code/gap2w_serve.py --splits validation,private --tags off8k_s0,off8k_s1,off8k_s2
#      --weights <27_gap2/g1_weights> --out $D/g1_pred_cpu   (CPU, 1 torch thread, ~1 min; recipe step 6)
#   2. G3 serve: src/27/g3_serve.py SPLIT model_v6r_off_SPLIT_{v,q}.txt names_v6r.json $D/g3  (both splits; saved LightGBM
#      models of recipe step 3 instead of the refit; eval features rebuilt as state6_gaptrain.load; ~5-10 min, ~2 GB)
#   3. src/27/g2_official.py (copy of exp/L_gap2/g2_official.py, recipe step 7) weights_g2.json --members $D/g1_pred_cpu
#      --v6 $D/g3 --v6a bench_validation.json,bench_private_aprfit.json  with G2_MEMBER_TAGS=off8k_s0,off8k_s1,off8k_s2;
#      parities p1-p3 (hook GAPW arm recomputed; == row 26 t1_gapwf gap rows; GAPW + m3 + shipped members == PARENT gap rows).
#      ~2 min, ~2.5 GB.
# Inputs: PARENT; row 26 intermediates: R26_HOOK (hooked S2b build npz, $WORK/26_bundle18/build/t1.csv.gaprows.npz; stored old
#   equivalent artifacts/claude_ref/l_gap2/build/t1.csv.gaprows.npz, same hook code; the row-26 file also passes p1/p2), R26_GAPWF
#   ($WORK/26_bundle18/post/t1_gapwf.state.parquet; stored bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet); row 19: R19_PRED_DIR
#   (shipped cnr0-2 served, $WORK/19_gap2w/pred_cpu; stored c_gap2c47/pred_cpu) + weights_mar.json; row 25: gap_cfg_gapw.json.
#   Base fitted inputs read in place under $TFB_OLD_REPO: artifacts/claude_ref/gap/offpred_v3.npz (gap rows table),
#   artifacts/claude_ref/gap/rows_off_<split>_<panel>.npz (gap v3 row tables, G3 eval features), release caches.
#   Weights rows final_solution/weights/rows/27.csv (62 files: 48 G1 .pt + 6 receipts, 4 v6r models + names, 2 a-fit json, weights_g2.json).
# Leaves in $WORK/27_gap2/: g1_pred_cpu/ (old artifacts/claude_ref/l_gap2/g1/pred_cpu), g3/pred_v6r_off_<split>.npz (old l_gap2/g3/).
source "$(dirname "$0")/lib/common.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
PARENT=${1:?parent csv}; OUT=${2:?out csv}
m() { cygpath -m "$1"; }
first() { local x; for x in "$@"; do [ -e "$x" ] && { echo "$x"; return; }; done; echo ""; }
SRC=$FS/pipeline/pieces/src/27; D=$WORK/27_gap2; mkdir -p "$D/g1_pred_cpu" "$D/g3"
CR=$OLD/artifacts/claude_ref
HOOK=${R26_HOOK:-$(first "$WORK/26_bundle18/build/t1.csv.gaprows.npz" "$CR/l_gap2/build/t1.csv.gaprows.npz")}
GAPWF=${R26_GAPWF:-$(first "$WORK/26_bundle18/post/t1_gapwf.state.parquet" "$CR/l_b6/bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet")}
R19=${R19_PRED_DIR:-$(first "$WORK/19_gap2w/pred_cpu" "$CR/c_gap2c47/pred_cpu")}
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" f1a5278f5f17335445ccd4fdc6852ecb760ff41e3dd5282040094846b05e5108
WD=$("$PY" "$FS/pipeline/pieces/src/19/wdir.py" 27 27_gap2/g1_weights/)
say "27 inputs: hook $HOOK | gapwf $GAPWF | shipped gap2 members $R19 | G1 weights $WD"
say "27 step 1: G1 serve"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$OLD/docs/reference_code/gap2w_serve.py" --splits validation,private \
  --tags off8k_s0,off8k_s1,off8k_s2 --weights "$(m "$WD")" --out "$(m "$D/g1_pred_cpu")"
"$PY" - "$(m "$D/g1_pred_cpu")" "$(m "$SRC/serve_expected.json")" "${STRICT_SERVE:-0}" <<'EOF'
import json, sys
from pathlib import Path
d, exp, strict = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["values_sha256"], sys.argv[3] == "1"
got = {}
for sp in ("validation", "private"):
    for f, v in json.loads((d / f"serve_{sp}.json").read_text())["outputs"].items():
        got[f] = v["values_sha256"]
bad = [f for f in exp if got.get(f) != exp[f]]
print("27 G1_SERVE_VALUES", "EQUAL" if not bad else "DIFFER " + ",".join(bad), flush=True)
sys.exit(1 if bad and strict else 0)
EOF
say "27 step 2: G3 serve"
for sp in validation private; do
  "$PY" "$SRC/g3_serve.py" $sp "$(m "$(wpath 27_gap2/g3/model_v6r_off_${sp}_v.txt)")" "$(m "$(wpath 27_gap2/g3/model_v6r_off_${sp}_q.txt)")" \
    "$(m "$(wpath 27_gap2/g3/names_v6r.json)")" "$(m "$D/g3")"
done
say "27 step 3: gap rows over PARENT"
rm -f "$OUT" "$OUT.tmp"
G2_MEMBER_TAGS=off8k_s0,off8k_s1,off8k_s2 G2O_PARENT=$(m "$PARENT") G2O_SKIP_PARENT_SHA=1 G2O_GAPWF=$(m "$GAPWF") G2O_HOOK=$(m "$HOOK") \
G2O_OUT=$(m "$OUT") G2O_GAPCFG=$(m "$(wpath 25_gapw/gap_cfg_gapw.json)") G2O_M3=$(m "$(wpath 19_gap2w/weights_mar.json)") G2O_SHIP=$(m "$R19") \
  "$PY" "$SRC/g2_official.py" "$(m "$(wpath 27_gap2/weights_g2.json)")" c_gap_on_c66 --members "$(m "$D/g1_pred_cpu")" --v6 "$(m "$D/g3")" \
  --v6a "$(m "$(wpath 27_gap2/g3/bench_validation.json)"),$(m "$(wpath 27_gap2/g3/bench_private_aprfit.json)")"
say "27 done: $OUT $(sha "$OUT")"
