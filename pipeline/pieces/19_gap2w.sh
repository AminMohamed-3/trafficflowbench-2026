#!/usr/bin/env bash
# Row 19 (ADOPTED.md row 19): gap2 member from saved weights, C47 -> C50 (c_gap2w_on_c47.csv, sha 9ddd8848).
# Usage: 19_gap2w.sh PARENT.csv OUT.csv          PARENT = C47 c_b6l24k_on_c44.csv (sha 1606b1c5)
#   env SKIP_PARENT_SHA=1 skips the parent sha check; STRICT_SERVE=1 makes a served-prediction difference fatal.
# Steps (original recipe exp/C_gap2c47/BUILD_STEP.md, g2w_build.py build; ~2 min serve + ~1 min blend on the laptop):
#   1. serve the 3 gap U-Net members (cnr0,1,2; 48 panel weights + 6 receipts) on CPU, 1 torch thread:
#      $TFB_CODE_ROOT/docs/reference_code/gap2w_serve.py --weights <weights dir> --out $WORK/19_gap2w/pred_cpu
#      (docs/reference_code/gap2w_serve.py sha 9333f8a6, gap2w_cnn.py sha 23bd8acb, used in place; import closure:
#      asm2_phys_lwr_diag, b6full_imp, flow_krige, flow_own, gap2_ramps, gap2w_cnn, gap_lib, phys_lwr_diag, tfb.py, src/tfb).
#      The served arrays are checked against the laptop serve of 2026-10-13 (src/19/serve_expected.json, values sha).
#   2. blend: src/19/g2_build.py (= exp/C_gap2c47/g2_build.py main, paths parameterised) on the 128,465 official gap&target
#      state rows of PARENT: clip((1-w) row + w mean(3 members, smooth 1)), w = weights_mar.json m3. Streamed, ~1 GB RAM.
# Inputs: PARENT; weights rows final_solution/weights/rows/19.csv (56 files: 48 .pt, 6 receipts, weights_mar.json,
#   keymap.npz = docs/reference_code/xfer_rows.py keymap index of the submission key);
#   base-build fitted inputs read in place under $TFB_CODE_ROOT (R1, docs/report/ASM2_DEPENDENCIES.md):
#   artifacts/claude_ref/gap/offpred_v3.npz (gap rows table M, sha 3f93f48c), release caches (scripts/build_cache.py),
#   data/raw/kaggle_public (task2/<panel>/<split>/window_index.csv, corridors/<panel>/network/lwr_mainline_topology.csv).
#   No intermediate of the base build run is needed (the step works on the finished C47 file).
# Leaves in $WORK/19_gap2w/: pred_cpu/pred_cnr{0,1,2}_off_{validation,private}.npz + serve_<split>.json (old equivalent
#   artifacts/claude_ref/c_gap2c47/pred_cpu/), delta.csv.gz + <OUT stem>.json (old artifacts/claude_ref/c_gap2c47/c_gap2w_on_c47.*).
source "$(dirname "$0")/lib/common.sh"
PARENT=${1:?parent csv}; OUT=${2:?out csv}
export TFB_CODE_ROOT=${TFB_CODE_ROOT:-$OLD}
SRC=$FS/pipeline/pieces/src/19; D=$WORK/19_gap2w; mkdir -p "$D"
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a
wpy check 19
WD=$("$PY" "$SRC/wdir.py" 19 19_gap2w/weights/)
say "19 serve: weights $WD"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$TFB_CODE_ROOT/docs/reference_code/gap2w_serve.py" --weights "$WD" --out "$D/pred_cpu"
"$PY" - "$D/pred_cpu" "$SRC/serve_expected.json" "${STRICT_SERVE:-0}" <<'EOF'
import json, sys
from pathlib import Path
d, exp, strict = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["values_sha256"], sys.argv[3] == "1"
got = {}
for sp in ("validation", "private"):
    for f, v in json.loads((d / f"serve_{sp}.json").read_text())["outputs"].items():
        got[f] = v["values_sha256"]
bad = [f for f in exp if got.get(f) != exp[f]]
print("19 SERVE_VALUES", "EQUAL" if not bad else "DIFFER " + ",".join(bad), flush=True)
sys.exit(1 if bad and strict else 0)
EOF
say "19 blend"
"$PY" "$SRC/g2_build.py" "$PARENT" "$OUT" "$D/pred_cpu" "$(wpath 19_gap2w/weights_mar.json)" "$(wpath 19_gap2w/keymap.npz)" "$D/delta.csv.gz"
say "19 done: $OUT $(sha "$OUT")"
