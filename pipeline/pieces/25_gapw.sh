#!/usr/bin/env bash
# Row 25 (ADOPTED.md row 25): C0 + L10 GAPW-W, C64 -> C65 (artifacts/claude_ref/l_b6/gapw/c_gapw_on_c64.csv, sha 7b2a2daa).
# The 128,465 official gap rows rebuilt over the S2 chain: gap stage with the GAPW W3 weights (gap_cfg_gapw.json, sha 0c9c1551,
# March all-dates fit), then the C48 gap2 step (row 19 members, weights_mar.json m3). Every other line of PARENT byte for byte.
# Usage: 25_gapw.sh PARENT.csv OUT.csv [--panel P] [--hook-from PREFIX]
#   PARENT = C64 c_b6s2_on_c59.csv (sha ddbd4591; SKIP_PARENT_SHA=1 skips the check)
#   --panel P        CHEAP PARITY: hooked S2 build of panel P only (both splits, asm2_final debug mode, no parquet), its hook
#                    arrays compared with the stored hook (STORED_HOOK, default artifacts/claude_ref/l_b6/gapw/build/
#                    t1.csv.gaprows.npz) -> exit; OUT is not written.
#   --hook-from PRE  skip step 1 and use an existing hooked build (PRE.state.parquet + PRE.gaprows.npz); the stored one is
#                    artifacts/claude_ref/l_b6/gapw/build/t1.csv (parity of steps 2-3 in ~4 min).
# Steps (original recipe exp/L_b6/gapw_finish.sh: gapw_build.py -> gapw_official.py -> c0_official.py; GAPW_RESULT.md):
#   1. src/25/gap_hook_build.py $D/build/t1.csv src/25/tune_s2.json --w3 gap_cfg_gapw.json --arm-dir ARM --s08-dir S08
#      = the S2 state build (s2_build.py path: asm2_final --state-only, C54 lm_build per-split knobs, stage 2 3M, tune_s2.json)
#      with the GAPW hook (copy of exp/L_gap2/g2_build.py, superset of gapw_build.py). ~34 min CPU (4 threads), peak RSS 3.7 GB.
#   2. src/25/gap_post.py $D/build/t1.csv REF $D/post t1_gapw  (gapw_official.py checks 1-2; gap rows <- W3 arm)    ~1 min
#   3. src/25/gap2_step.py PARENT $D/post/t1_gapw.state.parquet OUT --members R19 --gap2w weights_mar.json  (c0_official) ~2 min
# Inputs: PARENT; intermediates of earlier pieces (env var -> default; the stored old equivalents are the parity defaults):
#   R24_ARM_DIR   S2 B6R members served on the official view (row 24 serve step; s2_b6r.py serve), 60 imp_*.npz
#                 -> first "$WORK"/24_*/arm36k, else stored artifacts/claude_ref/l_b6/s2/arm36k
#   R24_STATE     the plain S2 build parquet (row 24), only for check 1 (determinism) -> first "$WORK"/24_*/build/t1.csv.state.parquet,
#                 else none (parity runner: stored artifacts/claude_ref/l_b6/s2/build/t1.csv.state.parquet)
#   R19_PRED_DIR  gap2 members cnr0-2 served on CPU (row 19) -> $WORK/19_gap2w/pred_cpu if present, else stored
#                 artifacts/claude_ref/c_gap2c47/pred_cpu;  weights_mar.json = weights row 19_gap2w/weights_mar.json
#   S08_DIR       base build (R1, asm2_run_all.sh --s08=127,4800): S08 official parquets official_{validation,private}.parquet
#                 = $A/s08/l127_r4800 (A = artifacts/claude_ref/asm2/run); default the folder the stored build used,
#                 artifacts/claude_ref/c_budget/s08/official/l127_r4800
#   other fitted inputs of the build read in place under $TFB_OLD_REPO (list: artifacts/claude_ref/l_b6/gapw/build/t1.csv.state.json
#   "inputs"): stage-2 models c_budget/s2/all_l63_m2000_n3000000_r600/m_{v,q}.txt, S lane (lane_s 0424/0426/0447), phys5 stage 4,
#   phys3 m368, phys4 s3_y{q,v}L, polish/lsa json, gap v3 config + gap/offpred_v3.npz, asm/b6w_s08.json, b6w_s2.json (row 24),
#   release caches. Weights rows: final_solution/weights/rows/25.csv (gap_cfg_gapw.json).
# Leaves in $WORK/25_gapw/: build/t1.csv.{state.parquet,state.json,gaprows.npz} (old artifacts/claude_ref/l_b6/gapw/build/t1.csv.*),
#   post/t1_old.state.parquet + post/t1_gapw.state.parquet (old gapw/build/t1_old, t1_gapw), OUT.info.json.
source "$(dirname "$0")/lib/common.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
PARENT=${1:?parent csv}; OUT=${2:?out csv}; shift 2
PANEL=""; HOOK_FROM=""
while [ $# -gt 0 ]; do case $1 in --panel) PANEL=$2; shift 2;; --hook-from) HOOK_FROM=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
m() { cygpath -m "$1"; }
SRC=$FS/pipeline/pieces/src/25; D=$WORK/25_gapw; mkdir -p "$D/build" "$D/post"
first() { local x; for x in "$@"; do [ -e "$x" ] && { echo "$x"; return; }; done; echo ""; }
ARM=${R24_ARM_DIR:-$(first "$WORK"/24_*/arm36k "$OLD/artifacts/claude_ref/l_b6/s2/arm36k")}
REF=${R24_STATE:-$(first "$WORK"/24_*/build/t1.csv.state.parquet)}; REF=${REF:-none}
R19=${R19_PRED_DIR:-$(first "$WORK/19_gap2w/pred_cpu" "$OLD/artifacts/claude_ref/c_gap2c47/pred_cpu")}
S08=${S08_DIR:-$OLD/artifacts/claude_ref/c_budget/s08/official/l127_r4800}
W3=$(wpath 25_gapw/gap_cfg_gapw.json); G2W=$(wpath 19_gap2w/weights_mar.json)
say "25 inputs: arm $ARM | ref $REF | gap2 members $R19 | S08 $S08 | W3 $W3"
if [ -n "$PANEL" ]; then
  LM_THREADS=${LM_THREADS:-3} LM_RSS_GB=${LM_RSS_GB:-4} "$PY" "$SRC/gap_hook_build.py" "$(m "$D/build/p_$PANEL.csv")" "$(m "$SRC/tune_s2.json")" \
    --w3 "$(m "$W3")" --arm-dir "$(m "$ARM")" --s08-dir "$(m "$S08")" --panels "$PANEL"
  "$PY" "$SRC/cmp_hook.py" "$(m "$D/build/p_$PANEL.csv.gaprows.npz")" "$(m "${STORED_HOOK:-$OLD/artifacts/claude_ref/l_b6/gapw/build/t1.csv.gaprows.npz}")"
  exit $?
fi
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" ddbd45915f12ded493816f913c4e28f31c4e6803d2f02a51359aa03f59fe4510
if [ -z "$HOOK_FROM" ]; then
  say "25 step 1: hooked S2 state build"
  LM_THREADS=${LM_THREADS:-3} LM_RSS_GB=${LM_RSS_GB:-4} "$PY" "$SRC/gap_hook_build.py" "$(m "$D/build/t1.csv")" "$(m "$SRC/tune_s2.json")" \
    --w3 "$(m "$W3")" --arm-dir "$(m "$ARM")" --s08-dir "$(m "$S08")"
  HOOK_FROM=$D/build/t1.csv
fi
say "25 step 2: post step (hook $HOOK_FROM)"
"$PY" "$SRC/gap_post.py" "$(m "$HOOK_FROM")" "$([ "$REF" = none ] && echo none || m "$REF")" "$(m "$D/post")" t1_gapw
say "25 step 3: gap2 step over PARENT"
"$PY" "$SRC/gap2_step.py" "$(m "$PARENT")" "$(m "$D/post/t1_gapw.state.parquet")" "$(m "$OUT")" --members "$(m "$R19")" --gap2w "$(m "$G2W")"
say "25 done: $OUT $(sha "$OUT")"
