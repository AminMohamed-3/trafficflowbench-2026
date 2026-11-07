#!/usr/bin/env bash
# Row 24 (ADOPTED.md row 24): L_b6 S2 = B6R (B6 imputer + ramp and occupancy channels), C59 -> C64 (c_b6s2_on_c59, sha ddbd4591).
# Usage: 24_b6s2.sh PARENT.csv OUT.csv [--control C54_STATE.parquet] [--state STATE.parquet] [--members DIR] [--panels P]
#   PARENT    = C59 c_tinyad_on_c57.csv (sha c9a635fd, output of row 23); SKIP_PARENT_SHA=1 skips the check.
#   --control = row 20's state parquet (default $WORK/20_lmicro/t1.csv.state.parquet, or env CTL20; old stored equivalent
#               artifacts/claude_ref/l_micro/a2/t1.csv.state.parquet). The C54 state rows carry unchanged to C59 (rows 21-23
#               change queue rows only), so parent rows equal to it are the non-gap state rows the S2 build replaces.
#   --members DIR = skip the serve step and take the 60 member files from DIR (old stored: artifacts/claude_ref/l_b6/s2/arm36k).
#   --state F = skip serve and build, use state parquet F (cheap parity of the file step).
#   --panels P = cheap parity: serve and build only panel(s) P (no OUT): $WORK/24_b6s2/arm36k (6 files per panel) and
#               $WORK/24_b6s2/t1.csv.panels.parquet.
# Steps (original exp/L_b6/s2_finish.sh step 5; laptop 2026-10-17: serve 5 min at 4 threads, build 34 min at 4 threads, peak RSS
#   3.2 GB, file step ~2 min / ~3 GB):
#   1. serve: src/24/s2_serve.py -> src/24/s2_b6r.py serve (unchanged copy of exp/L_b6/s2_b6r.py): 60 UNetS2 members (10 panels x
#      2 splits x seeds 0,1,2; 36k iterations = 24k parent + 12k warm) on the official view, CPU, OMP/MKL 4 threads;
#      asm2_b6l.py cannot serve them (cin 16). Inputs b6full_imp.inputs (7) + 9 ramp/occupancy channels (released data of the split;
#      ramp calibration and occ_ref fitted on the split's I-S-OWN visible cells, c_b6long_full_step.own_layer).
#   2. state build: src/24/s2_build.py (= exp/L_b6/s2_build.py) $WORK/24_b6s2/t1.csv src/24/tune_s2.json: the full Task 1 chain
#      (asm2_final --state-only) with row 20's L_micro knobs (lm_split stage 3 scale .8/.7, polish a 1.0) and the S2 b6 block
#      (members of step 1, blend weights b6w_s2.json = all-days fit with the C54 v_nb - .1). It rebuilds all 6,735,795 state rows.
#   3. file step: src/24/s2_probe.py: the 6,607,330 non-gap state rows of C59 take the S2 values; the 128,465 gap2 rows stay
#      from the parent (C0 / GAPW of row 25 rebuild them), all queue / ODME rows stay.
# Inputs from the base build run (as row 20): S08_DIR = $A/s08/l127_r4800 (old artifacts/claude_ref/c_budget/s08/official/
#   l127_r4800); release caches; base fitted inputs under $TFB_CODE_ROOT; serve code read in place from docs/reference_code:
#   c_b6long_full_step.py (sha 711540d5), c_b6long_kg_step.py, b6full_imp.py (s2_b6r has its own ramp code). The 24k B6 members
#   are NOT read (the S2 sd_*.pt files hold the full warm-started weights).
# Weights rows final_solution/weights/rows/24.csv (122): 60 sd_*.pt + 60 run_*.json (it36000_w32), b6w_s2.json, kg_ref_all.json.
# Leaves in $WORK/24_b6s2/ (rows 25-27 read these; their old stored equivalents in brackets):
#   arm36k/imp_{valown,privown}_<panel>_s{0,1,2}.npz + receipt.json   [artifacts/claude_ref/l_b6/s2/arm36k]
#   t1.csv.state.parquet + t1.csv.state.json (the S2 chain, all state rows incl. its own gap-stage rows, which C64 does not use)
#                                                                       [artifacts/claude_ref/l_b6/s2/build/t1.csv.state.parquet]
#   build_rss.txt, info.json                                            [artifacts/claude_ref/l_b6/s2/c_b6s2_on_c59.info.json]
#   (The gap-stage internals GAPW needs, t1.csv.gaprows.npz, are not written here: exp/L_b6/gapw_build.py re-runs this build with a
#   gap-row dump and checks it against the parquet above.)
source "$(dirname "$0")/lib/common.sh"
PARENT=${1:?parent csv}; OUT=${2:?out csv}; shift 2
D=$WORK/24_b6s2
CONTROL=${CTL20:-$WORK/20_lmicro/t1.csv.state.parquet}; STATE=""; MEMBERS=""; PANELS=""
while [ $# -gt 0 ]; do case $1 in --control) CONTROL=$2; shift 2;; --state) STATE=$2; shift 2;; --members) MEMBERS=$2; shift 2;;
  --panels) PANELS=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
export TFB_CODE_ROOT=${TFB_CODE_ROOT:-$OLD}
BASE_RUN=${BASE_RUN:-$TFB_CODE_ROOT/artifacts/claude_ref/asm2/run}
export S08_DIR=${S08_DIR:-$BASE_RUN/s08/l127_r4800}
SRC=$FS/pipeline/pieces/src/24; mkdir -p "$D"
export LM_THREADS=${LM_THREADS:-4} LM_RSS_GB=${LM_RSS_GB:-5} LM_RSS_FILE=$D/build_rss.txt PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
unset B6W_SMOKE_ITERS B6R_PIT B6R_PARENT
wpy check 24
export B6W=$(wpath 24_b6s2/b6w_s2.json)
serve() {   # $1 = panels ("" = all)
  local WR; WR=$("$PY" -c "import sys;from pathlib import Path;print(Path(sys.argv[1]).parent)" "$("$PY" "$SRC/wdir.py" 24 24_b6s2/it36000_w32/)")
  say "24 serve (weights root $WR, panels ${1:-all})"; rm -rf "$D/arm36k"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" "$SRC/s2_serve.py" "$D/arm36k" "$WR" "$(wpath 24_b6s2/kg_ref_all.json)" "$1"
}
if [ -n "$PANELS" ]; then
  if [ -n "$MEMBERS" ]; then export B6_DIR=$MEMBERS; else serve "$PANELS"; export B6_DIR=$D/arm36k; fi
  say "24 state build, panels $PANELS (cheap parity; B6 $B6_DIR)"
  "$PY" "$SRC/s2_build.py" "$D/t1.csv" "$SRC/tune_s2.json" --panels "$PANELS"
  exit 0
fi
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" c9a635fd35917ec279b09d8c034792051896592676848850eb670d89fd72019f
[ -f "$CONTROL" ] || { echo "control parquet missing: $CONTROL (row 20 intermediate; set --control or CTL20)"; exit 2; }
if [ -z "$STATE" ]; then
  if [ -n "$MEMBERS" ]; then export B6_DIR=$MEMBERS; else serve ""; export B6_DIR=$D/arm36k; fi
  say "24 state build (S08 $S08_DIR, B6 $B6_DIR, threads $LM_THREADS)"
  rm -f "$D/t1.csv.state.parquet"
  "$PY" "$SRC/s2_build.py" "$D/t1.csv" "$SRC/tune_s2.json"
  STATE=$D/t1.csv.state.parquet
  say "24 state build done: $(cat "$D/build_rss.txt") sha $(sha "$STATE")"
fi
say "24 file step (state $STATE, control $CONTROL)"
rm -f "$OUT"
"$PY" "$SRC/s2_probe.py" "$STATE" "$PARENT" "$CONTROL" "$OUT" "$D/info.json"
say "24 done: $OUT $(sha "$OUT")"
