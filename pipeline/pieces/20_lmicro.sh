#!/usr/bin/env bash
# Row 20 (ADOPTED.md row 20): L_micro file 2, C50 -> C54 (c_lmicro2_on_c50, sha b0cbce4e).
# Usage: 20_lmicro.sh PARENT.csv OUT.csv --control C47.csv|C47_STATE.parquet [--state STATE.parquet] [--panels P]
#   PARENT  = C50 c_gap2w_on_c47.csv (sha 9ddd8848, output of row 19); SKIP_PARENT_SHA=1 skips the check.
#   --control = the C47 state build: preferably the base build's <OUT>.state.parquet (asm2_final.py writes it next to its output
#             file, also without --state-only; old stored equivalent artifacts/claude_ref/c_b6long/p24k/official/t1.csv.state.parquet),
#             or the C47 file itself (row 19's parent; CSV path of lm_probe.py, not exercised by the parity runs). Only used to tell
#             the 128,465 gap2 rows of the parent (they differ from C47) from the rows the new build replaces.
#   --state F = skip the state build and use F (cheap parity of the file step with the stored build parquet).
#   --panels P = cheap parity of the build: only panel(s) P, writes $WORK/20_lmicro/t1.csv.panels.parquet, no OUT.
# Steps (original exp/L_micro/finish2.sh; laptop 2026-10-15: build 54 min at 6 threads, peak RSS 3.3 GB; file step ~5 min, ~3 GB):
#   1. state build: src/20/lm_build.py (= exp/L_micro/lm_build.py, paths parameterised) $WORK/20_lmicro/t1.csv src/20/tune_lm2.json
#      = asm2_final --state-only (full Task 1 chain: s08in, c10, s2 (3M), b6 blend, polish, lsa, s3p4, gap, clip, S lane, phys5
#      stage 4) with the knobs of tune_lm2.json: stage 3 flow scale .8 (validation) / .7 (private), polish a 1.0, B6 blend
#      weights b6w_lm_a2_vm01.json (= b6w_24k.json with v_nb - .1 on every panel), B6 members = the 24k set.
#   2. file step: src/20/lm_probe.py: parent state rows equal to the control take the new values; gap2 rows, queue, ODME stay.
# Inputs from the base build run (R1, docs/reference_code/asm2_run_all.sh, run dir A=artifacts/claude_ref/asm2/run by default;
#   set BASE_RUN, or S08_DIR / B6_DIR one by one):
#   S08_DIR = $A/s08/l127_r4800/official_{validation,private}.parquet   (c_s08_ship.py export; old stored equivalent
#             artifacts/claude_ref/c_budget/s08/official/l127_r4800/, sha c0ea341d / bfc24f2a)
#   B6_DIR  = $A/b6l/imp_{valown,privown}_<panel>_s{0,1,2}.npz, 60 files (asm2_b6l.py serve --it 24000, CPU; old stored
#             equivalent artifacts/claude_ref/c_b6long/p24k/arm24k/, same 24k weights, receipt.json holds the array shas)
#   plus the release caches (scripts/build_cache.py) and the base fitted inputs read in place under $TFB_CODE_ROOT (the 116
#   inputs of the stored build are listed in artifacts/claude_ref/l_micro/a2/t1.csv.state.json; all but two are the C47 build's).
#   All mapped dirs must be under $TFB_CODE_ROOT (asm2_build.Stages._in records paths relative to it).
# Weights rows final_solution/weights/rows/20.csv: b6w_lm_a2_vm01.json (1 file). Tune: src/20/tune_lm2.json (= exp/L_micro/tune_lm2.json, sha 8fbb6a04).
# Leaves in $WORK/20_lmicro/: t1.csv.state.parquet + t1.csv.state.json (old equivalent artifacts/claude_ref/l_micro/a2/t1.csv.state.*;
#   row 24 reads this parquet as its control), build_rss.txt, info.json.
source "$(dirname "$0")/lib/common.sh"
PARENT=${1:?parent csv}; OUT=${2:?out csv}; shift 2
CONTROL=""; STATE=""; PANELS=""
while [ $# -gt 0 ]; do case $1 in --control) CONTROL=$2; shift 2;; --state) STATE=$2; shift 2;; --panels) PANELS=$2; shift 2;; *) echo "unknown $1"; exit 2;; esac; done
export TFB_CODE_ROOT=${TFB_CODE_ROOT:-$OLD}
BASE_RUN=${BASE_RUN:-$TFB_CODE_ROOT/artifacts/claude_ref/asm2/run}
export S08_DIR=${S08_DIR:-$BASE_RUN/s08/l127_r4800} B6_DIR=${B6_DIR:-$BASE_RUN/b6l}
SRC=$FS/pipeline/pieces/src/20; D=$WORK/20_lmicro; mkdir -p "$D"
export LM_THREADS=${LM_THREADS:-6} LM_RSS_FILE=$D/build_rss.txt PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
wpy check 20
export B6W=$(wpath 20_lmicro/b6w_lm_a2_vm01.json)
if [ -n "$PANELS" ]; then
  say "20 state build, panels $PANELS (cheap parity)"
  "$PY" "$SRC/lm_build.py" "$D/t1.csv" "$SRC/tune_lm2.json" --panels "$PANELS"
  exit 0
fi
[ -n "$CONTROL" ] || { echo "--control required"; exit 2; }
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" 9ddd8848c235627c665b2a6732043485f7e2c14e3bbb45f09b6bb509f1d75bb4
if [ -z "$STATE" ]; then
  say "20 state build (S08 $S08_DIR, B6 $B6_DIR, threads $LM_THREADS)"
  rm -f "$D/t1.csv.state.parquet"
  "$PY" "$SRC/lm_build.py" "$D/t1.csv" "$SRC/tune_lm2.json"
  STATE=$D/t1.csv.state.parquet
  say "20 state build done: $(cat "$D/build_rss.txt") sha $(sha "$STATE")"
fi
say "20 file step (state $STATE, control $CONTROL)"
rm -f "$OUT"
"$PY" "$SRC/lm_probe.py" "$STATE" "$PARENT" "$CONTROL" "$OUT" "$D/info.json"
say "20 done: $OUT $(sha "$OUT")"
