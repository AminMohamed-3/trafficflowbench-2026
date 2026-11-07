#!/usr/bin/env bash
# Row 26 (ADOPTED.md row 26): BUNDLE18, C65 -> C66 (artifacts/claude_ref/l_b6/bundle18/c_bundle18_s2b_s7b.csv, sha f1a5278f).
# Parts: (1) S2b B6R members (48k) in the chain; (2) GAPW-F: 128,465 gap rows = GAPW W3 gap stage over the S2b chain + gap2 step;
# (3) S7b: stage 3 = shipped s3L + 60 own-month trees per split (C_*_mar_all March rows, C_*_apr_all April rows) on the S2b chain,
# giving the 6,607,330 non-gap state rows. Queue/ODME lines of PARENT byte for byte.
# Usage: 26_bundle18.sh PARENT.csv OUT.csv [--panel P]
#   PARENT = C65 c_gapw_on_c64.csv (sha 7b2a2daa; SKIP_PARENT_SHA=1 skips the check)
#   --panel P   CHEAP PARITY (no OUT): serve the 6 S2b members of panel P (both splits) and compare with the stored arm48k
#               arrays; hooked S2b build of panel P (stored arm48k) compared with the stored BUNDLE18 hook npz -> exit.
#   env USE_GAPWF / USE_S7B / USE_REF = existing parquets: skip steps 1-4 (cheap parity of step 5 on all rows uses the stored
#               bundle18/off_c_bundle18_s2b/t1_gapwf, bundle18/s7b/build/t1.csv, s2b/build/t1.csv state parquets).
# Steps (original: exp/L_b6/s2ab_serve.py b; bundle18_official.sh = bundle18_build.py + bundle18_official.py + c0_official.py on the
#   S2b-on-C64 file; bundle18_s7bbuild.py; bundle18_stack.py; BUNDLE18.md "RESULT"):
#   1. src/26/s2b_serve.py $D/arm48k --root <weights 26_bundle18/s2b>   60 members CPU (s2_b6r.serve)      ~5 min, peak ~3 GB
#   2. src/25/gap_hook_build.py $D/build/t1.csv <tune_s2b, arm $D/arm48k>   hooked S2b state build       ~35 min, peak 3.8 GB
#   3. src/25/gap_post.py $D/build/t1.csv REF $D/post t1_gapwf   (bundle18_official.py)                   ~1 min
#   4. src/26/s7b_build.py $D/s7b/t1.csv <bundle18_tune_s7b, arm $D/arm48k, registered C models>       ~35 min, peak ~3.5 GB
#   5. src/25/gap2_step.py PARENT $D/post/t1_gapwf.state.parquet OUT --nongap $D/s7b/t1.csv.state.parquet --ref $D/build/t1.csv.state.parquet
#      gap rows = gap2 step over t1_gapwf (c0_official), non-gap state rows = S7b build (bundle18_stack.py), check b (S7b
#      gap rows == S2b build gap rows). The old intermediate files (S2b-on-C64 c_b6s2b_on_s2.csv, FILE12 c_bundle18_s2b.csv)
#      are not needed: every state line they touched is rewritten here from the same values with the same repr() format, and
#      their other fields equal C65's (checked: C66 vs C65, 0 of 6,980,503 lines differ outside speed/flow of state rows).
# Inputs: PARENT; R19_PRED_DIR (row 19 gap2 members; -> $WORK/19_gap2w/pred_cpu, else stored c_gap2c47/pred_cpu) and
#   weights_mar.json (row 19 weights); gap_cfg_gapw.json (row 25 weights); S08_DIR (base build R1: $A/s08/l127_r4800 official
#   parquets; default stored artifacts/claude_ref/c_budget/s08/official/l127_r4800); R26_REF (optional, the S2b plain build parquet for
#   check 1; parity: stored artifacts/claude_ref/l_b6/s2b/build/t1.csv.state.parquet). Other fitted build inputs in place under
#   $TFB_OLD_REPO as row 25 (list: artifacts/claude_ref/l_b6/bundle18/s7b/build/t1.csv.state.json "inputs").
#   Weights rows final_solution/weights/rows/26.csv: 60 sd_*.pt + 60 run_*.json (S2b it48000_w32), b6w_s2b.json, 4 S7b models.
# Leaves in $WORK/26_bundle18/: arm48k/ (old artifacts/claude_ref/l_b6/s2b/arm48k), build/t1.csv.* (old bundle18/build_c_bundle18_s2b
#   + s2b/build parquet), post/t1_gapwf.state.parquet (old bundle18/off_c_bundle18_s2b/), s7b/t1.csv.state.parquet (old
#   bundle18/s7b/build/). Row 27 reads build/t1.csv.gaprows.npz and post/t1_gapwf.state.parquet.
source "$(dirname "$0")/lib/common.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
PARENT=${1:?parent csv}; OUT=${2:?out csv}; shift 2
PANEL=""; [ "${1:-}" = --panel ] && PANEL=$2
m() { cygpath -m "$1"; }
first() { local x; for x in "$@"; do [ -e "$x" ] && { echo "$x"; return; }; done; echo ""; }
S25=$FS/pipeline/pieces/src/25; SRC=$FS/pipeline/pieces/src/26; D=$WORK/26_bundle18; mkdir -p "$D/build" "$D/post" "$D/s7b"
R19=${R19_PRED_DIR:-$(first "$WORK/19_gap2w/pred_cpu" "$OLD/artifacts/claude_ref/c_gap2c47/pred_cpu")}
S08=${S08_DIR:-$OLD/artifacts/claude_ref/c_budget/s08/official/l127_r4800}
REF=${R26_REF:-none}
WROOT=$("$PY" -c "import sys, pathlib; print(pathlib.Path(sys.argv[1]).parent.as_posix())" \
  "$("$PY" "$FS/pipeline/pieces/src/19/wdir.py" 26 26_bundle18/s2b/it48000_w32/)")   # folder holding it48000_w32/ (wdir.py: row 19 helper)
W3=$(wpath 25_gapw/gap_cfg_gapw.json); G2W=$(wpath 19_gap2w/weights_mar.json); B6W=$(wpath 26_bundle18/b6w_s2b.json)
M=26_bundle18/s7b_models
say "26 inputs: S2b weights $WROOT | gap2 members $R19 | S08 $S08 | ref $REF"
if [ -n "$PANEL" ]; then
  OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" "$SRC/s2b_serve.py" "$(m "$D/arm_$PANEL")" --root "$(m "$WROOT")" --panels "$PANEL"
  "$PY" - "$(m "$D/arm_$PANEL")" "$(m "$OLD/artifacts/claude_ref/l_b6/s2b/arm48k")" <<'EOF'
import sys, json, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2]); bad = []
fs = sorted(a.glob("imp_*.npz"))
for f in fs:
    x, y = np.load(f), np.load(b / f.name)
    if not all(np.array_equal(x[k], y[k]) for k in ("speed", "flow", "dates")):
        bad.append(f.name)
print(json.dumps(dict(serve_files=len(fs), differ=bad, result="EXACT" if fs and not bad else "FAIL")))
sys.exit(1 if bad or not fs else 0)
EOF
  "$PY" "$SRC/mk_tune.py" "$(m "$SRC/tune_s2b.json")" "$(m "$D/tune_s2b_stored_arm.json")" --arm-dir "$(m "$OLD/artifacts/claude_ref/l_b6/s2b/arm48k")" --b6w "$(m "$B6W")"
  LM_THREADS=${LM_THREADS:-3} LM_RSS_GB=${LM_RSS_GB:-4.5} "$PY" "$S25/gap_hook_build.py" "$(m "$D/build/p_$PANEL.csv")" "$(m "$D/tune_s2b_stored_arm.json")" \
    --w3 "$(m "$W3")" --s08-dir "$(m "$S08")" --panels "$PANEL"
  "$PY" "$S25/cmp_hook.py" "$(m "$D/build/p_$PANEL.csv.gaprows.npz")" "$(m "${STORED_HOOK:-$OLD/artifacts/claude_ref/l_b6/bundle18/build_c_bundle18_s2b/t1.csv.gaprows.npz}")"
  exit $?
fi
[ "${SKIP_PARENT_SHA:-0}" = 1 ] || need_sha "$PARENT" 7b2a2daa0eac7310681f647a9d4562be56aebb48b8c089d0f6898f039f89d4f1
GAPWF=${USE_GAPWF:-}; S7B=${USE_S7B:-}; SREF=${USE_REF:-}
if [ -z "$GAPWF" ] || [ -z "$S7B" ]; then
  if [ "${R26_SERVE:-0}" = 1 ]; then
    say "26 step 1: serve 60 S2b members (R26_SERVE=1: CPU re-serve, a sibling of C66, not C66)"
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" "$SRC/s2b_serve.py" "$(m "$D/arm48k")" --root "$(m "$WROOT")"
  else
    # Stored-prediction exception (as rows 21-23, retrain/s2b_arm48k.md): the 60 member arrays of the original serve
    # (artifacts/claude_ref/l_b6/s2b/arm48k), sha-pinned in weights/rows/26.csv. A CPU re-serve from the saved weights
    # today differs by float noise (mean |d speed| ~3e-4 km/h, max ~.2), which the state stack carries into every state row.
    say "26 step 1: stored S2b member arrays (60 + receipt, sha checked)"
    mkdir -p "$D/arm48k"
    for f in $("$PY" -c "import csv,sys;[print(r['repo_path']) for r in csv.DictReader(open(sys.argv[1])) if r['repo_path'].startswith('26_bundle18/arm48k/')]" "$(m "$FS/weights/rows/26.csv")" | tr -d '\r'); do
      cp "$(wpath "$f")" "$D/arm48k/${f##*/}"
    done
    [ "$(ls "$D/arm48k"/imp_*.npz | wc -l)" = 60 ] || { say "26 stored arrays: expected 60 member files"; exit 2; }
  fi
  "$PY" "$SRC/mk_tune.py" "$(m "$SRC/tune_s2b.json")" "$(m "$D/tune_s2b.json")" --arm-dir "$(m "$D/arm48k")" --b6w "$(m "$B6W")"
  "$PY" "$SRC/mk_tune.py" "$(m "$SRC/bundle18_tune_s7b.json")" "$(m "$D/tune_s7b.json")" --arm-dir "$(m "$D/arm48k")" --b6w "$(m "$B6W")" \
    --s3 "validation=$(m "$(wpath $M/C_yq_mar_all.txt)"),$(m "$(wpath $M/C_yv_mar_all.txt)")" \
    --s3 "private=$(m "$(wpath $M/C_yq_apr_all.txt)"),$(m "$(wpath $M/C_yv_apr_all.txt)")"
  say "26 step 2: hooked S2b state build"
  LM_THREADS=${LM_THREADS:-3} LM_RSS_GB=${LM_RSS_GB:-4.5} "$PY" "$S25/gap_hook_build.py" "$(m "$D/build/t1.csv")" "$(m "$D/tune_s2b.json")" \
    --w3 "$(m "$W3")" --s08-dir "$(m "$S08")"
  say "26 step 3: post step"
  "$PY" "$S25/gap_post.py" "$(m "$D/build/t1.csv")" "$([ "$REF" = none ] && echo none || m "$REF")" "$(m "$D/post")" t1_gapwf
  say "26 step 4: S7b state build"
  LM_THREADS=${LM_THREADS:-3} LM_RSS_GB=${LM_RSS_GB:-4.5} "$PY" "$SRC/s7b_build.py" "$(m "$D/s7b/t1.csv")" "$(m "$D/tune_s7b.json")" --s08-dir "$(m "$S08")"
  GAPWF=$D/post/t1_gapwf.state.parquet; S7B=$D/s7b/t1.csv.state.parquet; SREF=$D/build/t1.csv.state.parquet
fi
say "26 step 5: gap2 step + S7b non-gap rows over PARENT (gapwf $GAPWF, s7b $S7B, ref $SREF)"
"$PY" "$S25/gap2_step.py" "$(m "$PARENT")" "$(m "$GAPWF")" "$(m "$OUT")" --members "$(m "$R19")" --gap2w "$(m "$G2W")" \
  --nongap "$(m "$S7B")" --ref "$(m "$SREF")"
say "26 done: $OUT $(sha "$OUT")"
