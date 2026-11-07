#!/usr/bin/env bash
# 20_pieces.sh: ADOPTED rows 19 to 35 CHAINED on the fresh base build (worker R4, 2026-10-21).
#
#   outputs/c47/final_A.csv -> 19 C50 -> 20 C54 -> 21 C55 -> 22 C57 -> 23 C59 -> 24 C64 -> 25 C65 -> 26 C66 -> 27 C69
#   -> 28 C70 -> 29 C71 -> 30 FV2 -> 31 C73/C73P/C74P -> 32 C75 -> 33 C76 -> 34 C77/C78 -> 35 C79/C80 -> outputs/C80.csv
#
# Every piece reads the output of the piece before it (no stored parent). The piece scripts are called unchanged; this
# runner only sets their environment (paths of the fresh base run and of earlier pieces) and turns the parent sha checks
# off (the fresh base is value-equal, not byte-equal, to C47). After each step the output is compared with the stored
# development file of that step and one line goes to outputs/logs/chain_19_35.tsv:
#   SHA_EQUAL        sha256 equal to the stored file
#   VALUE_EQUAL_TOL  pipeline/99_verify.py PASS: queue / ODME values exact, state speed / flow within 1e-8, same ids
#   DIFF             99_verify FAIL (max differences in the detail column); the run CONTINUES
#   NOT_COMPARED     no single-step stored reference on disk (sha difference still recorded)
#   PIECE_FAILED     the piece script exited non-zero: the run STOPS (exit 1)
# Rows 28 to 35 run with R3_SOFT=1 (a sha difference is recorded, not fatal) and R3_KEEP=1 (this runner compares every
# chain file a piece wrote, then removes the ones the piece itself would drop); their own check lines go to
# outputs/logs/chain_r3_checks.tsv (build parquets B_*, serves, hook parity).
#
# Usage: bash pipeline/20_pieces.sh [--dry-run] [--from NN] [--to NN] [--keep]
#   --dry-run   resolve paths, check every input and stored reference, print the plan; no lock, no build, exit 0/1
#   --from NN   resume at step NN (needs the earlier outputs in WORK: run the earlier part with --keep)
#   --to NN     stop after step NN
#   --keep      keep the work folder (default: intermediates no later step reads are removed as the chain goes)
# Environment (all optional):
#   CHAIN_WORK      work folder (default final_solution/work/chain; must lie inside the development repo for rows 28-35)
#   CHAIN_BASE      fresh base file (default outputs/c47/final_A.csv; its .state.parquet is row 20's control)
#   CHAIN_BASE_RUN  base build run folder (default outputs/c47/run, else artifacts/claude_ref/asm2/run)
#   CHAIN_S08_DIR   S08 exports (default <run>/s08/l127_r4800, else artifacts/claude_ref/c_budget/s08/official/l127_r4800 =
#                   the --s08-fitted exports the base build read)
#   TFB_CODE_ROOT / TFB_ROOT / TFB_OLD_REPO  code + fitted-file root of the pieces (default: the development repo)
# Laptop rules: the laptop lock (exp/L_b6/logs/laptop.lock.d, lib/common.sh protocol) is held during each of rows 19-27 and
# released between them; rows 28-35 take it themselves around their builds. Before every step: RAM >= 4 GiB and the
# 13.5 GiB disk floor (waits, does not fail).
# Runtime (laptop CPU, from the STATUS files): rows 19-27 about 3.5 h without contention (24: 45 min, 20: 31 min, 25: 40 min,
# 26: 80 min, rest minutes), rows 28-35 about 8.5 h, comparisons about 0 to 30 min: about 12 to 13 h in all.
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export FS
DRY=0; FROM=19; TO=35; KEEP=0
while [ $# -gt 0 ]; do case $1 in
  --dry-run) DRY=1; shift;; --from) FROM=$2; shift 2;; --to) TO=$2; shift 2;; --keep) KEEP=1; shift;;
  *) echo "unknown argument $1 (see header)"; exit 2;; esac; done
case "$FROM$TO" in *[!0-9]*) echo "--from / --to need a row number"; exit 2;; esac
export WORK=${CHAIN_WORK:-$FS/work/chain}
source "$FS/pipeline/pieces/lib/common.sh"     # FS, OLD, PY, WORK, OUTD, LOCK, say, lock/unlock, waitmem, waitdisk, sha
set +e; set -uo pipefail
PIECES=$FS/pipeline/pieces; R3T=$PIECES/src/r3lib/expected.tsv
LOGD=$FS/outputs/logs; PLOG=$LOGD/chain; TSV=$LOGD/chain_19_35.tsv; R3RES=$LOGD/chain_r3_checks.tsv
REFROOT=${CHAIN_REF_ROOT:-$OLD}                 # stored development files (the per-step references) live under this root
m() { cygpath -m "$1"; }
first() { local x; for x in "$@"; do [ -e "$x" ] && { echo "$x"; return; }; done; echo ""; }

# ---- inputs of the fresh base build -------------------------------------------------------------------------------
BASE=${CHAIN_BASE:-$FS/outputs/c47/final_A.csv}; BASE_STATE=$BASE.state.parquet
BASE_RUN=${CHAIN_BASE_RUN:-$(first "$FS/outputs/c47/run/mix/cnn_ens2.parquet" "$FS/artifacts/claude_ref/asm2/run/mix/cnn_ens2.parquet")}
BASE_RUN=${BASE_RUN%/mix/cnn_ens2.parquet}
S08=${CHAIN_S08_DIR:-$(first "$BASE_RUN/s08/l127_r4800/official_validation.parquet" "$FS/artifacts/claude_ref/c_budget/s08/official/l127_r4800/official_validation.parquet")}
S08=${S08%/official_validation.parquet}
# piece environment: base run intermediates and earlier pieces' intermediates (never the stored old equivalents)
export TFB_BASE_RUN=$BASE_RUN BASE_RUN=$BASE_RUN S08_DIR=$S08 B6_DIR=$BASE_RUN/b6l
export SKIP_PARENT_SHA=1 PARENT_SHA_CHECK=0
export R19_PRED_DIR=$WORK/19_gap2w/pred_cpu R24_ARM_DIR=$WORK/24_b6s2/arm36k R24_STATE=$WORK/24_b6s2/t1.csv.state.parquet
export R26_HOOK=$WORK/26_bundle18/build/t1.csv.gaprows.npz R26_GAPWF=$WORK/26_bundle18/post/t1_gapwf.state.parquet
CH=$WORK/r3chain; R3W=$WORK/r3

# ---- step table -----------------------------------------------------------------------------------------------------
# rows 19-27: NN|script|parent|out|expected sha256|stored reference (full:PATH or pd:PARENT:DELTA, paths under REFROOT)
CR=artifacts/claude_ref
R2STEPS=(
"19|19_gap2w.sh|C47|C50|9ddd8848c235627c665b2a6732043485f7e2c14e3bbb45f09b6bb509f1d75bb4|full:$CR/probes/c_gap2w_on_c47.csv.gz"
"20|20_lmicro.sh|C50|C54|b0cbce4e1352a38c41110225f9d04794c616d0a91c8b743ffadc895b722b729a|pd:$CR/probes/c_gap2w_on_c47.csv.gz:$CR/l_micro/c_lmicro2_on_c50.delta.csv.gz"
"21|21_tinyvm.sh|C54|C55|5ebd5de0438517e03a719ecf88545acd26152287441e66265bb3d81081ff3020|full:$CR/probes/c_tinyvm_on_c54.csv.gz"
"22|22_a3cold.sh|C55|C57|caaf4d8a3740a27db5cb5237164b3850c69761830002649ef4497f06ed7999a5|full:$CR/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv.gz"
"23|23_tinyad.sh|C57|C59|c9a635fd35917ec279b09d8c034792051896592676848850eb670d89fd72019f|full:$CR/l_tinyad/a1/on_c57/delta/c_tinyad_on_c57.csv.gz"
"24|24_b6s2.sh|C59|C64|ddbd45915f12ded493816f913c4e28f31c4e6803d2f02a51359aa03f59fe4510|full:$CR/l_b6/s2/c_b6s2_on_c59.csv.gz"
"25|25_gapw.sh|C64|C65|7b2a2daa0eac7310681f647a9d4562be56aebb48b8c089d0f6898f039f89d4f1|full:$CR/l_b6/gapw/c_gapw_on_c64.csv.gz"
"26|26_bundle18.sh|C65|C66|f1a5278f5f17335445ccd4fdc6852ecb760ff41e3dd5282040094846b05e5108|full:$CR/l_b6/bundle18/c_bundle18_s2b_s7b.csv"
"27|27_gap2.sh|C66|C69|d1ada3d6453ee9fd0b25ce9c24f48e4ce062a63302ab69b8af82e8f513a99c1e|full:$CR/l_gap2/c_gap_on_c66.csv"
)
R3PIECES=(28_fvft_mar 29_bundle19 30_fv2 31_fvinel 32_gapfv 33_qaccum 34_fvinel_w10 35_s7c_inel)
R3MAIN=([28]=C70 [29]=C71 [30]=FV2_BOTH_ON_C70 [31]=C74P [32]=C75 [33]=C76 [34]=C78 [35]=C80)
xcol() { awk -F'\t' -v n="$1" -v c="$2" '$1 == n {print $c}' "$R3T"; }
namebysha() { awk -F'\t' -v s="$1" '$2 == s {print $1}' "$R3T" | head -n 1; }
r3spec() {   # stored reference of R3 chain file NAME: its full file, else its parent's full file + its delta
  local n=$1 f d pn pf; f=$(xcol "$n" 3)
  if [ -n "$f" ] && [ "$f" != - ] && [ -f "$REFROOT/$f" ]; then echo "full:$f"; return; fi
  d=$(xcol "$n" 4); pn=$(namebysha "$(xcol "$n" 5)"); pf=$([ -n "$pn" ] && xcol "$pn" 3)
  if [ -n "$d" ] && [ "$d" != - ] && [ -f "$REFROOT/$d" ] && [ -n "$pf" ] && [ "$pf" != - ] && [ -f "$REFROOT/$pf" ]; then echo "pd:$pf:$d"; return; fi
  echo none; }
mkref() {    # SPEC EXPECTED -> REFARGS for 99_verify.py; return 1 when the reference is not on disk
  local s=$1 e=$2 a b; REFARGS=()
  case $s in
    full:*) a=$REFROOT/${s#full:}; [ -f "$a" ] || return 1; REFARGS=(--ref "$(m "$a")" --ref-sha "$e");;
    pd:*) a=${s#pd:}; b=$REFROOT/${a#*:}; a=$REFROOT/${a%%:*}; [ -f "$a" ] && [ -f "$b" ] || return 1
          REFARGS=(--ref-parent "$(m "$a")" --ref-delta "$(m "$b")" --ref-sha "$e")
          case $a in *.gz) REFARGS+=(--no-ref-sha-check);; esac;;   # 99_verify's sha rebuild reads the parent's raw line ending
    *) return 1;;
  esac; }
chainfile() { [ "$1" = C47 ] && echo "$BASE" || echo "$WORK/$1.csv"; }

# ---- logging and comparison ------------------------------------------------------------------------------------------
clean() { printf '%s' "$*" | tr '\t\r\n' '   ' | cut -c1-600; }
row() {      # STEP NAME KIND VERDICT SHA EXPECTED RUNTIME_S DETAIL
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "${5:0:16}" "${6:0:16}" "$7" "$(clean "${8:--}")" >> "$TSV"
  say "STEP $1 $2 ($3): $4 ${8:+- $(clean "$8" | cut -c1-200)}"; }
cmp_csv() {  # STEP NAME FILE EXPECTED SPEC RUNTIME NOTE [SHA]
  local st=$1 nm=$2 f=$3 e=$4 spec=$5 rt=$6 note=$7 s=${8:-} out v j
  [ -n "$s" ] || s=$(sha "$f")
  if [ "$s" = "$e" ]; then row "$st" "$nm" csv SHA_EQUAL "$s" "$e" "$rt" "$note"; return 0; fi
  if ! mkref "$spec" "$e"; then row "$st" "$nm" csv NOT_COMPARED "$s" "$e" "$rt" "sha differs; no stored single-step reference ($spec) $note"; return 0; fi
  j=$PLOG/verify_$nm.json; waitmem 35 "compare $nm"
  out=$("$PY" "$FS/pipeline/99_verify.py" "$(m "$f")" "${REFARGS[@]}" --state-atol 1e-8 --json "$(m "$j")" 2>&1 | tail -n 1)
  case $out in VERIFY_PASS*) v=VALUE_EQUAL_TOL;; VERIFY_FAIL*) v=DIFF;; *) v=VERIFY_ERROR;; esac
  row "$st" "$nm" csv "$v" "$s" "$e" "$rt" "$note $out"; }

# ---- dry run ---------------------------------------------------------------------------------------------------------
if [ $DRY = 1 ]; then
  bad=0; ok() { if [ -e "$2" ]; then printf '  ok    %s: %s\n' "$1" "$2"; else printf '  MISS  %s: %s\n' "$1" "$2"; bad=1; fi; }
  echo "20_pieces.sh DRY RUN (rows $FROM..$TO; keep=$KEEP)"
  echo "  FS=$FS"; echo "  OLD (TFB_OLD_REPO)=$OLD"; echo "  TFB_CODE_ROOT=${TFB_CODE_ROOT:-$OLD}"; echo "  PY=$PY"
  echo "  WORK=$WORK"; echo "  BASE_RUN=$BASE_RUN"; echo "  S08_DIR=$S08"; echo "  B6_DIR=$B6_DIR"; echo "  stored references under $REFROOT"
  echo "inputs:"
  ok "python" "$PY"; ok "fresh base file" "$BASE"; ok "fresh base state parquet (row 20 control)" "$BASE_STATE"
  for f in mix/cnn_ens2.parquet mix/sccnn_official.parquet ongoing.parquet ongoing_bb.parquet on/cnn_s0.parquet off/cnn_s0.parquet; do ok "base run" "$BASE_RUN/$f"; done
  nb=$(ls "$B6_DIR"/imp_*.npz 2>/dev/null | wc -l); echo "  $([ "$nb" = 60 ] && echo ok || { echo MISS; bad=1; })    base run B6 members: $nb of 60 in $B6_DIR"
  ok "S08 export" "$S08/official_validation.parquet"; ok "S08 export" "$S08/official_private.parquet"
  ok "99_verify" "$FS/pipeline/99_verify.py"; ok "R3 expected table" "$R3T"
  echo "plan:"
  for r in "${R2STEPS[@]}"; do IFS='|' read -r nn sc par out e spec <<< "$r"; [ "$nn" -ge "$FROM" ] && [ "$nn" -le "$TO" ] || continue
    ex=""; [ "$nn" = 20 ] && ex=" --control <base>.state.parquet"
    printf '  %s  lock; bash pipeline/pieces/%s %s %s%s\n' "$nn" "$sc" "$([ "$par" = C47 ] && echo '<base>' || echo "WORK/$par.csv")" "WORK/$out.csv" "$ex"
    ok "      piece" "$PIECES/$sc"; mkref "$spec" "$e" && printf '        compare %s vs %s (sha %s)\n' "$out" "${spec#*:}" "${e:0:8}" || { printf '  MISS        stored reference %s\n' "$spec"; bad=1; }
  done
  for p in "${R3PIECES[@]}"; do nn=${p%%_*}; [ "$nn" -ge "$FROM" ] && [ "$nn" -le "$TO" ] || continue
    printf '  %s  bash pipeline/pieces/%s.sh  (R3_SOFT=1 R3_KEEP=1, CH=WORK/r3chain; takes the lock around its builds)\n' "$nn" "$p"
    ok "      piece" "$PIECES/$p.sh"; n=${R3MAIN[$nn]}; s=$(r3spec "$n")
    printf '        main output %s (sha %s) vs %s; then drop: %s\n' "$n" "$(xcol "$n" 2 | cut -c1-8)" "$s" "$(sed -n 's/^drop //p' "$PIECES/$p.sh")"
    [ "$s" = none ] && { echo "  MISS        no stored reference for $n"; bad=1; }
  done
  [ "$TO" -ge 35 ] && echo "  end: WORK/r3chain/C80.csv -> outputs/C80.csv; summary line in outputs/logs/chain_19_35.tsv"
  echo "state now:"
  echo "  disk free C: $("$PY" -c "import shutil; print(round(shutil.disk_usage('C:/').free / 2**30, 2))") GiB (floor 13.5)"
  echo "  RAM available: $("$PY" -c "import psutil; print(round(psutil.virtual_memory().available / 2**30, 2))") GiB (need 4)"
  if [ -d "$LOCK" ]; then op=$(cat "$LOCK/pid" 2>/dev/null); echo "  laptop lock: held by pid $op ($(cat "$LOCK/owner" 2>/dev/null)) $(kill -0 "$op" 2>/dev/null && echo alive || echo STALE)"; else echo "  laptop lock: free"; fi
  echo "DRY_RUN $([ $bad = 0 ] && echo OK || echo MISSING_INPUTS)"; exit $bad
fi

# ---- real run --------------------------------------------------------------------------------------------------------
trap unlock EXIT
mkdir -p "$PLOG"
if [ "$FROM" = 19 ]; then   # clean start: the work folder must be this runner's own (marker) or empty
  if [ -n "$(ls -A "$WORK" 2>/dev/null)" ]; then
    [ -f "$WORK/.r4_chain" ] || { say "FAILED: $WORK is not empty and not a chain work folder (no .r4_chain marker)"; exit 2; }
    say "clean start: removing the previous chain work folder contents"; rm -rf "${WORK:?}"/* "$WORK"/.r4_chain
  fi
  [ -f "$TSV" ] && mv -f "$TSV" "$TSV.prev"; [ -f "$R3RES" ] && mv -f "$R3RES" "$R3RES.prev"
fi
mkdir -p "$WORK"; touch "$WORK/.r4_chain"
[ -f "$TSV" ] || printf 'time_utc\tstep\tname\tkind\tverdict\tsha256\texpected\truntime_s\tdetail\n' > "$TSV"
for f in "$BASE" "$BASE_STATE" "$BASE_RUN/mix/cnn_ens2.parquet" "$S08/official_validation.parquet"; do
  [ -f "$f" ] || { say "FAILED: input missing $f (run with --dry-run)"; exit 2; }; done
say "chain 19-35 start: rows $FROM..$TO, base $BASE, run $BASE_RUN, S08 $S08, WORK $WORK"
T00=$SECONDS
if [ "$FROM" = 19 ]; then
  bs=$(sha "$BASE"); bv=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('VERIFY_PASS' if not d.get('problems') else 'VERIFY_FAIL', 'state max|d| speed %.3g flow %.3g' % (d['columns']['state.speed_kmh']['max_abs'], d['columns']['state.flow_vph']['max_abs']))" "$(m "$FS/outputs/c47/verify.json")" 2>/dev/null || echo "verify.json not readable")
  row 10 C47 csv "$([ "$bs" = 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a ] && echo SHA_EQUAL || { case $bv in VERIFY_PASS*) echo VALUE_EQUAL_TOL;; *) echo DIFF;; esac; })" \
    "$bs" 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a - "fresh base build (outputs/c47/verify.json: $bv)"
fi

# rows 19-27 (R2 pieces: no lock of their own, so the runner holds it per piece)
for r in "${R2STEPS[@]}"; do
  IFS='|' read -r nn sc par out e spec <<< "$r"
  [ "$nn" -ge "$FROM" ] && [ "$nn" -le "$TO" ] || continue
  P=$(chainfile "$par"); O=$WORK/$out.csv
  [ -f "$P" ] || { row "$nn" "$out" csv PIECE_FAILED - "$e" 0 "parent $P missing (resume needs the earlier outputs: --keep)"; exit 1; }
  EX=(); [ "$nn" = 20 ] && EX=(--control "$BASE_STATE"); [ "$nn" = 24 ] && EX=(--control "$WORK/20_lmicro/t1.csv.state.parquet")
  waitmem 40 "row $nn"; waitdisk 1350 "row $nn"
  lock "R4 chain row $nn"; echo "final_solution R4 chain (row $nn, $sc)" > "$LOCK/owner"
  t0=$SECONDS; rm -f "$O"
  bash "$PIECES/$sc" "$P" "$O" ${EX[@]+"${EX[@]}"} > "$PLOG/${nn}_${out}.log" 2>&1; rc=$?
  unlock; rt=$((SECONDS - t0))
  if [ $rc -ne 0 ] || [ ! -s "$O" ]; then
    row "$nn" "$out" csv PIECE_FAILED - "$e" "$rt" "exit $rc; $(grep -v '^\s*$' "$PLOG/${nn}_${out}.log" | tail -n 2)"; exit 1; fi
  note=$(grep -hoE '(G1_)?SERVE_VALUES [A-Z]+.*' "$PLOG/${nn}_${out}.log" | head -n 1)
  cmp_csv "$nn" "$out" "$O" "$e" "$spec" "$rt" "${note:+$note;}"
  if [ $KEEP = 0 ]; then   # remove what no later step reads (C66 and C69 go on to rows 28-35)
    [ "$par" != C47 ] && [ "$par" != C66 ] && rm -f "$(chainfile "$par")"
    case $nn in 24) rm -rf "$WORK/20_lmicro";; 25) rm -rf "$WORK/24_b6s2" "$WORK/25_gapw";;
      27) rm -rf "$WORK/26_bundle18/build" "$WORK/26_bundle18/s7b" "$WORK/26_bundle18/post" "$WORK/27_gap2" "$WORK/19_gap2w";; esac
  fi
done

# rows 28-35 (R3 pieces: chain folder CH, their own lock around builds)
if [ "$TO" -ge 28 ]; then
  mkdir -p "$CH"
  for n in C66 C69; do
    [ -f "$CH/$n.csv" ] && continue
    [ -f "$WORK/$n.csv" ] || { row 28 "$n" csv PIECE_FAILED - - 0 "row 26/27 output $WORK/$n.csv missing"; exit 1; }
    if [ $KEEP = 1 ]; then ln "$WORK/$n.csv" "$CH/$n.csv" 2>/dev/null || cp "$WORK/$n.csv" "$CH/$n.csv"; else mv "$WORK/$n.csv" "$CH/$n.csv"; fi
  done
  S2B=$WORK/26_bundle18/arm48k
  if [ "$(ls "$S2B"/imp_*.npz 2>/dev/null | wc -l)" = 60 ]; then export R3_S2B_SERVE=$(m "$S2B")
  else say "NOTE: row 26 served S2b arm not in $S2B: rows 28-35 use the stored arm48k (row 26 parity: served == stored arrays)"; fi
  export R3_SOFT=1 R3_KEEP=1 R3_PARENTS=chain R3_BUILDS=run RES=$R3RES
  touch "$R3RES"
  for p in "${R3PIECES[@]}"; do
    nn=${p%%_*}; [ "$nn" -ge "$FROM" ] && [ "$nn" -le "$TO" ] || continue
    waitmem 40 "row $nn"; waitdisk 1350 "row $nn"
    n0=$(wc -l < "$R3RES"); t0=$SECONDS
    bash "$PIECES/$p.sh" > "$PLOG/${p}.log" 2>&1; rc=$?; rt=$((SECONDS - t0))
    # every check line the piece wrote: chain csv files (compared now), build parquets / serves / hooks (verdict as recorded)
    tail -n +$((n0 + 1)) "$R3RES" | while IFS=$'\t' read -r _t _pc nm res s e note; do
      if [ -n "$(xcol "$nm" 2)" ]; then
        if [ -f "$CH/$nm.csv" ]; then cmp_csv "$nn" "$nm" "$CH/$nm.csv" "$e" "$(r3spec "$nm")" "$rt" "" "$s"
        else row "$nn" "$nm" csv "$([ "$res" = PASS-SHA ] && echo SHA_EQUAL || echo NOT_COMPARED)" "$s" "$e" "$rt" "file no longer in the chain folder"; fi
      else
        case $res in PASS-SHA) v=SHA_EQUAL;; FAIL) v=SHA_DIFF;; *) v=$res;; esac
        row "$nn" "$nm" "$(case $nm in B_*) echo build;; SERVE_*) echo serve;; *) echo check;; esac)" "$v" "$s" "$e" "$rt" "$note"
      fi
    done
    if [ $rc -ne 0 ]; then row "$nn" "PIECE_$nn" piece PIECE_FAILED - - "$rt" "exit $rc; $(grep -h 'R3_FAIL' "$PLOG/${p}.log" | tail -n 1)"; exit 1; fi
    [ $KEEP = 1 ] || for n in $(sed -n 's/^drop //p' "$PIECES/$p.sh"); do rm -f "$CH/$n.csv" "$CH"/d_"$n"__* "$CH"/d_*__"$n".*; done   # = lib/r3.sh drop
    say "row $nn done in $rt s"
  done
fi

# end: final file, summary
if [ "$TO" -ge 35 ]; then
  [ -f "$CH/C80.csv" ] || { row 35 C80 csv PIECE_FAILED - - 0 "CH/C80.csv missing"; exit 1; }
  if [ $KEEP = 1 ]; then cp "$CH/C80.csv" "$OUTD/C80.csv"; else mv -f "$CH/C80.csv" "$OUTD/C80.csv"; fi
  say "final file $OUTD/C80.csv sha $(sha "$OUTD/C80.csv")"
  if [ $KEEP = 0 ]; then   # logs of rows 28-35 to outputs/logs/chain/r3, then the bulky work files
    ( cd "$R3W" 2>/dev/null && find . -name '*.log' -exec cp --parents {} "$PLOG/" \; ) 2>/dev/null
    rm -rf "$CH" "$R3W/serve" "$R3W/build" "$R3W/gapfv" "$R3W/qaccum" "$R3W/gap3" "$WORK"/2[0-9]_*; rm -f "$R3W"/*.csv
  fi
fi
S=$(awk -F'\t' 'NR > 1 && $3 != "summary" {c[$5]++} END {for (k in c) printf "%s=%d ", k, c[k]}' "$TSV")
row "$FROM-$TO" summary - CHAIN_DONE - - $((SECONDS - T00)) "$S"
say "CHAIN_19_35_DONE in $(( (SECONDS - T00) / 60 )) min: $S"
