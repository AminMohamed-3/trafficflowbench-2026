# sourced (after lib/common.sh) by the piece scripts and parity runners of rows 28 to 35 (worker R3, 2026-10-21).
#
# Chain folder CH ($WORK/r3chain): every chain file of rows 28-35 is CH/<NAME>.csv, NAME as in src/r3lib/expected.tsv
# (C66, C69, C70, ..., C80). A piece reads its parents with `cf NAME` and writes its outputs as CH/<NAME>.csv; every output is
# checked against the stored sha256 (expected.tsv) by `check NAME` (exit 1 on a mismatch unless R3_SOFT=1).
# Modes (environment):
#   R3_PARENTS=chain  (default) a parent must already be in CH (written by an earlier piece; the entry files C66 and C69 are
#                     linked in by run_28_35.sh from rows 26/27 or the stored files).
#   R3_PARENTS=stored a parent missing in CH is taken from the stored development file (sha checked; read in place), or rebuilt
#                     into CH from its stored delta (tools/probe.py expand, sha checked) when the full file was deleted.
#   R3_BUILDS=run     (default) members are served from saved weights and the state build runs (laptop lock, RAM >= 4 GiB).
#   R3_BUILDS=stored  the stored build parquet is used instead (sha checked against builds.tsv = the original receipts):
#                     glue-only parity (stack / transfer steps), minutes instead of hours.
#   R3_KEEP=1         keep intermediates (default: a piece deletes CH files that no later piece reads).
# TFB_ROOT = old-layout root the copied python reads inputs from (laptop: the development repo; clean machine: the folder where
# 00_fetch_data / 01_fetch_weights restored data and artifacts). WORK must lie inside TFB_ROOT (relative serve arms, rel paths).
# all paths in Windows mixed form (C:/...): MSYS_NO_PATHCONV=1 below, and Windows python cannot open /c/... paths
FS=$(cygpath -m "$FS"); OLD=$(cygpath -m "$OLD"); WORK=$(cygpath -m "$WORK"); OUTD=$(cygpath -m "$OUTD"); LOCK=$(cygpath -m "$LOCK")
export FS TFB_OLD_REPO=$OLD TFB_ROOT=$(cygpath -m "${TFB_ROOT:-$OLD}")
export MSYS_NO_PATHCONV=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONPATH=$TFB_ROOT/src CUDA_VISIBLE_DEVICES=-1
R3=$FS/pipeline/pieces/src/r3lib
CH=${CH:-$WORK/r3chain}; R3W=${R3W:-$WORK/r3}; RES=${RES:-$FS/pipeline/pieces/parity/r3_results.tsv}
CH=$(cygpath -m "$CH"); R3W=$(cygpath -m "$R3W"); RES=$(cygpath -m "$RES"); export CH R3W
CR=$TFB_ROOT/artifacts/claude_ref
R3_PARENTS=${R3_PARENTS:-chain}; R3_BUILDS=${R3_BUILDS:-run}
mkdir -p "$CH" "$R3W" "$(dirname "$RES")"
case "$WORK/" in "$TFB_ROOT/"*) ;; *) say "WORK ($WORK) must lie inside TFB_ROOT ($TFB_ROOT)"; exit 2;; esac
lock() { local k=0 op
  until mkdir "$LOCK" 2>/dev/null; do
    op=$(cat "$LOCK/pid" 2>/dev/null || true)
    if [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null; then say "stale lock of pid $op removed"; rm -rf "$LOCK"; continue; fi
    [ $((k % 30)) -eq 0 ] && say "waiting for the laptop lock (pid $op, $(cat "$LOCK/owner" 2>/dev/null))"; k=$((k + 1)); sleep 20
  done
  echo $$ > "$LOCK/pid"; echo "final_solution R3 rows 28-35 ($1)" > "$LOCK/owner"; say "lock taken ($1)"; }
r3fail() { say "R3_FAIL $*" >&2; exit 1; }
xsha() { awk -F'\t' -v n="$1" '$1 == n {print $2}' "$R3/expected.tsv"; }
xcol() { awk -F'\t' -v n="$1" -v c="$2" '$1 == n {print $c}' "$R3/expected.tsv"; }
bsha() { awk -F'\t' -v n="$1" '$1 == n {print $2}' "$R3/builds.tsv"; }
bstored() { awk -F'\t' -v n="$1" '$1 == n {print $3}' "$R3/builds.tsv"; }
namebysha() { awk -F'\t' -v s="$1" '$2 == s {print $1}' "$R3/expected.tsv" | head -n 1; }
# cf NAME: path of chain file NAME (see modes above); prints the path
cf() { local n=$1 e f d ps pn pp
  e=$(xsha "$n"); [ -n "$e" ] || r3fail "cf: $n not in expected.tsv"
  [ -f "$CH/$n.csv" ] && { echo "$CH/$n.csv"; return 0; }
  [ "$R3_PARENTS" = stored ] || r3fail "cf: $n not in $CH (chain mode: run the earlier piece first, or R3_PARENTS=stored)"
  f=$(xcol "$n" 3)
  if [ "$f" != "-" ] && [ "${f%.gz}" = "$f" ]; then
    [ "$(sha "$TFB_ROOT/$f")" = "$e" ] || r3fail "cf: stored $f sha differs from expected.tsv"
    echo "$TFB_ROOT/$f"; return 0
  fi
  if [ "$f" != "-" ]; then   # stored gz: gunzip into CH, content sha checked
    gzip -dc "$TFB_ROOT/$f" > "$CH/$n.csv.tmp" && mv "$CH/$n.csv.tmp" "$CH/$n.csv"
  else
    d=$(xcol "$n" 4); ps=$(xcol "$n" 5); pn=$(namebysha "$ps"); [ -n "$pn" ] || r3fail "cf: no parent for $n"
    pp=$(cf "$pn") || exit 1
    "$PY" "$TFB_ROOT/tools/probe.py" expand "$pp" "$TFB_ROOT/$d" "$CH/$n.csv" --sha "$e" >&2 || r3fail "cf: expand $n"
  fi
  [ "$(sha "$CH/$n.csv")" = "$e" ] || r3fail "cf: materialised $n sha differs"
  echo "$CH/$n.csv"; }
# check NAME FILE [what]: FILE sha == expected sha of NAME; records a line in RES
check() { local n=$1 f=$2 e s r; e=$(xsha "$n"); s=$(sha "$f")
  if [ "$s" = "$e" ]; then r=PASS-SHA; else r=FAIL; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${PIECE:-?}" "$n" "$r" "$s" "$e" "${3:-R3_PARENTS=$R3_PARENTS R3_BUILDS=$R3_BUILDS}" >> "$RES"
  say "PARITY $n $r ($s vs expected $e)"
  [ "$r" = PASS-SHA ] || [ "${R3_SOFT:-0}" = 1 ] || r3fail "parity $n"; }
# put NAME TMPFILE: move a written file into the chain folder as NAME (after check)
put() { check "$1" "$2" "${3:-}"; [ "$2" = "$CH/$1.csv" ] || mv -f "$2" "$CH/$1.csv"; }
# mkdelta NAME PARENTNAME: CH/d_NAME__PARENTNAME.delta.csv.gz (+ .json) = tools/probe.py make on the two chain files (cached)
mkdelta() { local n=$1 p=$2 d
  d=$CH/d_${n}__${p}.delta.csv.gz
  if [ -f "$d" ] && [ -f "${d%.delta.csv.gz}.json" ] && grep -q "$(xsha "$n")" "${d%.delta.csv.gz}.json" && grep -q "$(xsha "$p")" "${d%.delta.csv.gz}.json"; then echo "$d"; return 0; fi
  "$PY" "$TFB_ROOT/tools/probe.py" make "$(cf "$p")" "$(cf "$n")" "$d" >&2 || r3fail "probe make $n vs $p"
  echo "$d"; }
# disjoint OUTNAME NEWPARENT PIECE PIECEPARENT OTHER: tools/rebase/stack_disjoint.py (copy) NEWPARENT + delta(PIECE on PIECEPARENT),
# OTHER = NEWPARENT's own name (its delta on PIECEPARENT is the disjointness reference)
disjoint() { local o=$1 np=$2 pc=$3 pp=$4 dp dot
  waitmem 30 "disjoint $o" >&2; waitdisk 1350 "disjoint $o" >&2
  dp=$(mkdelta "$pc" "$pp"); dot=$(mkdelta "$np" "$pp")
  "$PY" "$R3/stack_disjoint.py" "$(cf "$np")" "$dp" "$dot" "$R3W/$o.csv" || r3fail "stack_disjoint $o"
  rm -f "$R3W/$o.delta.csv.gz" "$R3W/$o.json"
  put "$o" "$R3W/$o.csv"; }
# drop NAME...: remove chain files no later piece reads (only inside CH; never stored files)
drop() { [ "${R3_KEEP:-0}" = 1 ] && return 0; local n; for n in "$@"; do rm -f "$CH/$n.csv" "$CH"/d_"$n"__* "$CH"/d_*__"$n".*; done; }
# serve KEY: CPU serve of the 30 members of KEY into R3W/serve/KEY (parity vs the stored serve receipt); prints the folder
SERVES="FVFT_MAR:l_fvft/mar:it54000_w32:54000:validation FVFT_APR:l_fvft/apr:it54000_w32:54000:private
FV2_MAR:l_fvft/fv2/mar:it60000_w32:60000:validation FV2_APR:l_fvft/fv2/apr:it60000_w32:60000:private
FVI_MAR:l_fvinel/mar:it60000_w32:60000:validation FVI_APR:l_fvinel/apr:it60000_w32:60000:private
W10_MAR:l_fvinel/mar_w10:it60000_w32:60000:validation W10_APR:l_fvinel/apr_w10:it60000_w32:60000:private"
serve() { local k=$1 l d w it per out wd t0
  l=$(echo "$SERVES" | tr ' ' '\n' | grep "^$k:") || r3fail "serve key $k"
  IFS=: read -r _ d w it per <<< "$l"; out=$R3W/serve/$k
  if [ -f "$out/receipt.json" ] && grep -q '"exact": true' "$out/receipt.json"; then echo "$out"; return 0; fi
  wd=$("$PY" "$R3/wdir.py" "$d/w/$w") || r3fail "weights $d/w/$w"
  waitmem 30 "serve $k" >&2; t0=$(date +%s)
  "$PY" "$R3/serve.py" "$wd" "$it" "$per" "$out" "$CR/$d/serve/receipt.json" > "$R3W/serve_$k.log" 2>&1 || r3fail "serve $k: $(tail -n 1 "$R3W/serve_$k.log" | cut -c1-200)"
  say "$(grep SERVE_OK "$R3W/serve_$k.log") $(( $(date +%s) - t0 )) s" >&2
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${PIECE:-?}" "SERVE_$k" "PASS-ARRAYS" "30 members" "$d/serve/receipt.json" "$(( $(date +%s) - t0 )) s" >> "$RES"
  echo "$out"; }
# build BNAME TEMPLATE KEY...: state build (laptop lock) with the tune template rendered for the given serve keys; prints the parquet.
# R3_BUILDS=stored: the stored parquet (sha == builds.tsv).
build() { local b=$1 tpl=$2 out k sv t0 p; local -a args=()
  shift 2
  if [ "$R3_BUILDS" = stored ]; then p=$TFB_ROOT/$(bstored "$b"); [ "$(sha "$p")" = "$(bsha "$b")" ] || r3fail "stored build $b sha"; echo "$p"; return 0; fi
  out=$R3W/build/$b; mkdir -p "$out"
  if [ -f "$out/t1.csv.state.parquet" ] && [ "$(sha "$out/t1.csv.state.parquet")" = "$(bsha "$b")" ]; then echo "$out/t1.csv.state.parquet"; return 0; fi
  for k in "$@"; do sv=$(serve "$k") || exit 1; args+=("$k=$sv"); done
  "$PY" "$R3/tune.py" "$R3/tunes/$tpl" "$out/tune.json" ${args[@]+"${args[@]}"} >&2 || r3fail "tune $b"
  lock "build $b ~35 min" >&2; waitmem 40 "build $b" >&2; waitdisk 1350 "build $b" >&2; t0=$(date +%s)
  LM_THREADS=${LM_THREADS:-4} LM_RSS_GB=${LM_RSS_GB:-4.8} "$PY" "$R3/build.py" "$out/t1.csv" "$out/tune.json" > "$out/build.log" 2>&1 || { unlock; r3fail "build $b: $(grep -v '^\s*$' "$out/build.log" | tail -n 1 | cut -c1-200)"; }
  unlock
  p=$out/t1.csv.state.parquet; local s; s=$(sha "$p")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${PIECE:-?}" "$b" "$([ "$s" = "$(bsha "$b")" ] && echo PASS-SHA || echo FAIL)" "$s" "$(bsha "$b")" "build $(( $(date +%s) - t0 )) s; $(cat "$out/t1.csv.rss.txt" 2>/dev/null)" >> "$RES"
  say "BUILD $b $(( $(date +%s) - t0 )) s parquet $s (stored $(bsha "$b"))" >&2
  [ "$s" = "$(bsha "$b")" ] || [ "${R3_SOFT:-0}" = 1 ] || r3fail "build parity $b"
  echo "$p"; }
# splitstack OUTNAME BASENAME PARQUET SPLIT [BREFNAME]: src/r3lib/stack.py (isi_stack.py form) -> CH/OUTNAME.csv
splitstack() { local o=$1 b=$2 pq=$3 sp=$4 br=""
  waitmem 30 "stack $o"; waitdisk 1350 "stack $o"
  [ -n "${5:-}" ] && br=$(cf "$5")
  rm -f "$R3W/$o.csv" "$R3W/$o.info.json"
  "$PY" "$R3/stack.py" stack "$(cf "$b")" "$pq" none "$sp" "$R3W" "$o" ${br:+"$br"} > "$R3W/stack_$o.log" 2>&1 || r3fail "stack $o: $(tail -n 1 "$R3W/stack_$o.log" | cut -c1-300)"
  grep "R3_STACK check" "$R3W/stack_$o.log" | while read -r l; do say "$l"; done
  put "$o" "$R3W/$o.csv"; }
# takepriv OUTNAME C72NAME BOTHNAME: exp/L_fvft/fv2_apr_on_c72.py (copy): OUT = C72 with every private non-gap state row from BOTH
takepriv() { local o=$1; waitmem 30 "take $o"; waitdisk 1350 "take $o"
  "$PY" "$R3/take_private_rows.py" "$(cf "$2")" "$(cf "$3")" "$(cf C70)" "$R3W/$o.csv" > "$R3W/take_$o.log" 2>&1 || r3fail "take $o: $(tail -n 1 "$R3W/take_$o.log" | cut -c1-200)"
  put "$o" "$R3W/$o.csv"; }
