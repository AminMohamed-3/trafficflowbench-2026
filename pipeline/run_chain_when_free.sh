#!/usr/bin/env bash
# run_chain_when_free.sh: self-finishing launcher of the chained rows 19-35 on the fresh base (worker R4, 2026-10-21).
#   1. waits until the parity runs of the piece workers are over:
#        R2: pipeline/pieces/parity/logs/RUN_19_27_DONE exists;
#        R3: r3_parity.sh writes no done marker, so: no running bash whose script is r3_parity.sh, run_19_27.sh,
#            run_28_35.sh or NN_parity.sh (checked from /proc/<pid>/cmdline, Git Bash / MSYS processes),
#        and the laptop lock (exp/L_b6/logs/laptop.lock.d) is free or stale; both seen on two polls 60 s apart;
#   2. runs pipeline/20_pieces.sh (outputs/C80.csv, per-step lines in outputs/logs/chain_19_35.tsv);
#   3. pipeline/99_verify.py outputs/C80.csv against the stored C80 (sha 3e27d66c..., state atol 1e-8);
#   4. writes outputs/logs/chain_19_35.result: one line starting RESULT (chain done and VERIFY_PASS) or FAILED.
# Usage: bash pipeline/run_chain_when_free.sh [--no-wait] [--dry-run] [--max-wait-h H] [-- ARGS for 20_pieces.sh]
#   --no-wait     skip step 1 (20_pieces.sh still takes the laptop lock per piece)
#   --dry-run     print the wait state and 20_pieces.sh --dry-run; no lock, no build, no result file
#   --max-wait-h  give up waiting after H hours (default 30): FAILED line
# Log: outputs/logs/run_chain_when_free.log (appended). Launch from Git Bash, detached:
#   nohup bash "pipeline/run_chain_when_free.sh" > /dev/null 2>&1 &
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); OLD=$(cd "$FS/.." && pwd)
PY=${PY:-$OLD/.venv_tfb/Scripts/python.exe}
LOGD=$FS/outputs/logs; LOG=$LOGD/run_chain_when_free.log; RESULT=$LOGD/chain_19_35.result
LOCK=${LOCK:-$OLD/exp/L_b6/logs/laptop.lock.d}
DONE_R2=$FS/pipeline/pieces/parity/logs/RUN_19_27_DONE
REF_C80=${CHAIN_REF_C80:-$OLD/artifacts/claude_ref/l_isinel/cand/c80/c_s7c_both_aprw10_c80.csv}
SHA_C80=3e27d66cd2b3cfc4371fd218617ebf08a0fa16e481a7bf7108ea2942cd9ef650
WAIT=1; DRY=0; MAXH=30; ARGS=()
while [ $# -gt 0 ]; do case $1 in
  --no-wait) WAIT=0; shift;; --dry-run) DRY=1; shift;; --max-wait-h) MAXH=$2; shift 2;;
  --) shift; ARGS=("$@"); break;; *) echo "unknown argument $1 (see header)"; exit 2;; esac; done
mkdir -p "$LOGD"
[ $DRY = 1 ] || exec >> "$LOG" 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
m() { cygpath -m "$1"; }

busy() {    # running parity scripts of the piece workers (bash <script> ...), one "pid script" per line
  local p a
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = $$ ] && continue
    mapfile -d '' -t a < "$p/cmdline" 2>/dev/null || continue
    case "${a[0]:-}" in *bash|*bash.exe) ;; *) continue;; esac
    case "${a[1]:-}" in */r3_parity.sh|r3_parity.sh|*/run_19_27.sh|run_19_27.sh|*/run_28_35.sh|run_28_35.sh|*/[0-9][0-9]_parity.sh|[0-9][0-9]_parity.sh)
      echo "${p#/proc/} ${a[1]##*/} ${a[2]:-}";; esac
  done; }
lockstate() { local op
  [ -d "$LOCK" ] || { echo free; return; }
  op=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then echo "held pid $op ($(cat "$LOCK/owner" 2>/dev/null))"; else echo "stale (pid ${op:-?})"; fi; }
ready() { [ -f "$DONE_R2" ] && [ -z "$(busy)" ] && case $(lockstate) in free|stale*) true;; *) false;; esac; }
state() { echo "R2 done marker: $([ -f "$DONE_R2" ] && cat "$DONE_R2" || echo missing); parity scripts running: $(busy | awk '{printf "%s %s %s; ", $1, $2, $3}')lock: $(lockstate)"; }
fail() { say "FAILED $*"; echo "FAILED $(date -u +%FT%TZ) chain_19_35: $*" > "$RESULT"; exit 1; }

if [ $DRY = 1 ]; then
  echo "run_chain_when_free.sh DRY RUN"
  echo "  wait state: $(state)"; echo "  ready now: $(ready && echo yes || echo no)"
  echo "  stored C80 reference: $REF_C80 $([ -f "$REF_C80" ] && echo '(present)' || echo '(MISSING)')"
  echo "  then: bash pipeline/20_pieces.sh ${ARGS[*]:-}; python pipeline/99_verify.py outputs/C80.csv --ref <stored C80> --ref-sha ${SHA_C80:0:8} --state-atol 1e-8"
  echo "  result line -> $RESULT"
  bash "$FS/pipeline/20_pieces.sh" --dry-run ${ARGS[@]+"${ARGS[@]}"}; exit $?
fi

[ -f "$RESULT" ] && mv -f "$RESULT" "$RESULT.prev"
say "launcher start (pid $$): FS $FS"
[ -f "$REF_C80" ] || fail "stored C80 reference missing: $REF_C80"
if [ $WAIT = 1 ]; then
  T0=$SECONDS; k=0; streak=0
  while :; do
    if ready; then streak=$((streak + 1)); else streak=0; fi
    [ $streak -ge 2 ] && break
    [ $((k % 10)) -eq 0 ] && say "waiting: $(state)"
    [ $((SECONDS - T0)) -lt $((MAXH * 3600)) ] || fail "still waiting after $MAXH h: $(state)"
    k=$((k + 1)); sleep 60
  done
  say "free: $(state); R3 parity tail: $(tail -n 2 "$FS/pipeline/pieces/parity/r3_results.tsv" 2>/dev/null | cut -f2-4 | tr '\n' ';')"
fi

T1=$SECONDS
say "start: bash pipeline/20_pieces.sh ${ARGS[*]:-}"
bash "$FS/pipeline/20_pieces.sh" ${ARGS[@]+"${ARGS[@]}"}; rc=$?
CH_S=$(awk -F'\t' '$3 == "summary" {d = $9} END {print d}' "$LOGD/chain_19_35.tsv" 2>/dev/null)
MIN=$(( (SECONDS - T1) / 60 ))
[ $rc -eq 0 ] || fail "20_pieces.sh exit $rc after $MIN min; last step: $(tail -n 1 "$LOGD/chain_19_35.tsv" 2>/dev/null | cut -f2-5,9 | tr '\t' ' ' | cut -c1-300)"
[ -f "$FS/outputs/C80.csv" ] || fail "outputs/C80.csv missing after the chain ($MIN min)"
say "verify outputs/C80.csv against $REF_C80"
"$PY" "$FS/pipeline/99_verify.py" "$(m "$FS/outputs/C80.csv")" --ref "$(m "$REF_C80")" --ref-sha $SHA_C80 --state-atol 1e-8 \
  --json "$(m "$LOGD/chain_c80_verify.json")" > "$LOGD/chain_c80_verify.txt" 2>&1
V=$(tail -n 1 "$LOGD/chain_c80_verify.txt")
cat "$LOGD/chain_c80_verify.txt"
case $V in
  VERIFY_PASS*) echo "RESULT $(date -u +%FT%TZ) chain_19_35: chain $MIN min; steps: $CH_S; C80 $V" | tee "$RESULT";;
  *) fail "chain $MIN min; steps: $CH_S; C80 $(echo "$V" | cut -c1-400)";;
esac
