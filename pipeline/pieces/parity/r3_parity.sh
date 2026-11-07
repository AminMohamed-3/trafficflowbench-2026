#!/usr/bin/env bash
# Parity runner of rows 28 to 35 (worker R3, 2026-10-21). Results: parity/r3_results.tsv (one line per checked file), log on stdout.
#   bash parity/r3_parity.sh glue [FROM TO]   stored parents + STORED build parquets: every stack / transfer step, the gap3 and
#                                             GAP-ON-FV official steps (real recompute from the fitted files) and Q-ACCUM (real build
#                                             from the stored kernel outputs) must give the stored file sha. ~2 h, no state build.
#   bash parity/r3_parity.sh serve [KEY..]    serve the members of KEY (default all 8 sets) on CPU; arrays == stored receipts; the
#                                             served folders are removed after the check. ~3 min per set.
#   bash parity/r3_parity.sh build BNAME      one state build from saved weights (serve + build, laptop lock) == stored parquet sha.
#                                             ~40 min. BNAME in src/r3lib/builds.tsv (B_FVFT_MAR ... B_S7C_PRIV).
#   bash parity/r3_parity.sh full             = run_28_35.sh with the stored C66 / C69: the whole chain from weights (~8.5 h).
# All files go to $WORK/r3parity (deleted at the end unless R3_KEEP=1); nothing is written outside final_solution/.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd -W); source "$D/lib/common.sh"
export CH=$WORK/r3parity/chain R3W=$WORK/r3parity/w
source "$D/lib/r3.sh"
mode=${1:-glue}; shift || true
T0=$(date +%s)
case $mode in
  glue)
    export R3_PARENTS=stored R3_BUILDS=stored
    F=${1:-28}; T=${2:-35}
    for p in 28_fvft_mar 29_bundle19 30_fv2 31_fvinel 32_gapfv 33_qaccum 34_fvinel_w10 35_s7c_inel; do
      n=${p%%_*}; [ "$n" -ge "$F" ] && [ "$n" -le "$T" ] || continue
      t1=$(date +%s); bash "$D/$p.sh" || r3fail "glue parity $p"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$n" "PIECE_$n" "GLUE-DONE" "-" "-" "$(( $(date +%s) - t1 )) s (stored parents + stored builds)" >> "$RES"
    done;;
  serve)
    K=${*:-FVFT_MAR FVFT_APR FV2_MAR FV2_APR FVI_MAR FVI_APR W10_MAR W10_APR}
    lock "serve parity ~3 min per set"; for k in $K; do PIECE=serve serve "$k" > /dev/null || { unlock; exit 1; }; rm -rf "$R3W/serve/$k"; done; unlock;;
  build)
    export R3_BUILDS=run; PIECE=build
    b=$1; t=$(awk -F'\t' -v b="$b" '$1 == b {print $4}' "$R3/tunes/SOURCES.tsv" | tr ',' '\n' | cut -d= -f2 | grep -v '^S2B$' | sort -u | tr '\n' ' ')
    build "$b" "$b.json" $t > /dev/null
    [ "${R3_KEEP:-0}" = 1 ] || rm -rf "$R3W/build/$b" "$R3W/serve";;
  full)
    export R3_PARENTS=chain R3_BUILDS=run; bash "$D/run_28_35.sh";;
  *) echo "usage: r3_parity.sh glue|serve|build|full"; exit 2;;
esac
[ "${R3_KEEP:-0}" = 1 ] || { rm -rf "$WORK/r3parity/chain" "$WORK/r3parity/w/build" "$WORK/r3parity/w/serve" "$WORK/r3parity/w/gapfv" "$WORK/r3parity/w/qaccum" "$WORK/r3parity/w/gap3"; rm -f "$WORK"/r3parity/w/*.csv; }
say "R3_PARITY_$mode done in $(( $(date +%s) - T0 )) s; results in $RES"
