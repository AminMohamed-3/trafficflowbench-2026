#!/usr/bin/env bash
# ADOPTED row 32, GAP-ON-FV (C75 = .90073): official gap rows rebuilt with the current FV chain member (W3 + F1), L_gap3 form, no
# refit; March over FV-INEL (W .3), April over FV2. Worker R3, 2026-10-21.
# In:  C73P, C74P (row 31); serves FVI_MAR, FV2_APR (rows 30/31 weights); build parquets B_FVI_MAR, B_FV2_APR (build parity refs);
#      row 26/27 fitted files (l_gap2 S2b hook, members, v6r, weights; GAPW cfg; t1_gapwf) and l_gap3/knobs_march.json (row 29).
# Steps (original exp/L_gapfv/gfv_c73p.sh; C75 by C with stack_disjoint):
#  1 hook build, one panel per process (src/32/gfv73_build.py, tune B_GAPFV_HOOK = exp/L_gapfv/tune_gapfv_c73p.json), the 8 panels
#    with gap rows; R3_BUILDS=stored: the 8 stored per-panel hook files (l_gapfv/c73p/build) are copied instead
#  2 merge + build parity at the gap rows (src/32/gfv73_parity.py): hook old arm == FV-INEL March / FV2 April build gap rows EXACT
#  3 official gap rows over C73P (src/32/gfv73_official.py, parities p1/p2/pP inside) -> GAPFV_ON_C73P 7b936ed5
#  4 stack_disjoint C74P + delta(GAPFV_ON_C73P on C73P) -> C75 15280925
# Out: CH/C75.csv. Runtime ~70 min (hook builds ~55, official ~8, stack ~5).
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=32; T0=$(date +%s)
G=$R3W/gapfv; BD=$G/build; mkdir -p "$BD"
PANELS="D7_I405_N D7_I210_E D12_I5_N D7_I10_E D7_I405_S D7_I10_W D12_I5_S D7_I210_W"
if [ "$R3_BUILDS" = stored ]; then
  for pn in $PANELS; do cp "$CR/l_gapfv/c73p/build/t1.csv.$pn.gaprows.npz" "$BD/"; done
  RV=$(build B_FVI_MAR x) || exit 1; RP=$(build B_FV2_APR x) || exit 1
else
  RV=$(build B_FVI_MAR B_FVI_MAR.json FVI_MAR) || exit 1; RP=$(build B_FV2_APR B_FV2_APR.json FV2_MAR FV2_APR) || exit 1
  SV=$(serve FVI_MAR) || exit 1; SP=$(serve FV2_APR) || exit 1
  "$PY" "$R3/tune.py" "$R3/tunes/B_GAPFV_HOOK.json" "$G/tune.json" FVI_MAR="$SV" FV2_APR="$SP" >&2 || r3fail tune
  lock "gapfv hook builds ~55 min"
  for pn in $PANELS; do
    [ -f "$BD/t1.csv.$pn.gaprows.npz" ] && continue
    waitmem 36 "hook $pn"; t1=$(date +%s)
    LM_THREADS=4 LM_RSS_GB=3.5 "$PY" -B "$D/src/32/gfv73_build.py" "$BD/t1.csv" "$G/tune.json" --panels $pn > "$G/build_$pn.log" 2>&1 || { unlock; r3fail "hook $pn: $(grep -v '^\s*$' "$G/build_$pn.log" | tail -n 1 | cut -c1-200)"; }
    say "hook $pn $(( $(date +%s) - t1 )) s $(cat "$BD/t1.csv.rss.txt" 2>/dev/null || true)"
  done
  unlock
fi
GFV_BUILD_DIR=$BD GFV_REF_VAL=$RV GFV_REF_PRIV=$RP GFV_PARITY_JSON=$G/build_parity.json "$PY" -B "$D/src/32/gfv73_parity.py" > "$G/parity.log" 2>&1 || r3fail "hook parity: $(tail -n 2 "$G/parity.log" | cut -c1-300)"
say "$(grep 'GFV73_BUILD_PARITY {' "$G/parity.log" | cut -c1-300)"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" 32 GAPFV_HOOK PASS-EXACT "hook old arm == FV build gap rows" "$G/build_parity.json" "R3_BUILDS=$R3_BUILDS" >> "$RES"
lock "gapfv official ~8 min"; waitmem 36 official; waitdisk 1350 official; rm -f "$G"/c_gapfv_on_c73p.*
GFV_OUT=$G GFV_PARENT=$(cf C73P) GFV73_NEW_HOOK=$BD/t1.csv.gaprows.npz "$PY" -B "$D/src/32/gfv73_official.py" c_gapfv_on_c73p > "$G/official.log" 2>&1 || { unlock; r3fail "official: $(grep -v '^\s*$' "$G/official.log" | tail -n 1 | cut -c1-300)"; }
unlock
say "$(grep GFVOFF_PARITY "$G/official.log" | cut -c1-300)"
rm -f "$G/c_gapfv_on_c73p.delta.csv.gz" "$G/c_gapfv_on_c73p.json"
put GAPFV_ON_C73P "$G/c_gapfv_on_c73p.csv"
disjoint C75 C74P GAPFV_ON_C73P C73P
drop GAPFV_ON_C73P C73P C74P
say "PIECE_32_DONE C75 $(( $(date +%s) - T0 )) s"
