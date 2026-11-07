#!/usr/bin/env bash
# ADOPTED row 29, BUNDLE19 (C71 = .90044): L_gap3 EDGE-TAPER + UNSMOOTH gap rows + April FV-FT non-gap state rows. Worker R3, 2026-10-21.
# In:  C66, C69, C70 (row 28); weights 30 FV-FT April members l_fvft/apr/w/it54000_w32 (weights/rows/29.csv); gap3 knobs
#      l_gap3/knobs_march.json; row 26/27 fitted files read by the gap3 recipe (l_gap2 hook/members/v6r/weights, GAPW cfg,
#      bundle18 t1_gapwf parquet; R2's manifest rows).
# Steps (originals: exp/L_fvft/apr_finish.sh; exp/L_gap3/g3_official.sh; stacks by C, exp/L_gap3/stack_c70.log and l_bundle19 json):
#  1 April FV-FT: serve (arrays == stored receipt), build B_FVFT_APR (= exp/L_fvft/apr_tune.json), stack ONLY the private non-gap
#    state rows onto C70 (check b against C66: the March arm of that build is the S2b arm) -> FVFT_BOTH_ON_C69 6f7ce2b6
#  2 gap3 official gap rows over C69 (src/29/g3_official.py, parities p1, p2, pC69 EXACT inside) -> GAP3_ON_C69 819b26d1 (content)
#  3 stack_disjoint: C70 + delta(GAP3_ON_C69 on C69) -> GAP3_ON_C70 26bd6344
#  4 stack_disjoint: GAP3_ON_C70 + delta(FVFT_BOTH_ON_C69 on C70) -> C71 45c9cd41
# Out: CH/C71.csv. Runtime ~55 min (serve 3, build 33, gap3 ~8, stacks ~10).
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=29; T0=$(date +%s)
P=$(build B_FVFT_APR B_FVFT_APR.json FVFT_APR) || exit 1
splitstack FVFT_BOTH_ON_C69 C70 "$P" private C66
G=$R3W/gap3; mkdir -p "$G"; rm -f "$G"/c_gap3_on_c69.*
lock "gap3 official ~10 min"; waitmem 40 gap3; waitdisk 1350 gap3
G3_OUT=$G G3_PARENT=$(cf C69) "$PY" -B "$D/src/29/g3_official.py" c_gap3_on_c69 > "$G/g3_official.log" 2>&1 || { unlock; r3fail "gap3: $(grep -v '^\s*$' "$G/g3_official.log" | tail -n 1 | cut -c1-200)"; }
unlock
say "$(grep GAP3OFF_PARITY "$G/g3_official.log")"
gzip -dc "$G/c_gap3_on_c69.csv.gz" > "$R3W/GAP3_ON_C69.csv"; rm -f "$G/c_gap3_on_c69.csv.gz"
put GAP3_ON_C69 "$R3W/GAP3_ON_C69.csv"
disjoint GAP3_ON_C70 C70 GAP3_ON_C69 C69
disjoint C71 GAP3_ON_C70 FVFT_BOTH_ON_C69 C70
drop FVFT_BOTH_ON_C69 GAP3_ON_C69 GAP3_ON_C70
say "PIECE_29_DONE C71 $(( $(date +%s) - T0 )) s"
