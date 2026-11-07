#!/usr/bin/env bash
# ADOPTED row 30, FV2 (+6k more FV-FT iterations): March rows on C71 (C72 = .90046) and April rows by the declared method (C72P).
# Worker R3, 2026-10-21. NOTE: C72 and C72P are NOT in the C80 lineage (C73 = C71 + FV-INEL March); this piece is rebuilt because
# FV2_BOTH_ON_C70 (its April rows) is read by row 31 (C73P), and the FV2 April serve + build are read by row 32 (April gap rows).
# In:  C70, C71; weights l_fvft/fv2/{mar,apr}/w/it60000_w32 (weights/rows/30.csv); S2b arm48k (row 26).
# Steps (originals: exp/L_fvft/fv2_finish.sh validation|private, stack_b19 by C, exp/L_fvft/fv2_apr_on_c72.py):
#  1 serve FV2 March, build B_FV2_MAR (= fv2_tune_mar.json), stack validation non-gap rows onto C70 -> FV2_MAR_ON_C70 b994683a
#  2 stack_disjoint C71 + delta(FV2_MAR_ON_C70 on C70) -> C72 4315b61c
#  3 serve FV2 April, build B_FV2_APR (= fv2_tune_apr.json), stack private non-gap rows onto FV2_MAR_ON_C70 -> FV2_BOTH_ON_C70 042b3bce
#  4 fv2_apr_on_c72.py (copy src/r3lib/take_private_rows.py) C72, FV2_BOTH_ON_C70, C70 -> C72P d046c382
# Out: CH/FV2_BOTH_ON_C70.csv (kept for row 31). Runtime ~80 min (2 serves, 2 builds, 3 stacks).
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=30; T0=$(date +%s)
P=$(build B_FV2_MAR B_FV2_MAR.json FV2_MAR) || exit 1
splitstack FV2_MAR_ON_C70 C70 "$P" validation
disjoint C72 C71 FV2_MAR_ON_C70 C70
P=$(build B_FV2_APR B_FV2_APR.json FV2_MAR FV2_APR) || exit 1
splitstack FV2_BOTH_ON_C70 FV2_MAR_ON_C70 "$P" private
takepriv C72P C72 FV2_BOTH_ON_C70
drop FV2_MAR_ON_C70 C72 C72P
say "PIECE_30_DONE C72 C72P $(( $(date +%s) - T0 )) s"
