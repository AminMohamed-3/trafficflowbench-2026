#!/usr/bin/env bash
# ADOPTED row 35, S7C-INEL (+60 continued trees on the S7b C model per month, fitted on the INEL cells of the FV2 chain; the fitted
# trees are saved): March rows on C77 (C79 = .90088, BEST READ), April rows on C78; C80 = C79 + April W1.0 + S7c April.
# Worker R3, 2026-10-21.
# In:  C77, C78 (row 34); trees l_isinel/s7c/{yq,yv}_{mar,apr}_all.txt (weights/rows/35.csv); serves W10_MAR, W10_APR (row 34
#      weights), FVI_APR (row 31 weights); S7b C models (row 26).
# Steps (originals: exp/L_isinel/isi_cand_c77.sh; C80 by C with tools/rebase/stack_disjoint.py, jsons in l_isinel/cand/c80):
#  1 build B_S7C_VAL (= exp/L_isinel/tunes/cand_s7c_validation_c77.json), stack validation rows onto C77 -> C79 104c2c6f
#  2 build B_S7C_PRIV (= tunes/cand_s7c_private_c77.json), stack private rows onto C78 -> S7C_APR_ON_C78 70aa5aab
#  3 stack_disjoint C79 + delta(C78 on C77) -> X_C79_APRW10 4a124cfc
#  4 stack_disjoint X_C79_APRW10 + delta(S7C_APR_ON_C78 on C78) -> C80 3e27d66c
# Out: CH/C80.csv (final file of rows 28-35; slot 36 = S7C-FOLD if adopted). Runtime ~80 min.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=35; T0=$(date +%s)
"$PY" "$D/lib/weights.py" check 35 || r3fail "S7c trees (weights/rows/35.csv)"
P=$(build B_S7C_VAL B_S7C_VAL.json W10_MAR FVI_APR) || exit 1
splitstack C79 C77 "$P" validation
P=$(build B_S7C_PRIV B_S7C_PRIV.json W10_MAR W10_APR) || exit 1
splitstack S7C_APR_ON_C78 C78 "$P" private
disjoint X_C79_APRW10 C79 C78 C77
disjoint C80 X_C79_APRW10 S7C_APR_ON_C78 C78
drop C77 C78 S7C_APR_ON_C78 X_C79_APRW10
say "PIECE_35_DONE C80 $(( $(date +%s) - T0 )) s"
