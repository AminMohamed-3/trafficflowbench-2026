#!/usr/bin/env bash
# ADOPTED row 34, FV-INEL W1.0 (INEL label weight 1.0): March rows (read on C75 = .90075; C77 = C76 + W1.0 March) and April rows by
# the declared method (C78). Worker R3, 2026-10-21.
# In:  C70, FVI_MAR_ON_C70 (row 31), C75 (row 32), C76 (row 33); weights l_fvinel/{mar_w10,apr_w10}/w/it60000_w32
#      (weights/rows/34.csv); serve FVI_MAR (row 31 weights); S2b arm48k (row 26).
# Steps (originals: FVI_VAR=w10 exp/L_fvinel/fvi_finish.sh; exp/L_fvinel/fvi_lw_on_c75.py form onto C75; stack_disjoint onto C76;
# exp/L_fvinel/fvi_apr_w10_chain.sh; exp/L_fvinel/aprw10_apply.sh):
#  1 serve W1.0 March, build B_W10_MAR (= fvi_tune_mar_w10.json), stack validation rows onto C70 -> W10_MAR_ON_C70 eda9f5ea
#  2 src/34/w10_on_c75.py C75, FVI_MAR_ON_C70, W10_MAR_ON_C70 -> W10_ON_C75 67d02edf (the file read on the board)
#  3 stack_disjoint C76 + delta(W10_ON_C75 on C75) -> C77 10abf5ee
#  4 serve W1.0 April, build B_W10_APR (= fvi_tune_apr_w10.json), stack private rows onto FVI_MAR_ON_C70 -> W10_BOTH_ON_C70 fd2c9e88
#  5 take_private_rows C77, W10_BOTH_ON_C70, C70 -> C78 f5b8e9e0
# Out: CH/C77.csv, CH/C78.csv. Runtime ~85 min.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=34; T0=$(date +%s)
P=$(build B_W10_MAR B_W10_MAR.json W10_MAR) || exit 1
splitstack W10_MAR_ON_C70 C70 "$P" validation
rm -f "$R3W"/W10_ON_C75.*
"$PY" "$D/src/34/w10_on_c75.py" "$(cf C75)" "$(cf FVI_MAR_ON_C70)" "$(cf W10_MAR_ON_C70)" "$R3W/W10_ON_C75.csv" > "$R3W/w10_on_c75.log" 2>&1 || r3fail "w10 on C75: $(tail -n 1 "$R3W/w10_on_c75.log" | cut -c1-200)"
rm -f "$R3W/W10_ON_C75.delta.csv.gz" "$R3W/W10_ON_C75.json"
put W10_ON_C75 "$R3W/W10_ON_C75.csv"
disjoint C77 C76 W10_ON_C75 C75
P=$(build B_W10_APR B_W10_APR.json FVI_MAR W10_APR) || exit 1
splitstack W10_BOTH_ON_C70 FVI_MAR_ON_C70 "$P" private
takepriv C78 C77 W10_BOTH_ON_C70
drop W10_MAR_ON_C70 W10_ON_C75 C75 C76 W10_BOTH_ON_C70 FVI_MAR_ON_C70
say "PIECE_34_DONE C77 C78 $(( $(date +%s) - T0 )) s"
