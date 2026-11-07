#!/usr/bin/env bash
# ADOPTED row 31, FV-INEL (FV2 recipe + released-ineligible cells as labels, w .3): March rows (C73 = .90071), C73P = C73 + FV2
# April rows, April rows by the declared method (C74P). Worker R3, 2026-10-21.
# In:  C70, C71, FV2_BOTH_ON_C70 (row 30); weights l_fvinel/{mar,apr}/w/it60000_w32 (weights/rows/31.csv); S2b arm48k (row 26).
# Steps (originals: exp/L_fvinel/fvi_finish.sh, stack_b19 by C, fv2_apr_on_c72.py for C73P (logs/c73p.log), fvi_apr_chain.sh, then
# fv2_apr_on_c72.py onto C73P):
#  1 serve FV-INEL March, build B_FVI_MAR (= fvi_tune_mar.json), stack validation non-gap rows onto C70 -> FVI_MAR_ON_C70 01bff667
#  2 stack_disjoint C71 + delta(FVI_MAR_ON_C70 on C70) -> C73 219c0f93
#  3 take_private_rows C73, FV2_BOTH_ON_C70, C70 -> C73P 5827d508
#  4 serve FV-INEL April, build B_FVI_APR (= fvi_tune_apr.json), stack private rows onto FVI_MAR_ON_C70 -> FVI_BOTH_ON_C70 8dd8dafa
#  5 take_private_rows C73P, FVI_BOTH_ON_C70, C70 -> C74P d4376e1d
# Out: CH/C73P.csv, CH/C74P.csv (row 32), CH/FVI_MAR_ON_C70.csv (row 34). Runtime ~80 min.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=31; T0=$(date +%s)
P=$(build B_FVI_MAR B_FVI_MAR.json FVI_MAR) || exit 1
splitstack FVI_MAR_ON_C70 C70 "$P" validation
disjoint C73 C71 FVI_MAR_ON_C70 C70
takepriv C73P C73 FV2_BOTH_ON_C70
P=$(build B_FVI_APR B_FVI_APR.json FVI_MAR FVI_APR) || exit 1
splitstack FVI_BOTH_ON_C70 FVI_MAR_ON_C70 "$P" private
takepriv C74P C73P FVI_BOTH_ON_C70
drop C71 C73 FV2_BOTH_ON_C70 FVI_BOTH_ON_C70
say "PIECE_31_DONE C73 C73P C74P $(( $(date +%s) - T0 )) s"
