#!/usr/bin/env bash
# ADOPTED row 28, FV-FT March (C70 = .90037). Worker R3, 2026-10-21. Usage: bash 28_fvft_mar.sh   (chain folder: lib/r3.sh)
# In:  C66 (row 26), C69 (row 27); weights 30 FV-FT March B6R members l_fvft/mar/w/it54000_w32 (weights/rows/28.csv); S2b arm48k
#      served members of row 26 ($R3_S2B_SERVE, default the stored artifacts/claude_ref/l_b6/s2b/arm48k); S7b C models (row 26).
# Steps (original: exp/L_fvft/mar_finish.sh + C's stack onto C69, exp/L_fvft/mar_logs/stack_c69.log):
#  1 serve the 30 members on CPU (src/r3lib/serve.py = mar_serve.members), arrays == stored serve receipt
#  2 state build (src/r3lib/build.py = mar_build.py body) with tune B_FVFT_MAR (= exp/L_fvft/mar_tune.json, arm -> step 1)
#  3 stack ONLY the validation non-gap state rows onto C66 (src/r3lib/stack.py = mar_stack.py writing loop) -> FVFT_MAR_ON_C66 d5167911
#  4 tools/rebase/stack_disjoint.py: C69 + delta(FVFT_MAR_ON_C66 on C66), disjoint from C69's gap rows -> C70 5181d5ed
# Out: CH/C70.csv (+ FVFT_MAR_ON_C66 dropped). Runtime ~45 min (serve 3, build 33, stacks 6).
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=28; T0=$(date +%s)
P=$(build B_FVFT_MAR B_FVFT_MAR.json FVFT_MAR) || exit 1
splitstack FVFT_MAR_ON_C66 C66 "$P" validation
disjoint C70 C69 FVFT_MAR_ON_C66 C66
drop FVFT_MAR_ON_C66
say "PIECE_28_DONE C70 $(( $(date +%s) - T0 )) s"
