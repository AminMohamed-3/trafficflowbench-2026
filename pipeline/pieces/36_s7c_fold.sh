#!/usr/bin/env bash
# Slot 36, S7C-FOLD (exp/L_isinel/s7cf_run.sh): S7c trees refit on folds instead of the FV2-fit trees. NOT ADOPTED.
# Gate read 2026-10-21: S7CF_GATE KEEP-FV2FIT ("fold-fit not built, C80 keeps the FV2-fit trees"). C80 (row 35) is the final file.
# This placeholder only keeps the slot: run_28_35.sh calls it only with R3_WITH_36=1, and it stops with this explanation.
# If S7C-FOLD is ever adopted: build the fold-fit trees (weights row 36), render B_S7C_VAL / B_S7C_PRIV with the new tree paths,
# and repeat the 35 steps (C79 -> X -> C80) into C81.
set -euo pipefail
echo "slot 36 S7C-FOLD: not adopted (S7CF_GATE KEEP-FV2FIT, 2026-10-21); C80 from row 35 is final. Nothing to do." >&2
exit 3
