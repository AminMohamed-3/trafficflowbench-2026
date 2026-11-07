#!/bin/bash
# Parity runner, ADOPTED row 21 (C54 -> C55, ConvNeXt-Tiny VM member). R2 piece worker, 2026-10-21.
#   bash parity/21_parity.sh [--rows-only]
# Full: stored C54 = tools/probe.py expand(C50 c_gap2w_on_c47.csv.gz 9ddd8848, l_micro/c_lmicro2_on_c50.delta.csv.gz) with
# sha b0cbce4e, into $WORK/parity; 21_tinyvm.sh with the STORED base-build equivalents (env below, override to test R1's
# run dir); lib/compare.py vs C55 (5ebd5de0, artifacts/claude_ref/probes/c_tinyvm_on_c54.csv.gz); tools/window_guard.py
# --no-record vs the parent (reference C44). About 7 min, peak RAM about 2.5 GiB, disk about 1.3 GB temporary.
# --rows-only (cheap variant, about 30 s, under 1 GiB): every step but the CSV; the ongoing rows must equal the stored
# C55 rows artifacts/claude_ref/c_bb/vm_tiny_swap/ongoing_bb_vm.parquet bytewise (sha a13fff47).
# Result: parity/21.result (one JSON line). Deletes only its own $WORK/parity/21_* files.
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/ong_lib.sh"
S=$OLD/artifacts/claude_ref/c_cold/official/ship_c35/a0.75
export T21_BASE_MIX=${T21_BASE_MIX:-$S} T21_BASE_ONGOING=${T21_BASE_ONGOING:-$S/ongoing.parquet} T21_BASE_ONGOING_BB=${T21_BASE_ONGOING_BB:-$S/ongoing_bb.parquet}
parity_begin 21
if [ "${1:-}" = --rows-only ]; then
  bash "$PIECES/21_tinyvm.sh" - - || fail_result "piece script failed" rows-only
  rows_result 21 "$WORK/21_tinyvm/ongoing_bb.parquet" a13fff47130c9645175ab9de8e707f1b43167d535171260add685b79504c6dc4 \
    "$OLD/artifacts/claude_ref/c_bb/vm_tiny_swap/ongoing_bb_vm.parquet"
else
  gunzip -c "$OLD/artifacts/claude_ref/probes/c_gap2w_on_c47.csv.gz" > "$P/21_c50.csv"
  need_sha "$P/21_c50.csv" 9ddd8848c235627c665b2a6732043485f7e2c14e3bbb45f09b6bb509f1d75bb4
  "$PY" "$PIECES/src/21/probe.py" expand "$P/21_c50.csv" "$OLD/artifacts/claude_ref/l_micro/c_lmicro2_on_c50.delta.csv.gz" \
    "$P/21_parent.csv" --sha b0cbce4e1352a38c41110225f9d04794c616d0a91c8b743ffadc895b722b729a || fail_result "parent expand failed"
  rm -f "$P/21_c50.csv"
  bash "$PIECES/21_tinyvm.sh" "$P/21_parent.csv" "$P/21_out.csv" || fail_result "piece script failed"
  full_result 21 "$P/21_out.csv" 5ebd5de0438517e03a719ecf88545acd26152287441e66265bb3d81081ff3020 \
    "$OLD/artifacts/claude_ref/probes/c_tinyvm_on_c54.csv.gz" "$P/21_parent.csv"
fi
