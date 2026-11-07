#!/bin/bash
# Parity runner, ADOPTED row 22 (C55 -> C57, L_dose A3 cold members). R2 piece worker, 2026-10-21.
#   bash parity/22_parity.sh [--rows-only]
# Full: stored C55 = gunzip artifacts/claude_ref/probes/c_tinyvm_on_c54.csv.gz (sha 5ebd5de0) into $WORK/parity; 22_a3cold.sh
# with the STORED equivalents of the base-build members and of piece 21 (env below; override to test R1's run dir / $WORK);
# lib/compare.py vs C57 (caaf4d8a, artifacts/claude_ref/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv.gz); tools/window_guard.py
# --no-record vs the parent (reference C44). About 6 min (chain ~45 s), peak RAM about 2.5 GiB, disk about 0.9 GB temporary.
# --rows-only (cheap variant, about 1 min, under 1.5 GiB): every step but the CSV; the rows must equal the stored
# artifacts/claude_ref/l_dose/a3/rebase/ongoing_bb_c55.parquet bytewise (sha ff1113a7). Laptop check 2026-10-21: the chain
# files cnn_ens2/sccnn_official/ongoing/ongoing_bb_c55 were bytewise equal to the stored ones (c4976682, f5955d66, 1a3fa6dd, ff1113a7).
# Result: parity/22.result (one JSON line). Deletes only its own $WORK/parity/22_* files.
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/ong_lib.sh"
CR=$OLD/artifacts/claude_ref; M=$CR/asm2/e16/s01234_5_6e-4_none_out/rows; SW=$CR/c_bb/vm_tiny_swap
export T22_BASE_ON=${T22_BASE_ON:-$M/on} T22_BASE_OFF=${T22_BASE_OFF:-$M/off} T21_MEMBER=${T21_MEMBER:-$SW/bb_member_vm.parquet} \
  T21_CFG=${T21_CFG:-$PIECES/src/21/cfg_vm.json} T21_ROWS=${T21_ROWS:-$SW/ongoing_bb_vm.parquet}
parity_begin 22
if [ "${1:-}" = --rows-only ]; then
  bash "$PIECES/22_a3cold.sh" - - || fail_result "piece script failed" rows-only
  rows_result 22 "$WORK/22_a3cold/ongoing_bb_c55.parquet" ff1113a7d08fc18e9b0fac70a0c3f55e2823bffdd46112cc1eed43fb6fe16870 \
    "$CR/l_dose/a3/rebase/ongoing_bb_c55.parquet"
else
  gunzip -c "$CR/probes/c_tinyvm_on_c54.csv.gz" > "$P/22_parent.csv"
  need_sha "$P/22_parent.csv" 5ebd5de0438517e03a719ecf88545acd26152287441e66265bb3d81081ff3020
  bash "$PIECES/22_a3cold.sh" "$P/22_parent.csv" "$P/22_out.csv" || fail_result "piece script failed"
  full_result 22 "$P/22_out.csv" caaf4d8a3740a27db5cb5237164b3850c69761830002649ef4497f06ed7999a5 \
    "$CR/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv.gz" "$P/22_parent.csv"
fi
