#!/bin/bash
# Parity runner, ADOPTED row 23 (C57 -> C59, L_tinyad A1). R2 piece worker, 2026-10-21.
#   bash parity/23_parity.sh [--rows-only]
# Full: stored C57 = gunzip artifacts/claude_ref/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv.gz (sha caaf4d8a) into
# $WORK/parity; 23_tinyad.sh (member merged from the stored per-date adapted predictions) with the STORED equivalents of
# pieces 21 and 22 (env below; override to use $WORK); lib/compare.py vs C59 (c9a635fd,
# artifacts/claude_ref/l_tinyad/a1/on_c57/delta/c_tinyad_on_c57.csv.gz); tools/window_guard.py --no-record vs the parent
# (reference C44). About 5 min, peak RAM about 2.5 GiB, disk about 0.9 GB temporary.
# --rows-only (cheap variant, about 30 s, under 1 GiB): merge + decode; the rows must equal the stored
# artifacts/claude_ref/l_tinyad/a1/on_c57/rows_c57.parquet bytewise (sha 94f1c536; laptop check 2026-10-21: equal, and the
# merged member equal to tinyad_member.parquet a7c7c2a9 bytewise).
# Result: parity/23.result (one JSON line). Deletes only its own $WORK/parity/23_* files.
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/ong_lib.sh"
CR=$OLD/artifacts/claude_ref
export T21_MEMBER=${T21_MEMBER:-$CR/c_bb/vm_tiny_swap/bb_member_vm.parquet} T22_CHAIN=${T22_CHAIN:-$CR/l_dose/a3/a3_build/a0.75} \
  T22_ROWS=${T22_ROWS:-$CR/l_dose/a3/rebase/ongoing_bb_c55.parquet}
parity_begin 23
if [ "${1:-}" = --rows-only ]; then
  bash "$PIECES/23_tinyad.sh" - - || fail_result "piece script failed" rows-only
  rows_result 23 "$WORK/23_tinyad/ongoing_bb_c59.parquet" 94f1c53669db68fb20c705d7c71200277b783f9298346c2ef8097f3b988a1996 \
    "$CR/l_tinyad/a1/on_c57/rows_c57.parquet"
else
  gunzip -c "$CR/l_dose/a3/rebase/delta/c_a3xsd_on_c55.csv.gz" > "$P/23_parent.csv"
  need_sha "$P/23_parent.csv" caaf4d8a3740a27db5cb5237164b3850c69761830002649ef4497f06ed7999a5
  bash "$PIECES/23_tinyad.sh" "$P/23_parent.csv" "$P/23_out.csv" || fail_result "piece script failed"
  full_result 23 "$P/23_out.csv" c9a635fd35917ec279b09d8c034792051896592676848850eb670d89fd72019f \
    "$CR/l_tinyad/a1/on_c57/delta/c_tinyad_on_c57.csv.gz" "$P/23_parent.csv"
fi
