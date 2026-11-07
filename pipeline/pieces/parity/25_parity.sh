#!/usr/bin/env bash
# Parity of row 25 (GAPW-W + C0 gap rows on the S2 chain): stored C64 in, rebuilt C65 out, must equal c_gapw_on_c64 (sha 7b2a2daa).
# Usage: 25_parity.sh [--cheap]
#   full   : hooked S2 build (~34 min CPU, peak RSS 3.7 GB) + post (~1 min) + gap2 step (~2 min); ~40 min, peak ~4 GB.
#   --cheap: (a) hooked build of ONE panel (D12_I405_S, both splits; ~5-8 min, ~3 GB) whose hook arrays must equal the stored
#            hook npz; (b) post + gap2 step from the STORED hooked build (artifacts/claude_ref/l_b6/gapw/build/t1.csv) -> full file
#            sha (~4 min, ~2 GB). Proves steps 2-3 on all rows and step 1 on one panel.
# Earlier-piece intermediates are pinned to the stored old equivalents (R24_ARM_DIR, R24_STATE, R19_PRED_DIR).
# Writes parity/25.result (JSON line of lib/compare.py + runtime_s, variant); deletes $WORK/parity/25_* at the end.
source "$(dirname "$0")/../lib/common.sh"
trap unlock EXIT
CHEAP=0; [ "${1:-}" = --cheap ] && CHEAP=1
lock 25_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/25.result
CR=$OLD/artifacts/claude_ref
export R24_ARM_DIR=$CR/l_b6/s2/arm36k R24_STATE=$CR/l_b6/s2/build/t1.csv.state.parquet R19_PRED_DIR=$CR/c_gap2c47/pred_cpu
waitmem 40 "25 parent"; waitdisk 1350 "25 parent"
gzip -dc "$CR/l_b6/s2/c_b6s2_on_c59.csv.gz" > "$P/25_parent.csv"
rc=0; extra=""
if [ $CHEAP = 1 ]; then
  a=$(WORK=$P/25_work bash "$FS/pipeline/pieces/25_gapw.sh" "$P/25_parent.csv" "$P/25_out.csv" --panel D12_I405_S | tail -1) || rc=$?
  extra=$a
  [ $rc -eq 0 ] && { WORK=$P/25_work bash "$FS/pipeline/pieces/25_gapw.sh" "$P/25_parent.csv" "$P/25_out.csv" --hook-from "$CR/l_b6/gapw/build/t1.csv" || rc=$?; }
else
  WORK=$P/25_work bash "$FS/pipeline/pieces/25_gapw.sh" "$P/25_parent.csv" "$P/25_out.csv" || rc=$?
fi
if [ $rc -eq 0 ]; then
  res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/25_out.csv" 7b2a2daa0eac7310681f647a9d4562be56aebb48b8c089d0f6898f039f89d4f1 \
        "$CR/l_b6/gapw/c_gapw_on_c64.csv.gz") || rc=$?
else res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='25', variant=sys.argv[3], runtime_s=int(sys.argv[2]), panel_check=sys.argv[4][:300]); print(json.dumps(d))" \
  "$res" $((SECONDS - T0)) "$([ $CHEAP = 1 ] && echo cheap || echo full)" "$extra" | tee "$R"
rm -rf "$P/25_parent.csv" "$P/25_out.csv" "$P/25_out.csv.tmp" "$P/25_out.csv.info.json" "$P/25_work"
exit $rc
