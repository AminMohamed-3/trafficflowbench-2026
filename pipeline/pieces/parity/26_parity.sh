#!/usr/bin/env bash
# Parity of row 26 (BUNDLE18: S2b + GAPW-F + S7b): stored C65 in, rebuilt C66 out, must equal c_bundle18_s2b_s7b.csv (sha f1a5278f).
# Usage: 26_parity.sh [--cheap]
#   full   : serve 60 S2b members (~5 min) + hooked S2b build (~35 min) + post + S7b build (~35 min) + step 5 (~3 min) = ~80 min CPU,
#            peak RSS ~4 GB (one step at a time).
#   --cheap: (a) serve of panel D12_I405_S (6 members) == stored arm48k arrays, and the hooked S2b build of that panel == stored
#            BUNDLE18 hook npz (~8 min, ~3 GB); (b) step 5 on ALL rows from the stored intermediates (bundle18/off_c_bundle18_s2b/
#            t1_gapwf, bundle18/s7b/build/t1.csv, s2b/build/t1.csv state parquets) -> full file sha (~5 min, ~2.5 GB).
#            Not covered by --cheap: the S7b build itself (no file in debug mode) and the serve of the other 54 members.
# Writes parity/26.result; deletes $WORK/parity/26_* at the end.
source "$(dirname "$0")/../lib/common.sh"
trap unlock EXIT
CHEAP=0; [ "${1:-}" = --cheap ] && CHEAP=1
lock 26_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/26.result
CR=$OLD/artifacts/claude_ref; B=$CR/l_b6/bundle18
export R19_PRED_DIR=$CR/c_gap2c47/pred_cpu R26_REF=$CR/l_b6/s2b/build/t1.csv.state.parquet
waitmem 40 "26 parent"; waitdisk 1350 "26 parent"
gzip -dc "$CR/l_b6/gapw/c_gapw_on_c64.csv.gz" > "$P/26_parent.csv"
rc=0; extra=""
if [ $CHEAP = 1 ]; then
  extra=$(WORK=$P/26_work bash "$FS/pipeline/pieces/26_bundle18.sh" "$P/26_parent.csv" "$P/26_out.csv" --panel D12_I405_S | grep '"result"' | tr '\n' ' ') || rc=$?
  echo "$extra" | grep -q FAIL && rc=1
  [ $rc -eq 0 ] && { USE_GAPWF=$B/off_c_bundle18_s2b/t1_gapwf.state.parquet USE_S7B=$B/s7b/build/t1.csv.state.parquet \
    USE_REF=$CR/l_b6/s2b/build/t1.csv.state.parquet WORK=$P/26_work bash "$FS/pipeline/pieces/26_bundle18.sh" "$P/26_parent.csv" "$P/26_out.csv" || rc=$?; }
else
  WORK=$P/26_work bash "$FS/pipeline/pieces/26_bundle18.sh" "$P/26_parent.csv" "$P/26_out.csv" || rc=$?
fi
if [ $rc -eq 0 ]; then
  res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/26_out.csv" f1a5278f5f17335445ccd4fdc6852ecb760ff41e3dd5282040094846b05e5108 \
        "$B/c_bundle18_s2b_s7b.csv.gz") || rc=$?
else res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='26', variant=sys.argv[3], runtime_s=int(sys.argv[2]), panel_check=sys.argv[4][:400]); print(json.dumps(d))" \
  "$res" $((SECONDS - T0)) "$([ $CHEAP = 1 ] && echo cheap || echo full)" "$extra" | tee "$R"
rm -rf "$P/26_parent.csv" "$P/26_out.csv" "$P/26_out.csv.tmp" "$P/26_out.csv.info.json" "$P/26_work"
exit $rc
