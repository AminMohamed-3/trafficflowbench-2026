#!/usr/bin/env bash
# Parity of row 27 (L_gap2 gap rows): stored C66 in, rebuilt C69 out, must equal c_gap_on_c66.csv (sha d1ada3d6).
# Full run only (cheap enough): G1 serve ~1 min + G3 serve ~5-10 min + gap rows ~2 min; peak RSS ~2.5 GB.
# Row 26 / 19 intermediates pinned to the stored old equivalents (R26_HOOK, R26_GAPWF, R19_PRED_DIR); G1 served values must equal
# the 2026-10-18 laptop serve (STRICT_SERVE=1). The stored C66 csv is read in place (read only), so no parent copy is made.
# Writes parity/27.result; deletes $WORK/parity/27_* at the end.
source "$(dirname "$0")/../lib/common.sh"
trap unlock EXIT
lock 27_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/27.result
CR=$OLD/artifacts/claude_ref
export R26_HOOK=$CR/l_gap2/build/t1.csv.gaprows.npz R26_GAPWF=$CR/l_b6/bundle18/off_c_bundle18_s2b/t1_gapwf.state.parquet \
  R19_PRED_DIR=$CR/c_gap2c47/pred_cpu
waitmem 30 "27 piece"; waitdisk 1350 "27 piece"
rc=0
STRICT_SERVE=1 WORK=$P/27_work bash "$FS/pipeline/pieces/27_gap2.sh" "$CR/l_b6/bundle18/c_bundle18_s2b_s7b.csv" "$P/27_out.csv" || rc=$?
if [ $rc -eq 0 ]; then
  res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/27_out.csv" d1ada3d6453ee9fd0b25ce9c24f48e4ce062a63302ab69b8af82e8f513a99c1e \
        "$CR/l_gap2/c_gap_on_c66.csv") || rc=$?
else res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='27', variant='full', runtime_s=int(sys.argv[2])); print(json.dumps(d))" \
  "$res" $((SECONDS - T0)) | tee "$R"
rm -rf "$P/27_out.csv" "$P/27_out.csv.tmp" "$P/27_out.csv.info.json" "$P/27_work"
exit $rc
