#!/usr/bin/env bash
# Parity of row 19 (gap2 member from saved weights): stored C47 in, rebuilt C50 out, must equal c_gap2w_on_c47 (sha 9ddd8848).
# Full run only (cheap: ~3 min serve + blend after the parent expand of ~3 min; peak RAM ~3 GB in the expand, ~1.4 GB serve).
# Parent C47 = tools/probe.py expand of C44 (artifacts/claude_ref/probes/c_b6l12k_on_c43.csv) + c_b6l24k_on_c44.delta.csv.gz.
# Also asserts the served member arrays equal the laptop serve of 2026-10-13 (STRICT_SERVE=1).
# Writes parity/19.result (JSON line of lib/compare.py + runtime_s); deletes $WORK/parity/19_* at the end.
source "$(dirname "$0")/../lib/common.sh"
trap unlock EXIT
lock 19_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/19.result
waitmem 40 "19 expand"; waitdisk 1350 "19 expand"
"$PY" "$OLD/tools/probe.py" expand "$OLD/artifacts/claude_ref/probes/c_b6l12k_on_c43.csv" \
  "$OLD/artifacts/claude_ref/probes/c_b6l24k_on_c44.delta.csv.gz" "$P/19_parent.csv" \
  --sha 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a
waitmem 20 "19 piece"; waitdisk 1350 "19 piece"
rc=0
STRICT_SERVE=1 WORK=$P/19_work bash "$FS/pipeline/pieces/19_gap2w.sh" "$P/19_parent.csv" "$P/19_out.csv" || rc=$?
if [ $rc -eq 0 ]; then
  res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/19_out.csv" 9ddd8848c235627c665b2a6732043485f7e2c14e3bbb45f09b6bb509f1d75bb4 \
        "$OLD/artifacts/claude_ref/probes/c_gap2w_on_c47.csv.gz") || rc=$?
else res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='19', variant='full', runtime_s=int(sys.argv[2])); print(json.dumps(d))" \
  "$res" $((SECONDS - T0)) | tee "$R"
rm -rf "$P/19_parent.csv" "$P/19_out.csv" "$P/19_out.csv.tmp" "$P/19_work"
exit $rc
