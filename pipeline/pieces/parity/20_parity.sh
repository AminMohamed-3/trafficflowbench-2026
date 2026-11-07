#!/usr/bin/env bash
# Parity of row 20 (L_micro file 2): stored C50 in, rebuilt C54 out, must equal c_lmicro2_on_c50 (sha b0cbce4e).
# Usage: 20_parity.sh [--cheap [PANEL]]
#   full (default): state build of all panels (~55 min at 6 threads, peak RSS 3.3 GB; waits for 4 GiB free) + file step
#     (~5 min, ~3 GB) on the stored C50 (gunzip of artifacts/claude_ref/probes/c_gap2w_on_c47.csv.gz); base-build intermediates
#     taken from their stored equivalents: S08 artifacts/claude_ref/c_budget/s08/official/l127_r4800, B6 members
#     artifacts/claude_ref/c_b6long/p24k/arm24k; control = artifacts/claude_ref/c_b6long/p24k/official/t1.csv.state.parquet.
#     Also compares the rebuilt state parquet with the stored one (artifacts/claude_ref/l_micro/a2/t1.csv.state.parquet).
#   --cheap [PANEL] (default D12_I405_S, ~5-10 min build + ~6 min file step): (a) state build of one panel, both splits, equal
#     bit for bit to the stored parquet's rows of that panel; (b) file step with the STORED state parquet -> full file sha.
#     Proves the file step fully and the build on one panel.
# Writes parity/20.result (JSON line + runtime_s; the cheap variant adds the panel check); deletes $WORK/parity/20_* at the end.
source "$(dirname "$0")/../lib/common.sh"
CHEAP=0; PANEL=D12_I405_S
[ "${1:-}" = --cheap ] && { CHEAP=1; [ -n "${2:-}" ] && PANEL=$2; }
trap unlock EXIT
lock 20_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/20.result
CR=$OLD/artifacts/claude_ref; STORED_STATE=$CR/l_micro/a2/t1.csv.state.parquet
export S08_DIR=$CR/c_budget/s08/official/l127_r4800 B6_DIR=$CR/c_b6long/p24k/arm24k
CTL=$CR/c_b6long/p24k/official/t1.csv.state.parquet
rc=0; extra='{}'
if [ $CHEAP = 1 ]; then
  waitmem 40 "20 panel build"; waitdisk 1350 "20 panel build"
  WORK=$P/20_work bash "$FS/pipeline/pieces/20_lmicro.sh" x x --panels "$PANEL" || rc=$?
  if [ $rc -eq 0 ]; then extra=$("$PY" "$FS/pipeline/pieces/src/20/cmp_state.py" "$P/20_work/20_lmicro/t1.csv.panels.parquet" "$STORED_STATE") || rc=$?
  else extra="{\"result\": \"FAIL\", \"why\": \"panel build exit $rc\"}"; fi
  STATE_ARG=(--state "$STORED_STATE")
else STATE_ARG=(); fi
waitmem 40 "20 build"; waitdisk 1350 "20 build"
gunzip -c "$CR/probes/c_gap2w_on_c47.csv.gz" > "$P/20_parent.csv"
rc2=0
WORK=$P/20_work bash "$FS/pipeline/pieces/20_lmicro.sh" "$P/20_parent.csv" "$P/20_out.csv" --control "$CTL" ${STATE_ARG[@]+"${STATE_ARG[@]}"} || rc2=$?
if [ $rc2 -eq 0 ] && [ $CHEAP = 0 ]; then
  extra=$("$PY" "$FS/pipeline/pieces/src/20/cmp_state.py" "$P/20_work/20_lmicro/t1.csv.state.parquet" "$STORED_STATE") || rc=$?
fi
if [ $rc2 -eq 0 ]; then
  X=b0cbce4e1352a38c41110225f9d04794c616d0a91c8b743ffadc895b722b729a
  if ! res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/20_out.csv" $X); then   # sha differs: value check vs the stored file
    "$PY" "$OLD/tools/probe.py" expand "$P/20_parent.csv" "$CR/l_micro/c_lmicro2_on_c50.delta.csv.gz" "$P/20_stored.csv" || true
    res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/20_out.csv" $X "$P/20_stored.csv") || rc=1
  fi
else rc=$rc2; res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc2\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='20', variant=sys.argv[4], state_check=json.loads(sys.argv[2]), runtime_s=int(sys.argv[3])); print(json.dumps(d))" \
  "$res" "$extra" $((SECONDS - T0)) "$([ $CHEAP = 1 ] && echo "cheap panel $PANEL + file step from the stored state parquet" || echo full)" | tee "$R"
rm -rf "$P/20_parent.csv" "$P/20_out.csv" "$P/20_stored.csv" "$P/20_work"
exit $rc
