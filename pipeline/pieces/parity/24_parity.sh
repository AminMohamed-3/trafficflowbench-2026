#!/usr/bin/env bash
# Parity of row 24 (L_b6 S2 = B6R): stored C59 in, rebuilt C64 out, must equal c_b6s2_on_c59 (sha ddbd4591).
# Usage: 24_parity.sh [--cheap [PANEL]]
#   full (default): serve the 60 B6R members (~5 min, 4 threads, ~2 GB), state build of all panels (~35 min at 4 threads, peak RSS
#     3.2 GB; waits for 4 GiB free), file step (~3 min, ~3 GB) on the stored C59 (gunzip of artifacts/claude_ref/l_tinyad/a1/on_c57/
#     delta/c_tinyad_on_c57.csv.gz). Row 20's intermediate = its stored equivalent artifacts/claude_ref/l_micro/a2/t1.csv.state.parquet
#     (CTL20); S08 = artifacts/claude_ref/c_budget/s08/official/l127_r4800. Checks: served member arrays = stored receipt
#     (artifacts/claude_ref/l_b6/s2/arm36k/receipt.json speed/flow sha256), rebuilt state parquet = artifacts/claude_ref/l_b6/s2/build/
#     t1.csv.state.parquet bit for bit, output sha.
#   --cheap [PANEL] (default D12_I405_S, ~10 min): serve PANEL (6 members) = stored receipt; state build of PANEL from those members =
#     stored S2 parquet rows; file step with the STORED S2 state parquet -> full file sha.
# Writes parity/24.result (JSON line + runtime_s + member and state checks); deletes $WORK/parity/24_* at the end.
source "$(dirname "$0")/../lib/common.sh"
CHEAP=0; PANEL=D12_I405_S
[ "${1:-}" = --cheap ] && { CHEAP=1; [ -n "${2:-}" ] && PANEL=$2; }
trap unlock EXIT
lock 24_parity
T0=$SECONDS; P=$WORK/parity; mkdir -p "$P"; R=$FS/pipeline/pieces/parity/24.result
CR=$OLD/artifacts/claude_ref; STORED_STATE=$CR/l_b6/s2/build/t1.csv.state.parquet; STORED_REC=$CR/l_b6/s2/arm36k/receipt.json
export S08_DIR=$CR/c_budget/s08/official/l127_r4800 CTL20=$CR/l_micro/a2/t1.csv.state.parquet
W24=$P/24_work/24_b6s2
members_check() { "$PY" - "$W24/arm36k/receipt.json" "$STORED_REC" <<'EOF'
import json, sys
a, b = (json.load(open(x))["files"] for x in sys.argv[1:3])
bad = [f for f in a if (a[f]["speed_sha256"], a[f]["flow_sha256"]) != (b[f]["speed_sha256"], b[f]["flow_sha256"])]
print(json.dumps(dict(members=len(a), differ=bad, result="EQUAL" if a and not bad else "DIFFER"))); sys.exit(1 if bad or not a else 0)
EOF
}
rc=0; mem='{}'; st='{}'
waitmem 40 "24 serve/build"; waitdisk 1350 "24 serve/build"
if [ $CHEAP = 1 ]; then
  WORK=$P/24_work bash "$FS/pipeline/pieces/24_b6s2.sh" x x --panels "$PANEL" || rc=$?
  if [ $rc -eq 0 ]; then
    mem=$(members_check) || rc=1
    st=$("$PY" "$FS/pipeline/pieces/src/24/cmp_state.py" "$W24/t1.csv.panels.parquet" "$STORED_STATE") || rc=1
  fi
  STATE_ARG=(--state "$STORED_STATE")
else STATE_ARG=(); fi
waitmem 40 "24 piece"; waitdisk 1350 "24 piece"
gunzip -c "$CR/l_tinyad/a1/on_c57/delta/c_tinyad_on_c57.csv.gz" > "$P/24_parent.csv"
rc2=0
WORK=$P/24_work bash "$FS/pipeline/pieces/24_b6s2.sh" "$P/24_parent.csv" "$P/24_out.csv" ${STATE_ARG[@]+"${STATE_ARG[@]}"} || rc2=$?
if [ $rc2 -eq 0 ] && [ $CHEAP = 0 ]; then
  mem=$(members_check) || rc=1
  st=$("$PY" "$FS/pipeline/pieces/src/24/cmp_state.py" "$W24/t1.csv.state.parquet" "$STORED_STATE") || rc=1
fi
if [ $rc2 -eq 0 ]; then
  res=$("$PY" "$FS/pipeline/pieces/lib/compare.py" "$P/24_out.csv" ddbd45915f12ded493816f913c4e28f31c4e6803d2f02a51359aa03f59fe4510 \
        "$CR/l_b6/s2/c_b6s2_on_c59.csv.gz") || rc=1
else rc=$rc2; res="{\"result\": \"FAIL\", \"why\": \"piece script exit $rc2\"}"; fi
"$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(piece='24', variant=sys.argv[5], member_check=json.loads(sys.argv[2]), state_check=json.loads(sys.argv[3]), runtime_s=int(sys.argv[4])); print(json.dumps(d))" \
  "$res" "$mem" "$st" $((SECONDS - T0)) "$([ $CHEAP = 1 ] && echo "cheap panel $PANEL (serve + build) + file step from the stored state parquet" || echo full)" | tee "$R"
rm -rf "$P/24_parent.csv" "$P/24_out.csv" "$P/24_work"
exit $rc
