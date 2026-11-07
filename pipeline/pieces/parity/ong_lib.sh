# Shared by parity/21_parity.sh, 22_parity.sh, 23_parity.sh (R2 piece worker, 2026-10-21). Source after lib/common.sh.
PIECES=$FS/pipeline/pieces; P=$WORK/parity; GUARD=${TFB_CODE_ROOT:-$OLD}/tools/window_guard.py
parity_begin() { NN=$1; R=$PIECES/parity/$NN.result; mkdir -p "$P"; T0=$SECONDS
  lock "${NN}_parity"; trap 'rm -f "$P/${NN}_"*; unlock' EXIT
  waitmem 40 "$NN parity"; waitdisk 1350 "$NN parity"; }
# write_result JSON VARIANT EXTRA_JSON: adds piece, variant, runtime_s, extra fields; one line to $R and stdout
write_result() { local e=${3:-}; [ -n "$e" ] || e='{}'; "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[3])); d.update(piece=sys.argv[2], variant=sys.argv[4], runtime_s=int(sys.argv[5])); print(json.dumps(d))" \
  "$1" "$NN" "$e" "$2" $((SECONDS - T0)) | tee "$R"; }
# rows_result NN ROWS EXPECTED_SHA STORED_ROWS: the cheap variant (ongoing rows parquet, bytes or values)
rows_result() { local rc=0 res
  res=$("$PY" -c "
import json, sys, hashlib, pandas as pd
f, x, st = sys.argv[1:4]; s = hashlib.sha256(open(f, 'rb').read()).hexdigest(); out = dict(built=f, sha256=s, expected=x)
if s == x: out['result'] = 'PASS-SHA'
else:
    a = pd.read_parquet(f).set_index('submission_id').queue_pred.sort_index(); b = pd.read_parquet(st).set_index('submission_id').queue_pred.sort_index()
    n = int((a.to_numpy() != b.reindex(a.index).to_numpy()).sum()) if a.index.equals(b.index) else -1
    out.update(cells_differ=n, result='PASS-VALUE' if n == 0 else 'FAIL')
print(json.dumps(out)); sys.exit(0 if out['result'].startswith('PASS') else 1)" "$(cygpath -m "$2" 2>/dev/null || echo "$2")" "$3" "$(cygpath -m "$4" 2>/dev/null || echo "$4")") || rc=1
  write_result "$res" "rows-only (ongoing rows parquet, no CSV)"; return $rc; }
# full_result NN OUT EXPECTED_SHA STORED_GZ PARENT: lib/compare.py + window guard (--no-record) vs the parent
full_result() { local rc=0 res g grc=0
  res=$("$PY" "$PIECES/lib/compare.py" "$2" "$3" "$4") || rc=1
  if [ -f "$GUARD" ]; then g=$("$PY" "$GUARD" "$2" --against "$5" --no-record 2>&1) || grc=$?
    g=$("$PY" -c "import json,sys; print(json.dumps(dict(window_guard='PASS' if sys.argv[1] == '0' else 'REFUSE', guard_rc=int(sys.argv[1]), guard_out=sys.argv[2].strip().splitlines()[-2:])))" "$grc" "$g")
    [ $grc -eq 0 ] || rc=1
  else g='{"window_guard": "SKIPPED (tools/window_guard.py not found)"}'; fi
  write_result "$res" "full" "$g"; return $rc; }
# fail_result WHY
fail_result() { write_result "{\"result\": \"FAIL\", \"why\": \"$1\"}" "${2:-full}"; exit 1; }
