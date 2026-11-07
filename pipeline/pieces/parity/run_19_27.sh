#!/usr/bin/env bash
# Sequential full parity chain for rows 19 to 27 (worker R2, 2026-10-21). Self-finishing; each runner takes the laptop
# lock itself, waits for 4 GiB RAM and the 13.5 GiB disk floor, writes parity/<NN>.result and cleans its own files.
# After each runner the status table pipeline/pieces/STATUS_19_27.md is regenerated (make_status.py).
# Usage: run_19_27.sh [NN ...]   (default: 19 21 22 23 20 24 25 26 27; quick ones first)
# A full run that fails is followed by its --cheap variant when the runner has one (result kept as <NN>.cheap.result).
# Log: parity/logs/run_19_27.log; DONE marker parity/logs/RUN_19_27_DONE.
D=$(cd "$(dirname "$0")" && pwd); FS=$(cd "$D/../../.." && pwd); OLD=$(cd "$FS/.." && pwd)
PY=$OLD/.venv_tfb/Scripts/python.exe
mkdir -p "$D/logs"; rm -f "$D/logs/RUN_19_27_DONE"
LIST=("$@"); [ ${#LIST[@]} -eq 0 ] && LIST=(19 21 22 23 20 24 25 26 27)
for n in "${LIST[@]}"; do
  echo "$(date -u +%FT%TZ) start $n"
  bash "$D/${n}_parity.sh" > "$D/logs/${n}_full.log" 2>&1; rc=$?
  echo "$(date -u +%FT%TZ) end $n rc=$rc"
  if [ $rc -ne 0 ] && grep -q -- '--cheap' "$D/${n}_parity.sh"; then
    cp "$D/${n}.result" "$D/${n}.full_fail.result" 2>/dev/null || true
    echo "$(date -u +%FT%TZ) cheap $n"
    bash "$D/${n}_parity.sh" --cheap > "$D/logs/${n}_cheap.log" 2>&1; rc2=$?
    mv "$D/${n}.result" "$D/${n}.cheap.result" 2>/dev/null || true
    cp "$D/${n}.full_fail.result" "$D/${n}.result" 2>/dev/null || true
    echo "$(date -u +%FT%TZ) cheap end $n rc=$rc2"
  fi
  "$PY" "$D/../make_status.py" > /dev/null 2>&1 || true
done
date -u +%FT%TZ > "$D/logs/RUN_19_27_DONE"
