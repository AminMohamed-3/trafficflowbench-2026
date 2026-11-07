#!/usr/bin/env bash
# Full chain C47 -> C80 (rows 19-35) on the laptop with two read traces, to list every development-repo file the build reads.
#   1. Python open audit (sitecustomize.py on PYTHONPATH): every Python-level open outside final_solution.
#   2. NTFS last-access times: every file under the development repo whose atime moved during the run (covers native readers:
#      pyarrow, LightGBM, torch C loads). NTFS updates atime only when the stored value is over an hour old, so the cutoff is
#      the run start minus 2 h (a superset: it also keeps files read just before the start).
# Usage (Git Bash, laptop): bash checks/open_audit/run_chain_audited.sh [20_pieces.sh args, e.g. --from 28 --keep]
#   (self-finishing; ~12-13 h for the whole chain)
# Env: TFB_OLD_REPO (development repo root), CHAIN_BASE, CHAIN_BASE_RUN, CHAIN_S08_DIR, CHAIN_WORK as for pipeline/20_pieces.sh;
#   AUDIT_TAG (default chain_audit) names the log folder.
# The audit root is TFB_OPEN_AUDIT_ROOT (= final_solution). TFB_ROOT is NOT set here: rows 19-27 set their own, and rows
# 28-35 (pipeline/pieces/lib/r3.sh) default it to the development repo, where their work folder must lie.
# Writes outputs/logs/$AUDIT_TAG/{open_audit.log, atime_reads.tsv, atime_reads_summary.txt, run.log, RESULT}
set -uo pipefail
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OLD=${TFB_OLD_REPO:?set TFB_OLD_REPO to the development repo}
PY=${PY:-$OLD/.venv_tfb/Scripts/python.exe}
L=$FS/outputs/logs/${AUDIT_TAG:-chain_audit}; mkdir -p "$L"
T0=$(date +%s); echo "start $(date -u +%FT%TZ) epoch $T0 args: $*" > "$L/RESULT"
export TFB_OPEN_AUDIT_ROOT=$(cygpath -w "$FS") TFB_OPEN_AUDIT_LOG=$(cygpath -w "$L/open_audit.log")
export TFB_OPEN_AUDIT_ALLOW=$(cygpath -w "$OLD/data")
export PYTHONPATH="$(cygpath -w "$FS/checks/open_audit")${PYTHONPATH:+;$PYTHONPATH}"
bash "$FS/pipeline/20_pieces.sh" "$@" > "$L/run.log" 2>&1; rc=$?
echo "chain rc=$rc end $(date -u +%FT%TZ)" >> "$L/RESULT"
"$PY" -I "$FS/checks/open_audit/atime_reads.py" "$(cygpath -w "$OLD")" "$((T0 - 7200))" "$(cygpath -w "$L")" >> "$L/RESULT" 2>&1
tail -n 20 "$FS/outputs/logs/chain_19_35.tsv" | cut -f2-5 >> "$L/RESULT"
echo DONE >> "$L/RESULT"
