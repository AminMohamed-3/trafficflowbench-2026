#!/usr/bin/env bash
# Clean-room check of the chain C47 -> C80 (rows 19-35) on the author's laptop: everything the chain reads comes from a fresh
# folder made only from what is published (the exported repository, the weights release zips, the devroot extras zip,
# the competition data), never from the development repository.
#   bash checks/clean_room/run_clean_room.sh      (self-finishing, ~12-13 h; writes $CLEAN/RESULT)
# Env (all required): CLEAN (new empty folder, outside the development repo, same volume for hard links), DEV (development repo:
#   only its competition data folder and its Python venv are used), ZIPS (folder with the weights release zips), EXTRA_ZIP
#   (devroot_extra.zip), BASE_OUT (outputs/c47 of the base build made from this repository, with BASE_CACHE = that build's
#   artifacts/cache; the base build itself was checked separately from the repository in WSL, outputs/logs/base_c47.result),
#   REFS (folder holding only the stored C79 and C80 files at their artifacts/claude_ref paths, for the final comparison).
#   Optional RESUME=1 (setup kept) and CHAIN_ARGS (passed to 20_pieces.sh). Optional FETCH=1: discovery run (checks/open_audit/fetch_mode.py): any development file the chain touches under the devroot
#   that the devroot lacks is copied from DEV and listed in outputs/logs/clean_room/fetch.tsv (the extras list). Not clean.
# Checks: (1) Python open audit with root $CLEAN: every Python open outside $CLEAN, the venv, the data and system folders is
#   logged (must be empty); (2) NTFS atime of the development repo after the run (reads there other than data and venv are
#   listed); (3) the chain verdicts, C79 and C80 compared with the stored files.
set -uo pipefail
: "${CLEAN:?}" "${DEV:?}" "${ZIPS:?}" "${EXTRA_ZIP:?}" "${BASE_OUT:?}" "${BASE_CACHE:?}" "${REFS:?}"
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=$DEV/.venv_tfb/Scripts/python.exe
R=$CLEAN/RESULT; mkdir -p "$CLEAN"; T0=$(date +%s)
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R"; }
die() { say "CLEAN_ROOM_FAIL $*"; echo DONE >> "$R"; exit 1; }
# RESUME=1: setup steps 2-5 of an earlier start in this CLEAN are done (devroot exists); the code is exported again, then the chain
[ "${RESUME:-0}" = 1 ] && { [ -f "$CLEAN/devroot/DEVROOT.json" ] || die "RESUME=1 but no $CLEAN/devroot/DEVROOT.json"; }   || [ -z "$(ls -A "$CLEAN" | grep -v '^RESULT$')" ] || die "$CLEAN is not empty"
say "start epoch $T0; source repo $SRC commit $(git -C "$SRC" rev-parse --short HEAD)"
# 1 repository: the public export (tracked files only, no history)
"$PY" -I "$SRC/tools/release/export_public.py" "$CLEAN/repo" >> "$R" 2>&1 || die "export"
FS=$CLEAN/repo; cd "$FS" || die "cd"
if [ "${RESUME:-0}" != 1 ]; then
# 2 data: link to the competition release (the only development-repo folder used besides the venv)
bash pipeline/00_fetch_data.sh --link "$(cygpath -w "$DEV/data/raw/kaggle_public")" >> "$R" 2>&1 || die "data link"
# 3 weights: unzip the release parts, place them by the manifest (sha256 checked)
mkdir -p weights/download
for z in "$ZIPS"/*.zip; do unzip -q -o "$z" -d weights/download || die "unzip $z"; done
PY="$PY" bash pipeline/01_fetch_weights.sh --from-dir weights/download --pieces "" >> "$R" 2>&1 || die "weights"
# 4 base build outputs (hard links): the chain's parent C47 and the base run intermediates, panel cache
mkdir -p outputs artifacts
cp -rl "$BASE_OUT" outputs/c47 2>/dev/null || cp -r "$BASE_OUT" outputs/c47
cp -rl "$BASE_CACHE" artifacts/cache 2>/dev/null || cp -r "$BASE_CACHE" artifacts/cache
# 5 devroot
mkdir -p "$CLEAN/extra"; unzip -q -o "$EXTRA_ZIP" -d "$CLEAN/extra" || die "extra unzip"
"$PY" -I pipeline/15_make_devroot.py "$(cygpath -w "$CLEAN/devroot")" --extra "$(cygpath -w "$CLEAN/extra")" >> "$R" 2>&1 || die "devroot"
rm -rf weights/download
else say "RESUME: setup of the earlier start kept (data link, weights, base outputs, devroot)"; fi
# 6 chain, audited. The chain runs in a clean-room venv (same base interpreter and packages as the dev venv, without its editable
# install of the dev repo's src) whose site-packages holds a .pth naming checks/open_audit: every Python process loads the audit,
# also children started with their own PYTHONPATH (exp/L_lab/a3/a3.py chain_env sets PYTHONPATH=ROOT/src for asm2_step).
VB=$(sed -n 's/^home *= *//p' "$DEV/.venv_tfb/pyvenv.cfg" | tr -d '\r')
if [ ! -x "$CLEAN/venv/Scripts/python.exe" ]; then
  "$VB/python.exe" -m venv --system-site-packages "$(cygpath -w "$CLEAN/venv")" || die "venv"
  for p in lightgbm lightgbm-4.6.0.dist-info; do cp -r "$DEV/.venv_tfb/Lib/site-packages/$p" "$CLEAN/venv/Lib/site-packages/" || die "venv $p"; done
fi
cygpath -w "$FS/checks/open_audit" > "$CLEAN/venv/Lib/site-packages/tfb_open_audit.pth"
PY=$CLEAN/venv/Scripts/python.exe
say "chain python $PY (base $VB, + lightgbm 4.6.0, audit .pth)"
L=$FS/outputs/logs/clean_room; mkdir -p "$L"
export TFB_OPEN_AUDIT_ROOT=$(cygpath -w "$CLEAN") TFB_OPEN_AUDIT_LOG=$(cygpath -w "$L/open_audit.log")
export TFB_OPEN_AUDIT_ALLOW="$(cygpath -w "$DEV/data/raw/kaggle_public")"
unset PYTHONPATH
"$PY" -c "import sys, subprocess, os; e=dict(os.environ, PYTHONPATH='x'); r=subprocess.run([sys.executable,'-c','import sys; print(\"sitecustomize\" in sys.modules)'],env=e,capture_output=True,text=True).stdout.strip(); print('audit in child with own PYTHONPATH:', r); sys.exit(r!='True')" >> "$R" 2>&1 || die "audit not loaded in children"
export TFB_OLD_REPO=$CLEAN/devroot CHAIN_WORK=$CLEAN/devroot/work CHAIN_REF_ROOT=$REFS PY
if [ "${FETCH:-0}" = 1 ]; then
  export TFB_OPEN_AUDIT_FETCH="$(cygpath -w "$CLEAN/devroot")|$(cygpath -w "$DEV")" TFB_OPEN_AUDIT_FETCH_LOG=$(cygpath -w "$L/fetch.tsv")
  say "FETCH MODE (discovery run, not a clean check): missing devroot files are copied from DEV, list in $L/fetch.tsv"
fi
say "chain start"
bash pipeline/20_pieces.sh ${CHAIN_ARGS:-} > "$L/run.log" 2>&1; rc=$?   # CHAIN_ARGS (resume of a discovery run only): e.g. --from 33
say "chain rc=$rc"
"$PY" -I "$FS/checks/open_audit/atime_reads.py" "$(cygpath -w "$DEV")" "$((T0 - 60))" "$(cygpath -w "$L")" >> "$R" 2>&1
n=$( [ -f "$L/open_audit.log" ] && wc -l < "$L/open_audit.log" || echo 0)
say "python opens outside the clean room: $n (outputs/logs/clean_room/open_audit.log)"
tail -n 25 "$FS/outputs/logs/chain_19_35.tsv" | cut -f2-6 >> "$R"
[ -f "$FS/outputs/C80.csv" ] && say "C80 sha $(sha256sum "$FS/outputs/C80.csv" | cut -c1-16)"
[ -f "$L/fetch.tsv" ] && say "fetched from DEV: $(cut -f1 "$L/fetch.tsv" | sort | uniq -c | tr -s ' \n' ' ')"
say "CLEAN_ROOM_END rc=$rc opens_outside=$n minutes=$(( ($(date +%s) - T0) / 60 ))"
echo DONE >> "$R"
