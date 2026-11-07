#!/bin/bash
# 10_base_c47.sh: the base build = the submitted file C47 (public .89776), CPU only, deterministic, from the released data
# and the saved weights. It runs the proven entry docs/reference_code/asm2_run_all.sh (byte copy of the development repo's
# script, unchanged) from final_solution/ with default flags, output under outputs/c47/, then checks it with 99_verify.py.
#
#   bash pipeline/10_base_c47.sh [--reuse-cache DIR] [--s08-fitted] [--foreground] [--no-gates]
#
#   --reuse-cache DIR  hard-link an existing audited panel cache (DIR = <repo>/artifacts/cache with meta.json per
#                      panel/split) instead of building it (saves ~20 min and ~3.2 GiB of disk; tfb.data.build_cache
#                      skips a cache whose meta.json exists; the caches are byte-equal across machines, KBUILD wsl-cache)
#   --s08-fitted       take the fitted S08 exports (weights/base/00_base_s08fit.csv; fetch them with --pieces base_s08fit)
#                      instead of the 127 x 4800 refit (saves ~4 h and ~2.8 GiB peak disk; the refit reproduced them exactly)
#   --foreground       run in this process (use under `wsl.exe ... --`: wsl.exe must stay alive while the job runs);
#                      default: detach with nohup and return at once (pid and log printed)
#   --no-gates         skip the RAM / disk start gates (not the running disk brake)
#
# Needs: Linux, Python 3.12 venv with environment/requirements-cpu.txt at /workspace/tfb/venv (VENV=...; the LightGBM
# runtime guard src/tfb/lane_s/lgb_runtime.py accepts LightGBM on Linux only from /workspace/tfb/venv), the release
# (pipeline/00_fetch_data.sh) and the weights (pipeline/01_fetch_weights.sh). Why not Windows: asm2_inputs.py
# --environment checks all 35 pins (Linux wheels) and the guard path; both stop a Windows run at the first step.
# Settings (env): TFB_ASM2_COLD_NPAR (cold-start members at a time, 1.65 GiB each, default 2 here), TFB_ASM2_S08_THREADS
# (default 8), OMP/MKL/OPENBLAS threads (script default 3), FLOOR_GIB (13.5), BRAKE_GIB (13.0), NEED_GIB (start gate).
# Results: outputs/c47/final_A.csv (+ .manifest.json, .state.parquet, .build.json), outputs/c47/run/ (intermediates,
# moved from artifacts/claude_ref/asm2/run after the build), outputs/c47/verify.json, outputs/logs/base_c47.{log,hb,result},
# outputs/logs/open_audit.tsv (Python-level opens outside final_solution; expected empty).
# Expected: state rows of the rebuilt file == C47 to ~4e-13 km/h / ~4e-11 vph (Linux vs the Windows CPU that made C47),
# every queue and ODME row exact (KBUILD wsl-compare 2026-10-13). Runtime ~7 h (--s08-fitted, --reuse-cache) to ~11.5 h.
set -euo pipefail
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$FS"
LOGD=$FS/outputs/logs; OUTD=outputs/c47; mkdir -p "$LOGD" "$OUTD"
LOG=$LOGD/base_c47.log; HB=$LOGD/base_c47.hb; RES=$LOGD/base_c47.result
REUSE=""; S08FIT=0; FG=0; GATES=1; CHILD=0; ORIG=("$@")
while [ $# -gt 0 ]; do case $1 in
  --reuse-cache) REUSE=$2; shift 2;; --s08-fitted) S08FIT=1; shift;; --foreground) FG=1; shift;;
  --no-gates) GATES=0; shift;; --child) CHILD=1; shift;; *) echo "unknown argument $1"; exit 2;; esac; done
if [ $FG = 0 ] && [ $CHILD = 0 ]; then
  nohup setsid bash "$0" --child "${ORIG[@]}" > "$LOG" 2>&1 < /dev/null &
  echo "base build detached: pid $! log $LOG heartbeat $HB result $RES"; exit 0
fi
[ $CHILD = 1 ] || exec > >(tee "$LOG") 2>&1
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { say "FAILED $*"; echo "FAILED $(date -u +%FT%TZ) base_c47: $*" > "$RES"; exit 1; }
VENV=${VENV:-/workspace/tfb/venv}; export PATH=$VENV/bin:$PATH
FLOOR_GIB=${FLOOR_GIB:-13.5}; BRAKE_GIB=${BRAKE_GIB:-13.0}
NCACHE=$( (ls artifacts/cache/*/*/meta.json 2>/dev/null || true) | wc -l)   # 30 = every panel/split cache already in place (skipped by the build)
NEED_GIB=${NEED_GIB:-$(awk -v r="$REUSE" -v c=$NCACHE -v s=$S08FIT 'BEGIN { n = 2.6; if (r == "" && c < 30) n += 3.3; if (s == 0) n += 1.6; print n }')}
free_gib() { df -Pk "$FS" | awk 'NR == 2 { printf "%.2f", $4 / 1048576 }'; }
mem_gib() {   # host view when the Windows interop is there (WSL), else the VM / machine view
  local m=""; if command -v powershell.exe >/dev/null 2>&1; then
    m=$(timeout 20 powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" 2>/dev/null | tr -d '\r' || true); fi
  [ -n "$m" ] && echo "$m" || awk '/MemAvailable/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }

say "base build C47 from $FS (reuse-cache='${REUSE}', s08-fitted=$S08FIT)"
# 1. platform, venv, inputs
[ "$(uname -s)" = Linux ] || fail "Linux required (see header: environment pins and LightGBM guard)"
python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" || fail "python 3.12 venv not first on PATH (VENV=$VENV)"
python -c "import lightgbm, pathlib; p = pathlib.Path(lightgbm.__file__).resolve(); assert p.is_relative_to('/workspace/tfb/venv'), p" \
  || fail "LightGBM must come from /workspace/tfb/venv (src/tfb/lane_s/lgb_runtime.py)"
[ -f data/raw/kaggle_public/config/corridors.json ] || fail "release missing: bash pipeline/00_fetch_data.sh [--link DIR]"
[ -f PLAN.md ] && [ -d .git ] || fail "root markers PLAN.md and .git/ missing (tfb.paths and lane_q.windows stop at .git)"
if [ $GATES = 1 ]; then
  m=$(mem_gib); f=$(free_gib)
  ge "$m" 4 || fail "RAM gate: $m GiB free, need >= 4"
  ge "$f" "$(awk -v a=$FLOOR_GIB -v b=$NEED_GIB 'BEGIN { print a + b }')" || fail "disk gate: $f GiB free, need floor $FLOOR_GIB + build $NEED_GIB"
  say "gates ok: RAM $m GiB free, disk $f GiB free (floor $FLOOR_GIB + need $NEED_GIB)"
fi
PIECES=base; [ $S08FIT = 1 ] && PIECES=base,base_s08fit
python pipeline/lib/fetch_weights.py check --pieces $PIECES || fail "weights not in place: bash pipeline/01_fetch_weights.sh"
python pipeline/lib/fetch_weights.py snapshot "$LOGD/inputs_snapshot.json" --pieces $PIECES
A=artifacts/claude_ref/asm2/run
[ ! -e "$A" ] || fail "$A exists from an earlier run: move it away first (the build must start clean)"
[ ! -e "$OUTD/final_A.csv" ] || fail "$OUTD/final_A.csv exists: move it away first"
# 2. optional cache reuse (hard links; read-only memmaps)
if [ -n "$REUSE" ]; then
  python - "$REUSE" <<'EOF' || fail "cache reuse"
import os, sys, json
from pathlib import Path
src, dst = Path(sys.argv[1]), Path("artifacts/cache"); n = 0
for meta in sorted(src.glob("*/*/meta.json")):
    d = dst / meta.parent.relative_to(src)
    if (d / "meta.json").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    for f in meta.parent.iterdir():
        if f.is_file():
            try: os.link(f, d / f.name)
            except OSError: raise SystemExit(f"hard link failed (other volume?): {f}")
            n += 1
print(f"cache reuse: {n} files hard-linked from {src}; panels/splits: {len(list(dst.glob('*/*/meta.json')))}")
EOF
fi
cache_list() { (find artifacts/cache -type f -printf '%P %s %T@\n' 2>/dev/null || true) | sort; }
cache_list > "$LOGD/cache_before.txt"; say "panel caches in place: $(ls artifacts/cache/*/*/meta.json 2>/dev/null | wc -l) of 30 (missing ones are built)"
# 3. watcher: heartbeat, disk brake, S08 training matrices removed once their split is exported
BUILD_PGID=""
watch() {
  local cleaned_v=0 cleaned_p=0
  while sleep 30; do
    local f m; f=$(free_gib); m=$(awk '/MemAvailable/ { printf "%.2f", $2 / 1048576 }' /proc/meminfo)
    echo "$(date -u +%FT%TZ) disk_free_gib $f mem_avail_gib $m" > "$HB"
    if ! ge "$f" "$BRAKE_GIB"; then
      echo "FAILED $(date -u +%FT%TZ) base_c47: disk brake, $f GiB free (< $BRAKE_GIB); build stopped" > "$RES"
      [ -n "$BUILD_PGID" ] && kill -TERM -- -"$BUILD_PGID" 2>/dev/null; return; fi
    if [ $cleaned_v = 0 ] && grep -qs "^wrote .*official_validation.parquet" $A/s08/ship_validation.log; then
      rm -rf $A/s08/validation/data_seed*; cleaned_v=1; echo "$(date -u +%FT%TZ) removed S08 validation training matrices" >> "$HB.events"; fi
    if [ $cleaned_p = 0 ] && grep -qs "^wrote .*official_private.parquet" $A/s08/ship_private.log; then
      rm -rf $A/s08/private/data_seed*; cleaned_p=1; echo "$(date -u +%FT%TZ) removed S08 private training matrices" >> "$HB.events"; fi
  done
}
# 4. the proven entry, unchanged (default flags = C47), CPU only, deterministic
export TFB_ROOT=$FS TFB_ASM2_CPU_ONLY=1 TFB_ASM2_DET=1 CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export TFB_ASM2_COLD_NPAR=${TFB_ASM2_COLD_NPAR:-2} TFB_ASM2_S08_THREADS=${TFB_ASM2_S08_THREADS:-8}
export PYTHONPATH=$FS/checks/open_audit TFB_OPEN_AUDIT_LOG=$LOGD/open_audit.tsv TFB_OPEN_AUDIT_ALLOW=$(realpath data/raw/kaggle_public)
: > "$TFB_OPEN_AUDIT_LOG"
FLAGS=(); [ $S08FIT = 1 ] && FLAGS+=(--s08-from=artifacts/claude_ref/c_budget/s08/official/l127_r4800)
say "launch: bash docs/reference_code/asm2_run_all.sh $OUTD/final_A.csv ${FLAGS[*]:-} (COLD_NPAR $TFB_ASM2_COLD_NPAR, S08 threads $TFB_ASM2_S08_THREADS)"
T0=$SECONDS
setsid bash docs/reference_code/asm2_run_all.sh "$OUTD/final_A.csv" "${FLAGS[@]}" &
BPID=$!; BUILD_PGID=$BPID; watch & WPID=$!
set +e; wait $BPID; RC=$?; set -e
kill $WPID 2>/dev/null || true
BUILD_S=$((SECONDS - T0))
[ $RC = 0 ] || fail "asm2_run_all.sh exit $RC after $((BUILD_S / 60)) min (log above; $(cat "$RES" 2>/dev/null | head -c 200))"
say "build done in $((BUILD_S / 60)) min"
# 5. inputs untouched, intermediates to outputs/, verification against the submitted C47
python pipeline/lib/fetch_weights.py compare "$LOGD/inputs_snapshot.json" || fail "a weight / fitted input changed during the build"
python pipeline/lib/fetch_weights.py check --pieces $PIECES || fail "weights sha check after the build"
cache_list > "$LOGD/cache_after.txt"   # reused caches are hard links: they must not change (new ones may appear)
CCH=$(comm -23 "$LOGD/cache_before.txt" "$LOGD/cache_after.txt" | wc -l)
[ "$CCH" = 0 ] || fail "$CCH panel cache files changed during the build (see $LOGD/cache_before.txt / cache_after.txt)"
mv "$A" "$OUTD/run"
python pipeline/99_verify.py "$OUTD/final_A.csv" --ref-parent reference/c_b6l12k_on_c43.csv \
  --ref-delta reference/c_b6l24k_on_c44.delta.csv.gz --ref-sha 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a \
  --state-atol 1e-8 --json "$OUTD/verify.json" | tee "$LOGD/verify_c47.txt"
V=$(tail -n 1 "$LOGD/verify_c47.txt")
AUD=$(wc -l < "$TFB_OPEN_AUDIT_LOG")
echo "RESULT $(date -u +%FT%TZ) base_c47: build $((BUILD_S / 60)) min; $V; open audit $AUD lines outside final_solution" | tee "$RES"
case "$V" in VERIFY_PASS*) exit 0;; *) exit 1;; esac
