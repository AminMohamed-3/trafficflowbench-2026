# Step runner shared by the ongoing-queue pieces 21, 22, 23 (R2 piece worker, 2026-10-21). Source after lib/common.sh.
# CODE = the code + data root the shipped scripts run in (they bind ROOT = parents[2] of docs/reference_code/*.py and read
# ROOT/data/raw/kaggle_public/submission_key.csv, ROOT/artifacts/claude_ref/ong2/official_types.csv (written by the base
# build's ong2_official_types.py), the panel caches of scripts/build_cache.py). Laptop: the development repo ($OLD).
# Clean machine: the folder where R1's base build ran (final_solution with the copied code closure + fetched data).
CODE=${TFB_CODE_ROOT:-$OLD}
# BASE_RUN = the base build's run dir (asm2_run_all.sh A=artifacts/claude_ref/asm2/run, relative to CODE).
BASE_RUN=${TFB_BASE_RUN:-$CODE/artifacts/claude_ref/asm2/run}
SRC=$FS/pipeline/pieces/src
wp() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi; }   # path for Windows python
# step SCRIPT ARGS...: python docs/reference_code/asm2_step.py docs/reference_code/SCRIPT ARGS (cwd CODE), with the
# environment of the development chain (exp/L_lab/a3/a3.py chain_env: DET, 2 threads, CPU only).
step() { local s=$1; shift
  ( cd "$CODE" && env TFB_ASM2_DET=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$(wp "$CODE/src")" \
      PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="${STEP_CVD:--1}" COLD_PARITY_TOL="${COLD_PARITY_TOL:-1e-6}" \
      "$PY" -B docs/reference_code/asm2_step.py "docs/reference_code/$s" "$@" ); }
glue() { ( cd "$CODE" && env PYTHONPATH="$(wp "$CODE/src")" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 "$PY" -B "$@" ); }
fsha() { [ "${PARENT_SHA_CHECK:-1}" = 0 ] && { say "parent sha check skipped (PARENT_SHA_CHECK=0)"; return 0; }; need_sha "$1" "$2"; }
