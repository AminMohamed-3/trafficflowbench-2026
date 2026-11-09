#!/usr/bin/env bash
# run_all.sh: the single entry script. From the competition data and the public weight release it rebuilds C79 (our
# selected and scored Kaggle submission, public .90088, private .89874) and C80, then checks both by sha256.
#
#   bash run_all.sh [--data DIR] [--work DIR] [--from N] [--to N] [--dry-run]
#
#   --data DIR   the competition release already on disk (the folder with config/corridors.json); default: download it
#                with the Kaggle CLI and YOUR credentials (kaggle.json or `kaggle auth login`; accept the rules on Kaggle)
#   --work DIR   work folder on the SAME drive as this repository (hard links); default: <repo>/../tfb_run
#   --from/--to  run only steps N..M (each finished step leaves WORK/done_<N>, so a stopped run can resume with --from)
#   --dry-run    check the machine and print the plan; changes nothing
#   env PYTHON312=PATH  the Windows Python 3.12 for step 1 (default: py -3.12, uv python find 3.12, python on PATH)
#
# Where it runs: Git Bash on Windows 10/11 with WSL2 (Ubuntu 22.04). This is the split under which the files were made
# and checked: the base build (step 5) runs inside WSL, because its environment check and LightGBM guard accept only the
# pinned Linux runtime at /workspace/tfb/venv; the piece chain (step 7) runs in Git Bash with a Windows venv (its scripts
# use cygpath and python.exe). Needs about 40 GB free on the repository's drive, 16 GB RAM, no GPU, internet for the
# downloads. Runtime on our laptop (16 threads): about 3.5 h (base) + 11.6 h (chain) + downloads.
#
# Steps
#   1 win-env    Windows venv WORK/venv-win (Python 3.12) from environment/requirements-windows.txt
#   2 data       data/raw/kaggle_public: junction to --data DIR, or Kaggle download (pipeline/00_fetch_data.sh)
#   3 weights    pipeline/01_fetch_weights.sh --release: 3 weight parts + devroot_extra.zip, every file sha256-checked
#   4 linux-env  WSL venv /workspace/tfb/venv from environment/requirements-cpu.txt (uv; /workspace made once as root)
#   5 base       pipeline/10_base_c47.sh --s08-fitted --foreground inside WSL -> outputs/c47/final_A.csv (C47, VERIFY_PASS)
#   6 devroot    pipeline/15_make_devroot.py WORK/devroot --extra weights/devroot_extra (hard links)
#   7 chain      pipeline/20_pieces.sh (ADOPTED rows 19-35 on the fresh C47) -> outputs/C79.csv, outputs/C80.csv
#   8 verify     sha256 of outputs/C79.csv and outputs/C80.csv against the scored and the stored file
# Log: outputs/logs/run_all.log. Result line: outputs/logs/run_all.result (RUN_ALL_PASS or RUN_ALL_FAIL).
set -uo pipefail
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$FS"
C79_SHA=104c2c6fce3393b14ea219ce8ae3ada6230ec358bba37f137335615eb4db1f04
C80_SHA=3e27d66cd2b3cfc4371fd218617ebf08a0fa16e481a7bf7108ea2942cd9ef650
DATA=""; WORK=""; FROM=1; TO=8; DRY=0
while [ $# -gt 0 ]; do case $1 in
  --data) DATA=$2; shift 2;; --work) WORK=$2; shift 2;; --from) FROM=$2; shift 2;; --to) TO=$2; shift 2;;
  --dry-run) DRY=1; shift;; -h|--help) sed -n 2,33p "$0"; exit 0;; *) echo "unknown argument $1 (see bash run_all.sh --help)"; exit 2;; esac; done
WORK=${WORK:-$(cd "$FS/.." && pwd)/tfb_run}
mkdir -p outputs/logs; LOG=$FS/outputs/logs/run_all.log; RES=$FS/outputs/logs/run_all.result
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "RUN_ALL_FAIL step $STEP: $*"; echo "RUN_ALL_FAIL $(date -u +%FT%TZ) step $STEP: $*" > "$RES"; exit 1; }
STEP=0
VW=$WORK/venv-win; PYW=$VW/Scripts/python.exe

# ---------------------------------------------------------------- checks (always)
command -v cygpath >/dev/null 2>&1 || die "run this in Git Bash on Windows (cygpath not found); see the header"
command -v wsl.exe >/dev/null 2>&1 || die "WSL2 is needed for the base build (wsl.exe not found)"
for t in curl unzip sha256sum powershell.exe; do command -v $t >/dev/null 2>&1 || die "$t not found"; done
find_py312() {   # PYTHON312=<path to python.exe 3.12> overrides the search
  if [ -n "${PYTHON312:-}" ]; then echo "$PYTHON312"; return; fi
  if command -v py >/dev/null 2>&1 && py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" >/dev/null 2>&1; then echo "py -3.12"; return; fi
  if command -v uv >/dev/null 2>&1; then p=$(uv python find 3.12 2>/dev/null) && { echo "$p"; return; }; fi
  for p in python3.12 python; do command -v $p >/dev/null 2>&1 && $p -c "import sys; assert sys.version_info[:2] == (3, 12)" >/dev/null 2>&1 && { echo "$p"; return; }; done
  return 1
}
WFS=$(wsl.exe -e wslpath -a "$(cygpath -w "$FS")" 2>/dev/null | tr -d '\r') || true
[ -n "$WFS" ] || die "WSL did not answer (wsl.exe -e wslpath); install Ubuntu 22.04 under WSL2"
WUSER=$(wsl.exe -e whoami 2>/dev/null | tr -d '\r')
VOL_FS=$(cygpath -w "$FS" | cut -c1); VOL_W=$(cygpath -w "$WORK" | cut -c1)
[ "$VOL_FS" = "$VOL_W" ] || die "--work must be on the same drive as the repository ($VOL_FS:), for hard links"
FREE=$(df -BG --output=avail "$FS" | tail -1 | tr -dc 0-9)
say "run_all: repo $FS (WSL $WFS), work $WORK, steps $FROM..$TO, free ${FREE} GB, WSL user $WUSER"
done_() { touch "$WORK/done_$1"; }
want() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }
if [ $DRY = 1 ]; then
  PY312=$(find_py312) && say "Windows Python 3.12: $PY312" || say "MISSING: Windows Python 3.12 (py launcher, uv or python3.12 on PATH) for step 1"
  [ -n "$DATA" ] && { [ -f "$DATA/config/corridors.json" ] && say "data: $DATA (release folder ok)" || say "MISSING: $DATA is not a release folder"; } \
    || say "data: Kaggle download in step 2 (needs your Kaggle credentials)"
  [ "${FREE:-0}" -ge 40 ] || say "WARNING: ${FREE} GB free; about 40 GB are needed"
  wsl.exe -e bash -lc "cat /etc/os-release | grep PRETTY_NAME; nproc; free -g | sed -n 2p; command -v uv || ls ~/.local/bin/uv 2>/dev/null || echo 'uv: will be installed in step 4'"
  for n in 1 2 3 4 5 6 7 8; do want $n && echo "  step $n $( [ -f "$WORK/done_$n" ] && echo '(done earlier, skipped)' || echo '(to run)')"; done
  exit 0
fi
mkdir -p "$WORK" || die "cannot make $WORK"

# ---------------------------------------------------------------- 1 Windows venv
STEP=1
if want 1 && [ ! -f "$WORK/done_1" ]; then
  if [ ! -x "$PYW" ]; then
    PY312=$(find_py312) || die "no Python 3.12 on Windows: install it (python.org, or uv python install 3.12)"
    say "step 1: venv $VW from $PY312"; $PY312 -m venv "$(cygpath -w "$VW")" >> "$LOG" 2>&1 || die "venv"
  fi
  "$PYW" -m pip install -q --disable-pip-version-check -r environment/requirements-windows.txt >> "$LOG" 2>&1 || die "pip install (log $LOG)"
  "$PYW" -c "import numpy, pandas, scipy, lightgbm, torch, pyarrow, yaml; print('win-env ok', numpy.__version__, pandas.__version__, lightgbm.__version__, torch.__version__)" | tee -a "$LOG" || die "imports"
  done_ 1
fi

# ---------------------------------------------------------------- 2 data
STEP=2
if want 2 && [ ! -f "$WORK/done_2" ]; then
  if [ -n "$DATA" ]; then bash pipeline/00_fetch_data.sh --link "$DATA" >> "$LOG" 2>&1 || die "data link"
  else
    "$PYW" -m pip install -q --disable-pip-version-check kaggle >> "$LOG" 2>&1 || die "kaggle CLI install"
    PATH="$VW/Scripts:$PATH" bash pipeline/00_fetch_data.sh >> "$LOG" 2>&1 || die "Kaggle download (credentials? rules accepted?)"
  fi
  [ -f data/raw/kaggle_public/config/corridors.json ] || die "data/raw/kaggle_public has no config/corridors.json"
  say "step 2: data in place"; done_ 2
fi

# ---------------------------------------------------------------- 3 weights
STEP=3
if want 3 && [ ! -f "$WORK/done_3" ]; then
  say "step 3: weight release (about 4.2 GB, sha256 checked)"
  PY="$PYW" bash pipeline/01_fetch_weights.sh --release >> "$LOG" 2>&1 || die "weights (log $LOG)"
  [ -d weights/devroot_extra ] || die "weights/devroot_extra missing after the release fetch"
  done_ 3
fi

# ---------------------------------------------------------------- 4 Linux venv in WSL
STEP=4
if want 4 && [ ! -f "$WORK/done_4" ]; then
  say "step 4: WSL venv /workspace/tfb/venv"
  wsl.exe -u root -e sh -c "mkdir -p /workspace/tfb && chown $WUSER /workspace/tfb" >> "$LOG" 2>&1 || die "mkdir /workspace/tfb in WSL"
  wsl.exe -e bash -lc "set -e; cd '$WFS'; export PATH=\$HOME/.local/bin:\$PATH
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\$HOME/.local/bin:\$PATH
    [ -x /workspace/tfb/venv/bin/python ] || uv venv --python 3.12 /workspace/tfb/venv
    uv pip install --python /workspace/tfb/venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
    uv pip install --python /workspace/tfb/venv/bin/python -r environment/requirements-cpu.txt
    /workspace/tfb/venv/bin/python -c 'import lightgbm, torch, pandas; print(\"linux-env ok\", lightgbm.__version__, torch.__version__, pandas.__version__)'" \
    2>&1 | tee -a "$LOG" | grep -q 'linux-env ok' || die "WSL venv (log $LOG)"
  done_ 4
fi

# ---------------------------------------------------------------- 5 base build C47 (WSL)
STEP=5
if want 5 && [ ! -f "$WORK/done_5" ]; then
  say "step 5: base build C47 in WSL (about 3.5 h; log outputs/logs/base_c47.log)"
  wsl.exe -e bash -lc "cd '$WFS' && export PATH=/workspace/tfb/venv/bin:\$PATH TFB_ASM2_CPU_ONLY=1 && bash pipeline/10_base_c47.sh --s08-fitted --foreground" >> "$LOG" 2>&1
  grep -q VERIFY_PASS outputs/logs/base_c47.result 2>/dev/null || die "base build: $(cat outputs/logs/base_c47.result 2>/dev/null || echo 'no result file')"
  say "step 5: $(head -c 300 outputs/logs/base_c47.result)"; done_ 5
fi

# ---------------------------------------------------------------- 6 devroot
STEP=6
if want 6 && [ ! -f "$WORK/done_6" ]; then
  say "step 6: devroot $WORK/devroot"
  "$PYW" pipeline/15_make_devroot.py "$(cygpath -w "$WORK/devroot")" --extra "$(cygpath -w "$FS/weights/devroot_extra")" >> "$LOG" 2>&1 || die "devroot"
  done_ 6
fi

# ---------------------------------------------------------------- 7 chain rows 19-35
STEP=7
if want 7 && [ ! -f "$WORK/done_7" ]; then
  say "step 7: piece chain rows 19-35 (about 12 h; per step outputs/logs/chain_19_35.tsv)"
  export TFB_OLD_REPO=$WORK/devroot CHAIN_WORK=$WORK/devroot/work PY=$PYW
  bash pipeline/20_pieces.sh >> "$LOG" 2>&1 || die "chain (outputs/logs/chain_19_35.tsv, $LOG)"
  [ -f "$WORK/devroot/work/r3chain/C79.csv" ] || die "chain finished without r3chain/C79.csv"
  cp "$WORK/devroot/work/r3chain/C79.csv" outputs/C79.csv
  done_ 7
fi

# ---------------------------------------------------------------- 8 verify
STEP=8
if want 8; then
  s79=$(sha256sum outputs/C79.csv 2>/dev/null | cut -d' ' -f1); s80=$(sha256sum outputs/C80.csv 2>/dev/null | cut -d' ' -f1)
  v79=$([ "$s79" = $C79_SHA ] && echo BYTE_EQUAL || echo "DIFFERS ($s79)"); v80=$([ "$s80" = $C80_SHA ] && echo BYTE_EQUAL || echo "DIFFERS ($s80)")
  say "step 8: C79 (scored submission) $v79; C80 $v80"
  [ "$s79" = $C79_SHA ] || die "C79 differs from the scored file. Score outputs/C79.csv on Kaggle or compare by value: python pipeline/99_verify.py outputs/C79.csv --ref <your download of the submission> --state-atol 1e-6"
  echo "RUN_ALL_PASS $(date -u +%FT%TZ) C79 $v79 C80 $v80" > "$RES"; say "RUN_ALL_PASS: outputs/C79.csv is the scored submission, byte for byte"
fi
