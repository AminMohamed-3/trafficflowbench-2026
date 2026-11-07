# sourced by every piece script and parity runner of rows 19 to 27 (worker R2, 2026-10-21).
# Paths: FS = final_solution root; OLD = development repo root (laptop mode; on a clean machine set TFB_OLD_REPO
# to a folder holding the fetched data and leave weights in $FS/weights).
# Laptop lock (same protocol as exp/L_isinel/isi_lock.sh): exp/L_b6/logs/laptop.lock.d with pid; stale = pid gone.
set -euo pipefail
FS=${FS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OLD=${TFB_OLD_REPO:-$(cd "$FS/.." && pwd)}
export FS TFB_OLD_REPO=$OLD
PY=${PY:-$OLD/.venv_tfb/Scripts/python.exe}
[ -x "$PY" ] || PY=python
W="$PY $FS/pipeline/pieces/lib/weights.py"
OUTD=${OUTD:-$FS/outputs}
WORK=${WORK:-$FS/work}
mkdir -p "$OUTD" "$WORK"
say() { echo "$(date -u +%FT%TZ) $*"; }
wpy() { "$PY" "$FS/pipeline/pieces/lib/weights.py" "$@"; }   # use this, not $W (PY path has spaces)
wpath() { wpy path "$1"; }                     # resolved weights file (sha checked)
LOCK=${LOCK:-$OLD/exp/L_b6/logs/laptop.lock.d}
unlock() { [ -d "$LOCK" ] && [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; return 0; }
lock() { local k=0 op
  until mkdir "$LOCK" 2>/dev/null; do
    op=$(cat "$LOCK/pid" 2>/dev/null || true)
    if [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null; then say "stale lock of pid $op removed"; rm -rf "$LOCK"; continue; fi
    [ $((k % 30)) -eq 0 ] && say "waiting for the laptop lock (pid $op, $(cat "$LOCK/owner" 2>/dev/null))"; k=$((k + 1)); sleep 20
  done
  echo $$ > "$LOCK/pid"; echo "final_solution R2 ($1)" > "$LOCK/owner"; say "lock taken ($1)"; }
diskok() { local f; f=$("$PY" -c "import shutil; print(int(shutil.disk_usage('C:/').free / 2**30 * 100))"); [ "$f" -ge "${1:-1350}" ]; }
waitdisk() { local k=0; until diskok "${1:-1350}"; do [ $((k % 10)) -eq 0 ] && say "waiting: disk below floor (${2:-})"; k=$((k + 1)); sleep 60; done; }
memok() { local f; f=$("$PY" -c "import psutil; print(int(psutil.virtual_memory().available / 2**30 * 10))"); [ "$f" -ge "$1" ]; }
waitmem() { local k=0; until memok "$1"; do [ $((k % 10)) -eq 0 ] && say "waiting: available RAM below $(($1 / 10)).$(($1 % 10)) GiB (${2:-})"; k=$((k + 1)); sleep 60; done; }
sha() { "$PY" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');[h.update(c) for c in iter(lambda:f.read(1<<22),b'')];print(h.hexdigest())" "$1"; }
need_sha() { local s; s=$(sha "$1"); [ "$s" = "$2" ] || { say "input sha mismatch $1: $s != $2"; exit 3; }; }
