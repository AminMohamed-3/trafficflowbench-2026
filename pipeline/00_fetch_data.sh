#!/bin/bash
# 00_fetch_data.sh: the organiser's competition release under data/raw/kaggle_public (unchanged; ~12 GB unpacked).
#
#   bash pipeline/00_fetch_data.sh                 download with the Kaggle CLI (needs YOUR credentials: run `kaggle auth login`
#                                                  or place your own kaggle.json; nothing is stored in this repo)
#   bash pipeline/00_fetch_data.sh --link DIR      use a release that is already on disk (DIR = the kaggle_public folder);
#                                                  makes data/raw/kaggle_public a link to it (Windows: junction, Linux: symlink)
#
# The build reads only data/raw/kaggle_public (tfb.paths.release_root) and checks the calendars and layout itself
# (tfb.data.build_cache). Competition: 2026-ieee-big-data-traffic-flow-bench (accept its rules on Kaggle first).
set -euo pipefail
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$FS"
COMP=2026-ieee-big-data-traffic-flow-bench
DEST=data/raw/kaggle_public
if [ -e "$DEST" ]; then echo "exists: $DEST (remove it to fetch again)"; ls "$DEST" | head; exit 0; fi
mkdir -p data/raw
if [ "${1:-}" = --link ]; then
  SRC=${2:?--link needs the kaggle_public directory}
  [ -f "$SRC/config/corridors.json" ] || { echo "not a release folder (no config/corridors.json): $SRC"; exit 2; }
  if command -v cygpath >/dev/null 2>&1; then   # Git Bash on Windows: directory junction (no admin rights needed)
    powershell.exe -NoProfile -Command "New-Item -ItemType Junction -Path '$(cygpath -w "$FS/$DEST")' -Target '$(cygpath -w "$(cd "$SRC" && pwd)")' | Out-Null"
  else
    ln -s "$(cd "$SRC" && pwd)" "$DEST"
  fi
  ls "$DEST/config"; exit 0
fi
KAGGLE=${KAGGLE:-kaggle}
"$KAGGLE" competitions download -c "$COMP" -p data/raw/download
mkdir -p "$DEST"
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "data/raw/download/$COMP.zip" "$DEST"
[ -f "$DEST/config/corridors.json" ] || { echo "unexpected archive layout: look into $DEST and move the release root there"; exit 1; }
echo "release in $DEST"
