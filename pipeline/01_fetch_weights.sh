#!/bin/bash
# 01_fetch_weights.sh: put every saved weight / fitted file where the pipeline reads it, each checked by sha256 against
# weights/MANIFEST.csv (= weights/base/*.csv + weights/rows/*.csv; rebuild with `python pipeline/lib/fetch_weights.py merge`).
#
#   bash pipeline/01_fetch_weights.sh --from-repo OLD_REPO [--pieces base,reference]   development laptop (hard links + archive members)
#   bash pipeline/01_fetch_weights.sh --from-dir DIR       [--pieces ...]              unpacked public weight release
#   bash pipeline/01_fetch_weights.sh --check              [--pieces ...]              sha256 of everything in place
#   bash pipeline/01_fetch_weights.sh --release [TAG]                                    public GitHub release (all pieces):
#        downloads the parts in weights/RELEASE.tsv one at a time (curl, sha256 checked), unzips each into weights/download,
#        deletes the zip, then places every file by --from-dir (hard links) and sha-checks all 2,084 files. Needs ~4.2 GB.
#
# Base build (piece base): 890 files, 1,594,271,323 bytes (the build's own input manifest
# artifacts/claude_ref/asm2/serving_inputs_sha256.txt, 888 files, + that manifest + the package index).
# 408 of them are loose files in the development repo (hard links, no disk use); 480 (the 96 x 5 dated adapted CNN weights,
# 837 MiB) exist there only inside the package archives artifacts/claude_ref/asm2/package/e16mh5/*.tar.gz and are extracted.
# Piece reference: the submitted C47 file as C44 full CSV + C47 delta (547 MB, hard links) for pipeline/99_verify.py.
# Piece base_s08fit: the two fitted S08 exports, only for 10_base_c47.sh --s08-fitted.
#
# TODO (user decision 2 of docs/report/FINAL_PLAN_2026-10-21.md): public download. Planned form: the private Kaggle
# weight datasets made public, or GitHub Release assets; then
#   kaggle datasets download -d <owner>/<dataset> -p weights/download --unzip   (user credentials, never stored here)
#   bash pipeline/01_fetch_weights.sh --from-dir weights/download
# The release must keep the repo_path layout of weights/MANIFEST.csv (root pieces under their artifacts/... path).
set -euo pipefail
FS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$FS"
PY=${PY:-python}
ARGS=(); MODE=fetch
while [ $# -gt 0 ]; do case $1 in
  --from-repo|--from-dir) ARGS+=("$1" "$2"); shift 2;;
  --pieces) PIECES=$2; shift 2;;
  --check) MODE=check; shift;;
  --release) MODE=release; TAG=${2:-weights-v1}; [ $# -ge 2 ] && [ "${2#--}" = "$2" ] && shift; shift;;
  *) echo "unknown argument $1"; exit 2;; esac; done
REPO_SLUG=${REPO_SLUG:-AminMohamed-3/trafficflowbench-2026}
if [ $MODE = release ]; then
  PIECES=${PIECES:-}; DL=$FS/weights/download; mkdir -p "$DL"
  tail -n +2 weights/RELEASE.tsv | tr -d '' | while IFS=$'	' read -r name bytes want; do
    url=https://github.com/$REPO_SLUG/releases/download/$TAG/$name
    echo "download $name ($((bytes / 1000000)) MB)"; curl -fL --retry 3 -o "$DL/$name" "$url"
    got=$("$PY" -c "import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],'rb');[h.update(c) for c in iter(lambda:f.read(1<<22),b'')];print(h.hexdigest())" "$DL/$name")
    [ "$got" = "$want" ] || { echo "sha256 mismatch for $name"; exit 3; }
    "$PY" -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DL/$name" "$DL" && rm -f "$DL/$name"
  done || exit 3
  ARGS=(--from-dir "$DL")
fi
PIECES=${PIECES-base,reference}
"$PY" pipeline/lib/fetch_weights.py merge
if [ $MODE = check ]; then exec "$PY" pipeline/lib/fetch_weights.py check --pieces "$PIECES"; fi
[ ${#ARGS[@]} -gt 0 ] || { echo "give --from-repo OLD_REPO or --from-dir DIR (public download: see TODO in this file)"; exit 2; }
"$PY" pipeline/lib/fetch_weights.py fetch "${ARGS[@]}" --pieces "$PIECES"
"$PY" pipeline/lib/fetch_weights.py check --pieces "$PIECES"
[ $MODE = release ] && echo "weights in place; weights/download holds the unpacked release (hard-linked, safe to delete)"
exit 0
