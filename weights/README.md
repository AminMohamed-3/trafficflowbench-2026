# Weights and fitted files

`MANIFEST.csv` lists every saved weight / fitted file (repo_path, sha256, bytes, old_repo_path, piece, location) and is
rebuilt from `base/*.csv` (base build, worker R1) and `rows/*.csv` (pieces 19-35, workers R2/R3) by
`python pipeline/lib/fetch_weights.py merge`.
- location `root`: the file sits at `<repo root>/<repo_path>` (the base build binds `artifacts/...` paths; `reference/` = the
  submitted C47 file as C44 + delta, for verification only).
- location `weights`: the file sits at `weights/<repo_path>`.
- old_repo_path: where the file is in the development repository; `<archive>::<member>` = a member of a package archive
  there (the 480 dated adapted CNN weights of the base build exist only inside artifacts/claude_ref/asm2/package/e16mh5/*.tar.gz).
Public download: the `weights-v1` release of this repository, 3 zip parts listed with sha256 in `RELEASE.tsv`
(`RELEASE_PARTS.csv` maps every file to its part). `bash pipeline/01_fetch_weights.sh --release` downloads, checks and
unpacks them; `--from-dir DIR` takes an already unpacked copy.
