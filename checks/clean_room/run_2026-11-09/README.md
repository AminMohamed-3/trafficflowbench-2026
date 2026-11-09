# Clean-folder check, 2026-11-08/09

`bash checks/clean_room/run_clean_room.sh` on the author's laptop (16 threads, 15 GB RAM), folder `C:\Data\tfb_cleanroom\run3`.
The development repository path is written `<DEV>` here; from it only the competition data and the Python base install were used.

- `RESULT_first_start`: the first start (2026-11-08 18:39Z). Setup from published material only: export of this repository,
  the `weights-v1` release zips (2084 files, sha256 checked), `devroot_extra.zip`, the base build C47. That chain run stopped at row 32
  (a memory guard set too tight, 3.5 GiB) and its audit showed one helper (`docs/reference_code/phys_lwr_diag.py` and its
  portable copy) loading 10 `tfb` modules from an old development worktree named in a stored config (same code up to line
  endings; every row sha equal). Both fixed in commits 28b7c80b and c6d4a601.
- `RESULT`, `run.log`, `chain_19_35.tsv`: the rerun of rows 19-35 in the same setup (RESUME=1, empty work folder), 2026-11-09
  04:15Z to 15:51Z (696 min). Every build step SHA_EQUAL to development, every serve PASS-ARRAYS, C79 and C80 VERIFY_PASS
  byte_equal=True against the submitted files.
- `open_audit.log`: every Python file open outside the clean folder, venv, data and system folders. Two lines: the stored
  C79 and C80 submission files read by `pipeline/99_verify.py` for the final comparison.
- `atime_reads.tsv`: files of the development repository read during the run by NTFS access time (one: `.gitattributes` at 13:36Z,
  touched by a git process on the laptop; no Python process of the chain opened it, see open_audit.log).
