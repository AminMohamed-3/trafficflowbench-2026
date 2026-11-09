# TrafficFlowBench 2026 (IEEE BigData Cup): 8th place solution

Author: Amin Mohamed. Private score .89874 (8th), public .90088.

- Write-up: `report/KAGGLE_WRITEUP.md`; full method report: `report/REPORT.md` (PDF: `report/REPORT.pdf`, built by `report/build_pdf.py`).
- Interactive site (all board reads, the idea registry, the analysis): https://aminmohamed-3.github.io/trafficflowbench-2026/
- Video, the approach in 3 minutes (narrated, 1080p): https://aminmohamed-3.github.io/trafficflowbench-2026/#video (built from `report/video/` by `report/video/build_video.py`; narration voice: ElevenLabs).
- Weights: the `weights-v1` release of this repository (4.1 GB in 3 zip parts plus `devroot_extra.zip`, 0.7 GB of stored intermediate files; all sha256-checked by the fetch script).
- Verified on a clean folder (2026-11-09): from this repository, the release assets and the competition data alone, the chain
  rebuilt C79 byte-equal to the scored submission and C80 byte-equal to our stored file, with an audit showing no file read from outside that folder
  (`checks/clean_room/run_clean_room.sh`; logs in `checks/clean_room/run_2026-11-09/`).

Code that rebuilds our submitted files from the organisers' release and our saved weights.
Competition: Kaggle `2026-ieee-big-data-traffic-flow-bench`. Score: S_total = .35 state + .15 onset + .15 ongoing
+ .15 physics + .20 ODME. Public board = validation split (March 2031), private = April 2031.

## Results

| file | what | public S_total | private S_total | rebuilt by |
|---|---|---|---|---|
| C47 | base build: state chain + B6 imputer (24k it), onset Q03 + Q0436, ongoing CNN ensemble + cold-start pools + ConvNeXt member, ODME NNLS | .89776 | .89697 | `pipeline/10_base_c47.sh` |
| C79 | C47 + ADOPTED rows 19-35 except the April rows of 34 and 35: **the scored file** (sha256 104c2c6f...) | .90088 | **.89874** | `pipeline/20_pieces.sh` |
| C80 | C79 + the April rows of rows 34 and 35 (sha256 3e27d66c...; built after our last upload, never uploaded) | .90088 (same March rows) | not scored | `pipeline/20_pieces.sh` |

The reproducibility gate of the award is 1% of the private score (about .009). C47 alone is within .002 of C79 on the private board.

## Hardware and runtime

- Development: laptop, 16 logical CPUs, 15.3 GB RAM, NVIDIA RTX 3050 Ti 4 GB, Windows 11 + WSL2 Ubuntu 22.04; rented
  Vast.ai GPU VMs and Kaggle kernels for training (GPU hours per piece in `retrain/`).
- The base build needs NO GPU. Measured on the laptop under WSL2 (at most 10 threads), 2026-10-12/13: caches 20 min, S08
  refit 238 min at 10 threads, onset + ODME 21 min, ongoing 196 min with 3 cold-start members at a time (peak 7.5 GiB),
  Task 1 chain 38-53 min (peak 4.6 GiB), assembly 8-11 min: about 9 h. `10_base_c47.sh` defaults to 2 cold-start members
  at a time (lower RAM, about +1.8 h); `--reuse-cache` saves 20 min, `--s08-fitted` saves about 4 h.
- Disk: release 12 GB; fetched weights 1.6 GB (+ .55 GB reference); build peak about 2.6 to 7.5 GB (flags).

## Single entry script

```bash
bash run_all.sh --data /path/to/competition/release   # or no --data: downloads it with your Kaggle credentials
cat outputs/logs/run_all.result                        # RUN_ALL_PASS: outputs/C79.csv and outputs/C80.csv match by sha256
```

`run_all.sh` runs the eight steps below in order, from the competition data and the public `weights-v1` release, and
checks the result: (1) Windows venv from `environment/requirements-windows.txt`, (2) data, (3) weights (sha256-checked),
(4) Linux venv in WSL from `environment/requirements-cpu.txt`, (5) base build C47 inside WSL, (6) development layout,
(7) piece chain rows 19-35, (8) sha256 of C79 and C80. It runs in Git Bash on Windows 10/11 with WSL2 (Ubuntu 22.04),
the split under which the files were made: the base build accepts only the pinned Linux runtime, the piece chain uses
Windows paths. Each finished step leaves a marker, so a stopped run resumes with `--from N`; `--dry-run` checks the
machine and prints the plan; `bash run_all.sh --help` lists the options. Needs about 40 GB free disk on the repository's
drive, 16 GB RAM, no GPU; about 15 h on our laptop. Each stage was verified on its own (base build 2026-10-21, piece chain
in the clean folder 2026-11-09, both byte- or value-equal); the script chains exactly those commands.

## Commands, step by step (reproduce C79, the scored file, and C80)

Five steps, run from the repository root. Steps 0-2 run on Linux (or WSL2); steps 3-4 run in Git Bash on Windows with
the Windows venv (the piece scripts use `cygpath` and `python.exe`), on the same checkout.

```bash
# 0. environment (Linux, Python 3.12; venv path is required, see environment/README.md)
uv venv --python 3.12 /workspace/tfb/venv
uv pip install --python /workspace/tfb/venv/bin/python -r environment/requirements-cpu.txt
export PATH=/workspace/tfb/venv/bin:$PATH
# 1. data and weights
bash pipeline/00_fetch_data.sh                      # Kaggle CLI with your own credentials (or --link DIR)
bash pipeline/01_fetch_weights.sh --release         # downloads the 3 weight parts + devroot_extra.zip, checks sha256, unpacks
#   (or --from-dir DIR --pieces "" for an unpacked release: "" = all pieces, the default is base,reference only)
# 2. base build C47 (Linux / WSL, CPU only, self-detaching; log and result in outputs/logs/)
bash pipeline/10_base_c47.sh --s08-fitted           # optional --reuse-cache DIR
cat outputs/logs/base_c47.result                    # RESULT ... VERIFY_PASS ... -> outputs/c47/final_A.csv (+ .state.parquet)
# 3. ADOPTED rows 19-35 chained on the fresh C47 (Git Bash): each piece reads the previous piece's output.
#    The pieces run development code at its development paths: first lay out that folder ("devroot", same volume = hard links)
python pipeline/15_make_devroot.py C:/tfb/devroot --extra weights/devroot_extra
export TFB_OLD_REPO=C:/tfb/devroot CHAIN_WORK=C:/tfb/devroot/work PY=<your venv>/Scripts/python.exe
bash pipeline/20_pieces.sh --dry-run                # checks inputs and stored references, prints the plan
bash pipeline/20_pieces.sh                          # -> $CHAIN_WORK/r3chain/C79.csv (scored file) and outputs/C80.csv
# 4. verify. C79 is the selected and scored Kaggle submission (public .90088, private .89874); its sha256 must be
#    104c2c6fce3393b14ea219ce8ae3ada6230ec358bba37f137335615eb4db1f04 (byte-equal in our clean-folder run, 2026-11-09).
#    C80 (never uploaded) must be 3e27d66cd2b3cfc4371fd218617ebf08a0fa16e481a7bf7108ea2942cd9ef650.
sha256sum "$CHAIN_WORK/r3chain/C79.csv" outputs/C80.csv
#    value check against a copy of the submission downloaded from Kaggle (C79_SUBMITTED.csv):
python pipeline/99_verify.py "$CHAIN_WORK/r3chain/C79.csv" --ref C79_SUBMITTED.csv --ref-sha 104c2c6fce3393b14ea219ce8ae3ada6230ec358bba37f137335615eb4db1f04 --state-atol 1e-8
# steps 3 + 4 unattended (waits for a free laptop lock, writes outputs/logs/chain_19_35.result: RESULT ... or FAILED ...):
nohup bash pipeline/run_chain_when_free.sh > /dev/null 2>&1 &
```

Runtimes (laptop: 16 logical CPUs, 15.3 GB RAM, CPU only; from `outputs/logs/base_c47.result` and
`pipeline/pieces/STATUS_19_27.md`, `STATUS_28_35.md`):

| step | runtime | notes |
|---|---|---|
| 10 base C47 | 203 min measured (WSL, `--s08-fitted`, cache reuse); about 7 to 11.5 h without those flags | peak RAM ~7.5 GiB with 3 cold-start members at a time |
| 19 gap2 U-Net member | 2 min | 48 panel weights served on CPU |
| 20 L_micro state build | 31 min | full Task 1 state build |
| 21-23 ongoing members | 1-2 min each | stored member predictions (below) |
| 24 B6R S2 | about 45 min (191 min measured under lock and RAM contention) | 60 members served + state build |
| 25 GAPW gap rows | about 40 min | hooked state build |
| 26 BUNDLE18 | about 80 min | 60 members + 2 state builds |
| 27 L_gap2 gap rows | about 2 to 10 min | G1 serve + G3 LightGBM serve |
| 28-35 | about 8.5 h | 11 state builds of ~33 min, 8 hook builds, Q-ACCUM ~30 min |
| rows 19-35 in all | about 12 to 13 h | peak work disk about 3 GB; keeps a 13.5 GiB free-disk floor |

The fresh base is value-equal to C47, not byte-equal (state rows differ by at most 4.2e-13 km/h / 3.6e-11 vph). So
`20_pieces.sh` does not stop on a sha difference: it records for every step SHA_EQUAL, VALUE_EQUAL_TOL (queue and
ODME exact, state within 1e-8), DIFF or NOT_COMPARED against the stored development file of that step, and stops only
when a piece fails.

Stored-prediction exceptions on the path to C80 (sha-pinned, rerun recipes in `retrain/`):
- row 21 (ConvNeXt-Tiny VM member): stored probabilities of the bf16 GPU serve of the saved `model.pt`; a CPU fp32 serve
  gives a sibling file, not C80 (`retrain/tiny_vm.md`).
- row 22 (L_dose A3 cold members): stored member predictions of the saved per-date fits; a CPU re-serve differs by ~1e-6
  (`retrain/a3_cold.md`).
- row 23 (L_tinyad A1): stored per-date adapted probabilities; the saved weights serve bitwise only on an RTX 5070 with
  torch 2.7.1+cu128 bf16 (`retrain/tinyad_a1.md`, ~.8 GPU h).
- row 33 (Q-ACCUM): no weights, per-date online fits inside 3 Kaggle kernels; the default path uses the stored kernel
  outputs (`retrain/qaccum.md`; a kernel rerun reproduces the decoded cells, not the bits).

Expected verification result (measured 2026-10-13, Linux CPU against the Windows CPU run that made the submitted file):
every queue row (174,000) and every ODME row (70,708) equal; of 6,735,795 state rows 77,942 speed and 101,773 flow
values differ in the last bits (max 4.2e-13 km/h, 3.6e-11 vph), inside the build's own tolerance 1e-8. The file is not
byte-identical (line endings and those last bits); `99_verify.py` reports both.

## Known exceptions (stated, not hidden)

- ConvNeXt-Tiny ongoing member in the base build (ADOPTED row 6): served from stored probabilities (sha-pinned
  `artifacts/claude_ref/c_bb/serve/convnext_tiny_up2_e8_s0_serve0228.parquet`); its fine-tuned weights were lost with a VM.
  Row 21 replaces it with a retrain whose weights are saved.
- Q-ACCUM (row 33): per-date online fits inside 3 Kaggle kernels, no weights; default path uses the stored kernel
  outputs (sha checked). Tiny adaptation A1 (row 23): per-date weights not saved; stored probabilities with sha, rerun
  recipe in `retrain/`.
- S0631 TabPFN (CUDA only) and hedge file B are off in the default build.

## Layout notes

- Code keeps the development repository's relative paths (`docs/reference_code/`, `src/tfb/`, `scripts/`, `tools/`):
  the scripts find the root as `parents[2]` of their own file, put `ROOT/src` and `ROOT/docs/reference_code` on
  `sys.path`, and the shipped adaptation receipts bind source hashes. Files are byte copies (`checks/code_files.csv`).
- `PLAN.md` and an empty `.git/` are root markers: `tfb.paths.project_root` accepts `TFB_ROOT` only with a PLAN.md, and
  several modules find the root by walking up to `.git`. After `git init` the real `.git` takes that role.
- `checks/open_audit/` (on PYTHONPATH during the build) logs any Python-level file open outside the repository.
- Compliance and disclosures: `docs/COMPLIANCE.md`, `docs/DISCLOSURES.md`.
