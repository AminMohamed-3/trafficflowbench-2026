# Q-ACCUM (ADOPTED row 33): rebuild and rerun notes

Q-ACCUM = DW-ADAPT + ONG-TODR-FIT + MARGIN-ADAPT. Each is a per-date online fit of the ongoing (queue) members on the target
month. The bundle sums their k1 - k0 member deltas on the A3 cold members (MARGIN also on the warm mh5 members), then runs the
A3 chain and the w8 .2 decode. Result on the board: C76 = C75 + 11 ongoing queue cells (March 4 windows, April 6 windows).
C76 read .90079, +.00006 vs C75 (forecast +.00007). Private: the same declared method on April rows, forecast about +.00007.

There are no saved weights. The fits are refit per date inside the serve kernels from the base ongoing weights. So the frozen
inputs of this piece are the stored kernel OUTPUTS (member parquets + json receipts), listed with sha256 in
`final_solution/weights/rows/33.csv` (200 files):

| source kernel | outputs used | files |
|---|---|---|
| tfb-kg-dwadapt-serve (T4 x2) | l_dwadapt/kg_serve/.../l_dwadapt/official/{k0,k1} | 40 |
| tfb-kg-todr-serve (T4 x2) | l_todr/kg_serve/.../l_todr/official/{k0,k1} | 40 |
| tfb-kg-margin-inst (T4 x2) | l_margin/kg_inst/.../l_margin/{official,official_warm}/{k0,k1,k0b} | 120 |

## Default path (used by `pipeline/pieces/33_qaccum.sh`, laptop CPU, ~30 min, no GPU)
1. `lib/weights.py check 33`: every stored kernel output == its sha256.
2. `src/33/qa_build.py` (copy of exp/L_qaccum/qa_build.py): members, chain, rows, then delta on C75 -> C76 (sha fc3576fc).
   Input folders can be pointed elsewhere with env `QA_DW`, `QA_TODR`, `QA_MARGIN` (default: the old paths under TFB_ROOT).
   `QA_OUT` must lie inside TFB_ROOT (mg_build uses paths relative to the root).

## Rerun of the kernels (only if the stored outputs are lost or a full from-scratch rebuild is wanted)
Kernels (self-contained: code blob + recipe, no data, no weights, no predictions) are copied in
`final_solution/kaggle_kernels/q_accum/` (SHA256SUMS there):
- `tfb-kg-dwadapt-serve/`, `tfb-kg-todr-serve/`, `tfb-kg-margin-inst/` (kernel-metadata.json: private, competition attached,
  dataset `tfb-dwadapt-base-weights`, internet ON for the pinned uv venv, GPU T4 x2).
- The private weights-only dataset `tfb-dwadapt-base-weights` (41 files, 70.2 MiB: 10 e16 ongoing members, 10 Dec-cutoff
  members, A2 speed heads): sha list `tfb-dwadapt-base-weights.sha256.json`, staging script `make_dataset.sh` (original
  exp/L_dwadapt/make_dataset.sh, run from the old-layout root).
- Pull and unpack: `dw_pull.sh`, `todr_pull.sh`, `mg_pull.sh` (originals; they unpack into artifacts/claude_ref/l_*/kg_*).

Steps (the account owner runs these; kaggle CLI, from the old-layout root):
```
.venv/Scripts/kaggle.exe kernels push -p final_solution/kaggle_kernels/q_accum/tfb-kg-dwadapt-serve
.venv/Scripts/kaggle.exe kernels push -p final_solution/kaggle_kernels/q_accum/tfb-kg-todr-serve
.venv/Scripts/kaggle.exe kernels push -p final_solution/kaggle_kernels/q_accum/tfb-kg-margin-inst
# when COMPLETE: bash <copy of>/dw_pull.sh serve ; todr_pull.sh serve ; mg_pull.sh inst
# then QA_DW / QA_TODR / QA_MARGIN -> the pulled official folders, and run pipeline/pieces/33_qaccum.sh
```
The kernels check the code blob against code_manifest.json; they need the old repo files listed there unchanged.

### GPU cost (measured, the October 19 runs)
| kernel | jobs | wall (summary minutes_total) | GPU quota |
|---|---|---|---|
| tfb-kg-dwadapt-serve | 20 (10 members x k0/k1, DET mode) | 314 min | ~5.2 GPU session-h |
| tfb-kg-todr-serve | 20 (DET mode) | 291 min | ~4.9 GPU session-h |
| tfb-kg-margin-inst | 30 (20 member-split + 10 OFF250; nondet mode + k0b) | 246 min | ~4.1 GPU session-h |
Total about 14 GPU session-h of the 30 h weekly Kaggle quota. A CPU laptop alternative for DW and TODR exists (dw_official.py,
~12 process-h each, see exp/L_dwadapt/PUSH.md section 5), not used for the shipped files.

### Reproducibility of a rerun
- DW-ADAPT and TODR ran in deterministic mode (TFB_ASM2_DET, poison checks bitwise on ctl_s0 / sc_s0). A rerun on the same
  image is expected to give the same parquets; a different CUDA / torch build may change the last bits.
- MARGIN ran NON-deterministic (det mode was 7.4x slower; repeat diff 1.4e-05 in one process). Its noise reference
  k0b - k0 changed 0 decoded cells (artifacts/claude_ref/l_margin/build/rows_noise.json), so k1 - k0 is used as served.
  A MARGIN rerun will NOT be bitwise equal; it is expected to give the same 11 decoded queue cells (0-cell noise band), not
  the same C76 sha. Check: the decoded official queue cells of the rerun C76 vs the stored C76.
- So the exact C76 sha fc3576fc is guaranteed only from the stored kernel outputs (default path).
