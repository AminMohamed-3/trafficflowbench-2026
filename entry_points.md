# Entry points

All commands run from the repository root on Linux with the pinned venv first on PATH
(`export PATH=/workspace/tfb/venv/bin:$PATH`). Nothing needs a GPU.

| step | command | reads | writes | time |
|---|---|---|---|---|
| data | `bash pipeline/00_fetch_data.sh` (or `--link DIR`) | Kaggle competition `2026-ieee-big-data-traffic-flow-bench` (your credentials) | `data/raw/kaggle_public/` | download |
| weights | `bash pipeline/01_fetch_weights.sh --release` (GitHub release weights-v1), `--from-dir DIR` or `--from-repo OLD` (development laptop) | `weights/MANIFEST.csv` | `artifacts/...` (890 files, 1.59 GB), `reference/` (547 MB) | minutes |
| base build | `bash pipeline/10_base_c47.sh [--reuse-cache DIR] [--s08-fitted] [--foreground]` | release, `artifacts/` | `outputs/c47/final_A.csv` (+ manifest, state parquet, build receipt), `outputs/c47/run/`, `outputs/logs/` | 7 to 11.5 h CPU |
| verify | `python pipeline/99_verify.py outputs/c47/final_A.csv --ref-parent reference/c_b6l12k_on_c43.csv --ref-delta reference/c_b6l24k_on_c44.delta.csv.gz --ref-sha 1606b1c59e9a59f46605db19a25e4b83ba493b47e46b3ff41fbdeec255c2a61a` (run by 10_base_c47.sh at the end) | rebuilt file, reference | `outputs/c47/verify.json` | 2 min |
| pieces 19-35 | `pipeline/pieces/` (workers R2, R3) | `outputs/c47/`, `weights/` | the final file | see pieces |

## Inside the base build (`docs/reference_code/asm2_run_all.sh`, default flags = C47)

Every step runs as `python docs/reference_code/asm2_step.py SCRIPT ARGS` (= `python SCRIPT ARGS`, clean process exit).

| # | step | script | output (under `artifacts/claude_ref/asm2/run/`, moved to `outputs/c47/run/`) |
|---|---|---|---|
| 1 | environment pins + sha of 888 fitted inputs | `asm2_inputs.py --environment` | `environment.json` |
| 2 | panel caches from the release | `scripts/build_cache.py --panel all --split all` | `artifacts/cache/` |
| 3 | Task 2 onset Q03 refit + Task 4 ODME (NNLS, lambda .05) | `asm2_qo.py` | `qo.parquet`, `qo_onset.parquet`, `qo_odme.parquet` |
| 4 | onset site choices Q0436 | `scripts/q0436_official.py build --serving-only`, `asm2_onset_rows.py` | `q0436/`, `onset.parquet` |
| 5 | ongoing types A/B/C | `ong2_official_types.py` | |
| 6 | ongoing CNN members, 5 seeds x (base, sccnn), per-date adapted weights served + unadapted | `c_e16_adapt.py serve`, `c_noadapt_ong.py off` | `on/`, `off/` |
| 7 | cold-start pools XSD (10 members, CPU) | `c_cold_split.py official`, `c_cold_ship.py build` | `cold_xsd/`, `on_xsd/` |
| 8 | mix a = .75, ensemble, decode (eIoU, sccnn type A, selcond cap, band raise) | `c_noadapt_blend.py`, `qcnn_avg_probs.py`, `ong2_probe.py build --noT` | `mix/`, `ongoing.parquet` |
| 9 | ConvNeXt-Tiny member (a = .2, stored probabilities, sha-pinned) | `asm2_bbmember.py serve/decode` | `ongoing_bb.parquet` |
| 10 | S08 stack input refit, 127 leaves x 4800 rounds, seeds 4701-4703 | `c_s08_ship.py data/fit/export` | `s08/l127_r4800/official_*.parquet` |
| 11 | B6 imputer members (24k iterations, 3 seeds, 10 panels, 2 splits) served on CPU | `asm2_b6l.py serve` | `b6l/imp_*.npz` (60) |
| 12 | Task 1 chain (K5 -> S0424/S0426/S0447 -> phys5 stage 4, gap v3) and assembly of all rows | `asm2_final.py` | the submission CSV |
| 13 | build receipt | `asm2_build_receipt.py` | `<out>.build.json` |
