# Rows 28 to 35 (C69 -> C80): status of the pieces (worker R3, 2026-10-21)

Chain runner: `bash pipeline/pieces/run_28_35.sh [C66.csv C69.csv]` (default: stored C66 / C69, sha checked) -> `outputs/C80.csv`,
sha 3e27d66cd2b3cfc4371fd218617ebf08a0fa16e481a7bf7108ea2942cd9ef650. Every chain file is checked against its stored sha256
(`src/r3lib/expected.tsv`) before the next piece runs. Parity runner: `parity/r3_parity.sh glue|serve|build|full`, results in
`parity/r3_results.tsv`.

Parity columns: GLUE = stored parents + stored build parquets (sha == original receipts), every stack / transfer / gap / Q-ACCUM step
re-run now from code and fitted files; ORIG = the original build parity of the line (receipt / build_parity.json; the build code
here is a path-only edit of the same code); SERVE = member arrays served now from the saved weights == stored receipt sha.

| row | piece | script | input sha | output sha | parity | runtime (from weights, laptop CPU) | weights |
|---|---|---|---|---|---|---|---|
| 28 | FV-FT March | 28_fvft_mar.sh | C66 f1a5278f, C69 d1ada3d6 | FVFT_MAR_ON_C66 d5167911, C70 5181d5ed | GLUE PASS-SHA (FVFT_MAR_ON_C66, C70); ORIG build parity PASS (l_fvft/mar) | ~45 min (serve 3 + build 33 + stack 3 + disjoint 4) | rows/28.csv: 60 files, 54.8 MiB |
| 29 | BUNDLE19 (gap3 + April FV-FT) | 29_bundle19.sh | C66, C69, C70 | FVFT_BOTH_ON_C69 6f7ce2b6, GAP3_ON_C69 819b26d1, GAP3_ON_C70 26bd6344, C71 45c9cd41 | GLUE PASS-SHA (4 files); gap3 p1 / p2 / pC69 EXACT re-run now; ORIG build parity PASS (l_fvft/apr) | ~60 min (build 33, gap3 ~10, 2 disjoint) | rows/29.csv: 61 files, 54.8 MiB (gap3 knobs + April members) |
| 30 | FV2 | 30_fv2.sh | C70, C71 | FV2_MAR_ON_C70 b994683a, C72 4315b61c, FV2_BOTH_ON_C70 042b3bce, C72P d046c382 | GLUE PASS-SHA (4 files); ORIG build parity PASS (l_fvft/fv2) | ~80 min (2 builds) | rows/30.csv: 120 files, 109.5 MiB |
| 31 | FV-INEL (W .3) | 31_fvinel.sh | C70, C71, FV2_BOTH_ON_C70 | FVI_MAR_ON_C70 01bff667, C73 219c0f93, C73P 5827d508, FVI_BOTH_ON_C70 8dd8dafa, C74P d4376e1d | GLUE PASS-SHA (5 files); ORIG build parity PASS (l_fvinel) | ~85 min (2 builds) | rows/31.csv: 122 files, 109.6 MiB |
| 32 | GAP-ON-FV | 32_gapfv.sh | C73P, C74P | GAPFV_ON_C73P 7b936ed5, C75 15280925 | GLUE PASS-SHA (GAPFV_ON_C73P, C75); hook vs FV build gap rows EXACT (0 of 64,720 + 63,745 mismatches) and official p1 / p2 / pP EXACT re-run now (stored hook files); hook build itself: ORIG GFV73 build parity | ~70 min (8 hook builds, official, disjoint) | none new (rows 26/27/29-31 fitted files) |
| 33 | Q-ACCUM | 33_qaccum.sh | C75 | C76 fc3576fc | PASS-SHA C76 re-run now from the stored kernel outputs (members, chain, rows, delta; 11 cells, guard PASS) | ~30 min | rows/33.csv: 200 stored Kaggle kernel outputs, 81.1 MiB (no weights; retrain/qaccum.md) |
| 34 | FV-INEL W1.0 | 34_fvinel_w10.sh | C70, FVI_MAR_ON_C70, C75, C76 | W10_MAR_ON_C70 eda9f5ea, W10_ON_C75 67d02edf, C77 10abf5ee, W10_BOTH_ON_C70 fd2c9e88, C78 f5b8e9e0 | GLUE PASS-SHA (5 files; confirms the reconstructed W10_ON_C75 and C77 steps); ORIG build parity PASS (l_fvinel w10) | ~85 min (2 builds) | rows/34.csv: 122 files, 109.6 MiB |
| 35 | S7C-INEL | 35_s7c_inel.sh | C77, C78 | C79 104c2c6f, S7C_APR_ON_C78 70aa5aab, X_C79_APRW10 4a124cfc, C80 3e27d66c | GLUE PASS-SHA (4 files, C80 3e27d66c); ORIG ISI parity_parmar / parapr PASS | ~80 min (2 builds) | rows/35.csv: 4 tree files, 36.3 MiB |
| 36 | S7C-FOLD (slot) | 36_s7c_fold.sh | - | - | not adopted (S7CF_GATE KEEP-FV2FIT) | - | - |

Glue parity 28-35 run 2026-10-21 (stored C66 / C69 / build parquets): all 27 output files PASS-SHA, C80 == 3e27d66c.
Measured glue step times: stack / transfer 2.5-3 min each, gap3 ~6 min, GAP-ON-FV official ~1 min, Q-ACCUM 3.6 min.
Serve parity run 2026-10-21: all 8 member sets (FVFT / FV2 / FVI / W10, March and April; 30 members each) served now on CPU from the
saved weights give speed + flow arrays EXACT == the stored receipts, 146-157 s per set.
Build parity (single state builds from saved weights, `parity/r3_parity.sh build`): NOT run here. C's instruction (2026-10-21):
replaced by the chained run pipeline/run_chain_when_free.sh (fresh C47 -> C80), which runs every build and compares every step
with the stored chain files. One B_FVFT_MAR build was started and stopped during its serve step (no lock taken, folders removed).

Whole chain from weights: about 8.5 h laptop CPU (11 state builds ~33 min each, peak RSS ~3.4 GB; 4 threads), peak WORK disk ~3 GB.

Scores (board reads, ADOPTED.md): C70 .90037, C71 .90044, C72 .90046, C73 .90071, C75 .90073, C76 .90079, C77 fc .90081
(unread), C79 .90088 (BEST READ). C80 = C79 + April rows (private only): public .90088 expected exactly (validation rows equal
C79), private forecast = C79 private + April W1.0 + S7c April (each declared-method April piece, forecasts +.00002 / +.00001).

Notes
- Transfer steps by C that had no script (C70, GAP3_ON_C70, C71, C72, C73, C75, C77, X, C80: tools/rebase/stack_disjoint.py;
  C72P, C73P, C74P, C78: exp/L_fvft/fv2_apr_on_c72.py; W10_ON_C75: the exp/L_fvinel/fvi_lw_on_c75.py form) were reconstructed
  from the logs and jsons; the glue parity confirms each one by sha.
- C72 / C72P are side files (board reads of row 30); the lineage to C80 goes C71 -> C73 -> C73P (FV2 April rows) -> C74P.
- Q-ACCUM: exact C76 only from the stored kernel outputs; a kernel rerun reproduces the decoded cells, not the bits (MARGIN
  ran non-deterministic; 0-cell noise band).
