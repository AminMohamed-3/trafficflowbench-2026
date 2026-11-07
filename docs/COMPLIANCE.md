# Compliance (DRAFT, 2026-10-21)

Sources in the development repository: RULES.md, DECISIONS_DIGEST.md, docs/report/TASK2_ONLINE_AUDIT_2026-10-11.md,
exp/C_audit748789/AUDIT.md (organiser thread 748789), docs/report/ADOPTED.md.

## 1. Data use per task

| task | rule we applied | where it is enforced |
|---|---|---|
| Task 1 state, Task 3 physics | offline: any released observation of the split at any time of day, plus train in full. Station constants are split-local (never from another month). | structure of the Task 1 chain (one split per fit) |
| Task 4 ODME | offline: the split's own released counts and prior, network files, train. NNLS ridge, lambda .05. | `asm2_qo.py` |
| Task 2 queue (onset, ongoing) | ONLINE. A forecast with origin T on date d may use values stamped at or before T on day d, released data of earlier days of the same split, and train. Nothing after T (horizon ramp_states, mainline after the blanked buffer), no later date, no Task 1 output that used such values. April windows may use released March data and train, also to fit or select a model, if everything used is stamped at or before the window's T (organiser, forum thread 748789, 2026-10-15). March windows never use April files. | accessors that cut at T and assert (`src/tfb/lane_q/windows.py`), future-poison tests (below) |

Forbidden and not used anywhere: submission_id values or order, row order, window_id numbering, ID numbering beyond
the documented maps, generator seeds, file timestamps, other teams' scores, organiser-only files, matching a hidden day
to a train day to copy values.

## 2. Audits

- Task 2 online audit of the build (2026-10-11): every same-day read is cut at step T or earlier; cross-day reads use
  earlier dates of the same split or train; no piece loads two evaluation splits together; no Task 1/3/4 output feeds a
  Task 2 input; no Task 2 model reads ramp_states. Violations found: none.
- Organiser ruling 2026-10-15: everything used for a window, including model or threshold selection, is stamped at or
  before its T, and April windows may use released March data. April (private) choices use train months and released
  March data only (exp/C_audit748789/AUDIT.md sections 3 and 4). A March board verdict of a METHOD may choose it for
  April rows (declared in the report).
- U1 (selection-implied inputs): the ongoing decode uses how the organisers selected windows (persistence IoU at most
  .9), computed from data at or before T; disclosed in the report.

## 3. Guards in the code

| guard | where | what it checks |
|---|---|---|
| online accessors | `src/tfb/lane_q/windows.py` | every Task 2 read filters `timestamp <= T` and asserts; prior-day archive asserts date < window date |
| cold-start XSD poison | `c_cold_split.py` (runs in the build), asserted in `c_cold_ship.py` | later dates and day d after each checkpoint set to a poison value: every refit bitwise equal |
| CNN future poison | `qcnn_predict_adapt.py`, `ong2_sccnn_official.py` (`--poison`) | later dates and day d after the last origin poisoned: identical decode / max dprob < 1e-6. TODO (audit gap 1): the default build serves without `--poison`; the check ran in the serving run on the VM |
| Q0436 poison | `scripts/q0436_official.py` | day d after t and later days set to NaN: features identical |
| shipped input check | `asm2_inputs.py` | sha256 of all 888 fitted inputs before any computation |
| receipts | `c_e16_adapt.py` / `qcnn_predict_adapt.py` | each dated adapted weight file is bound to base/config/source hashes, date, seed, recipe, device, threads |
| open audit | `checks/open_audit/sitecustomize.py` | the build opens nothing outside this repository (Python-level opens) |

Open guard items from the audit (not leaks; tests that would catch a future leak): poison also ramp_flow, window_history,
window_index; poison per window origin rather than after the last origin of the day; a mechanical test that a March
step never opens an April file.

## 4. Leaderboard use

Board reads were aggregate verdicts on a method or a scalar, one change per upload, with a written forecast first
(5 uploads per day; 91 submissions, 90 with a forecast, in report/data/board_calibration.csv). No read was decoded into
the truth of single cells or windows.

## 5. Reproducibility

One pipeline from the raw release (`pipeline/`), pinned environment (`environment/`), configs and code in git, saved
weights by sha256 manifest (`weights/MANIFEST.csv`), verification against the submitted file (`pipeline/99_verify.py`).
