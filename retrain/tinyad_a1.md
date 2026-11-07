# Row 23: L_tinyad A1 (per-date adaptation of the VM ConvNeXt-Tiny member), rerun recipe

Default build path (pipeline/pieces/23_tinyad.sh): STORED adapted predictions, sha checked. The 24 per-date files
(`weights/rows/23.csv`, `23_tinyad/dates/<split>_<date>.parquet` + `.json`; columns window_id, timestamp, link_id, p_un, p_ad,
kind, H) are merged into the member `p = shipped Tiny + .75 (p_ad - p_un)` (src/23/tinyad_merge.py = exp/L_tinyad/tinyad.py
merge(), paths only), which must equal the stored member `tinyad_member.parquet` (sha a7c7c2a9; laptop 2026-10-21: bytewise).
The decode of the C57 chain with this member gives C59 (sha c9a635fd).

## What exists
- Base model: row 21 `model.pt` (sha 43da1ffc, `weights/rows/21.csv`), config.json of the same run.
- Per-date adapted models: SAVED on 2026-10-17 (exp/L_tinyad/A1_WEIGHTS.md, verdict EXACT): development repo
  `artifacts/claude_ref/l_tinyad/a1_weights/weights/` = 65 fp32 state dicts (53 serving: `<split>_<date>_warm.pt` on warm
  dates, `<split>_<date>_H###.pt` per hour checkpoint on cold dates; 12 `_xall.pt` stage-1 models the cold H fits start
  from), 7.40 GB, sha + state key + seeds + pool per file in `a1_weights/RECEIPT.json`, sha list `a1_weights/sha256.txt`.
  NOT listed in weights/rows/23.csv (7.4 GB; the default path does not read them). Publishing them is the user's call.
- Reload proof (same VM, fresh process, models rebuilt from the saved files only): 24 of 24 dates bitwise equal to the
  A1 rows, member sha a7c7c2a9 (a1_weights/reload.json, verdict.json).

## Serve from the saved per-date weights (GPU, about 1 min of GPU after data load)
`exp/L_tinyad/aw_tinyad.py reload` then `merge` (venv: torch 2.7.1+cu128, timm 1.0.15, torchvision 0.22.1+cu128,
huggingface_hub 0.30.2, safetensors 0.5.3; the `l-tiny` venv of exp/C_backbone/vm_tiny/run.sh env_setup). Serving a
window at t on a cold date uses `<split>_<date>_H<12 floor(t/12)>.pt`, a warm date `<split>_<date>_warm.pt`;
tinyad.predict_date with bf16 autocast. Bitwise only on an NVIDIA GeForce RTX 5070 with torch 2.7.1+cu128 (CUDA 12.8)
bf16 autocast, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic algorithms; on other hardware a sibling (prob differences of
order 1e-3, a few decoded cells may flip). CPU fp32 serving: a sibling as well.

## Full rerun of the adaptation (gives a sibling unless the hardware above is used)
- Code: exp/L_tinyad/tinyad.py (sha 23071a3061696efc39c652fa7e2855cf414553ac0bc15fdb71a18ec68d19c477, unchanged),
  driver exp/L_tinyad/tinyad_vm.sh (smoke -> run -> merge -> sha list) or exp/L_tinyad/aw_vm.sh (run with every adapted
  model saved -> reload -> merge -> verdict). Recipe exp/L_tinyad/LINE.md, fixed before any test:
  - warm date (> 2 earlier same-day-type dates): qcnn_predict_adapt.finetune form, same-type pool of dates < d, cap none,
    bs 32, AdamW wd 1e-4, 5 epochs, lr 6e-4 head / 1.5e-4 backbone, BN running stats frozen;
  - cold date (<= 2): stage 1 xall on all earlier split dates 3 epochs 3e-4 / 7.5e-5 (seed k), stage 2 per hour checkpoint
    H = 12 floor(t/12) on same-day windows t' + 6 <= H, 3 epochs same lr (seed 1000 k + H, flag RNG 20000 + 100 k + H/12);
  - seed per date = official-date index ko, flag RNG default_rng(1000 + ko); bf16 autocast on CUDA;
  - output p = shipped + .75 (adapted - unadapted).
- Inputs: row 21 model.pt + config.json, the shipped Tiny probabilities (sha c02523d2), the qcnn cubes of qcnn_data.py
  (windows.parquet sha a6efcbc9..., as exp/C_backbone/vm_tiny), official types (ong2_official_types.py).
- Online rule and checks inside: zero-epoch parity (unadapted = shipped bitwise), future poison on the middle warm and cold
  date of each split (checks_<split>.json, smoke.json).
- Hardware and time (A1, 2026-10-16, VM RTX 5070 12 GB): smoke 5 min + run 41 min + merge (46 min total, about .8 GPU h);
  the saved-weights rerun (aw_vm.sh) took 22 min (run 20 min, reload .8 min). VRAM about 5 GB.
- Expected result: the same member bitwise on that hardware (shown twice); elsewhere a sibling: per-date fits drift by
  bf16/cuDNN numerics, about tens of decoded cells may differ from C59 (the piece moves 53 cells).
- Score (ADOPTED.md row 23): board C59 .89956 (+.00049 vs C57); private expected +.0002 (-.0003 to +.0007). A sibling
  carries the same expectation, not the same exact board read.
