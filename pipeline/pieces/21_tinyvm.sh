#!/bin/bash
# ADOPTED row 21 (C54 -> C55): ConvNeXt-Tiny ongoing member from the VM retrain with SAVED weights (model.pt sha 43da1ffc),
# replacing the C28 member (weights lost) in the shipped final ongoing decode. R2 piece worker, 2026-10-21.
#   21_tinyvm.sh PARENT.csv OUT.csv [--serve-weights cpu|cuda]
# PARENT.csv OUT.csv = "- -": rows only (every step but the CSV apply; the cheap parity variant).
# PARENT = C54 (c_lmicro2_on_c50.csv, sha b0cbce4e; PARENT_SHA_CHECK=0 skips the check); OUT = C55 (sha 5ebd5de0).
# Original: exp/C_backbone/vm_tiny/swap/swap.py (sha of the chain step: docs/reference_code/asm2_bbmember.py 5da84228,
# config artifacts/claude_ref/c_bb/vm_tiny_swap/cfg_vm.json 1dc98b80 = src/21/cfg_vm.json) + exp/L_lab/a3/a3.py delta.
# Steps (shipped code, unchanged, run in $CODE through asm2_step.py):
#   1. src/21/make_cfg.py         member config (a = .2) with the resolved file paths
#   2. asm2_bbmember.py serve     default: stored probabilities (weights row 21_tinyvm/bb_member_vm.parquet, sha c02523d2 =
#                                 the VM RTX 5070 bf16 serve of model.pt), byte copy with sha check.
#                                 --serve-weights cpu|cuda: serve from model.pt (needs timm 1.0.15 + torchvision 0.22.1 in
#                                 $PY; CPU: fp32, about 2 to 5 min, gives a SIBLING, not C55; cuda: bf16 autocast, bitwise
#                                 only on the VM card/torch 2.7.1+cu128 that made c02523d2).
#   3. asm2_bbmember.py decode    p = .8 f + .2 member per official ongoing window (f = sccnn prob on type A, cnn_ens2 prob on
#                                 B/C), eiou count, selcond .92 vs t-12..t-1 persistence, band raise on A. Gate inside: a = 0
#                                 reproduces the base build's ongoing rows exactly.
#   4. src/21/ongoing_apply.py    OUT = PARENT with the new ongoing queue_pred (65 cells vs C54); asserts PARENT's ongoing rows
#                                 == the base build's ongoing_bb.parquet (C47 = C54 ongoing rows).
# Base-build intermediates read (R1, docs/reference_code/asm2_run_all.sh, run dir $TFB_BASE_RUN, default
# $CODE/artifacts/claude_ref/asm2/run); env overrides point to the stored equivalents for parity runs:
#   T21_BASE_MIX        $BASE_RUN/mix          cnn_ens2.parquet + sccnn_official.parquet    stored: artifacts/claude_ref/c_cold/official/ship_c35/a0.75 (cbfb2850, f965f381)
#   T21_BASE_ONGOING    $BASE_RUN/ongoing.parquet                                         stored: .../ship_c35/a0.75/ongoing.parquet (e176f49d)
#   T21_BASE_ONGOING_BB $BASE_RUN/ongoing_bb.parquet (C28-member rows = C47/C54 ongoing)  stored: .../ship_c35/a0.75/ongoing_bb.parquet
# Writes $WORK/21_tinyvm/{cfg.json, bb_member.parquet, ongoing_bb.parquet (= C55 ongoing rows; stored equivalent
# artifacts/claude_ref/c_bb/vm_tiny_swap/ongoing_bb_vm.parquet a13fff47)}; pieces 22 and 23 read bb_member.parquet and cfg.json.
# Runtime about 3 min (decode ~40 s, CSV apply ~2 min), RAM about 2.5 GiB peak (the 400 MB CSV in pandas).
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/src/21/steps.sh"
[ $# -ge 2 ] || { echo "usage: $0 PARENT.csv OUT.csv [--serve-weights cpu|cuda]"; exit 2; }
PARENT=$1; OUT=$2; SERVE=""
[ "${3:-}" = --serve-weights ] && SERVE=${4:?cpu or cuda}
B_MIX=${T21_BASE_MIX:-$BASE_RUN/mix}; B_ONG=${T21_BASE_ONGOING:-$BASE_RUN/ongoing.parquet}; B_ONGBB=${T21_BASE_ONGOING_BB:-$BASE_RUN/ongoing_bb.parquet}
D=$WORK/21_tinyvm; mkdir -p "$D"; t0=$(date +%s)
[ "$PARENT" = - ] || fsha "$PARENT" b0cbce4e1352a38c41110225f9d04794c616d0a91c8b743ffadc895b722b729a
say "21 start: parent $PARENT, base mix $B_MIX"
if [ -z "$SERVE" ]; then
  glue "$(wp "$SRC/21/make_cfg.py")" "$(wp "$D/cfg.json")" --probs "$(wp "$(wpath 21_tinyvm/bb_member_vm.parquet)")"
else
  glue "$(wp "$SRC/21/make_cfg.py")" "$(wp "$D/cfg.json")" --weights "$(wp "$(wpath 21_tinyvm/model.pt)")" --device "$SERVE" --name c28vm_serve_$SERVE
  [ "$SERVE" = cuda ] && export STEP_CVD=${CUDA_VISIBLE_DEVICES:-0}
fi
step asm2_bbmember.py serve --cfg "$(wp "$D/cfg.json")" --out "$(wp "$D/bb_member.parquet")" | tee "$D/serve.log"
unset STEP_CVD
step asm2_bbmember.py decode --cfg "$(wp "$D/cfg.json")" --member "$(wp "$D/bb_member.parquet")" \
  --base "$(wp "$B_MIX/cnn_ens2.parquet")" --sc "$(wp "$B_MIX/sccnn_official.parquet")" --parent-ongoing "$(wp "$B_ONG")" \
  --out "$(wp "$D/ongoing_bb.parquet")" | tee "$D/decode.log"
if [ "$OUT" = - ]; then say "rows only (OUT = -): no CSV"; else glue "$(wp "$SRC/21/ongoing_apply.py")" "$(wp "$PARENT")" "$(wp "$D/ongoing_bb.parquet")" "$(wp "$OUT")" \
  --old-rows "$(wp "$B_ONGBB")" --report "$(wp "$D/apply.json")"; fi
say "21 done in $(( $(date +%s) - t0 )) s: $OUT"
