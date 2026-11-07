#!/bin/bash
# ADOPTED row 23 (C57 -> C59): L_tinyad A1 = per-date adaptation of the row 21 VM Tiny member (warm dates mh5 analog, cold
# dates XSD analog; recipe exp/L_tinyad/LINE.md, code exp/L_tinyad/tinyad.py sha 23071a30), member
#   p = shipped Tiny + .75 (adapted - unadapted)
# in the shipped final decode of the C57 chain. R2 piece worker, 2026-10-21.
#   23_tinyad.sh PARENT.csv OUT.csv [--member stored]
# PARENT.csv OUT.csv = "- -": rows only (every step but the CSV apply; the cheap parity variant).
# PARENT = C57 (c_a3xsd_on_c55.csv, sha caaf4d8a); OUT = C59 (c_tinyad_on_c57.csv, sha c9a635fd).
# Originals: exp/L_tinyad/tinyad.py merge (member) + exp/L_tinyad/ta_finish.py (tools/rebase/rebase_c57.decode_all on the A3
# chain with the adapted member, a3.py delta on C57). decode_all with P0 = A3 = P1 is exactly asm2_bbmember.py decode on the
# A3 chain (same per-window order, same .8/.2 blend and decoder), which is what runs here.
# Weights: the per-date adapted models WERE saved later (artifacts/claude_ref/l_tinyad/a1_weights/weights, 65 fp32 state dicts,
# 6.9 GiB; exp/L_tinyad/A1_WEIGHTS.md verdict EXACT), but serving them is bitwise only on the RTX 5070 + torch 2.7.1+cu128 bf16
# (GPU busy, none here). Default path = STORED adapted predictions with sha checks:
#   1. src/23/tinyad_merge.py   the 24 per-date files (p_ad, p_un; weights rows 23_tinyad/dates/*) + the row 21 shipped Tiny
#                               member -> the A1 member; must equal the stored member 23_tinyad/tinyad_member.parquet
#                               (a7c7c2a9) bytewise or value for value (laptop: bytewise).
#      --member stored          skip the merge, use the stored member (sha checked).
#   2. asm2_bbmember.py decode  --base/--sc/--parent-ongoing = piece 22's chain ($D22/a0.75), member = step 1, a = .2
#   3. src/21/ongoing_apply.py  OUT = PARENT with these rows (53 cells vs C57); asserts PARENT's ongoing rows == piece 22's rows.
# Rerun recipe (GPU, sibling): final_solution/retrain/tinyad_a1.md.
# Inputs from earlier pieces (env overrides = the stored equivalents for parity runs):
#   T21_MEMBER  $WORK/21_tinyvm/bb_member.parquet       stored: artifacts/claude_ref/c_bb/vm_tiny_swap/bb_member_vm.parquet (c02523d2)
#   T22_CHAIN   $WORK/22_a3cold/a0.75 (cnn_ens2, sccnn_official, ongoing .parquet)  stored: artifacts/claude_ref/l_dose/a3/a3_build/a0.75
#   T22_ROWS    $WORK/22_a3cold/ongoing_bb_c55.parquet  stored: artifacts/claude_ref/l_dose/a3/rebase/ongoing_bb_c55.parquet (ff1113a7)
# No base-build file is read directly (the A3 chain already carries the base build's on/off members).
# Writes $WORK/23_tinyad/{tinyad_member.parquet, cfg.json, ongoing_bb_c59.parquet (stored equivalent
# artifacts/claude_ref/l_tinyad/a1/on_c57/rows_c57.parquet)}. Runtime about 3 min, RAM about 2.5 GiB peak (CSV apply).
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/src/21/steps.sh"
[ $# -ge 2 ] || { echo "usage: $0 PARENT.csv OUT.csv [--member stored]"; exit 2; }
PARENT=$1; OUT=$2; MODE=dates
[ "${3:-}" = --member ] && MODE=${4:?stored}
SHIP=${T21_MEMBER:-$WORK/21_tinyvm/bb_member.parquet}; CH=${T22_CHAIN:-$WORK/22_a3cold/a0.75}; OLDROWS=${T22_ROWS:-$WORK/22_a3cold/ongoing_bb_c55.parquet}
D=$WORK/23_tinyad; mkdir -p "$D"; t0=$(date +%s)
[ "$PARENT" = - ] || fsha "$PARENT" caaf4d8a3740a27db5cb5237164b3850c69761830002649ef4497f06ed7999a5
say "23 start: parent $PARENT, chain $CH, member from $MODE"
if [ "$MODE" = dates ]; then
  glue "$(wp "$SRC/23/tinyad_merge.py")" "$(wp "$SHIP")" "$(wp "$D/tinyad_member.parquet")" --ref "$(wp "$(wpath 23_tinyad/tinyad_member.parquet)")"
else
  [ "$MODE" = stored ] || { echo "--member stored"; exit 2; }
  cp "$(wpath 23_tinyad/tinyad_member.parquet)" "$D/tinyad_member.parquet"
  need_sha "$D/tinyad_member.parquet" a7c7c2a94712c8cc0ad1d5bf32fac292f5df92a501faa8d2bbe49e8e54de3895
fi
glue "$(wp "$SRC/21/make_cfg.py")" "$(wp "$D/cfg.json")" --probs "$(wp "$D/tinyad_member.parquet")" --name tinyad_a1_vm_tiny
step asm2_bbmember.py decode --cfg "$(wp "$D/cfg.json")" --member "$(wp "$D/tinyad_member.parquet")" \
  --base "$(wp "$CH/cnn_ens2.parquet")" --sc "$(wp "$CH/sccnn_official.parquet")" --parent-ongoing "$(wp "$CH/ongoing.parquet")" \
  --out "$(wp "$D/ongoing_bb_c59.parquet")" | tee "$D/decode.log"
if [ "$OUT" = - ]; then say "rows only (OUT = -): no CSV"; else glue "$(wp "$SRC/21/ongoing_apply.py")" "$(wp "$PARENT")" "$(wp "$D/ongoing_bb_c59.parquet")" "$(wp "$OUT")" \
  --old-rows "$(wp "$OLDROWS")" --report "$(wp "$D/apply.json")"; fi
say "23 done in $(( $(date +%s) - t0 )) s: $OUT"
