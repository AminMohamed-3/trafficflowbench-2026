#!/bin/bash
# ADOPTED row 22 (C55 -> C57): L_dose A3 = speed term inside both XSD stages of the cold-date per-date fits, put into the
# shipped ongoing chain in place of the shipped XSD cold members, decoded with the row 21 VM Tiny member. R2 piece worker,
# 2026-10-21.
#   22_a3cold.sh PARENT.csv OUT.csv
# PARENT.csv OUT.csv = "- -": rows only (every step but the CSV apply; the cheap parity variant).
# PARENT = C55 (c_tinyvm_on_c54.csv, sha 5ebd5de0); OUT = C57 (c_a3xsd_on_c55.csv, sha caaf4d8a).
# Originals: exp/L_dose/a3_finish.sh (python part: exp/L_dose/a1_build.py chain with A3.COLD_MEMBERS = the A3 members,
# exp/L_lab/a3/a3.py chain) then tools/rebase/rebase_c55.py (decode with the VM member config, a3.py delta on C55).
# "Saved weights" of this piece: the A3 per-date fits ARE saved (artifacts/claude_ref/l_dose/a3/weights: 530 state dicts
# <member>/<split>/<date>/{xall,xsd_H###}.pt, 1.9 GiB, sha list weights_sha256.txt); the members
# (weights rows 22_a3cold/members/{ctl,sc}_s{0..4}.parquet + .json receipts) are their served per-window probabilities
# (exp/L_dose/a3_cold.py official serves from the reloaded saved files, VM CPU, deterministic, 2 threads; columns p_un,
# p_serve, p_xall, p_xsd). This script reads the members (stored predictions of the saved weights, sha checked): a CPU
# re-serve of the weights differs by ~1e-6 on another CPU (A3_RESULT.md: laptop vs VM), so it would be a sibling.
# See final_solution/retrain/a3_cold.md for the refit and the re-serve.
# Steps (shipped code, unchanged, run in $CODE through asm2_step.py; COLD_PARITY_TOL=1e-5 as a3_finish.sh):
#   1. copy the 10 members + receipts to $D/cold (c_cold_ship build writes cold_windows.csv / build_parity.json there)
#   2. c_cold_ship.py build $D/cold BASE/on BASE/off $D/on_xsd     cold windows: p_xsd (sccnn: 5-seed mean), ctl
#      queue_pred by eiou_mask; checks poison/ft-parity receipts, un parity vs BASE/off
#   3. c_noadapt_blend.py $D/on_xsd BASE/off 0.75 $D/a0.75         .75 adapted + .25 unadapted per member file
#   4. qcnn_avg_probs.py $D/a0.75/cnn_ens2.csv cnn_s0..4             seed mean + eiou
#   5. ong2_probe.py build --sc --noT                                 ongoing rows before the Tiny member ($D/a0.75/ongoing.parquet)
#   6. asm2_bbmember.py decode with the row 21 member and config (a = .2) -> $D/ongoing_bb_c55.parquet (stored equivalent
#      artifacts/claude_ref/l_dose/a3/rebase/ongoing_bb_c55.parquet ff1113a7). The C28-member decode that a1_build also ran
#      (a3_build/a0.75/ongoing_bb.parquet) is not on the path to C57 and is skipped.
#   7. src/21/ongoing_apply.py: OUT = PARENT with these rows (79 cells vs C55); asserts PARENT's ongoing rows == row 21's rows.
# Inputs from the base build (R1 asm2_run_all.sh run dir $TFB_BASE_RUN) / earlier pieces, with the stored equivalents for parity:
#   T22_BASE_ON   $BASE_RUN/on   served mh5 per-date adapted members (cnn_s0..4.parquet, sccnn_official.parquet), BEFORE the
#                                shipped cold XSD   stored: artifacts/claude_ref/asm2/e16/s01234_5_6e-4_none_out/rows/on
#   T22_BASE_OFF  $BASE_RUN/off  unadapted members  stored: artifacts/claude_ref/asm2/e16/s01234_5_6e-4_none_out/rows/off
#   T21_MEMBER    $WORK/21_tinyvm/bb_member.parquet  stored: artifacts/claude_ref/c_bb/vm_tiny_swap/bb_member_vm.parquet (c02523d2)
#   T21_CFG       $WORK/21_tinyvm/cfg.json           stored: src/21/cfg_vm.json
#   T21_ROWS      $WORK/21_tinyvm/ongoing_bb.parquet stored: artifacts/claude_ref/c_bb/vm_tiny_swap/ongoing_bb_vm.parquet (a13fff47)
# Writes $WORK/22_a3cold/{cold/, on_xsd/, a0.75/ (chain probabilities, read by piece 23), ongoing_bb_c55.parquet (= C57
# ongoing rows, read by piece 23)}. Runtime about 5 to 8 min, RAM about 2.5 GiB peak (CSV apply).
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/src/21/steps.sh"
[ $# -ge 2 ] || { echo "usage: $0 PARENT.csv OUT.csv"; exit 2; }
PARENT=$1; OUT=$2
B_ON=${T22_BASE_ON:-$BASE_RUN/on}; B_OFF=${T22_BASE_OFF:-$BASE_RUN/off}
T21=$WORK/21_tinyvm; MEMBER=${T21_MEMBER:-$T21/bb_member.parquet}; CFG=${T21_CFG:-$T21/cfg.json}; OLDROWS=${T21_ROWS:-$T21/ongoing_bb.parquet}
D=$WORK/22_a3cold; t0=$(date +%s)
[ "$PARENT" = - ] || fsha "$PARENT" 5ebd5de0438517e03a719ecf88545acd26152287441e66265bb3d81081ff3020
need_sha "$MEMBER" c02523d2e9464f9e4bda710ad889375c21c44dab1df9ee30d52b41d4cb7ddf6a
say "22 start: parent $PARENT, base on $B_ON, off $B_OFF"
rm -rf "$D/cold" "$D/on_xsd" "$D/a0.75"; mkdir -p "$D/cold" "$D/a0.75"
for k in ctl sc; do for i in 0 1 2 3 4; do for e in "" .json; do
  cp "$(wpath 22_a3cold/members/${k}_s$i.parquet$e)" "$D/cold/${k}_s$i.parquet$e"; done; done; done
COLD_PARITY_TOL=1e-5 step c_cold_ship.py build "$(wp "$D/cold")" "$(wp "$B_ON")" "$(wp "$B_OFF")" "$(wp "$D/on_xsd")" | tee "$D/chain.log"
grep -q COLD_XSD_BUILD_OK "$D/chain.log" || { say "cold XSD build failed"; exit 1; }
step c_noadapt_blend.py "$(wp "$D/on_xsd")" "$(wp "$B_OFF")" 0.75 "$(wp "$D/a0.75")" | tee -a "$D/chain.log"
cs=(); for s in 0 1 2 3 4; do cs+=("$(wp "$D/a0.75/cnn_s$s.parquet")"); done
step qcnn_avg_probs.py "$(wp "$D/a0.75/cnn_ens2.csv")" "${cs[@]}" | tee -a "$D/chain.log"
step ong2_probe.py build "--sc=$(wp "$D/a0.75/sccnn_official.parquet")" --noT "--base-probs=$(wp "$D/a0.75/cnn_ens2.parquet")" \
  "--ongoing-out=$(wp "$D/a0.75/ongoing.parquet")" | tee -a "$D/chain.log"
step asm2_bbmember.py decode --cfg "$(wp "$CFG")" --member "$(wp "$MEMBER")" --base "$(wp "$D/a0.75/cnn_ens2.parquet")" \
  --sc "$(wp "$D/a0.75/sccnn_official.parquet")" --parent-ongoing "$(wp "$D/a0.75/ongoing.parquet")" \
  --out "$(wp "$D/ongoing_bb_c55.parquet")" | tee -a "$D/chain.log"
if [ "$OUT" = - ]; then say "rows only (OUT = -): no CSV"; else glue "$(wp "$SRC/21/ongoing_apply.py")" "$(wp "$PARENT")" "$(wp "$D/ongoing_bb_c55.parquet")" "$(wp "$OUT")" \
  --old-rows "$(wp "$OLDROWS")" --report "$(wp "$D/apply.json")"; fi
say "22 done in $(( $(date +%s) - t0 )) s: $OUT"
