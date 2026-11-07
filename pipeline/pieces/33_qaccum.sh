#!/usr/bin/env bash
# ADOPTED row 33, Q-ACCUM (C76 = .90079): DW-ADAPT + ONG-TODR-FIT + MARGIN-ADAPT k1-k0 member deltas summed on the A3 cold members
# (+ MARGIN warm), A3 chain + w8 .2 decode; 11 ongoing queue cells. Worker R3, 2026-10-21.
# NO WEIGHTS: the three adaptations are per-date online fits inside Kaggle kernels. DEFAULT PATH = the stored kernel outputs
# (sha256 checked against weights/rows/33.csv), then the unchanged build (src/33/qa_build.py = exp/L_qaccum/qa_build.py, folders
# redirected). Rerun of the kernels: final_solution/retrain/qaccum.md (outputs then via QA_DW / QA_TODR / QA_MARGIN).
# In:  C75 (row 32); stored pulls l_dwadapt/kg_serve, l_todr/kg_serve, l_margin/kg_inst; A3 members, C59 rows, e16 on/ files
#      (rows 22/23 and base, read in place from TFB_ROOT).
# Steps (originals: qa_build.py members; exp/L_qaccum/qa_run.sh = chain + rows; qa_delta.sh = delta on C75):
#  members -> chain (A3, a0.75) -> rows (parity A3 chain -> C59 rows 0 cells) -> delta on C75 -> candidate C76 fc3576fc
# Out: CH/C76.csv. Runtime ~30 min CPU.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"; PIECE=33; T0=$(date +%s)
"$PY" "$D/lib/weights.py" check 33 || r3fail "stored kernel outputs (weights/rows/33.csv)"
Q=$R3W/qaccum; rm -rf "$Q"; mkdir -p "$Q"; export QA_OUT=$Q
lock "Q-ACCUM members/chain/rows/delta ~30 min"; waitmem 35 qaccum
for st in members chain rows; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" "$D/src/33/qa_build.py" $st > "$Q/$st.log" 2>&1 || { unlock; r3fail "qa $st: $(tail -n 1 "$Q/$st.log" | cut -c1-200)"; }
  say "qa $st: $(grep -E 'QA_PIECES|MG_CHAIN_DONE|MG_ROWS' "$Q/$st.log" | tail -n 1 | cut -c1-250)"
done
PW=$(cygpath -m "$(cf C75)")
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" "$D/src/33/qa_build.py" delta "$PW" c_qaccum_on_c75 > "$Q/delta.log" 2>&1 || { unlock; r3fail "qa delta: $(tail -n 1 "$Q/delta.log" | cut -c1-200)"; }
unlock
# the delta step deletes its full csv after writing the delta (its json holds full_sha256): expand the delta onto C75 (sha checked)
QD=$Q/build/delta/c_qaccum_on_c75.delta.csv.gz; [ -f "$QD" ] || r3fail "qa: delta not found ($QD)"
say "qa delta: $(grep -o 'A3_DELTA OK.*' "$Q/delta.log" | cut -c1-250)"
rm -f "$R3W/C76.csv"; waitdisk 1350 "expand C76"
"$PY" "$TFB_ROOT/tools/probe.py" expand "$PW" "$QD" "$R3W/C76.csv" --sha "$(xsha C76)" >&2 || r3fail "qa: expand C76"
put C76 "$R3W/C76.csv"
rm -rf "$Q/build/tmp"
say "PIECE_33_DONE C76 $(( $(date +%s) - T0 )) s"
