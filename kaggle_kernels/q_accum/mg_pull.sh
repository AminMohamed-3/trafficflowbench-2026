#!/bin/bash
# L_margin: C pulls one finished MARGIN-ADAPT kernel (mg_pull.tar, mg_summary.json, kg_log.jsonl), unpacks it into
# artifacts/claude_ref/l_margin/kg_<preset>[_p2]/ and prints the mode choice and check lines (exp/L_margin/PUSH.md). Helpers never run this.
#   bash exp/L_margin/mg_pull.sh inst|serve [p2]
set -u
cd "$(dirname "$0")/../.." || exit 1
U=aminmohamed333345; P=${1:?preset}; R=${2:-}
K=tfb-kg-margin-$P${R:+-p2}
DST=artifacts/claude_ref/l_margin/kg_${P}${R:+_p2}
KG() { PYTHONIOENCODING=utf-8 timeout 1800 .venv/Scripts/kaggle.exe "$@" 2>&1; }
st=$(KG kernels status $U/$K | tail -1); echo "status: $st"
[[ $st == *complete* || $st == *COMPLETE* || $st == *error* || $st == *ERROR* ]] || { echo "not finished yet"; exit 2; }
FREE=$(df -BM --output=avail /c | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -ge 14336 ] || { echo "laptop disk under 14 GiB free: not pulling"; exit 1; }
mkdir -p $DST
KG kernels output $U/$K -p $DST -q --page-size 200 --file-pattern "(mg_pull\.tar|mg_summary\.json|kg_log\.jsonl)$" > /dev/null
[ -f $DST/mg_summary.json ] || { echo "no mg_summary.json in the output (see $DST/kg_log.jsonl)"; exit 1; }
tar -xf $DST/mg_pull.tar -C $DST
.venv_tfb/Scripts/python.exe exp/L_margin/mg_summary.py $DST
echo "left > 0: .venv_tfb/Scripts/python.exe exp/L_margin/mg_make.py $P --resume && bash exp/L_margin/mg_push.sh $P p2; then pull p2 (it holds part 1's outputs too)"
echo "next: bash exp/L_margin/mg_gate_run.sh $DST   |   after MG_GATE GO: bash exp/L_margin/mg_build_run.sh <pull folder with serve-grade official outputs>"
