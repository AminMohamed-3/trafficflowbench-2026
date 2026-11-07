#!/bin/bash
# L_dwadapt: C pulls one finished DW-ADAPT kernel (dw_pull.tar ~20-60 MB, dw_summary.json, kg_log.jsonl), unpacks it into
# artifacts/claude_ref/l_dwadapt/kg_<preset>/ and prints the verdict lines (exp/L_dwadapt/PUSH.md). Helpers never run this.
#   bash exp/L_dwadapt/dw_pull.sh chk|inst|serve [p2]
set -u
cd "$(dirname "$0")/../.." || exit 1
U=aminmohamed333345; P=${1:?preset}; R=${2:-}
K=tfb-kg-dwadapt-$P${R:+-p2}
DST=artifacts/claude_ref/l_dwadapt/kg_${P}${R:+_p2}
KG() { PYTHONIOENCODING=utf-8 timeout 1800 .venv/Scripts/kaggle.exe "$@" 2>&1; }
st=$(KG kernels status $U/$K | tail -1); echo "status: $st"
[[ $st == *complete* || $st == *COMPLETE* || $st == *error* || $st == *ERROR* ]] || { echo "not finished yet"; exit 2; }
FREE=$(df -BM --output=avail /c | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -ge 14336 ] || { echo "laptop disk under 14 GiB free: not pulling"; exit 1; }
mkdir -p $DST
KG kernels output $U/$K -p $DST -q --page-size 200 --file-pattern "(dw_pull\.tar|dw_summary\.json|kg_log\.jsonl)$" > /dev/null
[ -f $DST/dw_summary.json ] || { echo "no dw_summary.json in the output (see $DST/kg_log.jsonl)"; exit 1; }
tar -xf $DST/dw_pull.tar -C $DST
.venv_tfb/Scripts/python.exe -c "import json; s = json.load(open('$DST/dw_summary.json')); print('summary', json.dumps({k: s[k] for k in ('minutes_total', 'jobs', 'done', 'left', 'error', 'worker_rc', 'code_sha')}), 'failed', len(s['failed']), s['failed'][:3])"
grep -h -o 'DW_REF_CHECK [A-Z]* [0-9]*\|DW_FTCHECK [A-Z]*[^"]*\|"step": "fatal"[^}]*' $DST/kg_logs/*.log $DST/kg_log.jsonl 2>/dev/null | sort | uniq
if [ $P = chk ]; then
  PYTHONPATH=src .venv_tfb/Scripts/python.exe - <<PY
import pandas as pd, numpy as np, json
a = pd.read_parquet("$DST/artifacts/claude_ref/l_dwadapt/parity/kg_ctl_s0_k0_0302.parquet")
for ref, tag in (("artifacts/claude_ref/l_dwadapt/parity/ctl_s0_k0_0302.parquet", "laptop k0"), ("artifacts/claude_ref/l_dose/a3/members/ctl_s0.parquet", "shipped A3 (VM CPU)")):
    s = pd.read_parquet(ref); K = ["window_id", "timestamp", "link_id"]; s = s.set_index(K).reindex(pd.MultiIndex.from_frame(a[K]))
    print("DW_CHK_PARITY vs", tag, json.dumps({c: float(np.abs(a[c].to_numpy(np.float64) - s[c].to_numpy(np.float64)).max()) for c in ("p_un", "p_xall", "p_xsd")}))
PY
  grep -q "DW_REF_CHECK PASS" $DST/kg_logs/check.log && grep -q "DW_FTCHECK PASS" $DST/kg_logs/sample.log && echo "DW_CHK PASS (digests == laptop, k0 bitwise == A3 on Kaggle CPU)" || { echo "DW_CHK FAIL"; exit 1; }
  exit 0
fi
echo "left > 0: .venv_tfb/Scripts/python.exe exp/L_dwadapt/dw_make.py $P --resume && bash exp/L_dwadapt/dw_push.sh $P p2; then pull p2 (outputs of both parts: pass both folders to the reader)"
