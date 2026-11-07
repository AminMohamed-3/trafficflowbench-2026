#!/bin/bash
# L_todr: C pulls one finished TODR kernel (todr_pull.tar, todr_summary.json, kg_log.jsonl), unpacks it into
# artifacts/claude_ref/l_todr/kg_<preset>/ and prints the verdict lines (exp/L_todr/PUSH.md). Helpers never run this.
#   bash exp/L_todr/todr_pull.sh chk|inst|serve [p2]
set -u
cd "$(dirname "$0")/../.." || exit 1
U=aminmohamed333345; P=${1:?preset}; R=${2:-}
K=tfb-kg-todr-$P${R:+-p2}
DST=artifacts/claude_ref/l_todr/kg_${P}${R:+_p2}
KG() { PYTHONIOENCODING=utf-8 timeout 1800 .venv/Scripts/kaggle.exe "$@" 2>&1; }
st=$(KG kernels status $U/$K | tail -1); echo "status: $st"
[[ $st == *complete* || $st == *COMPLETE* || $st == *error* || $st == *ERROR* ]] || { echo "not finished yet"; exit 2; }
FREE=$(df -BM --output=avail /c | tail -1 | tr -dc 0-9); [ "${FREE:-0}" -ge 14336 ] || { echo "laptop disk under 14 GiB free: not pulling"; exit 1; }
mkdir -p $DST
KG kernels output $U/$K -p $DST -q --page-size 200 --file-pattern "(todr_pull\.tar|todr_summary\.json|kg_log\.jsonl)$" > /dev/null
[ -f $DST/todr_summary.json ] || { echo "no todr_summary.json in the output (see $DST/kg_log.jsonl)"; exit 1; }
tar -xf $DST/todr_pull.tar -C $DST
.venv_tfb/Scripts/python.exe -c "import json; s = json.load(open('$DST/todr_summary.json')); print('summary', json.dumps({k: s[k] for k in ('minutes_total', 'jobs', 'done', 'left', 'error', 'worker_rc', 'code_sha')}), 'failed', len(s['failed']), s['failed'][:3])"
grep -h -o 'DW_REF_CHECK [A-Z]* [0-9]*\|TODR_REF_CHECK [A-Z]* [0-9]*\|TODR_FTCHECK [A-Z]*[^"]*\|"kind": "poison"[^}]*\|"kind": "serve_poison[^}]*\|"step": "fatal"[^}]*' $DST/kg_logs/*.log $DST/kg_log.jsonl $DST/logs_todr/*.log 2>/dev/null | sort | uniq
if [ $P = chk ]; then
  PYTHONPATH=src .venv_tfb/Scripts/python.exe - <<PY
import pandas as pd, numpy as np, json
K = ["window_id", "timestamp", "link_id"]
for kk, refs in (("k0", (("artifacts/claude_ref/l_todr/parity/ctl_s0_k0_0302.parquet", "laptop k0"), ("artifacts/claude_ref/l_dose/a3/members/ctl_s0.parquet", "shipped A3 (VM CPU)"))),
                 ("k1", (("artifacts/claude_ref/l_todr/parity/ctl_s0_k1_0302_poison.parquet", "laptop k1"),))):
    a = pd.read_parquet(f"$DST/artifacts/claude_ref/l_todr/parity/kg_ctl_s0_{kk}_0302.parquet")
    for ref, tag in refs:
        s = pd.read_parquet(ref).set_index(K).reindex(pd.MultiIndex.from_frame(a[K]))
        print("TODR_CHK_PARITY", kk, "vs", tag, json.dumps({c: float(np.abs(a[c].to_numpy(np.float64) - s[c].to_numpy(np.float64)).max()) for c in ("p_un", "p_xall", "p_xsd")}))
PY
  grep -q "DW_REF_CHECK PASS" $DST/kg_logs/check.log && grep -q "TODR_REF_CHECK PASS" $DST/kg_logs/check_ramp.log && grep -q "TODR_FTCHECK PASS" $DST/kg_logs/sample.log \
    && grep -q "TODR_OFFICIAL_DONE ctl_s0 .*k1_0302" $DST/kg_logs/sample.log && echo "TODR_CHK PASS (digests == laptop, k0 bitwise == A3 on Kaggle CPU, k1 poison 0)" || { echo "TODR_CHK FAIL"; exit 1; }
  exit 0
fi
echo "left > 0: .venv_tfb/Scripts/python.exe exp/L_todr/todr_make.py $P --resume && bash exp/L_todr/todr_push.sh $P p2; then pull p2 (part 2 holds part 1's outputs too)"
