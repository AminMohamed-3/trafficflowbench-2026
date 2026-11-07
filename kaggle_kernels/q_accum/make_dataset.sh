#!/usr/bin/env bash
# L_dwadapt: stage ONE private weights-only Kaggle dataset (CLAUDE.md 10-18 rule: weights, fitted parameter files, receipts/sha lists
# only; no data, no predictions). Hard links (no extra disk) of the BASE weights the DW-ADAPT jobs start from:
#   10 shipped e16 ongoing members (qcnn/run_full_s0..4_npt_e16, ong2/run_sc_s0..4_e16: model.pt + config.json),
#   10 Dec-cutoff OFF250 members (c_bb/runs/jft_{sc,ctl}16_s0..4: model.pt + config.json),
#   the A2 speed heads (l_dose/a2/warm/heads.pt, sha b842413a...).
# File names flatten the repo path with "__" (the kernel links them back). sha256 checked against exp/L_dwadapt/dw_ref.json.
# Usage (repo root): bash exp/L_dwadapt/make_dataset.sh   -> DW_DATASET_OK + the create command (C runs it)
set -euo pipefail
D=exp/L_dwadapt/dataset; mkdir -p $D
FILES=$(.venv_tfb/Scripts/python.exe -c "import sys; sys.path.insert(0, 'exp/L_dwadapt'); import dw_ref; print(' '.join(dw_ref.WEIGHTS))")
for f in $FILES; do n=$(echo $f | sed 's#^artifacts/claude_ref/##; s#/#__#g'); [ -e $D/$n ] || cp -l $f $D/$n; done
cat > $D/dataset-metadata.json <<'JSON'
{"title": "tfb-dwadapt-base-weights", "id": "aminmohamed333345/tfb-dwadapt-base-weights", "licenses": [{"name": "CC0-1.0"}],
 "description": "TrafficFlowBench L_dwadapt: base weights only (10 e16 ongoing members, 10 Dec-cutoff members, A2 speed heads) + configs. Private. No data, no predictions."}
JSON
.venv_tfb/Scripts/python.exe - <<'PY'
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, "exp/L_dwadapt"); import dw_ref
D = Path("exp/L_dwadapt/dataset"); ref = json.loads(Path("exp/L_dwadapt/dw_ref.json").read_text()); lst = {}
for f in dw_ref.WEIGHTS:
    n = f.replace("artifacts/claude_ref/", "").replace("/", "__"); h = hashlib.sha256((D / n).read_bytes()).hexdigest()
    assert h == ref[f"weights/{f}"], n; lst[n] = h
extra = [p.name for p in D.iterdir() if p.name not in lst and p.name != "dataset-metadata.json"]
assert not extra, extra
Path("exp/L_dwadapt/dataset_sha256.json").write_text(json.dumps(lst, indent=1))
mb = sum(p.stat().st_size for p in D.iterdir()) / 2**20
print(f"DW_DATASET_OK {len(lst)} files, sha = dw_ref.json; {mb:.1f} MiB (list exp/L_dwadapt/dataset_sha256.json)")
PY
echo "create (C runs it from PowerShell; absolute path, private by default):"
echo '  .venv/Scripts/kaggle.exe datasets create -p (Resolve-Path exp/L_dwadapt/dataset).Path --dir-mode skip'
