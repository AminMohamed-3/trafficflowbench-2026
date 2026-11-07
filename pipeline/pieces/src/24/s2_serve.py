"""Row 24 serve step: the 60 B6R (S2) members on the official view, CPU (final_solution glue, R2 worker).

Calls serve() of src/24/s2_b6r.py, an UNCHANGED copy of exp/L_b6/s2_b6r.py (sha256 e87672ae038ef799149976e00c05fcdc778139e5ad2f5bed78fe9ebcdbe1a5b7),
as `s2_b6r.py serve --out RUNDIR` did in exp/L_b6/s2_finish.sh (OMP/MKL threads 4). Only module globals are set before the call:
  - s2_b6r.ROOT = $TFB_CODE_ROOT (the copy would compute it from its own location);
  - c_b6long_full_step.REF_ALL = the kg_ref_all.json weights row (OWN-mask seed, fraction and digests; original exp/C_b6long/kg_ref_all.json);
  - root = the directory holding it36000_w32/ (sd_*.pt + run_*.json, resolved and sha-checked by lib/weights.py).
The OWN view (c_b6long_full_step.own_layer) is only used to fit the per-(panel, split) ramp calibration and occ_ref, as in training.
Writes RUNDIR/imp_{valown,privown}_<panel>_s{0,1,2}.npz + receipt.json (array sha256 per file).
Usage: python s2_serve.py RUNDIR WEIGHTS_ROOT KG_REF_ALL.json [PANEL,...]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU serving (exp/L_b6/s2_finish.sh ran with CUDA_VISIBLE_DEVICES=-1); b6full_imp picks cuda if visible
from pathlib import Path
CODE = Path(os.environ["TFB_CODE_ROOT"]).resolve()
sys.path.insert(0, str(CODE / "docs/reference_code")); sys.path.insert(1, str(CODE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_b6r as S
import c_b6long_full_step as F
out, root, ref = (str(Path(x).resolve()) for x in sys.argv[1:4])
panels = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else None
S.ROOT = CODE
F.REF_ALL = Path(ref)
assert S.CFG == "it36000_w32" and (Path(root) / S.CFG).is_dir(), (S.CFG, root)
assert S.F.K.b6().DEV == "cpu", "a GPU is visible: the stored members were served on CPU"
S.serve(out, root, panels)
