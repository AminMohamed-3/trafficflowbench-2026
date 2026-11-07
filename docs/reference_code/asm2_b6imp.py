"""asm2 (ledger 0475): refit the B6 per-split U-Net imputer (b6full_imp.py, ledger 0399) into artifacts/claude_ref/asm2/b6refit/
instead of b6full/ (which holds the frozen board files and is skipped-if-exists by b6full_imp). Uses the portable phys_lwr_diag
shim. Mode priv needs only the official private cache (raw release). Usage: python asm2_b6imp.py <mode> <panel|all> <seed> [iters]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
DET = os.environ.get("TFB_ASM2_DET") == "1"   # deterministic cuDNN/cuBLAS kernels (same-GPU bitwise repeatability)
if DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asm2_phys_lwr_diag as _P
sys.modules["phys_lwr_diag"] = _P
import b6full_imp as M

M.OUT = _P.ROOT / os.environ.get("TFB_ASM2_B6OUT", "artifacts/claude_ref/asm2/b6refit")
if DET:
    import torch
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
if __name__ == "__main__":
    M.main(*sys.argv[1:])
