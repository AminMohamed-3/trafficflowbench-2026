"""C budget helper 2026-10-09: official Task 1 state rows with a budget stage-2 form (ledger 0768), for a board delta.

Runs asm2_final.py (default organizer chain, --state-only, no S0631 overlay = the C26..C29 Task 1 state) with s2stage.OUT
pointed at the form directory, so asm2_build.Stages registers and loads m_{v,q}.txt from there (TFB_ASM2_ART cannot do
this: Stages registers the override path but s2stage.load_all() reads s2stage.OUT). Everything else is the shipped build.
Usage: python c_budget_s2build.py <form dir under artifacts/claude_ref/c_budget/s2> <out.csv> [asm2_final args ...]"""
import sys
from pathlib import Path
VM_CODE = Path("/workspace/work/main/docs/reference_code")   # VM worktree runs use the main mirror code + artifacts
sys.path.insert(0, str(VM_CODE if VM_CODE.is_dir() else Path(__file__).resolve().parent))
import s2stage as S2

form, out = sys.argv[1], sys.argv[2]
S2.OUT = S2.P.ROOT / "artifacts/claude_ref/c_budget/s2" / form
assert (S2.OUT / "m_v.txt").exists() and (S2.OUT / "m_q.txt").exists(), S2.OUT
import asm2_final
sys.argv = ["asm2_final.py", "--out", out, "--queue-base", "none", "--state-only"] + sys.argv[3:]
asm2_final.main()
