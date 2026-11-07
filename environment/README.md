# Environment

- Python 3.12.x (`python_version.txt`), Linux. The build was qualified on Ubuntu 22.04 (WSL2 and a Vast.ai VM) with
  Python 3.12.3 and 3.12.15.
- `requirements.txt` = the build's own pin file (byte copy of `docs/report/requirements-final.txt`, which
  `asm2_inputs.py --environment` reads and checks): **35 package pins** + the PyTorch cu118 index line.
- `requirements-cpu.txt` = the CPU box: 23 pins (the 35 without the 11 `nvidia-*` packages and `triton`), torch as the
  `2.7.1+cpu` wheel. With it set `TFB_ASM2_CPU_ONLY=1` (the environment check then expects that wheel, and the ongoing
  serving step accepts the shipped per-date receipts, which store `2.7.1+cu118`, for CPU serving by the same torch
  release: `docs/reference_code/c_e16_adapt.py`, `serve_torch_rule`).
- The venv must live at `/workspace/tfb/venv`: `src/tfb/lane_s/lgb_runtime.py` accepts LightGBM on Linux only from that
  path and with the native library hash of the PyPI wheel 4.6.0 (the S08 refit fits through this guard).

## Pin count: 35 is right

Older text says 34 pins (`docs/report/REPRO_README.md`: "matches all 34 committed pins", "all 34 release pins"); the VM
note in `docs/report/FINAL_BUILD_PLAN.md` says "all 35 pins". The file itself has 35 `==` lines. The 34 is the count
before commit 2e3091ed (2026-10-08, S0631 TabPFN piece and the C28 backbone member) added `packaging==26.3`; the same
README states that addition in its last section ("The focused pins add packaging==26.3"). The environment check
compares every `==` line of the file, so a box must match all 35 (23 on the CPU box). `packaging` is imported only by
the TabPFN runtime (S0631, off by default), but the check still requires it.

Recipe used for the qualified CPU box (`docs/reference_code/c_kbuild_wsl_setup.sh venv` in the development repo):

```bash
uv venv --python 3.12 /workspace/tfb/venv
uv pip install --python /workspace/tfb/venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
uv pip install --python /workspace/tfb/venv/bin/python -r environment/requirements-cpu.txt
```
