"""L_dose A1 (worker 1, 2026-10-15), laptop side: served on/ member files -> shipped cold XSD step -> shipped chain -> rows,
and the candidate delta on C50. Reuses the L_lab A3 helpers (exp/L_lab/a3/a3.py: chain, table, delta; read only).

  python exp/L_dose/a1_build.py chain ON_DIR TAG     ON_DIR = served adapted files (cnn_s0..4.parquet, sccnn_official.parquet)
        -> artifacts/claude_ref/l_dose/a1/TAG/{cold,on_xsd,a0.75}: c_cold_ship.py build (shipped XSD members on cold windows,
           un parity vs the stored off/), then c_noadapt_blend .75 with the stored unadapted off/, qcnn_avg_probs,
           ong2_probe build --noT, asm2_bbmember decode (ConvNeXt member from stored probabilities). Compared with the
           shipped rows (c_cold/official/ship_c35/a0.75/ongoing_bb.parquet = the ongoing rows of C50).
  python exp/L_dose/a1_build.py delta TAG NAME       candidate = C50 with TAG's ongoing rows; tools/probe.py make/expand,
           stream check, tools/window_guard.py --no-record -> artifacts/claude_ref/l_dose/a1/delta/
"""
import sys, json, shutil, subprocess, filecmp, importlib.util
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("a3", ROOT / "exp/L_lab/a3/a3.py"); A3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A3)
OUT = ROOT / "artifacts/claude_ref/l_dose/a1"; REL = "artifacts/claude_ref/l_dose/a1"


def chain(on_dir, tag):
    T = OUT / tag; T.mkdir(parents=True, exist_ok=True); cold = T / "cold"; cold.mkdir(exist_ok=True)
    for f in Path(ROOT / A3.COLD_MEMBERS).glob("*_s[0-4].parquet*"):
        shutil.copy2(f, cold / f.name)                       # copy: c_cold_ship build writes cold_windows.csv into its input dir
    log = open(T / "chain.log", "w")
    subprocess.run([sys.executable, "-B", f"{A3.R}/asm2_step.py", f"{A3.R}/c_cold_ship.py", "build", f"{REL}/{tag}/cold", on_dir,
                    f"{A3.M}/off", f"{REL}/{tag}/on_xsd"], cwd=str(ROOT), env=A3.chain_env(), stdout=log, stderr=subprocess.STDOUT, check=True)
    log.flush(); assert "COLD_XSD_BUILD_OK" in (T / "chain.log").read_text(), "cold XSD build failed"
    A3.chain(f"{REL}/{tag}/on_xsd", f"{REL}/{tag}/a0.75", log); log.close()
    ref = A3.rp(f"{A3.C36}/a0.75/ongoing_bb.parquet"); new = A3.rp(f"{REL}/{tag}/a0.75/ongoing_bb.parquet")
    res, w = A3.table(ref, new, f"{tag} rows vs the shipped C50 ongoing rows")
    res["bytes_equal_to_shipped_rows"] = bool(filecmp.cmp(OUT / tag / "a0.75/ongoing_bb.parquet", ROOT / A3.C36 / "a0.75/ongoing_bb.parquet", shallow=False))
    w.to_csv(T / "windows_vs_c50.csv", index=False); (T / "rows_vs_c50.json").write_text(json.dumps(res, indent=1))
    print("A1_CHAIN_DONE", tag, "changed cells", res["changed_cells"], "windows", res["changed_windows"], "bytes equal", res["bytes_equal_to_shipped_rows"], flush=True)


def delta(tag, name):
    A3.delta(new_rows=OUT / tag / "a0.75/ongoing_bb.parquet", name=name, tmpdir=OUT / "tmp", outdir=OUT / "delta")


if __name__ == "__main__":
    {"chain": chain, "delta": delta}[sys.argv[1]](*sys.argv[2:])
