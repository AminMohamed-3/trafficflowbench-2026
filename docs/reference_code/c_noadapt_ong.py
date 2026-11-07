"""C 2026-10-09 (ledger 0765 helper, SIGNAL probe): the shipped ongoing members with per-date adaptation ON or OFF.

Runs the production code unchanged (qcnn_predict_adapt.main for the base CNN seeds, ong2_sccnn_official.main for the
selection-conditioned CNN seeds). Mode `on` patches nothing, so it is the shipped chain (fitted per-date weights via
TFB_ASM2_USE_ADAPTED=1). Mode `off` replaces qcnn_predict_adapt.checkpointed_adaptation by "return the base model":
every official date is served by the unadapted base weights, with the same official_x inputs, the same online rule,
the same selection-flag RNG (S._rng reseeded per date before the swap, as in production) and the same decoders after.

  python c_noadapt_ong.py {on|off} qcnn  RUN_DIR OUT.csv          # writes OUT.csv + OUT.parquet (prob)
  python c_noadapt_ong.py {on|off} sccnn RUN_DIR[,RUN_DIR] OUT.parquet
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import qcnn_predict_adapt as qpa


def main(mode, which, runs, out):
    assert mode in ("on", "off") and which in ("qcnn", "sccnn"), (mode, which)
    if mode == "off":
        qpa.checkpointed_adaptation = lambda base, run, split, date, seed, fit, settings=None: base
    print("adaptation", mode, which, runs, flush=True)
    if which == "qcnn":
        qpa.main(runs, "eiou", out)
    else:
        import ong2_sccnn_official as O
        O.main(runs, out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
