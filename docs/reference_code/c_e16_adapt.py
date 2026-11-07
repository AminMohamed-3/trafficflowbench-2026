"""C e16 (2026-10-09, ledger 0771): fill the per-date adaptation store in parallel, date-split, with the production code.

Runs the unchanged production main (qcnn_predict_adapt.main for a base CNN run, ong2_sccnn_official.main for the sccnn runs)
with TFB_ASM2_USE_ADAPTED=0 (fit + save weights and receipts under TFB_ASM2_ADAPT_ROOT), but only the (split, date) pairs of
part PART of NPARTS are fitted; every other date is served by the base model and its output is discarded (this process's
output file is scratch). Per-date fits are independent in production: finetune seeds torch and numpy with the date index,
and the sccnn selection RNG is reseeded per date (S._rng = default_rng(1000 + k)) before the ensemble members are fitted
in order, so a date fitted here equals the same date fitted in one sequential process.
Afterwards the production main with TFB_ASM2_USE_ADAPTED=1 (receipts asserted, saved weights served) and --poison gives the
serving probabilities and the future-poison check (c_e16_serve.sh).
  python c_e16_adapt.py qcnn RUN_DIR PART NPARTS SCRATCH_OUT.csv
  python c_e16_adapt.py sccnn RUN_DIR[,RUN_DIR] PART NPARTS SCRATCH_OUT.parquet
  python c_e16_adapt.py serve qcnn|sccnn RUNS OUT [--poison]     all dates, production main with the recipe (build step)
Parts: dates of both splits sorted by fit cost (same-day-type pool dates before d, x members) and dealt greedily.
Recipe (env E16_RECIPE="epochs,lr,cap", default "3,3e-4,none" = shipped: then nothing is patched and the production code
runs as is). Another recipe sets qcnn_predict_adapt.EPOCHS/LR (recorded in every receipt) and the cap: sccnn through the
production --cap=N flag (recorded as settings.cap), base CNN by injecting cap into finetune and recording it in settings.cap
(the production base main has no cap flag). Use a separate TFB_ASM2_ADAPT_ROOT per recipe (the store path has no recipe).
Torch build rule of `serve` (KB2 2026-10-13, exp/C_kbuild/KBUILD.md 7c W11): see serve_torch_rule below.
"""
import os, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import qcnn_predict_adapt as qpa
from qcnn_adapt import weekend
from qcnn_train import PANELS


def official_dates(split):
    d = set()
    for p in PANELS:
        wi = pd.read_csv(qpa.REL / p / split / "window_index.csv")
        d |= {str(pd.Timestamp(o).date()) for o in wi[wi.condition == "queue_ongoing"].forecast_origin}
    return sorted(d)


def plan(nparts):
    items = []
    for split in ("validation", "private"):
        ds = official_dates(split)
        for k, d in enumerate(ds):
            items.append((1 + sum(weekend(x) == weekend(d) for x in ds[:k]), split, d))
    load = [0] * nparts; part = [[] for _ in range(nparts)]
    for c, split, d in sorted(items, reverse=True):
        i = load.index(min(load)); load[i] += c; part[i].append((split, d))
    return part, load


def recipe():
    ep, lr, cap = os.environ.get("E16_RECIPE", "3,3e-4,none").split(",")
    ep, lr, cap = int(ep), float(lr), (None if cap == "none" else int(cap))
    if (ep, lr, cap) == (3, 3e-4, None):
        assert (qpa.EPOCHS, qpa.LR) == (3, 3e-4); return None
    qpa.EPOCHS, qpa.LR = ep, lr
    print("recipe epochs", ep, "lr", lr, "cap", cap, flush=True)
    return cap


def serve_torch_rule():
    """Which torch build may SERVE the shipped per-date weights (the rule of the receipt check, build step `serve`).

    Every shipped receipt (inputs.json) stores the full version string of the torch that FITTED the weights; the
    production check qcnn_predict_adapt.checkpointed_adaptation compares it with the installed string, so a box with
    another wheel of the same release (the CPU wheel 2.7.1+cpu instead of the pinned 2.7.1+cu118) was refused although
    it only loads the stored weights and runs CPU inference. The rule:
      1. Installed torch == the pin of docs/report/requirements-final.txt: nothing changes (exact string comparison).
      2. Otherwise serving is allowed only if ALL of these hold, else the step stops with the reason:
         - stored weights are served, nothing is fitted (TFB_ASM2_USE_ADAPTED=1; a fit always records the real string);
         - the box is declared a CPU build box, TFB_ASM2_CPU_ONLY=1 (the same switch asm2_inputs.py --environment
           takes for the CPU wheel, recorded as cpu_only in the build's environment receipt);
         - same release: the public version (before "+") equals the pin's, and the installed local label is "cpu";
         - no CUDA device is visible, and every receipt's own torch_version is exactly the pin (checked per date).
         Then the comparison of that ONE field uses the pinned string; every other receipt field (base, config and
         the five source hashes, split, date, seed, recipe, settings, device cpu, threads, deterministic mode) and the
         weight file's sha256 are compared by the unchanged production check.
    Measured (WSL2 Ubuntu, torch 2.7.1+cpu, 2026-10-12/13): all 240 dated weights served under this rule give the
    87,000 ongoing rows of the reference file exactly. The production module is not edited (its sha256 is one of the
    five source hashes in every receipt); the pinned string is shown to it for the duration of the check only."""
    import json
    torch = qpa.torch; have = str(torch.__version__)
    pins = dict(l.strip().split("==", 1) for l in (ROOT / "docs/report/requirements-final.txt").read_text().splitlines()
                if "==" in l and not l.lstrip().startswith("#"))
    pin = pins["torch"]
    if have == pin or os.environ.get("TFB_ASM2_USE_ADAPTED") != "1":
        return
    assert os.environ.get("TFB_ASM2_CPU_ONLY") == "1", (f"torch {have} is not the pinned {pin}: install the pinned wheel, or, on a CPU "
                                                         "build box with the CPU wheel of the same release, set TFB_ASM2_CPU_ONLY=1")
    assert have.split("+")[0] == pin.split("+")[0] and have.split("+")[1:] == ["cpu"], f"torch {have}: only the CPU wheel of release {pin.split('+')[0]} may serve the shipped weights"
    assert not torch.cuda.is_available(), "TFB_ASM2_CPU_ONLY=1 with a visible CUDA device"
    inner = qpa.checkpointed_adaptation

    class Pinned:   # the installed torch, with the pinned version string for the receipt comparison
        __version__ = pin
        def __getattr__(self, k): return getattr(torch, k)

    def ca(base, run, split, date, seed, fit, settings=None):
        rf = Path(os.environ["TFB_ASM2_ADAPT_ROOT"]) / Path(run).name / split / date / "inputs.json"
        got = json.loads(rf.read_text())["inputs"].get("torch_version") if rf.is_file() else None
        assert got == pin, f"receipt torch_version {got!r} is not the pinned {pin}: {rf}"
        qpa.torch = Pinned()
        try:
            return inner(base, run, split, date, seed, fit, settings)
        finally:
            qpa.torch = torch
    qpa.checkpointed_adaptation = ca
    print(f"SERVE_TORCH_RULE installed torch {have}, receipts and pin {pin}: same release, CPU serving of stored weights (TFB_ASM2_CPU_ONLY=1)", flush=True)


def run_main(which, runs, out, cap, gate=None, poison=False):
    """Production main with the recipe; gate(split, date) -> fit this date (else serve the base, output discarded)."""
    orig_ca, orig_ft = qpa.checkpointed_adaptation, qpa.finetune
    if which == "qcnn" and cap is not None:
        qpa.finetune = lambda *a, **k: orig_ft(*a, **dict(k, cap=k.get("cap", cap)))

    def ca(base, run, split, date, seed, fit, settings=None):
        if gate is not None and not gate(split, date):
            return base
        if which == "qcnn" and cap is not None:
            settings = dict(settings or {}, cap=cap)
        return orig_ca(base, run, split, date, seed, fit, settings)
    if gate is not None or (which == "qcnn" and cap is not None):
        qpa.checkpointed_adaptation = ca
    if which == "qcnn":
        qpa.main(runs, "eiou", out, poison)
    else:
        import ong2_sccnn_official as O
        O.main(runs, out, *(["--poison"] if poison else []), *([f"--cap={cap}"] if cap is not None else []))


def main(which, runs, part, nparts, out):
    part, nparts = int(part), int(nparts)
    assert os.environ.get("TFB_ASM2_USE_ADAPTED") == "0" and os.environ.get("TFB_ASM2_ADAPT_ROOT")
    parts, load = plan(nparts); mine = set(parts[part])
    print("part", part, "of", nparts, "cost", load[part], "dates", sorted(mine), flush=True)
    run_main(which, runs, out, recipe(), gate=lambda split, date: (split, date) in mine)
    print("E16_ADAPT_PART_DONE", which, runs, part, nparts, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        assert os.environ.get("TFB_ASM2_ADAPT_ROOT")   # USE_ADAPTED=1: replay saved weights; 0: fit + save (build refit)
        serve_torch_rule()
        run_main(sys.argv[2], sys.argv[3], sys.argv[4], recipe(), poison="--poison" in sys.argv[5:])
        print("E16_SERVE_MAIN_DONE", sys.argv[2], sys.argv[3], flush=True)
    else:
        main(*sys.argv[1:6])
    sys.stdout.flush(); os._exit(0)
