"""C e16 serving retrain (2026-10-09): the shipped ongoing CNN members with 16 epochs instead of 8 (full OneCycle over 16).

Same trainers and fit cutoff (full mode, train split <= 2031-02-28) as the shipped serving members, only --epochs changes:
  ctl: qcnn_train.py full --seed S --drop prior,pos,tod --epochs 16 --tag _npt_e16  (shipped run_full_s{0,1}_npt, C03 recipe)
       -> artifacts/claude_ref/qcnn/run_full_s<S>_npt_e16
  sc:  ong2_sccnn_train.py --seed S --epochs 16                                    (shipped run_sc_s{0,1}, sccnn)
       -> artifacts/claude_ref/ong2/run_sc_s<S>_e16 (trained into a staging dir, then renamed: the per-date adaptation
       store is keyed by the run directory name, so e16 runs must not reuse run_sc_s<S>)
Both trainers read artifacts/claude_ref/qcnn/{panel}.npz + windows.parquet (qcnn_data.py). If absent, they are symlinked
from artifacts/T_noT/qcnn_data (same qcnn_data.py source, sha-checked against its prerequisite receipt).
Usage: python c_e16_train.py ctl|sc SEED [EPOCHS=16] [--replace]
  --replace (build --retrain-cnn): an existing run directory is renamed to <name>.shipped first (kept, not deleted).
"""
import hashlib, json, os, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
Q = ROOT / "artifacts/claude_ref/qcnn"
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
WSHA = "a6efcbc910be864548a3f87e82973c96b609a8ebd33e9da36b71afd2ff47f6d9"
DATA_SRC = "cee39dfe01e8126ea92e308914ecccc9bb2b64d8757f29490608b77298092752"   # qcnn_data.py


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def ensure_data():
    names = [f"{p}.npz" for p in PANELS] + ["windows.parquet"]
    if all((Q / n).exists() for n in names):
        assert sha(Q / "windows.parquet") == WSHA; print("data: artifacts/claude_ref/qcnn", flush=True); return
    src = ROOT / "artifacts/T_noT/qcnn_data"; rec = json.loads((src / "prerequisite_receipt.json").read_text())
    assert rec["status"] == "PASS" and rec["source_sha"] == DATA_SRC and rec["files"]["windows.parquet"] == WSHA
    Q.mkdir(parents=True, exist_ok=True)
    for n in names:
        assert sha(src / n) == rec["files"][n], n
        if not (Q / n).exists(): (Q / n).symlink_to((src / n).resolve())
    print("data: symlinked from", src, "(receipt shas verified)", flush=True)


def set_aside(out, replace):
    if out.exists():
        assert replace, f"{out} exists (use --replace)"
        old = out.with_name(out.name + ".shipped"); assert not old.exists(), old
        out.rename(old); print("existing run kept as", old, flush=True)


def main(kind, seed, epochs="16", *flags):
    assert kind in ("ctl", "sc") and set(flags) <= {"--replace"}; seed = int(seed); epochs = int(epochs)
    replace = "--replace" in flags
    skip = ROOT / f"artifacts/claude_ref/asm2/e16/skip_{kind}_s{seed}"   # moved to another GPU slot after launch (C, 0771)
    if skip.exists() and not os.environ.get("E16_NO_SKIP"):
        print("E16_SKIP", kind, seed, "(trained by another job)", flush=True); return
    ensure_data()
    if kind == "ctl":
        import qcnn_train
        out = Q / f"run_full_s{seed}_npt_e{epochs}"; set_aside(out, replace)
        sys.argv = ["qcnn_train.py", "full", "--seed", str(seed), "--drop", "prior,pos,tod", "--epochs", str(epochs), "--tag", f"_npt_e{epochs}"]
        qcnn_train.main()
    else:
        import ong2_sccnn_train as M
        out = ROOT / f"artifacts/claude_ref/ong2/run_sc_s{seed}_e{epochs}"; set_aside(out, replace)
        M.OUT = ROOT / f"artifacts/claude_ref/ong2/e{epochs}_staging_s{seed}"
        sys.argv = ["ong2_sccnn_train.py", "--seed", str(seed), "--epochs", str(epochs)]
        M.main()
        shutil.move(str(M.OUT / f"run_sc_s{seed}"), str(out)); M.OUT.rmdir()
    cfg = json.loads((out / "config.json").read_text()); assert cfg["epochs"] == epochs and cfg["mode"] == "full"
    print("E16_TRAINED", kind, seed, out, "config", json.dumps(cfg), "model sha256", sha(out / "model.pt"), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
