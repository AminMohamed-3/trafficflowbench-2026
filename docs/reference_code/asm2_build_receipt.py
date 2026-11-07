"""Record final-entry source, frozen inputs and saved adaptation lineage."""
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import import_closure as C


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--adapt-root", type=Path, required=True)
    ap.add_argument("--queue-mode", choices=("adaptation-replay", "adapted-weights", "fitted-probabilities", "strict-noT-rows"), default="adaptation-replay")
    ap.add_argument("--base-probs", type=Path)
    ap.add_argument("--selection-probs", type=Path)
    ap.add_argument("--runs", default="", help="comma list of the CNN run names the entry served (C 0779, e16: 10 runs x 24 "
                    "dated weights); default: every run under --adapt-root, 96 weights (C26 form)")
    a = ap.parse_args()
    a.out = a.out.resolve(); a.adapt_root = a.adapt_root.resolve()
    entry = ROOT / "docs/reference_code/asm2_run_all.sh"
    # Reuse the shared import walker for local source lineage; add helpers whose
    # imports are dynamic and the fit-only producers documented in the guide.
    todo = [ROOT / "docs/reference_code" / name for name in
            ("asm2_final.py", "asm2_qo.py", "asm2_onset_rows.py", "asm2_inputs.py",
             "asm2_build_receipt.py", "asm2_state5_weights.py", "qcnn_predict_adapt.py",
             "qcnn_avg_probs.py", "ong2_official_types.py", "ong2_sccnn_official.py", "ong2_probe.py",
             "asm2_s0631.py", "s_task1_tabpfn_runtime.py", "asm2_bbmember.py", "c_bb_serve.py", "c_bb_arch.py", "asm2_step.py",
             "c_e16_adapt.py", "c_e16_train.py", "c_noadapt_ong.py", "c_noadapt_blend.py")]
    todo += [ROOT / "scripts/q0436_official.py", ROOT / "scripts/build_cache.py"]
    seen = set()
    while todo:
        p = todo.pop().resolve()
        if p in seen: continue
        seen.add(p)
        for name in C.imports(p):
            q = C.resolve(name)
            if q is not None and q.resolve() not in seen: todo.append(q)
    seen.add(entry)
    sources = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(seen)}
    runs = [r for r in a.runs.split(",") if r]
    roots = [a.adapt_root / r for r in runs] if runs else [a.adapt_root]
    checkpoint_files = {p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size, sha256=sha(p))
                        for r in roots for p in sorted(r.rglob("*")) if p.is_file()}
    if a.queue_mode != "fitted-probabilities":
        assert checkpoint_files and any(p.endswith("/model.pt") for p in checkpoint_files), "missing adaptation checkpoints"
    if a.queue_mode in ("adapted-weights", "strict-noT-rows"):
        need = 24 * len(runs) if runs else 96
        assert sum(p.endswith("/model.pt") for p in checkpoint_files) == need, f"all {need} selected date weights required"
    probability_inputs = {}
    for label, path in (("base", a.base_probs), ("selection_conditioned", a.selection_probs)):
        if path is not None:
            path = path.resolve()
            probability_inputs[label] = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
    if a.queue_mode == "fitted-probabilities":
        assert len(probability_inputs) == 2, "canonical probability inputs required"
    final_manifest = Path(str(a.out) + ".manifest.json")
    manifest = json.loads(final_manifest.read_text())
    environment_path = ROOT / "artifacts/claude_ref/asm2/run/environment.json"
    tracked = set((git("ls-files") or "").splitlines())
    result = dict(schema=1, runtime_seconds=a.seconds, python=platform.python_version(),
                  git_head=git("rev-parse", "HEAD"), source_sha256=sources,
                  untracked_sources=[p for p in sources if p not in tracked],
                  tracked_source_changes=git("diff", "--name-only", "HEAD", "--", *sources),
                  serving_inventory_sha256=sha(ROOT / "artifacts/claude_ref/asm2/serving_inputs_sha256.txt"),
                  queue_mode=a.queue_mode, fitted_probability_inputs=probability_inputs,
                  noT_delivery=manifest.get("log", {}).get("noT"),
                  state_rows=manifest.get("log", {}).get("state_rows"),
                  s0631_receipt=(json.loads(Path(manifest["log"]["state_rows"]["file"] + ".json").read_text())
                                 if manifest.get("log", {}).get("state_rows") and Path(manifest["log"]["state_rows"]["file"] + ".json").is_file() else None),
                  environment_receipt_sha256=sha(environment_path) if environment_path.is_file() else None,
                  environment=json.loads(environment_path.read_text()) if environment_path.is_file() else None,
                  adaptation_files=checkpoint_files, output_sha256=sha(a.out),
                  output_manifest_sha256=sha(final_manifest), parity=manifest.get("check"),
                  source_walk_limit="Static imports only; runtime modules are also covered by the committed source tree")
    target = Path(str(a.out) + ".build.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print("build receipt", target, "seconds", a.seconds, "checkpoint files", len(checkpoint_files), flush=True)


if __name__ == "__main__": main()
