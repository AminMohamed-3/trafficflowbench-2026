"""Write the ConvNeXt-Tiny member config for docs/reference_code/asm2_bbmember.py (rows 21 and 23; R2 piece worker 2026-10-21).

  python make_cfg.py OUT.json (--probs FILE | --weights MODEL.pt [--device cpu|cuda]) [--name NAME]

Template: cfg_vm.json next to this file = artifacts/claude_ref/c_bb/vm_tiny_swap/cfg_vm.json, unchanged (sha 1dc98b80; the
config of the C55 build, a = .2, spec convnext_tiny.fb_in1k up2, drop prior/pos/tod, amp_bf16). Only the file paths are
replaced by the resolved ones (lib/weights.py, sha checked there and again inside asm2_bbmember.py serve):
  default      source = probs   (stored member probabilities, sha c02523d2; served on the VM RTX 5070 from model.pt, bf16)
  --weights    source = weights (asm2_bbmember.py serve builds the net with pretrained=False and loads model.pt; needs timm
               1.0.15 + torchvision 0.22.1; on CPU amp is off -> fp32 -> a SIBLING of the stored probabilities, not bitwise)
asm2_bbmember.py decode reads only "a" (and checks "source"); the paths matter only for serve."""
import argparse, hashlib, json
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--probs")
    ap.add_argument("--weights"); ap.add_argument("--device", default="cuda"); ap.add_argument("--name")
    a = ap.parse_args()
    cfg = json.loads((Path(__file__).resolve().parent / "cfg_vm.json").read_text())
    assert float(cfg["a"]) == .2 and cfg["weights_sha256"].startswith("43da1ffc")
    assert a.probs or a.weights, "--probs or --weights"
    if a.probs:
        cfg["probs"] = Path(a.probs).resolve().as_posix(); cfg["probs_sha256"] = sha(a.probs)
    if a.weights:
        assert sha(a.weights) == cfg["weights_sha256"], "model.pt sha"
        cfg.update(source="weights", weights=Path(a.weights).resolve().as_posix(), device=a.device)
    if a.name:
        cfg["name"] = a.name
    Path(a.out).write_text(json.dumps(cfg, indent=1) + "\n")
    print("MAKE_CFG", json.dumps(dict(out=a.out, source=cfg["source"], probs_sha256=cfg["probs_sha256"][:8], device=cfg["device"])), flush=True)


if __name__ == "__main__":
    main()
