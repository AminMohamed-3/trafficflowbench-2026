"""Rebuild b6w_s08 through state5_c13apr with portable, science-preserving OWN metadata checks."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs/reference_code"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import asm2_phys_lwr_diag as P
    sys.modules["phys_lwr_diag"] = P
    fit = ROOT / "artifacts/claude_ref/asm2/fit"
    protocol = json.loads((fit / "protocol.json").read_text())
    protocol.update(main_root=str(ROOT), D_worktree=str(ROOT), S_source_root=str(ROOT))
    protocol["OWN_metadata_binding"]["manifest"] = str(ROOT / protocol["OWN_metadata_binding"]["manifest"])
    P.P = protocol
    import state5_c13apr as CA
    import flow_own as FO
    metadata = json.loads((fit / "own_metadata.json").read_text())
    checks = []

    def portable_binding(p, item):
        # Every array/mask pin remains checked by own_instrument.load_item.
        # Only timing-only cache metadata is rebound after checking every other field.
        key = item["panel"] + "/" + item["split"]
        snap = metadata[key]
        assert item["source"]["meta.json"] == snap["sha256"]
        path = ROOT / "artifacts/cache" / key / "meta.json"
        current = json.loads(path.read_text()); frozen = deepcopy(snap["content"])
        current.pop("runtime_seconds", None); frozen.pop("runtime_seconds", None)
        assert current == frozen, f"OWN scientific metadata changed: {key}"
        result = deepcopy(item)
        result["source"]["meta.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append(dict(population=key, scientific_metadata_equal=True))
        return result

    FO.binding.bound_own_item = portable_binding
    CA.c11(base="s08", period="validation", tag="mar")
    import state5_probe_base as S
    weights = S.s08_weights("s08")
    reference = ROOT / "artifacts/claude_ref/asm/b6w_s08.json"
    original = json.loads(reference.read_text())
    new = deepcopy(original)
    for pn, w in weights.items(): new["per_panel"][pn]["weights_all_days"] = w
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Original was json.dumps(indent=1) in Windows text mode. Preserve those bytes on Linux.
    a.out.write_bytes(json.dumps(new, indent=1).replace("\n", "\r\n").encode())
    result = dict(checks=checks, panels=len(weights), weights_equal=new == original,
                  sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  bytes_equal=a.out.read_bytes() == reference.read_bytes())
    Path(str(a.out) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["weights_equal"] and result["bytes_equal"], result


if __name__ == "__main__": main()
