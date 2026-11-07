"""Build one or all audited panel/split caches. Safe to restart completed builds."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfb.data import build_cache, SPLITS
from tfb.paths import panels, project_root

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--panel", required=True, help="Panel ID or all")
    cli.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    cli.add_argument("--root", type=Path)
    args = cli.parse_args()
    root = args.root or project_root()
    selected = panels(root) if args.panel == "all" else [args.panel]
    splits = SPLITS if args.split == "all" else [args.split]
    results = []
    receipt = root / "artifacts" / "jobs" / "0104.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    job = {"owner": "L", "pid": os.getpid(), "command": sys.argv, "start_utc":
           datetime.now(timezone.utc).isoformat(), "eta": "panel/split checkpoints",
           "checkpoint_path": str(root / "artifacts" / "cache"), "status": "RUNNING"}
    receipt.write_text(json.dumps(job, indent=2), encoding="utf-8")
    try:
        for panel in selected:
            for split in splits:
                metadata = build_cache(panel, split, root=root)
                summary = {k: metadata[k] for k in ("panel", "split", "shape",
                    "target_rows", "target_rows_by_regime", "stations_links_one_to_one",
                    "gap_cells", "runtime_seconds")}
                results.append(summary)
                print(json.dumps(summary), flush=True)
        job["status"] = "COMPLETE"
        job["results"] = results
    except Exception:
        job["status"] = "FAIL"
        raise
    finally:
        receipt.write_text(json.dumps(job, indent=2), encoding="utf-8")
    out = root / "exp" / "0104_l04_cache" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"id": "0104", "owner": "L", "formulation": "L-04 cache",
        "utc": datetime.now(timezone.utc).isoformat(), "results": results,
        "verdict": "PASS"}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
