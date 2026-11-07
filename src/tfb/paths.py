"""Paths shared by the Lead and all lane worktrees."""
from pathlib import Path
import os

def project_root():
    override = os.environ.get("TFB_ROOT")
    if override:
        root = Path(override).resolve()
        if not (root / "PLAN.md").is_file():
            raise ValueError(f"TFB_ROOT is not the project root: {root}")
        return root
    current = Path(__file__).resolve().parent
    for root in (current, *current.parents):
        marker = root / ".git"
        if marker.is_dir():
            return root
        if marker.is_file():
            text = marker.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                raise ValueError(f"Invalid worktree marker: {marker}")
            gitdir = Path(text.partition(":")[2].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
            common = gitdir.parent.parent
            if common.name != ".git":
                raise ValueError(f"Unexpected Git worktree layout: {gitdir}")
            return common.parent
    raise FileNotFoundError("Could not locate the main project root")

def release_root(root=None):
    return Path(root or project_root()) / "data" / "raw" / "kaggle_public"

def cache_root(root=None):
    return Path(root or project_root()) / "artifacts" / "cache"

def panels(root=None):
    import json
    contract = json.loads((release_root(root) / "config" / "corridors.json").read_text(encoding="utf-8"))
    return [p["corridor_id"] for p in contract["panels"]]
