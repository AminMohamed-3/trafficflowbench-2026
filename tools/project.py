"""Resolve shared files to the main worktree."""
from pathlib import Path

def main_root(start=None):
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for root in (current, *current.parents):
        git = root / ".git"
        if git.is_dir():
            return root
        if git.is_file():
            marker = git.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                raise ValueError(f"Bad worktree pointer: {git}")
            gitdir = Path(marker.partition(":")[2].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
            common = gitdir.parent.parent
            if common.name != ".git":
                raise ValueError(f"Unexpected worktree layout: {gitdir}")
            return common.parent
    raise FileNotFoundError("No Git worktree found")
