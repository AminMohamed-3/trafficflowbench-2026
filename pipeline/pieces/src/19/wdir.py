"""Resolve every weights row of a piece whose repo_path starts with PREFIX (sha checked by lib/weights.py) and print the one
directory that holds them all (the serve scripts take a directory). Usage: python wdir.py PIECE PREFIX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import weights as W
piece, prefix = sys.argv[1:3]
rs = [r for r in W.rows(piece) if r["repo_path"].startswith(prefix)]
assert rs, f"no weights rows {piece} {prefix}"
d = {W.resolve(r["repo_path"]).parent for r in rs}
assert len(d) == 1, f"weights of {prefix} resolve to several directories: {d}"
print(d.pop())
