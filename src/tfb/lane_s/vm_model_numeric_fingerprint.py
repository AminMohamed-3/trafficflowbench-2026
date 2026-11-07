"""Strict structural fingerprint with bounded native leaf serialization noise."""
import hashlib,math

def fingerprints(text):
    lines=[]
    for line in text.splitlines():
        if line.startswith("[num_threads:") or line.startswith("tree_sizes="):continue
        if line.startswith("leaf_value="):
            values=[float(v) for v in line.split("=",1)[1].split()]
            assert all(math.isfinite(v) for v in values)
            line="leaf_value="+" ".join(format(v,".11f") for v in values)
        lines.append(line)
    canonical="\n".join(lines)+"\n"
    trees=canonical.split("Tree=0\n",1)[1].split("\nend of trees",1)[0]
    return dict(numeric_leaf11_canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),numeric_leaf11_trees_sha256=hashlib.sha256(trees.encode()).hexdigest())
