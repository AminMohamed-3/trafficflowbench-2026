"""Upload guard for the hard rule "no single-window or single-cell board reads" (RULES.md 3 and 4, user 2026-10-12).

  python tools/window_guard.py FILE [--against OTHER ...] [--no-record]

Compares FILE with the reference file (C44) and with every --against file on the VALIDATION rows (the rows the public
board scores) and counts what differs: queue windows, ODME rows, state rows. A board read of FILE would reveal single
answers when a comparison differs in 1 or 2 queue windows, 1 or 2 ODME rows or 1 or 2 state rows: verdict REFUSE.
0 or 3 and more = PASS. Files may be .csv or .csv.gz. One line per run goes to artifacts/claude_ref/probes/
window_guard.csv (file, sha256, other, counts, verdict); tools/upload_day.py and the project hook read it.
Exit code 0 = PASS, 3 = REFUSE. About 20 s and under 100 MB per comparison (line by line)."""
import argparse, csv, gzip, hashlib, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "artifacts/claude_ref/probes"
KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"
REFERENCE = "c_b6l12k_on_c43.csv"
RECORD = PROBES / "window_guard.csv"
MIN_UNITS = 3
FIELDS = ["utc", "file", "sha256", "other", "queue_windows", "queue_window_ids", "odme_rows", "state_rows", "verdict"]


def resolve(name):
    p = Path(name)
    for c in (p, Path(str(p) + ".gz"), PROBES / p.name, PROBES / (p.name + ".gz")):
        if c.exists():
            return c
    raise FileNotFoundError(name)


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8", newline="") if p.suffix == ".gz" else open(p, encoding="utf-8", newline="")


def sha256(p):
    h = hashlib.sha256()
    with (gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")) as f:  # sha of the csv content, as uploaded
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def same(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def compare(fa, fb):
    """Validation rows of fa that differ from fb: queue window ids, count of ODME rows, count of state rows."""
    windows, odme, state = set(), 0, 0
    with opener(fa) as a, opener(fb) as b, open(KEY, encoding="utf-8", newline="") as k:
        ha, hb, hk = (next(x).rstrip("\r\n").split(",") for x in (a, b, k))
        assert ha == hb, (ha, hb)
        ks, kw, kt = hk.index("split"), hk.index("window_id"), hk.index("task")
        n = 0
        for la, lb, lk in zip(a, b, k):
            n += 1
            la, lb = la.rstrip("\r\n"), lb.rstrip("\r\n")
            if la == lb:
                continue
            va, vb = la.split(","), lb.split(",")
            if va[0] != vb[0]:
                raise ValueError(f"row order differs at line {n}: {va[0]} vs {vb[0]}")
            if all(same(x, y) for x, y in zip(va, vb)) and len(va) == len(vb):
                continue
            kk = lk.rstrip("\r\n").split(",")
            if kk[0] != va[0]:
                raise ValueError(f"key order differs at line {n}: {kk[0]} vs {va[0]}")
            if kk[ks] != "validation":
                continue
            if kk[kt] == "queue":
                windows.add(kk[kw])
            elif kk[kt] == "odme":
                odme += 1
            else:
                state += 1
        if next(a, None) is not None or next(b, None) is not None:
            raise ValueError("files have different row counts")
    return sorted(windows), odme, state


def verdict(windows, odme, state):
    bad = [f"{n} {what}" for n, what in ((len(windows), "queue windows"), (odme, "ODME rows"), (state, "state rows")) if 0 < n < MIN_UNITS]
    return ("REFUSE: " + ", ".join(bad)) if bad else "PASS"


def passed(name, digest=None):
    """True if the record holds a PASS line against the reference for this file name (and sha, if given) and no REFUSE line."""
    if not RECORD.exists():
        return False
    with RECORD.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["file"] == name and (digest is None or r["sha256"] == digest)]
    return any(r["other"] == REFERENCE and r["verdict"] == "PASS" for r in rows) and not any(r["verdict"].startswith("REFUSE") for r in rows)


def run(file, against=(), record=True):
    f = resolve(file)
    name = f.name[:-3] if f.suffix == ".gz" else f.name
    digest = sha256(f)
    others, ok = [REFERENCE] + [o for o in against if o and Path(o).name.replace(".gz", "") != REFERENCE], True
    for o in others:
        fo = resolve(o)
        oname = fo.name[:-3] if fo.suffix == ".gz" else fo.name
        if oname == name:
            continue
        w, od, st = compare(f, fo)
        v = verdict(w, od, st)
        ok &= v == "PASS"
        print(f"{name} vs {oname}: {len(w)} validation queue windows, {od} ODME rows, {st} state rows -> {v}")
        if record:
            new = not RECORD.exists()
            with RECORD.open("a", encoding="utf-8", newline="") as g:
                wr = csv.DictWriter(g, FIELDS)
                if new:
                    wr.writeheader()
                wr.writerow(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), file=name, sha256=digest, other=oname,
                                 queue_windows=len(w), queue_window_ids=" ".join(w[:6]), odme_rows=od, state_rows=st, verdict=v))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--against", nargs="*", default=[]); ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if run(a.file, a.against, not a.no_record) else 3)


if __name__ == "__main__":
    main()
