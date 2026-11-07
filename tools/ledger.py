"""Claim, close and show ten-field entries in the shared LEDGER.md."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
import time
from project import main_root

OWNERS = {"L", "S", "Q", "D", "C", "U", "F", "O", "T", "B", "P"}
VERDICTS = {"RUNNING", "CARRY", "PASS", "NULL", "FAIL", "CLOSED", "REF"}
ID = re.compile(r"^(\d{4}) \| ")

@contextmanager
def locked(root):
    with (root / "LEDGER.lock").open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("LEDGER.lock timed out")
                time.sleep(0.05)
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)

def validate(line):
    fields = line.split(" | ")
    if len(fields) != 10 or any(not x.strip() for x in fields):
        raise ValueError("Ledger entry needs exactly ten nonempty fields")
    if not ID.match(line) or fields[2] not in OWNERS or fields[8] not in VERDICTS:
        raise ValueError("Invalid ledger ID, owner or verdict")
    return fields

def clean(value):
    if not value or not value.strip() or " | " in value or "\n" in value or "\r" in value:
        raise ValueError("Invalid ledger field")
    return value.strip()

def read_ledger(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        lines = file.readlines()
    fences = [i for i, line in enumerate(lines) if line.strip() == chr(96) * 3]
    if len(fences) < 2:
        raise ValueError("Missing fenced entries in LEDGER.md")
    first, last = fences[-2:]
    newline = "\r\n" if lines[last].endswith("\r\n") else "\n"
    for line in lines[first + 1:last]:
        if line.strip():
            validate(line.strip())
    return lines, first, last, newline

def save(path, lines):
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix="LEDGER.", suffix=".tmp", delete=False
    ) as file:
        temp = Path(file.name)
        file.writelines(lines)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def claim(args):
    root = main_root()
    path = root / "LEDGER.md"
    with locked(root):
        lines, first, last, newline = read_ledger(path)
        ids = [int(ID.match(line).group(1)) for line in lines[first + 1:last] if line.strip()]
        number = f"{max([99, *ids]) + 1:04d}"
        fields = [
            number, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            args.owner, args.component, args.formulation, args.instrument,
            args.result, args.reads, "RUNNING", args.path.replace("{id}", number),
        ]
        entry = " | ".join(clean(x) for x in fields)
        validate(entry)
        lines.insert(last, entry + newline)
        save(path, lines)
    print(number)

def close(args):
    root = main_root()
    path = root / "LEDGER.md"
    number = f"{int(args.id):04d}"
    with locked(root):
        lines, first, last, newline = read_ledger(path)
        matches = [i for i in range(first + 1, last) if lines[i].startswith(number + " | ")]
        if len(matches) != 1:
            raise ValueError(f"Expected one entry for {number}; found {len(matches)}")
        i = matches[0]
        fields = validate(lines[i].strip())
        if fields[2] != args.owner:
            raise PermissionError(f"{args.owner} cannot edit {fields[2]}'s entry")
        for index, value in [(5, args.instrument), (6, args.result), (7, args.reads), (9, args.path)]:
            if value is not None:
                fields[index] = clean(value)
        fields[8] = args.verdict
        entry = " | ".join(fields)
        validate(entry)
        lines[i] = entry + newline
        save(path, lines)
    print(entry)

def show(args):
    lines, first, last, _ = read_ledger(main_root() / "LEDGER.md")
    entries = [x.strip() for x in lines[first + 1:last] if x.strip()]
    print("\n".join(entries[-args.tail:]))

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    c = actions.add_parser("claim")
    c.add_argument("--owner", required=True, choices=sorted(OWNERS))
    for name in ("component", "formulation", "instrument", "result", "reads", "path"):
        c.add_argument("--" + name, required=True)
    c.set_defaults(func=claim)
    c = actions.add_parser("close")
    c.add_argument("--id", required=True)
    c.add_argument("--owner", required=True, choices=sorted(OWNERS))
    c.add_argument("--verdict", required=True, choices=sorted(VERDICTS))
    for name in ("instrument", "result", "reads", "path"):
        c.add_argument("--" + name)
    c.set_defaults(func=close)
    c = actions.add_parser("show")
    c.add_argument("--tail", type=int, default=30)
    c.set_defaults(func=show)
    args = cli.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
