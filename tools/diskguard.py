"""Check that a large write will leave at least 10 GiB free."""
import argparse
from pathlib import Path
import shutil

GIB = 1024 ** 3

def check_free_space(destination, expected_bytes=0, reserve_gb=10.0):
    path = Path(destination).resolve()
    while not path.exists():
        if path.parent == path:
            raise FileNotFoundError(destination)
        path = path.parent
    if expected_bytes < 0 or reserve_gb < 0:
        raise ValueError("Expected write size and reserve must be nonnegative")
    free = shutil.disk_usage(path).free
    if free - expected_bytes < reserve_gb * GIB:
        raise RuntimeError(
            f"Disk guard: {free / GIB:.2f} GiB free, "
            f"{expected_bytes / GIB:.2f} GiB expected; retain {reserve_gb:.2f} GiB"
        )
    return free

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("path", type=Path)
    cli.add_argument("--expected-gb", type=float, default=0)
    cli.add_argument("--reserve-gb", type=float, default=10)
    args = cli.parse_args()
    free = check_free_space(args.path, int(args.expected_gb * GIB), args.reserve_gb)
    print(f"{free / GIB:.2f} GiB free; write permitted")

if __name__ == "__main__":
    main()
