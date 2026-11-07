"""asm2 entry step runner: `python asm2_step.py SCRIPT.py ARGS...` == `python SCRIPT.py ARGS...`, plus a clean exit.

pyarrow 23.0.1 under pandas 3 (Arrow-backed strings) can abort a finished process at interpreter teardown
("terminate called without an active exception", exit 134), intermittently and depending on machine state (VM1,
2026-10-09: 27/40 short reads in one window, 0/600 later; never mid-run, never wrong values). The step's work and
files are complete by then, so on success this runner runs the Python-level cleanup (gc.collect, logging.shutdown,
registered atexit functions, after joining non-daemon threads), flushes, and leaves with os._exit(0), skipping only the native static teardown.
An exception still prints its traceback and exits 1; sys.exit(n) keeps its code. Science code is untouched.
"""
import atexit, gc, logging, os, runpy, sys, threading
from pathlib import Path


def main():
    assert len(sys.argv) >= 2, "usage: asm2_step.py SCRIPT.py [ARGS...]"
    script = str(Path(sys.argv[1]).resolve())
    sys.argv = sys.argv[1:]                       # the script sees argv exactly as when run directly
    sys.path[0] = str(Path(script).parent)        # and the same sys.path[0] (its own directory)
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise                                  # nonzero exits propagate through normal teardown
    for t in threading.enumerate():                # as normal shutdown: wait for non-daemon threads first
        if t is not threading.main_thread() and not t.daemon:
            t.join()
    gc.collect()                                   # finalizers of unreferenced objects (open writers, temp files)
    logging.shutdown()                             # flush and close logging handlers
    atexit._run_exitfuncs()                        # registered cleanup still runs (it is cleared, so it runs once)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
