"""Open audit for the final_solution build (worker R1, 2026-10-21). Loaded at start-up of every Python process of the build
when checks/open_audit is on PYTHONPATH (pipeline/10_base_c47.sh does that; TFB_OPEN_AUDIT_LOG names the log).

Logs, one line per event, every file a Python process OPENS (module loads included: io.open_code raises the same event) from a path that is neither inside
TFB_ROOT (final_solution), nor inside the interpreter (sys.prefix / base_prefix), nor in a system / temp location.
The release link data/raw/kaggle_public resolves outside TFB_ROOT; it is allowed explicitly (TFB_OPEN_AUDIT_ALLOW,
os.pathsep-separated resolved prefixes). An empty log = the build read nothing of the development repo through Python.
Native readers (pyarrow C++, LightGBM, torch.load via C) do not raise audit events, so this covers Python-level opens only.
It never changes behaviour: every error inside the hook is swallowed.
"""
import os, sys

_LOG = os.environ.get("TFB_OPEN_AUDIT_LOG")
_ROOT = os.environ.get("TFB_OPEN_AUDIT_ROOT") or os.environ.get("TFB_ROOT")   # audit root may differ from the build root
if _LOG and _ROOT:
    _real = os.path.realpath
    _ok = [_real(_ROOT), _real(sys.prefix), _real(sys.base_prefix), "/usr", "/lib", "/etc", "/proc", "/sys", "/dev", "/tmp",
           "/var", "/run", os.path.expanduser("~")] + [_real(p) for p in os.environ.get("TFB_OPEN_AUDIT_ALLOW", "").split(os.pathsep) if p]
    if os.name == "nt":
        _ok += [os.environ.get("SystemRoot", r"C:\Windows"), os.environ.get("TEMP", ""), os.environ.get("ProgramData", "")]
    _ok = tuple(os.path.normcase(p) for p in _ok if p)
    _seen = set()

    def _hook(event, args):
        if event != "open":   # module source / bytecode loads arrive here too (io.open_code raises "open")
            return
        try:
            p = args[0]
            if not isinstance(p, (str, bytes, os.PathLike)):
                return
            p = os.fsdecode(p)
            if not p or p.startswith("<") or os.path.normcase(p) in ("nul", "\\\\.\\nul", "/dev/null"):
                return
            r = os.path.normcase(_real(os.path.abspath(p)))
            if r.startswith(_ok) or r in _seen:
                return
            _seen.add(r)
            with open(_LOG, "a", encoding="utf-8") as f:   # the log itself is in _seen (no recursion)
                f.write(f"{os.getpid()}\t{event}\t{r}\t{' '.join(sys.argv[:2])[:200]}\n")
        except Exception:
            pass

    _seen.add(os.path.normcase(_real(os.path.abspath(_LOG))))
    sys.addaudithook(_hook)
