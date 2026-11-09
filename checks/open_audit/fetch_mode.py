"""Fetch mode of the open audit (C, 2026-11-08): discovery run of the clean room only. Imported by sitecustomize.py when
TFB_OPEN_AUDIT_FETCH = "DEVROOT|DEV" (Windows paths) is set.

When a Python process of the chain touches a path under DEVROOT that does not exist there (open for reading, os.stat,
os.path.exists|isfile|isdir, os.listdir / os.scandir / glob, pyarrow parquet reads, LightGBM model files), the same relative
path is COPIED from the development repository DEV first (copies, never hard links, so a later write cannot reach DEV) and one
line goes to TFB_OPEN_AUDIT_FETCH_LOG (default DEVROOT/FETCH.tsv): how<TAB>relative path<TAB>bytes.
  how = open | native (read) | stat (existence check) | list | glob (directory listing) | use (a fetched file read later)
        | *_dir (a directory created) | native_tree (a directory read by pyarrow, copied whole)
DEVROOT/work (the chain's own work folder) is never fetched. The run then behaves like the development run, and the log is
the list of development files the chain needs (the devroot extras). A second run without fetch mode proves the list sufficient.
Every error inside is swallowed: the hook never changes what a step computes, only which development files exist.
"""
import os, sys, shutil, threading

_FR, _FD = (os.path.normcase(os.path.abspath(p)) for p in os.environ["TFB_OPEN_AUDIT_FETCH"].split("|", 1))
_FL = os.environ.get("TFB_OPEN_AUDIT_FETCH_LOG") or os.path.join(_FR, "FETCH.tsv")
_FWORK = os.path.join(_FR, "work")
# never fetched (optional, ;-separated relative prefixes): development-only folders whose absence selects a portable code path
_SKIP = tuple(os.path.normcase(s.replace("/", os.sep)) for s in
              os.environ.get("TFB_OPEN_AUDIT_FETCH_SKIP", "").split(";") if s)
_done, _used, _lock, _tl = set(), set(), threading.RLock(), threading.local()
_stat, _exists, _isfile, _isdir, _listdir = os.stat, os.path.exists, os.path.isfile, os.path.isdir, os.listdir


def _log(how, rel, n=""):
    try:
        with open(_FL, "a", encoding="utf-8") as f:
            f.write(f"{how}\t{rel.replace(os.sep, '/')}\t{n}\n")
    except Exception:
        pass


def _rel(p):
    if p is None or isinstance(p, int):
        return None
    try:
        p = os.path.normcase(os.path.abspath(os.fsdecode(p)))
    except Exception:
        return None
    if not p.startswith(_FR + os.sep) or p == _FWORK or p.startswith(_FWORK + os.sep):
        return None
    rel = p[len(_FR) + 1:]
    if any(rel == s or rel.startswith(s + os.sep) for s in _SKIP):
        return None
    return p, rel


def fetch(p, how, tree=False):
    if getattr(_tl, "on", False) or isinstance(p, int):
        return
    _tl.on = True
    try:
        r = _rel(p)
        if r is None:
            return
        p, rel = r
        with _lock:
            if p in _done:
                if how in ("open", "native") and p not in _used and _isfile(p):
                    _used.add(p); _log("use", rel)
                return
            _done.add(p)
            if _exists(p):
                return
            src = os.path.join(_FD, rel)
            if _isfile(src):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                n = _stat(src).st_size
                if n > 200e6:   # big stored files (candidate CSVs): hard link to save disk; the chain never rewrites them in place
                    try:
                        os.link(src, p); how += "_link"
                    except OSError:
                        shutil.copyfile(src, p)
                else:
                    shutil.copyfile(src, p)
                _log(how, rel, n)
                how = how.replace("_link", "")
                if how in ("open", "native"):
                    _used.add(p)
            elif _isdir(src):
                if tree:
                    shutil.copytree(src, p, dirs_exist_ok=True)
                    for d, _, fs in os.walk(p):
                        for f in fs:
                            q = os.path.normcase(os.path.join(d, f)); _done.add(q)
                            _log(how + "_tree", q[len(_FR) + 1:], _stat(q).st_size)
                else:
                    os.makedirs(p, exist_ok=True)
                    _log(how + "_dir", rel)
    except Exception:
        pass
    finally:
        _tl.on = False


def _list(d, how):
    r = _rel(d)
    if r is None:
        return
    src = os.path.join(_FD, r[1])
    if not _isdir(src):
        return
    fetch(d, how)
    _tl.on = True
    try:
        names = _listdir(src)
    except Exception:
        names = []
    finally:
        _tl.on = False
    for e in names:
        fetch(os.path.join(r[0], e), how)


def _hook(event, args):
    if getattr(_tl, "on", False):
        return
    try:
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                return
            if mode is None and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                return
            fetch(args[0], "open")
        elif event in ("os.listdir", "os.scandir"):
            d = args[0] if args and args[0] is not None else "."
            if not isinstance(d, int):
                _list(d, "list")
        elif event.startswith("glob.glob"):
            pat = os.fsdecode(args[0]); root = args[2] if len(args) > 2 else None
            pat = os.path.abspath(os.path.join(os.fsdecode(root), pat) if root else pat)
            if _rel(os.path.dirname(pat)) is None and _rel(pat) is None:
                return
            import glob as _g
            _tl.on = True
            try:
                hits = _g.glob(os.path.join(_FD, os.path.relpath(pat, _FR)), recursive=bool(args[1]))
            finally:
                _tl.on = False
            for h in hits:
                fetch(os.path.join(_FR, os.path.relpath(h, _FD)), "glob")
    except Exception:
        _tl.on = False


def _wrap(f, how):
    def g(path, *a, **k):
        fetch(path, how)
        return f(path, *a, **k)
    g.__wrapped__ = f
    return g


def _srcs(x):
    return [s for s in (x if isinstance(x, (list, tuple)) else [x]) if isinstance(s, (str, os.PathLike))]


def _patch_pq(m):
    for name in ("read_table", "read_schema", "read_metadata", "read_pandas"):
        f = getattr(m, name, None)
        if f is not None and not hasattr(f, "__wrapped__"):
            def g(source, *a, __f=f, **k):
                for s in _srcs(source):
                    fetch(s, "native", tree=True)
                return __f(source, *a, **k)
            g.__wrapped__ = f
            setattr(m, name, g)
    for cname in ("ParquetFile", "ParquetDataset"):
        c = getattr(m, cname, None)
        if c is not None and not hasattr(c.__init__, "__wrapped__"):
            def init(self, source, *a, __i=c.__init__, **k):
                for s in _srcs(source):
                    fetch(s, "native", tree=True)
                return __i(self, source, *a, **k)
            init.__wrapped__ = c.__init__
            c.__init__ = init


def _patch_lgb(m):
    c = getattr(m, "Booster", None)
    if c is not None and not hasattr(c.__init__, "__wrapped__"):
        def init(self, *a, __i=c.__init__, **k):
            if k.get("model_file") is not None:
                fetch(k["model_file"], "native")
            return __i(self, *a, **k)
        init.__wrapped__ = c.__init__
        c.__init__ = init


_PATCH = {"pyarrow.parquet.core": _patch_pq, "pyarrow.parquet": _patch_pq, "lightgbm.basic": _patch_lgb}


class _Finder:
    def find_spec(self, name, path, target=None):
        if name not in _PATCH:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                orig = spec.loader.exec_module

                def exec_module(module, __o=orig, __n=name):
                    __o(module)
                    try:
                        _PATCH[__n](module)
                    except Exception:
                        pass
                spec.loader.exec_module = exec_module
                return spec
        return None


def install():
    os.stat = _wrap(_stat, "stat")
    os.path.exists, os.path.isfile, os.path.isdir = _wrap(_exists, "stat"), _wrap(_isfile, "stat"), _wrap(_isdir, "stat")
    try:   # sys.path entries: the import system tests directories with its own _path_isdir (not os.stat)
        import _frozen_importlib_external as _fie
        _fie._path_isdir = _wrap(_fie._path_isdir, "import")
    except Exception:
        pass
    sys.meta_path.insert(0, _Finder())
    sys.addaudithook(_hook)
