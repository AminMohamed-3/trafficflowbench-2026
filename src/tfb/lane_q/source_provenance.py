"""Preserve fitted source bytes and accept Git newline normalization only."""
import hashlib
from pathlib import Path

SOURCES={'features_source':'ongoing_features.py','learner_source':'ongoing_model.py'}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def normalized(raw):
    return raw.decode('utf-8-sig').replace('\r\n','\n').encode('utf-8')

def source_changes(out,before,after,source_dir=None):
    directory=Path(source_dir) if source_dir is not None else Path(__file__).parent
    accepted=set()
    for key,name in SOURCES.items():
        if before.get(key)==after.get(key):continue
        original=Path(out)/'fitted_source'/name
        current=(directory/name).read_bytes()
        if not original.exists() or digest(original.read_bytes())!=before[key] or normalized(original.read_bytes())!=normalized(current):
            raise RuntimeError('Changed learned source; refuse stale checkpoint')
        accepted.add(key)
    return accepted

def retain_sources(out,manifest,source_dir=None):
    directory=Path(source_dir) if source_dir is not None else Path(__file__).parent
    destination=Path(out)/'fitted_source';destination.mkdir(exist_ok=True)
    for key,name in SOURCES.items():
        path=destination/name
        if not path.exists():
            raw=(directory/name).read_bytes()
            if digest(raw)!=manifest[key]:raise RuntimeError('Missing original fitted-source bytes')
            path.write_bytes(raw)
        if digest(path.read_bytes())!=manifest[key]:raise RuntimeError('Fitted-source snapshot checksum mismatch')
