"""Verify the ordinary shared LightGBM runtime; no process path overrides."""
from functools import lru_cache
import hashlib
import json
import os
import sys
from pathlib import Path
import yaml
from .windows import project_root

@lru_cache(maxsize=1)
def pin_lightgbm():
    """Compatibility name for the old pin helper; now only verifies imports."""
    root=project_root()
    config=yaml.safe_load((root/'configs/lightgbm_runtime.yaml').read_text())
    if not config.get('shared_venv_modified'):raise RuntimeError('Shared verified LightGBM runtime required under D0012')
    import lightgbm as lgb
    if os.environ.get('TFB_VM') == '1' and sys.platform.startswith('linux'):
        vm = json.loads(Path(__file__).with_name('vm_runtime.json').read_text())
        prefix = Path(vm['prefix']).resolve()
        package = Path(lgb.__file__).resolve().parent
        if Path(sys.prefix).resolve() != prefix or prefix not in package.parents:
            raise RuntimeError('Use the approved shared VM Python environment')
        if lgb.__version__ != vm['version'] or vm['version'] != str(config['version']):
            raise RuntimeError('Pinned shared VM LightGBM version changed')
        expected = vm['python_wrappers_lf_sha256']
        actual = {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  for p in package.glob('*.py')}
        if actual != expected:
            raise RuntimeError('Shared VM LightGBM Python wrapper parity failed')
        library = package / 'lib/lib_lightgbm.so'
        if hashlib.sha256(library.read_bytes()).hexdigest() != vm['linux_library_sha256']:
            raise RuntimeError('Pinned shared VM LightGBM library changed')
        return {**config, 'host': 'vm', 'vm_verification': vm}
    path=(root/config['path']).resolve()
    if lgb.__version__!=str(config['version']) or path not in Path(lgb.__file__).resolve().parents:
        raise RuntimeError('Use main .venv_tfb Python with the pinned shared LightGBM version')
    receipt=json.loads((root/config['shared_verification']).read_text())
    for name,digest in receipt['runtime_file_parity'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Shared LightGBM runtime differs from independently verified files')
    return config
