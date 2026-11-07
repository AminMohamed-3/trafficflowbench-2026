"""Pinned Windows and provisioned VM LightGBM runtimes."""
import hashlib,importlib,os
from pathlib import Path
from tfb.paths import project_root

WHEEL_SHA='37089ee95664b6550a7189d887dbf098e3eadab03537e411f52c63c121e3ba4b'
DLL_SHA='d2a41567e16b127b2486b417cc25813588fe015a0481e06fe8ce662ebed1ec61'
VM_LIBRARY_SHA='6a12e1a7b13824d14c252d2a4d2e36af467383331a114eb0e7448eecbf02d7b1'

def load_lightgbm():
    lgb=importlib.import_module('lightgbm')
    assert lgb.__version__=='4.6.0','Use the pinned LightGBM 4.6.0 runtime'
    if os.name=='nt':
        shared=(project_root()/'.venv_tfb/Lib/site-packages').resolve()
        assert Path(lgb.__file__).resolve().is_relative_to(shared),'Remove isolated overrides and restart the learner'
        assert hashlib.sha256(Path(lgb.basic._LIB._name).read_bytes()).hexdigest()==DLL_SHA,'Shared LightGBM native integrity'
    else:
        assert Path(lgb.__file__).resolve().is_relative_to(Path('/workspace/tfb/venv')),'Only the provisioned pinned VM runtime is allowed'
        assert hashlib.sha256(Path(lgb.basic._LIB._name).read_bytes()).hexdigest()==VM_LIBRARY_SHA,'VM LightGBM native integrity'
    return lgb
