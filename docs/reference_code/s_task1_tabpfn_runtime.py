"""0600 official v2 download/isolated runtime and synthetic-only fixture."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '3')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import hashlib
import importlib.metadata as md
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get('TFB_ROOT', '/workspace/work/main'))
OUT = ROOT / 'artifacts/lane_s/S_task1/tabpfn_v2'
RUNTIME = OUT / 'runtime'
PACKAGES = {'tabpfn': '2.0.9', 'scikit-learn': '1.6.1', 'pandas': '2.3.3',
            'einops': '0.8.1', 'huggingface-hub': '0.36.0',
            'pytz': '2025.2', 'tzdata': '2025.2', 'hf-xet': '1.1.10'}
MANIFEST_NAME = 'download_manifest_v2.json'
REVISION = '4972a65a1b30806315c6f92499959ffbfc69a673'
MODEL_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
MODEL_BYTES = 44390977
CODE_SHA = '04e3bb989e9328d510ea4fccb6c6a36c8269630685d460af4df5eb268206bf21'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['packages'] == PACKAGES and manifest['model_sha256'] == MODEL_SHA
        for r in manifest['downloads']:
            assert sha(OUT / r['path']) == r['sha256']
        return manifest
    wheels = OUT / 'wheels'
    wheels.mkdir(exist_ok=True)
    # The shared VM runtime has no pip. Resolve official compatible wheels
    # directly, preserving its environment and all downloaded sources.
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    compatible = set(sys_tags())
    RUNTIME.mkdir(exist_ok=True)
    downloads = []
    for package, version in PACKAGES.items():
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=45) as response:
            metadata = json.load(response)
        candidates = [r for r in metadata['urls'] if r['filename'].endswith('.whl')
                      and parse_wheel_filename(r['filename'])[3] & compatible]
        assert candidates, 'No official compatible binary wheel: ' + package
        published = sorted(candidates, key=lambda r: r['filename'])[0]
        wheel = wheels / published['filename']
        if not wheel.exists():
            with urllib.request.urlopen(published['url'], timeout=90) as response:
                wheel.write_bytes(response.read())
        assert sha(wheel) == published['digests']['sha256'] and wheel.stat().st_size == published['size']
        if package == 'tabpfn':
            assert sha(wheel) == CODE_SHA
            with zipfile.ZipFile(wheel) as z:
                licence = next(n for n in z.namelist() if n.endswith('/LICENSE'))
                (OUT / 'CODE_LICENSE.txt').write_bytes(z.read(licence))
        downloads.append(dict(path=str(wheel.relative_to(OUT)), url=published['url'],
                              sha256=sha(wheel), bytes=wheel.stat().st_size))
        with zipfile.ZipFile(wheel) as z:
            for member in z.infolist():
                part = Path(member.filename)
                assert not part.is_absolute() and '..' not in part.parts
                assert '.data' not in part.parts[0], 'Wheel relocation requires separate qualification'
                assert (member.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink in wheel'
            z.extractall(RUNTIME)
    for name in ('tabpfn-v2-regressor.ckpt', 'LICENSE.txt', 'config.json'):
        url = f'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/{REVISION}/{name}'
        path = OUT / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response, path.open('wb') as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        downloads.append(dict(path=name, url=url, sha256=sha(path), bytes=path.stat().st_size))
    model = OUT / 'tabpfn-v2-regressor.ckpt'
    assert sha(model) == MODEL_SHA and model.stat().st_size == MODEL_BYTES
    licence = (OUT / 'LICENSE.txt').read_text()
    assert 'Version 1.1' in licence and '10. Additional attribution.' in licence
    manifest = dict(experiment='0600', packages=PACKAGES, revision=REVISION,
                    model_sha256=MODEL_SHA, source_sha256=sha(__file__), downloads=downloads,
                    provenance='Official PyPI wheels and ungated official Prior-Labs v2 repository; VM only',
                    scope='Synthetic technical fixture only, no external dataset or traffic labels')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def fixture(manifest):
    sys.path.insert(0, str(RUNTIME))
    import numpy as np
    import torch
    from tabpfn import TabPFNRegressor
    import tabpfn.model.loading as loading
    def forbidden_download(*args, **kwargs):
        raise RuntimeError('Explicit qualified checkpoint required; library download forbidden')
    loading.download_model = forbidden_download
    versions = {p: md.version(p) for p in PACKAGES}
    assert versions == PACKAGES, versions
    assert torch.cuda.is_available()
    model_path = OUT / 'tabpfn-v2-regressor.ckpt'
    assert sha(model_path) == MODEL_SHA
    # This exact official checkpoint contains the v2 criterion object. Its old
    # loader passes weights_only=None; torch2.7 changed that default to True.
    # Restrict legacy compatibility to this process after the published hash.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    rng = np.random.default_rng(600)
    x = rng.normal(size=(96, 18)).astype(np.float32)
    y = x[:, 0] + .3 * x[:, 1] * x[:, 2] + .1 * rng.normal(size=96)
    x[[1, 10, 81, 91], [3, 7, 3, 7]] = np.nan
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    model = TabPFNRegressor(n_estimators=2, random_state=0, device='cuda',
                           model_path=model_path, fit_mode='fit_preprocessors',
                           inference_precision=torch.float32, n_jobs=3)
    model.fit(x[:80], y[:80])
    fitted = time.time()
    pred = model.predict(x[80:])
    assert pred.shape == (16,) and np.isfinite(pred).all()
    again = model.predict(x[80:])
    small = model.predict(x[80:82])
    np.testing.assert_allclose(again, pred, rtol=0, atol=1e-6)
    np.testing.assert_allclose(small, pred[:2], rtol=0, atol=1e-5)
    elapsed = time.time() - t
    memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    assert elapsed < 300 and memory < 8
    result = dict(experiment='0600', verdict='REF', technical_pass=True, versions=versions,
                  torch=torch.__version__, numpy=np.__version__, scipy=md.version('scipy'),
                  model_sha256=MODEL_SHA, manifest_sha256=sha(OUT / MANIFEST_NAME),
                  source_sha256=sha(__file__), fit_seconds=fitted-t, fixture_seconds=elapsed,
                  peak_cuda_allocated_gib=memory, finite=True, repeated_prediction=True,
                  query_batch_invariance=True, traffic_or_private_read=False,
                  limitation='Synthetic runtime only; no model quality or traffic gain evidence')
    (OUT / 'fixture.json').write_text(json.dumps(result, indent=2) + '\n')
    print('R25 TECHNICAL FIXTURE', json.dumps(result), flush=True)


def main():
    assert not (OUT / 'fixture.json').exists(), 'Preserve completed technical fixture'
    fixture(prepare())


if __name__ == '__main__':
    main()
