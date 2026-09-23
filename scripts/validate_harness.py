"""Verify that the published code matches the imported v72 snapshot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'harness-source-manifest.json').read_text())
files = manifest['files']
assert len({f['path'] for f in files}) == len(files)
for record in files:
    relative = Path(record['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = ROOT / 'harness' / relative
    assert path.is_file() and not path.is_symlink(), relative
    content = path.read_bytes()
    assert len(content) == record['size'], relative
    assert hashlib.sha256(content).hexdigest() == record['sha256'], relative
    assert len(content) < 100 * 1024**2
assert sum(f['path'].startswith('seeds/') and f['path'].endswith('/result.json') for f in files) == 50
print(f'Validated {len(files)} original source/configuration hashes and 50 seed manifests.')
