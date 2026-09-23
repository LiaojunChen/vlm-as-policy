"""Copy the v72 code snapshot, retaining original bytes and a SHA-256 inventory.

Source/configuration is selected explicitly; external simulator meshes, model
weights, runtime binaries, caches and result artifacts are not source code.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.txt', '.md',
              '.j2', '.jinja', '.html', '.css', '.js', '.cu', '.cpp', '.h', '.c',
              '.urdf', '.srdf', '.xacro', '.xrdf', '.cfg', '.in', '.cff', '.typed',
              '.dockerfile', '.example', '.mtl', '.usd', '.usda', '.svg'}
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.git', '.cache', '.venv', 'node_modules', 'build', 'dist'}
NAMES = {'LICENSE', 'NOTICE', 'COPYING', 'AUTHORS', 'Dockerfile', 'Makefile',
         '.gitignore', '.gitmodules', 'METADATA', 'RECORD', 'WHEEL', 'INSTALLER'}
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')


def sync(source):
    dest = ROOT / 'harness'
    manifest = ROOT / 'harness-source-manifest.json'
    records, excluded = [], []
    for directory, dirs, files in os.walk(source, followlinks=False):
        rel_dir = Path(directory).relative_to(source)
        for name in list(dirs):
            p = Path(directory) / name
            if name in SKIP_DIRS or (rel_dir == Path('.') and name == 'results') or p.is_symlink():
                dirs.remove(name)
                excluded.append({'path': str(p.relative_to(source)), 'reason': 'external_link' if p.is_symlink() else 'runtime_or_results'})
        for name in sorted(files):
            p = Path(directory) / name
            rel = p.relative_to(source)
            if p.is_symlink() or (p.suffix not in EXTENSIONS and name not in NAMES):
                excluded.append({'path': str(rel), 'reason': 'external_link' if p.is_symlink() else 'asset_or_binary'})
                continue
            if name.startswith('.env'):
                excluded.append({'path': str(rel), 'reason': 'local_environment'})
                continue
            content = p.read_bytes()
            if SECRET.search(content):
                raise ValueError(f'Credential-like content found in {rel}; review before copying')
            if len(content) >= 100 * 1024**2:
                raise ValueError(f'Source exceeds GitHub limit: {rel}')
            records.append({'path': str(rel), 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    records.sort(key=lambda x: x['path'])
    previous = json.loads(manifest.read_text())['files'] if manifest.exists() else []
    stale = {x['path'] for x in previous} - {x['path'] for x in records}
    if stale:
        raise ValueError(f'Review removed source paths before syncing: {sorted(stale)}')
    for record in records:
        p = source / record['path']
        target = dest / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']
    data = {'source': 'phyRSI/robodawn_robotwin_harness_v72',
            'synced_utc': datetime.now(timezone.utc).isoformat(), 'files': records,
            'excluded': sorted(excluded, key=lambda x: x['path']),
            'note': 'Original code/configuration bytes preserved; package names and upstream licenses retained. External assets and runtime environments must be provisioned separately.'}
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'source_mib': sum(r['size'] for r in records)/1024**2,
                      'excluded': dict(Counter(x['reason'] for x in excluded))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    sync(args.source.resolve())
