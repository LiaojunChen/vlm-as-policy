"""Fail deployment if any referenced episode evidence is missing."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'site'
data = json.loads((root / 'data/snapshot.json').read_text())
assert len(data['episodes']) == data['aggregate']['total'] == 50
assert sum(e['success'] for e in data['episodes']) == data['aggregate']['successes'] == 36
assert (root / 'vendor/echarts.min.js').stat().st_size > 100000
files = 0
for e in data['episodes']:
    media = root / 'media' / e['task']
    assert (media / 'video.mp4').stat().st_size > 0
    frames = set(e['final_frames'].values())
    frames.update(f for row in e['trace'] for f in row['frames'].values())
    for frame in frames:
        assert (media / 'frame' / frame).stat().st_size > 0
    files += len(frames) + 1
assert 'api/repeat' not in (root / 'app.js').read_text()
assert 'server.py' not in (root / 'index.html').read_text()
assert not any(p.is_symlink() for p in root.rglob('*'))
assert max(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 100 * 1024 ** 2
print(f'Validated 50 episodes, 36 successes, {files} media files, static paths and chart bundle.')
