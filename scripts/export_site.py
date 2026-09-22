"""Export a standalone Pages site from the local v72 observatory snapshot."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source match: {old[:70]}')
    return text.replace(old, new, 1)


def export(source, results, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'app.js', 'styles.css']:
        shutil.copy2(source / name, dest / name)
    (dest / 'data').mkdir(exist_ok=True)
    for name in ['snapshot.json', 'evidence.json', 'results.csv']:
        shutil.copy2(source / 'data' / name, dest / 'data' / name)
    data = json.loads((dest / 'data/snapshot.json').read_text())
    repeat = json.loads((results / data['repeat']['cohort'] / 'summary.json').read_text())
    keys = ['requested', 'terminated', 'completed', 'successes', 'failures', 'errors', 'updated_utc']
    data['repeat'].update({k: repeat.get(k, 0) for k in keys})
    data['repeat']['statuses'] = {status: sum(e.get('status') == status for e in repeat['episodes'])
                                 for status in sorted({e.get('status') for e in repeat['episodes']})}
    data['published_snapshot_utc'] = datetime.now(timezone.utc).isoformat()
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    (dest / 'data/snapshot.json').write_text(encoded)
    (dest / 'data/snapshot.js').write_text('window.REPORT_DATA = ' + encoded.replace('<', '\\u003c') + ';\n')
    (dest / 'data/repeat.json').write_text(json.dumps(data['repeat'], ensure_ascii=False))
    files = 0
    for episode in data['episodes']:
        task = episode['task']
        assert task.replace('_', '').isalnum()
        origin = results / data['cohort'] / 'robodawn' / task
        output = dest / 'media' / task
        (output / 'frame').mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin / 'continuous.mp4', output / 'video.mp4')
        frames = set(episode['final_frames'].values())
        frames.update(frame for row in episode['trace'] for frame in row['frames'].values())
        for frame in sorted(frames):
            assert Path(frame).name == frame
            shutil.copy2(origin / 'frames' / frame, output / 'frame' / frame)
        files += len(frames) + 1
    app = (dest / 'app.js').read_text()
    app = replace_once(app, '`media/${encodeURIComponent(task)}/${file}`',
                       '`media/${encodeURIComponent(task)}/${file === "video" ? "video.mp4" : file}`')
    app = replace_once(app, "fetch('api/repeat'", "fetch('data/repeat.json'")
    app = app.replace("live?'已读取当前汇总':'页面快照'", "live?'已加载发布快照':'发布快照'")
    app = app.replace('刷新未成功，仍显示页面快照。请通过 server.py 启动，以读取重复实验的当前进度。',
                      '快照读取失败，仍显示页面内置数据。请检查网络后重试。')
    app = app.replace('。未完成项不计为任务失败；最终总体成功率待全部结束后计算。',
                      '。这是发布时的静态快照，不实时连接评测服务器；未完成项不计为失败。')
    (dest / 'app.js').write_text(app)
    html = (dest / 'index.html').read_text()
    html = html.replace('https://sf3-scmcdn-cn.feishucdn.com/obj/feishu-static/miaoda/coding-unpkg-sdk/echarts@5.6.0/dist/echarts.min.js', 'vendor/echarts.min.js')
    html = html.replace('录像暂时无法加载，请通过本地 server.py 启动页面并确认源数据目录可读。',
                        '录像暂时无法加载，请检查网络，或通过“打开原始录像”重试。')
    html = html.replace('刷新进度 ↻', '加载发布快照 ↻')
    html = html.replace('规则见 build_data.py。', '规则见仓库中的 scripts/build_data.py。')
    (dest / 'index.html').write_text(html)
    (dest / '.nojekyll').touch()
    print(json.dumps({'episodes': len(data['episodes']), 'media_files': files,
                      'repeat_completed': data['repeat']['completed'],
                      'repeat_updated_utc': data['repeat']['updated_utc']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'site')
    args = parser.parse_args()
    export(args.source.resolve(), args.results.resolve(), args.output.resolve())
