"""Build a reproducible, read-only snapshot from v72 evaluation evidence."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'phyRSI/robodawn_robotwin_harness_v72/results'
MAIN = 'full50_completion_review_restored_protocol_standard_20260922'
REPEAT = 'repeat500_maxsteps20_protocol_standard_20260922'


def read(path):
    return json.loads(path.read_text())


def family(task):
    if task.startswith(('place_', 'put_', 'move_')):
        return '放置与搬运'
    if task.startswith(('stack_', 'blocks_')):
        return '堆叠与排序'
    if task.startswith(('pick_', 'handover_', 'lift_', 'adjust_', 'grab_')):
        return '抓取与双臂协作'
    if task.startswith(('open_', 'click_', 'press_', 'turn_', 'beat_', 'stamp_')):
        return '接触与工具操作'
    return '姿态与特殊目标'


def repeat_snapshot():
    s = read(SOURCE / REPEAT / 'summary.json')
    counts = Counter(e.get('status', 'pending') for e in s['episodes'])
    return {**{k: s.get(k, 0) for k in ['requested', 'terminated', 'completed', 'successes', 'failures', 'errors']},
            'updated_utc': s.get('updated_utc'), 'statuses': dict(counts), 'max_steps': 20,
            'cohort': REPEAT}


def build():
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)
    episodes, failures, evidence = [], Counter(), []
    for folder in sorted((SOURCE / MAIN / 'robodawn').iterdir()):
        if not (folder / 'result.json').exists():
            continue
        r = read(folder / 'result.json')
        assert r['status'] == 'completed' and isinstance(r['success'], bool)
        rows = [json.loads(line) for line in (folder / 'trace.jsonl').read_text().splitlines() if line.strip()]
        trace = []
        for row in rows:
            a, o, result = row.get('action', {}), row.get('observation', {}), row.get('result', {})
            failure = result.get('failure')
            if failure:
                failures[failure] += 1
            frames = {p.name.split('_', 1)[1].replace('.png', ''): p.name for p in
                      (Path(v) for v in o.get('image_paths', [])) if (folder / 'frames' / p.name).is_file()}
            trace.append({'step': row['step'], 'observation_id': o.get('observation_id'),
                          'skill': a.get('skill', a.get('name')), 'arm': a.get('arm'),
                          'target': a.get('target'), 'destination': a.get('destination', {}).get('target') if isinstance(a.get('destination'), dict) else a.get('destination'),
                          'skill_success': result.get('skill_success'), 'official_success': result.get('success'),
                          'failure': failure, 'recovery': a.get('mission_recovery'), 'frames': frames})
        fields = ['task', 'seed', 'status', 'success', 'instruction', 'policy_decisions', 'model_calls',
                  'end_reason', 'error', 'last_failure', 'elapsed_s', 'started_at', 'task_contract']
        episode = {k: r.get(k) for k in fields}
        episode.update(family=family(r['task']), trace=trace, video=(folder / 'continuous.mp4').is_file(),
                       failed_actions=sum(bool(x['failure']) for x in trace),
                       final_frames={Path(p).name.split('_', 1)[1].replace('.png', ''): Path(p).name
                                     for p in r.get('final_observation', {}).get('image_paths', [])
                                     if (folder / 'frames' / Path(p).name).is_file()})
        episodes.append(episode)
        for name in ['result.json', 'trace.jsonl']:
            p = folder / name
            evidence.append({'file': str(p.relative_to(SOURCE)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    source_summary = read(SOURCE / MAIN / 'summary.json')
    assert len(episodes) == source_summary['completed'] == 50
    assert sum(e['success'] for e in episodes) == source_summary['successes']
    groups = []
    for name in dict.fromkeys(e['family'] for e in episodes):
        group = [e for e in episodes if e['family'] == name]
        groups.append({'name': name, 'total': len(group), 'successes': sum(e['success'] for e in group)})
    cohorts = []
    for folder in sorted(SOURCE.iterdir()):
        if folder.name == REPEAT:
            continue
        s = read(folder / 'summary.json')
        cohorts.append({'id': folder.name, **{k: s[k] for k in ['requested', 'completed', 'successes', 'errors', 'updated_utc']}})
    manifest = read(SOURCE / MAIN / 'manifest.json')
    aggregate = {'total': len(episodes), 'successes': sum(e['success'] for e in episodes),
                 'mean_elapsed_s': mean(e['elapsed_s'] for e in episodes),
                 'median_elapsed_s': median(e['elapsed_s'] for e in episodes),
                 'mean_decisions': mean(e['policy_decisions'] for e in episodes),
                 'model_calls': sum(e['model_calls'] for e in episodes),
                 'failed_actions': sum(failures.values()),
                 'recovered_episodes': sum(e['success'] and e['failed_actions'] > 0 for e in episodes),
                 'end_reasons': dict(Counter(e['end_reason'] for e in episodes)),
                 'by_outcome': {str(success).lower(): {'mean_elapsed_s': mean(e['elapsed_s'] for e in episodes if e['success'] == success),
                               'mean_decisions': mean(e['policy_decisions'] for e in episodes if e['success'] == success)} for success in [True, False]}}
    data = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'updated_utc': source_summary['updated_utc'],
            'source': str(SOURCE.relative_to(ROOT.parent)), 'cohort': MAIN, 'aggregate': aggregate,
            'config': {k: manifest[k] for k in ['model_checkpoint', 'temperature', 'enable_thinking', 'max_steps', 'timeout_s', 'head_resolution', 'json_schema_mode']},
            'groups': groups, 'cohorts': cohorts, 'repeat': repeat_snapshot(),
            'failures': [{'message': k, 'count': v} for k, v in failures.most_common()], 'episodes': episodes}
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    (data_dir / 'snapshot.json').write_text(encoded)
    (data_dir / 'snapshot.js').write_text('window.REPORT_DATA = ' + encoded.replace('<', '\\u003c') + ';\n')
    (data_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    with (data_dir / 'results.csv').open('w', newline='') as out:
        columns = ['task', 'family', 'seed', 'success', 'policy_decisions', 'model_calls', 'elapsed_s', 'end_reason', 'last_failure']
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(episodes)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
