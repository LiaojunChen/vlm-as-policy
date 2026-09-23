"""Validate the published denominator, cases, all videos and every referenced frame."""
import csv
import hashlib
import json
from validate_workflow import check as check_workflow
from collections import Counter
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parents[1]/'site'
data=json.loads((root/'data/snapshot.json').read_text())
episodes=data['episodes'];agg=data['aggregate']
assert data['schema_version']==2
assert len(episodes)==agg['total']==500
assert len({e['id'] for e in episodes})==500
assert sum(e['success'] for e in episodes)==agg['successes']==233
assert agg['errors']==0
assert Counter(e['end_reason'] for e in episodes)==agg['end_reasons']==dict(environment_success=233,invalid_model_action=109,decision_budget=106,wall_time_budget=52)
assert agg['elapsed_count']==448 and agg['elapsed_missing']==52
assert agg['mean_elapsed_s']==mean(e['elapsed_s'] for e in episodes if e['elapsed_s'] is not None)
assert all(e['end_reason']=='wall_time_budget' for e in episodes if e['elapsed_s'] is None)
assert len(data['tasks'])==50
for task in data['tasks']:
    es=[e for e in episodes if e['task']==task['task']]
    assert len(es)==10 and len({e['seed'] for e in es})==10
    assert sorted(e['repeat_index'] for e in es)==list(range(10))
    assert sum(e['success'] for e in es)==task['successes']
    assert task['episodes']==[e['id'] for e in es]
frames_count=0;traces={}
for e in episodes:
    assert e['seed_manifest']['seed']==e['seed']
    assert e['seed_manifest']['status']=='expert_validated'
    assert e['seed_manifest']['official_seed_selection']=='official_sequential_seed_scan_with_expert_filter'
    assert e['media']==f"media/official500/{e['task']}/repeat_{e['repeat_index']:02d}"
    p=root/e['media']; assert (p/'video.mp4').stat().st_size>0
    detail=json.loads((root/e['trace_url']).read_text()); assert detail['id']==e['id']
    trace=detail['trace'];traces[e['id']]=trace
    assert len(trace)==e['trace_count']
    assert sum(bool(t['failure']) for t in trace)==e['failed_actions']
    frames=set(e['final_frames'].values())|set(e['first_frames'].values())
    frames.update(f for t in trace for f in t['frames'].values())
    assert frames
    for frame in frames:
        assert Path(frame).name==frame and (p/'frame'/frame).stat().st_size>0
    frames_count+=len(frames)
for c in data['cases']:
    e=next(e for e in episodes if e['id']==c['id']);assert not e['success']
    if c['comparison_id']:
        other=next(e for e in episodes if e['id']==c['comparison_id'])
        assert other['success'] and other['task']==e['task'] and other['seed']!=e['seed']
    assert c['matching_steps']==[t['step'] for t in traces[c['id']] if c['signal'] and c['signal'] in (t['failure'] or '')]
assert [len(c['matching_steps']) for c in data['cases']]==[0,26,0,13,23,0]
evidence=json.loads((root/'data/evidence.json').read_text())
published=[x for x in evidence if 'published_video' in x]
assert len(published)==500
for e in published:
    assert hashlib.sha256((root/e['published_video']).read_bytes()).hexdigest()==e['sha256']
for name,count in [('results.csv',500),('tasks.csv',50)]:
    with (root/'data'/name).open() as f: assert len(list(csv.DictReader(f)))==count
js=(root/'data/snapshot.js').read_text()
assert json.loads(js[len('window.REPORT_DATA='):-2])==data
assert (root/'vendor/echarts.min.js').stat().st_size>100000
assert 'official.js' in (root/'index.html').read_text()
assert not any(p.is_symlink() for p in root.rglob('*'))
sizes=[p.stat().st_size for p in root.rglob('*') if p.is_file()]
assert max(sizes)<100*1024**2
assert sum(sizes)<1024**3, 'Published site exceeds Pages 1 GiB limit'
print(f'Validated 500 episodes, 233 successes, 50 distinct-seed sets, 500 video hashes, {frames_count} frames, 6 cases; site {sum(sizes)/1024**2:.1f} MiB.')
check_workflow()
