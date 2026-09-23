"""Validate model calls, associations, copied audit files and sensor assets."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]/'site'
def read(p):return json.loads(p.read_text())
def check():
    index=read(ROOT/'data/workflows/index.json')
    assert [x['outcome'] for x in index]==['success','failure']
    assert [(x['decisions'],x['calls']) for x in index]==[(8,10),(30,37)]
    files=0
    for item in index:
        d=read(ROOT/item['url']);assert d['id']==item['id']
        assert len(d['steps'])==d['result']['policy_decisions']==item['decisions']
        assert len(d['calls'])==d['result']['model_calls']==item['calls']
        assert bool(d['result']['success'])==(d['outcome']=='success')
        assert Counter(c for s in d['steps'] for c in s['calls'])==Counter(range(len(d['calls'])))
        for s in d['steps']:
            calls=[c for c in d['calls'] if c['step']==s['step']]
            assert [c['number'] for c in calls]==s['calls']
            if s['action'].get('raw') or s['action'].get('grounding_raw'):assert any(c['matched_action_fields'] for c in calls)
            assert len(s['depth'])==3
            for depth in s['depth']:assert (ROOT/depth['preview']).is_file() and (ROOT/depth['raw_url']).is_file()
            for url in s['rgb'].values():assert (ROOT/url).is_file()
        for c in d['calls']:
            assert read(ROOT/c['request_url'])==c['request']
            assert read(ROOT/c['response_url'])==c['response']
            assert c['raw_text']==c['response']['response']['choices'][0]['message']['content']
            assert len(c['attachments'])==sum(p.get('type')=='image_url' for m in c['request']['messages'] if isinstance(m['content'],list) for p in m['content'])
            for a in c['attachments']:assert (ROOT/a['url']).is_file()
        for e in read(ROOT/item['evidence_url']):
            assert hashlib.sha256((ROOT/e['published']).read_bytes()).hexdigest()==e['sha256'];files+=1
        assert read(ROOT/d['raw']['result.json'])==d['result']
        raw=[json.loads(l) for l in (ROOT/d['raw']['trace.jsonl']).read_text().splitlines()]
        for a,b in zip(raw,d['steps']):assert all(a[k]==b[k] for k in ['step','observation','action','result'])
    failure=read(ROOT/index[1]['url'])
    assert failure['calls'][7]['rejection_feedback']['reason']=='Unknown skill; use the available skill names.'
    assert failure['result']['end_reason']=='decision_budget'
    assert not any(s['result'].get('failure') for s in failure['steps'])
    success=read(ROOT/index[0]['url'])
    assert success['result']['end_reason']=='environment_success'
    assert success['calls'][1]['attachments']==[]
    print(f'Validated 2 full workflows: 38 decisions, 47 unabridged requests/responses, {files} source-file hashes, 114 RGB-D snapshots.')

if __name__=='__main__':check()
