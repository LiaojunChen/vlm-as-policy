"""Verify frozen sources, model-call evidence, traces and continuous videos."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/reviewed_v1'

def main():
    expected=json.loads((ROOT/'RETEST_CODE_SHA256.json').read_text())
    changed=[name for name,digest in expected.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    baseline=json.loads((ROOT/'SOURCE_BASELINE_SHA256.json').read_text())
    source=Path(json.loads((ROOT/'COPY_MANIFEST.json').read_text())['source'])
    source_changed=[name for name,digest in baseline.items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest]
    integrity={'files_checked':len(baseline),'changed_original_files':source_changed,'original_source_unchanged':not source_changed}
    (ROOT/'SOURCE_INTEGRITY_CHECK.json').write_text(json.dumps(integrity,indent=2))
    rows=[];videos=[];total_calls=0;frames=0
    for task in ('lift','pick_place'):
        for project in ('robodawn','show_harness'):
            for seed in range(5):
                directory=OUT/task/project/f'seed_{seed}'
                r=json.loads((directory/'result.json').read_text())
                assert r['status'] in ('completed','error'),directory
                assert not r.get('video_error') and not r.get('final_observation_error'),r
                trace=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
                assert len(trace)==r['policy_decisions']
                requests=sorted((directory/'calls').glob('*.request.json'))
                assert len(requests)==r['model_calls']
                for p in requests:
                    request=json.loads(p.read_text());assert request['model']=='qwen3-vl-plus'
                    assert set(request)<={'model','messages','temperature','max_tokens','response_format'}
                    assert p.with_name(p.name.replace('.request.json','.response.json')).is_file()
                    for message in request['messages']:
                        if isinstance(message['content'],list):
                            for part in message['content']:
                                if part.get('type')=='image_url':assert (p.parent/part['image_url']['url']).is_file()
                for event in trace:
                    obs=event['observation'];action=event['action']
                    for image in obs['image_paths']:assert Path(image).is_file()
                    if 'pixel' in action:assert action['observation_id']==obs['observation_id']
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=nb_frames,r_frame_rate,width,height:format=duration','-of','json',r['video']]))
                n=int(probe['streams'][0]['nb_frames']);assert n>=80
                videos.append({'task':task,'project':project,'seed':seed,'frames':n,**probe})
                total_calls+=r['model_calls'];frames+=n
                rows.append({k:r[k] for k in ('task','project','seed','model','status','success','end_reason',
                    'policy_decisions','model_calls','max_lift_m','stable_goal_s','elapsed_s','video')})
    assert len(rows)==20
    with (OUT/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for task in ('lift','pick_place'):
        for project in ('robodawn','show_harness'):
            part=[r for r in rows if r['task']==task and r['project']==project]
            summary.append({'task':task,'project':project,'episodes':len(part),
                'successes':sum(r['success'] is True for r in part),'errors':sum(r['status']=='error' for r in part),
                'model_calls':sum(r['model_calls'] for r in part),
                'mean_episode_s':sum(r['elapsed_s'] for r in part)/len(part)})
    audit={'episodes':len(rows),'model_calls':total_calls,'video_frames_total':frames,
           'frozen_source_files_checked':len(expected),'frozen_source_changed':changed,
           'original_integrity':integrity,'complete':True,'summary':summary}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'artifact_audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'video_metadata.json').write_text(json.dumps(videos,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
