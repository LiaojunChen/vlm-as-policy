"""Frozen, bounded model-free release diagnostics after a full cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from source_layout import ROOT,evaluation_sources


def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evaluation_sources(root)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after-cohort',type=Path,required=True)
    p.add_argument('--source-cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True,help='task:prefix_length:first_intervention_row')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('New results must remain in isolated candidate')
    cases=[]
    for spec in a.cases:
        task,length,start=spec.split(':');length,start=int(length),int(start)
        if not task.replace('_','').isalnum() or not 0<=start<length<=12:raise ValueError('Invalid diagnostic case')
        target=ROOT/'results'/f'measured_release_prefix_{task}_20260921'
        if target.exists():raise ValueError('Never overwrite an earlier diagnostic')
        cases.append(dict(task=task,actions=length,enable_from_row=start,output=str(target)))
    out.mkdir(parents=True,exist_ok=False);frozen=hashes(ROOT)
    (out/'code_hashes.json').write_text(json.dumps(frozen,indent=2))
    state=dict(diagnostic_only=True,no_new_model_calls=True,pid=os.getpid(),source_files=len(frozen),
               waiting_for=str(a.after_cohort.resolve()),source_cohort=str(a.source_cohort.resolve()),
               cases=cases,status='waiting',queued_utc=now(),results=[])
    (out/'queue.json').write_text(json.dumps(state,indent=2));print('QUEUED',json.dumps(state),flush=True)
    try:
        manifest=json.loads((a.after_cohort/'manifest.json').read_text())
        if len(manifest['jobs'])!=50:raise ValueError('Expected original full50 predecessor')
        deadline=time.monotonic()+21600
        while True:
            terminal=0
            for job in manifest['jobs']:
                path=a.after_cohort/job['project']/job['task']/'result.json'
                if path.exists() and json.loads(path.read_text()).get('status') in ('completed','error'):terminal+=1
            if terminal==50:break
            if time.monotonic()>deadline:raise TimeoutError('Predecessor did not finish in queue bound')
            time.sleep(15)
        if json.loads((a.after_cohort/'code_hashes.json').read_text())!=hashes(a.after_cohort.resolve().parent.parent):
            raise RuntimeError('Predecessor source changed during full evaluation')
        state.update(status='running',started_utc=now())
        (out/'queue.json').write_text(json.dumps(state,indent=2))
        for case in cases:
            if frozen!=hashes(ROOT):raise RuntimeError('Candidate changed after freeze')
            command=[sys.executable,'-u',str(ROOT/'probe_measured_release.py'),
                     '--episode',str((a.source_cohort/'robodawn'/case['task']).resolve()),
                     '--actions',str(case['actions']),'--enable-from-row',str(case['enable_from_row']),
                     '--continue-on-error','--output',case['output']]
            with (out/(case['task']+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                      stdout=log,stderr=subprocess.STDOUT,timeout=660)
            report_path=Path(case['output'])/'report.json'
            report=json.loads(report_path.read_text()) if report_path.exists() else {}
            state['results'].append(dict(task=case['task'],returncode=result.returncode,
                report_path=str(report_path),official_success_after_retained_prefix=report.get('official_success_after_retained_prefix'),
                error=report.get('error'),actions_executed=report.get('actions_executed')))
            (out/'queue.json').write_text(json.dumps(state,indent=2))
            print('CASE_TERMINAL',json.dumps(state['results'][-1]),flush=True)
        if frozen!=hashes(ROOT):raise RuntimeError('Candidate changed during diagnostic')
        state['status']='finished'
    except Exception as exc:state.update(status='error',error=str(exc),error_type=type(exc).__name__)
    state['updated_utc']=now()
    (out/'queue.json').write_text(json.dumps(state,indent=2));print('QUEUE_TERMINAL',json.dumps(state),flush=True)


if __name__=='__main__':main()
