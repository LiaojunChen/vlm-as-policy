"""Retained-observation model/geometry diagnostic, not a scoring rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark_clients import AuditedClient
from robodawn.mission_planner import RobodawnPlanner
from robotwin_harness_v3 import Bridge,estimate_table_height


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier diagnostics; choose a new output')
    args.output.mkdir(parents=True)
    from source_layout import evaluation_sources,ROOT
    (args.output/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources(ROOT)},indent=2))
    for task in args.tasks:
        prior=args.cohort/'robodawn'/task
        result=json.loads((prior/'result.json').read_text())
        instruction=result['instruction']+'\nCompletion requirements: '+result['task_contract']
        client=AuditedClient('zdtaichu',args.output/task/'calls')
        obs=dict(success=False,instruction=instruction,observation_id=1,
                 image_paths=[str(prior/'frames/0001_head_camera.png')],
                 sensors={s:dict(holding=False) for s in ('left','right')})
        trace=prior/'trace.jsonl'
        if trace.exists():
            with trace.open() as stream:
                row=json.loads(next(stream))
            obs.update(row['observation'])
            obs['image_paths']=[str(prior/'frames/0001_head_camera.png')]
        planner=RobodawnPlanner(client,instruction);record={'diagnostic_only':True}
        try:
            planner._replan(obs,[])
            record['steps']=planner.steps
            action=planner.plan(obs,[]);record.update(steps=planner.steps,action=action,memory=planner.memory.context())
            if action['skill'] in ('pick','grasp_handle'):
                d=np.load(prior/'frames/0001_rgbd.npz');b=Bridge.__new__(Bridge)
                b.cloud=d['world_xyz'];b.valid=d['valid'];b.rgb=np.zeros_like(b.cloud)
                b.table=estimate_table_height(b.cloud,b.valid);b.depth_id=1;b.landmarks={}
                record['geometry']=b.geometry(action)
        except Exception as exc:record['error']=str(exc)
        (args.output/task/'preflight.json').write_text(json.dumps(record,indent=2))
        print(task,json.dumps(record),flush=True)


if __name__=='__main__':main()
