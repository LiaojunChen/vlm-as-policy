"""Replay measured history into mission repair, without executing any action."""
import argparse
import json
from pathlib import Path
from benchmark_clients import AuditedClient
from robodawn.mission_planner import RobodawnPlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--after',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier attempts')
    prior=args.cohort/'robodawn'/args.task
    result=json.loads((prior/'result.json').read_text())
    history=[json.loads(x) for x in (prior/'trace.jsonl').read_text().splitlines()[:args.after]]
    assert len(history)==args.after and args.after>0
    observation_id=history[-1]['action']['observation_id']+1
    paths=[str(prior/'frames'/f'{observation_id:04d}_{camera}.png') for camera in ('head_camera','left_camera','right_camera')]
    assert all(Path(path).is_file() for path in paths)
    instruction=result['instruction']+'\nCompletion requirements: '+result['task_contract']
    obs=dict(success=False,instruction=instruction,observation_id=observation_id,image_paths=paths,
             sensors=history[-1]['result']['sensors'])
    client=AuditedClient('zdtaichu',args.output/'calls');planner=RobodawnPlanner(client,instruction)
    record=dict(diagnostic_only=True,source_cohort=str(args.cohort),observation_id=observation_id)
    try:
        planner._replan(obs,history);record['steps']=planner.steps
    except Exception as exc:record['error']=str(exc)
    (args.output/'report.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
