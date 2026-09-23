"""Replay only recorded policy observations and intentions; not task scoring."""
import argparse
import json
from pathlib import Path
from benchmark_clients import AuditedClient
from robodawn.episodic_memory import object_key
from robodawn.mission_planner import RobodawnPlanner,validate_mission
from robotwin_harness_v3 import parse_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--initial-plan',type=Path,required=True)
    parser.add_argument('--after',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (args.episode/'trace.jsonl').read_text().splitlines()]
    history=rows[:args.after];obs=rows[args.after]['observation']
    obs['image_paths']=[str(args.episode/'frames'/f"{obs['observation_id']:04d}_head_camera.png")]
    result=json.loads((args.episode/'result.json').read_text())
    instruction=result['instruction']+'\nCompletion requirements: '+result['task_contract']
    response=json.loads(args.initial_plan.read_text())['response']['choices'][0]['message']['content']
    steps=validate_mission(parse_json(response))
    planner=RobodawnPlanner(AuditedClient('zdtaichu',args.output/'calls'),instruction)
    planner.inventory_attempted=True
    context=obs['episodic_memory']
    planner.memory.objects={object_key(e['description']):e for e in context['objects']}
    planner.memory.events=context['recent_outcomes']
    planner.memory.intentions.initialize(steps)
    for row in history:planner.memory.intentions.feedback(row['action'],row['result'])
    report=dict(diagnostic_only=True,source_episode=str(args.episode),
                observation_id=obs['observation_id'],intentions_before=planner.memory.intentions.context())
    try:
        planner._replan(obs,history)
        report.update(steps=planner.steps,remaining_coverage_valid=True)
    except Exception as exc:report.update(error=str(exc),remaining_coverage_valid=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
