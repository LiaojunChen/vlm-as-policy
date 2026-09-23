"""Recorded-plan semantic review diagnostic; no motion or task scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_clients import AuditedClient
from robodawn.plan_review import review_plan
from robodawn.mission_planner import validate_mission
from robotwin_harness_v3 import parse_json
from source_layout import ROOT,evaluation_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--plans',nargs='+',required=True,help='Recorded task:call-index diagnostic selectors')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repair-invalid',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain in this version')
    out.mkdir(parents=True,exist_ok=False)
    (out/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in evaluation_sources()},indent=2))
    results=[]
    for selector in args.plans:
        task,index=selector.split(':');source=args.cohort/'robodawn'/task
        prior=json.loads((source/'result.json').read_text())
        obs=json.loads((source/'trace.jsonl').read_text().splitlines()[0])['observation']
        response=json.loads((source/'calls'/f'call_{int(index):04d}.response.json').read_text())
        raw=response['response']['choices'][0]['message']['content'];steps=validate_mission(parse_json(raw))
        row=dict(diagnostic_only=True,source_episode=str(source.resolve()),source_call=int(index),plan=steps)
        try:
            instruction=prior['instruction']+'\nCompletion requirements: '+prior['task_contract']
            row['review']=review_plan(AuditedClient('zdtaichu',out/task/'calls'),instruction,steps,obs,[])
            if args.repair_invalid and not row['review']['valid']:
                from robodawn.mission_planner import RobodawnPlanner
                planner=RobodawnPlanner(AuditedClient('zdtaichu',out/task/'repair_calls'),instruction)
                planner.last_perception_failure='Semantic review of candidate '+json.dumps(steps)+': '+json.dumps(row['review']['issues'])
                planner._replan(obs,[]);row['revised_plan']=planner.steps
                row['revised_review']=review_plan(AuditedClient('zdtaichu',out/task/'revised_review_calls'),instruction,planner.steps,obs,[])
        except Exception as exc:row.update(error=str(exc),error_type=type(exc).__name__)
        results.append(row);(out/'report.json').write_text(json.dumps(results,indent=2))
        print(task,json.dumps(row),flush=True)


if __name__=='__main__':main()
