"""Replay a real model-authored transfer with memory, without physics/scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_clients import AuditedClient
from robodawn.mission_planner import RobodawnPlanner
from source_layout import ROOT, evaluation_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.episode.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                                  for f in evaluation_sources()},indent=2))
    rows=[json.loads(line) for line in (source/'trace.jsonl').read_text().splitlines()]
    pick=next(row for row in rows if row['action'].get('skill')=='pick' and row['result'].get('skill_success'))
    place=next(row for row in rows if row['action'].get('skill')=='place')
    before=dict(pick['observation']);after=dict(place['observation'])
    for obs in (before,after):
        obs['image_paths']=[str(source/'frames'/f"{obs['observation_id']:04d}_head_camera.png")]
    planner=RobodawnPlanner(AuditedClient('zdtaichu',out/'calls'),before['instruction'])
    planner.steps=[place['action']['mission_intent']]
    report=dict(diagnostic_only=True,retained_model_intent=planner.steps[0],original_place=place['action'])
    try:
        planner._bind_transfer_identities(before)
        report['initial_memory']=planner.memory.context()
        planner.memory.feedback(pick['action'],pick['result'])
        report['new_place']=planner._destination(after,planner.steps[0],place['action']['arm'])
        report['final_memory']=planner.memory.context()
    except Exception as exc:
        report.update(error=str(exc),error_type=type(exc).__name__)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
