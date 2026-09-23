"""Current-image grounding from recorded observation/action memory; no scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_clients import AuditedClient
from robodawn.episodic_memory import EpisodicMemory,object_key
from robodawn.semantic_planner import ground
from source_layout import ROOT,evaluation_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--row',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources()},indent=2))
    rows=[json.loads(line) for line in (args.episode/'trace.jsonl').read_text().splitlines()]
    row=rows[args.row];obs=row['observation'];context=obs['episodic_memory']
    obs['image_paths']=[str(args.episode/'frames'/f"{obs['observation_id']:04d}_head_camera.png")]
    memory=EpisodicMemory();memory.objects={object_key(e['description']):e for e in context['objects']}
    memory.events=context['recent_outcomes']
    action={k:row['action'][k] for k in ('skill','arm','target','relation') if k in row['action']}
    report=dict(diagnostic_only=True,original_action=row['action'],recorded_observation_id=obs['observation_id'])
    try:
        report['new_action']=ground(AuditedClient('zdtaichu',out/'calls'),obs,action,memory=memory)
        report['memory']=memory.context()
    except Exception as exc:report.update(error=str(exc),error_type=type(exc).__name__)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
