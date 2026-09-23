"""Ground a named target in recorded current cameras; diagnostic, not scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark_clients import AuditedClient
from source_layout import ROOT,evaluation_sources
from robodawn.semantic_planner import ground
from robodawn.camera_views import in_camera,evidence_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--target',required=True)
    parser.add_argument('--camera',choices=['head_camera','left_camera','right_camera'],required=True)
    parser.add_argument('--skill',choices=['pick','reach','place'],default='pick')
    parser.add_argument('--part',choices=['body','rim','handle','edge'],default='body')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain in isolated version')
    out.mkdir(parents=True,exist_ok=False)
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources()},indent=2))
    report=json.loads((args.episode/'report.json').read_text());obs=report['final_observation']
    obs['image_paths']=[str(args.episode/'frames'/f"{obs['observation_id']:04d}_{camera}.png") for camera in ('head_camera','left_camera','right_camera')]
    result=dict(diagnostic_only=True,physical_execution=False,source=str(args.episode),observation_id=obs['observation_id'])
    try:
        action=ground(AuditedClient('zdtaichu',out/'calls'),obs,dict(skill=args.skill,arm='right',target=args.target,grasp_part=args.part,camera=args.camera))
        view=in_camera(obs,args.camera)
        with np.load(evidence_path(view['image_paths'][0],obs['observation_id'],args.camera),allow_pickle=False) as data:
            height,width=data['valid'].shape
            x,y=np.rint(np.asarray(action['point'])*[width-1,height-1]/1000).astype(int)
            result.update(action=action,current_depth_valid=bool(data['valid'][y,x]),observed_world_point=data['world_xyz'][y,x].tolist())
    except Exception as exc:result.update(error=str(exc),error_type=type(exc).__name__)
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
