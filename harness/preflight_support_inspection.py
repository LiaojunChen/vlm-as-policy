"""Current recorded-frame policy diagnostic, not a physical task score."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark_clients import AuditedClient
from flat_support_memory import refresh_landmarks
from robodawn.mission_planner import RobodawnPlanner
from source_layout import ROOT,evaluation_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Keep diagnostic in this version')
    output.mkdir(parents=True,exist_ok=False)
    (output/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in evaluation_sources()},indent=2))
    obs=json.loads(args.trace.read_text().splitlines()[0])['observation']
    prior=json.loads((args.episode/'result.json').read_text())
    instruction=prior['instruction']+'\nCompletion requirements: '+prior['task_contract']
    views={};index=obs['observation_id']
    for camera,path in zip(('head_camera','left_camera','right_camera'),obs['image_paths']):
        suffix='rgbd' if camera=='head_camera' else camera+'_rgbd'
        with np.load(Path(path).with_name(f'{index:04d}_{suffix}.npz')) as d:
            views[camera]=dict(rgb=np.asarray(Image.open(path)),cloud=d['world_xyz'],valid=d['valid'])
    obs['visual_landmarks']=refresh_landmarks({},views,obs['table_height_m'],index)
    planner=RobodawnPlanner(AuditedClient('zdtaichu',output/'calls'),instruction)
    report=dict(diagnostic_only=True,source_trace=str(args.trace.resolve()),source_episode=str(args.episode.resolve()))
    try:report.update(action=planner.plan(obs,[]),steps=planner.steps)
    except Exception as exc:report.update(error=str(exc),error_type=type(exc).__name__)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
