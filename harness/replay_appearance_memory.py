"""Model-free current-pixel diagnostic, not end-to-end policy or task scoring."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from robodawn.appearance_memory import repair_robot_point
from robodawn.episodic_memory import EpisodicMemory,object_key
from source_layout import ROOT,evaluation_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--rows',type=int,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay in this development version')
    out.mkdir(parents=True,exist_ok=False)
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources()}
    (out/'code_hashes.json').write_text(json.dumps(hashes,indent=2))
    rows=[json.loads(line) for line in (args.episode/'trace.jsonl').read_text().splitlines()]
    reports=[]
    for index in args.rows:
        row=rows[index];obs=dict(row['observation']);action=row['action']
        obs['image_paths']=[str(args.episode/'frames'/f"{obs['observation_id']:04d}_head_camera.png")]
        memory=EpisodicMemory();memory.objects={object_key(e['description']):e for e in obs['episodic_memory']['objects']}
        rgb=np.asarray(Image.open(obs['image_paths'][0]).convert('RGB'))
        result=repair_robot_point(obs,memory,rgb,action['bbox'],action['target'])
        reports.append(dict(row=index,observation_id=obs['observation_id'],original_point=action['point'],recovery=result))
    report=dict(diagnostic_only=True,no_model_calls=True,no_physics_or_scoring=True,reports=reports)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
