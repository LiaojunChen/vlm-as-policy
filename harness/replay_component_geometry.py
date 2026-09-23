"""Replay one recorded model contact against its exact RGB-D; no simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robotwin_harness_v3 import Bridge


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--row',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (args.episode/'trace.jsonl').read_text().splitlines()]
    row=rows[args.row];action=dict(row['action']);oid=action['observation_id']
    data=np.load(args.episode/'frames'/f'{oid:04d}_rgbd.npz')
    bridge=Bridge.__new__(Bridge);bridge.cloud=data['world_xyz'];bridge.valid=data['valid']
    bridge.rgb=np.asarray(Image.open(args.episode/'frames'/f'{oid:04d}_head_camera.png'))
    bridge.table=row['observation']['table_height_m'];bridge.depth_id=oid;bridge.obs_id=oid
    bridge.landmarks={};bridge.held={};bridge.failed={}
    report=dict(diagnostic_only=True,episode=str(args.episode),observation_id=oid,
                original_action=action,original_geometry=row['result'].get('geometry'))
    try:
        report['new_geometry']=bridge.geometry(action)
    except Exception as exc:report['error']=str(exc)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('original_action','original_geometry')}))


if __name__=='__main__':main()
