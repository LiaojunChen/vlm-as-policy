"""Read-only regression against recorded successful contact actions."""
import argparse
import hashlib
import json
from pathlib import Path
from robodawn.camera_views import in_camera
from robodawn.grounding_evidence import admit_wrist_contact
from source_layout import ROOT,evaluation_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-spatial-references',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain in this version')
    out.mkdir(parents=True,exist_ok=False)
    (out/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in evaluation_sources()},indent=2))
    report=dict(diagnostic_only=True,source_cohort=str(args.cohort.resolve()),checked=0,accepted=0,no_saved_depth=0,rejections=[],spatial_references=0)
    for path in sorted((args.cohort/'robodawn').glob('*/trace.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);action=row['action'];obs=row['observation']
            from robotwin_harness_v3 import RELATIVE_DIRECTIONS
            spatial=(args.include_spatial_references and action.get('skill')=='place' and action.get('relation') in RELATIVE_DIRECTIONS)
            if (action.get('skill') not in ('pick','grasp_handle','push','handover') and not spatial) or not row['result'].get('skill_success'):continue
            report['checked']+=1
            if spatial:report['spatial_references']+=1
            try:
                evidence=admit_wrist_contact(in_camera(obs,action.get('camera','head_camera')),action,include_head=True)
                report['no_saved_depth' if evidence is None else 'accepted']+=1
            except Exception as exc:
                report['rejections'].append(dict(task=path.parent.name,observation_id=obs['observation_id'],action=action,error=str(exc)))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
