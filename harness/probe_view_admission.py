"""Fixed-frame source-view diagnostic; not a task run or official score."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_clients import AuditedClient
from robodawn.camera_views import in_camera
from robodawn.episodic_memory import EpisodicMemory
from robodawn.grounding_evidence import admit_wrist_contact
from robodawn.semantic_planner import ground
from robotwin_harness_v3 import SkillError
from source_layout import ROOT,evaluation_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--row',type=int,required=True)
    parser.add_argument('--retain-component-crop',action='store_true')
    parser.add_argument('--restore-observed-memory',action='store_true')
    parser.add_argument('--restore-held-search',action='store_true',help='Reconstruct motion search memory from this episode prior observed acquisition/contact history')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Keep diagnostic in this version')
    output.mkdir(parents=True,exist_ok=False)
    (output/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in evaluation_sources()},indent=2))
    rows=[json.loads(line) for line in (args.episode/'trace.jsonl').read_text().splitlines()]
    row=rows[args.row]
    obs=row['observation'];old=row['action'];memory=EpisodicMemory()
    report=dict(diagnostic_only=True,official_task_run=False,source_episode=str(args.episode.resolve()),
                row=args.row,observation_id=obs['observation_id'],memory_initialization='empty',original_action=old)
    if args.restore_observed_memory:
        from robodawn.episodic_memory import object_key
        recorded=obs['episodic_memory']
        memory.objects={object_key(entry['description']):entry for entry in recorded.get('objects',[])}
        memory.events=recorded.get('recent_outcomes',[])
        report['memory_initialization']='same_observation_recorded_episode_memory'
    if args.restore_held_search:
        for previous in rows[:args.row]:
            memory.held_search.feedback(previous['observation'],previous['action'],previous['result'])
        report['held_search_reconstructed_from_prior_rows']=args.row
    try:
        report['original_admission']=admit_wrist_contact(in_camera(obs,old.get('camera','head_camera')),old,include_head=True)
    except SkillError as exc:report['original_rejected']=str(exc)
    # Retain only the original semantic request; no old pixels or camera choice.
    request={key:old[key] for key in ('skill','arm','target','grasp_part','approach','donor','grounding_role','relation','axis','angle') if key in old}
    crop=None
    if args.retain_component_crop:
        parent=old['parent_grounding'];crop=parent['bbox'];request['camera']=parent.get('camera','head_camera')
        report['same_observation_parent_crop']=parent
    try:
        report['new_grounding']=ground(AuditedClient('zdtaichu',output/'calls'),obs,request,memory=memory,image_region=crop)
        report['memory']=memory.context()
    except SkillError as exc:report['new_failure']=str(exc)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
