"""Physical regression of recorded model actions; diagnostic, never scoring.

This isolates executor geometry changes from model variability. Historical
actions are deliberately replayed, so this is NOT an end-to-end policy run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parent


class DiagnosticBudgetExceeded(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--actions',type=int,required=True)
    parser.add_argument('--continue-on-error',action='store_true',help='Diagnostic only: retain failed prefix actions instead of stopping at the first one')
    parser.add_argument('--observed-memory-on-reground',action='store_true',help='Restore the retained row\'s already observed episode identity memory for a same-prefix grounding comparison')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-fk',action='store_true')
    parser.add_argument('--audit-motion-status',action='store_true',help='Read-only recording of original motion planner failure categories')
    parser.add_argument('--audit-table-clearance',action='store_true',help='Read-only robot FK/collision-shape audit against the observed table plane')
    parser.add_argument('--contact-stop',action='store_true')
    parser.add_argument('--mesh-collision',action='store_true')
    parser.add_argument('--appearance-memory',action='store_true')
    parser.add_argument('--inspect-arm',choices=['left','right'])
    parser.add_argument('--inspect-anchor-from-row',type=int)
    parser.add_argument('--inspect-flat-colour',help='Diagnostic anchor on a currently observed flat colour component')
    parser.add_argument('--home-after-inspection',action='store_true')
    parser.add_argument('--inspect-pick-target',help='Diagnostic-only fresh model pick after the explicit inspection')
    parser.add_argument('--inspect-pick-part',choices=['body','rim','handle','edge'],default='rim')
    parser.add_argument('--inspect-pick-camera',choices=['head_camera','left_camera','right_camera'])
    parser.add_argument('--inspect-place-target',help='Diagnostic-only current-grounded container delivery after the fresh pickup')
    parser.add_argument('--support-preview',action='store_true',help='Replay only support identity bindings already observed before pickup')
    parser.add_argument('--retained-trace',type=Path,help='Optional previous diagnostic action trace for the same source task/seed')
    parser.add_argument('--reground-from-row',type=int,help='Diagnostic: remeasure contact actions from this prefix row in the current frame')
    parser.add_argument('--dual-lift-after-prefix',action='store_true',help='Diagnostic only: lift after actual bilateral holding')
    args=parser.parse_args();source=args.episode.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay in this isolated copy')
    out.mkdir(parents=True,exist_ok=False)
    from source_layout import evaluation_sources
    (out/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in evaluation_sources()},indent=2))
    prior=json.loads((source/'result.json').read_text());task=prior['task']
    trace_path=(args.retained_trace or source/'trace.jsonl').resolve()
    recorded=[json.loads(line) for line in trace_path.read_text().splitlines()][:args.actions]
    if len(recorded)!=args.actions:raise ValueError('Requested prefix not recorded')
    seed=json.loads((ROOT/'seeds'/task/'result.json').read_text())
    assert seed['seed']==prior['seed']
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1';os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    install();env=class_decorator(task);bridge=None
    config,_=load_task_args(dict(task_name=task,policy_name='robodawn',task_config='demo_clean'))
    config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
    report=dict(diagnostic_only=True,retained_model_actions_not_end_to_end=True,source_episode=str(source),task=task,retained_trace=str(trace_path.resolve()))
    rows=[]
    def timeout(signum,frame):raise DiagnosticBudgetExceeded('Diagnostic exceeded 600 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
        if args.audit_fk:
            import numpy as np
            entity=env.robot.right_entity;links=entity.get_links()
            original=entity.get_qpos().copy();changed=original.copy()
            names=[joint.name for joint in entity.get_active_joints()]
            changed[names.index(env.robot.right_arm_joints[0].name)]+=.2
            before=np.array([link.get_pose().p for link in links])
            model=entity.create_pinocchio_model();root=entity.get_root_pose().to_transformation_matrix()
            model.compute_forward_kinematics(original)
            predicted=np.array([(root@model.get_link_pose(i).to_transformation_matrix())[:3,3] for i in range(len(links))])
            model.compute_forward_kinematics(changed)
            moved=np.array([(root@model.get_link_pose(i).to_transformation_matrix())[:3,3] for i in range(len(links))])
            try:
                entity.set_qpos(changed)
                direct=np.array([link.get_pose().p for link in links])
            finally:entity.set_qpos(original)
            report['robot_fk_audit']=dict(pinocchio_current_max_error_m=float(np.max(np.linalg.norm(predicted-before,axis=1))),
                set_qpos_direct_max_displacement_m=float(np.max(np.linalg.norm(direct-before,axis=1))),
                pinocchio_max_displacement_m=float(np.max(np.linalg.norm(moved-predicted,axis=1))))
            print('ROBOT_FK_AUDIT',json.dumps(report['robot_fk_audit']),flush=True)
        bridge=Bridge(env,seed['instruction'],out)
        if args.audit_table_clearance:
            from robot_table_clearance import RobotTableClearance
            from robotwin_harness_v3 import component_opening
            report['table_clearance_audit']=[]
            original_choose=bridge.choose_grasp
            def audited_choose(action,geometry):
                result=original_choose(action,geometry)
                audit=RobotTableClearance(env.robot,action['arm'],bridge.table)
                item=dict(arm=action['arm'],target=action['target'],
                          robot_fk_current_max_error_m=audit.calibration_error(),candidates=[])
                for plan in bridge.plan_cache:
                    item['candidates'].append(dict(pose=plan['pose'],
                        clearance=audit.evaluate(plan['result']['position'][-1],component_opening(geometry))))
                report['table_clearance_audit'].append(item)
                print('TABLE_CLEARANCE_AUDIT',json.dumps(item),flush=True)
                return result
            bridge.choose_grasp=audited_choose
        if args.audit_motion_status:
            # Observe original results without changing planner settings,
            # validity checks, RNG, input states, returned paths or execution.
            for side in ('left','right'):
                engine=getattr(env.robot,side+'_planner').motion_gen
                original=engine.plan_single
                def audited(*call_args,_original=original,_side=side,**call_kwargs):
                    result=_original(*call_args,**call_kwargs)
                    audit=dict(arm=_side,action_index=bridge.index,
                               success=bool(result.success.all().item()),
                               status=str(result.status),valid_query=bool(result.valid_query))
                    if call_args:
                        audit['start_joint_positions']=call_args[0].position.detach().cpu().tolist()
                    with (out/'motion_status.jsonl').open('a') as stream:stream.write(json.dumps(audit)+'\n')
                    return result
                engine.plan_single=audited
        bridge.experimental_contact_stop=args.contact_stop
        bridge.screen_robot_mesh_collision=args.mesh_collision
        if args.inspect_arm:
            observation=bridge.observe()
            action=dict(skill='inspect',arm=args.inspect_arm,observation_id=observation['observation_id'])
            if args.inspect_flat_colour:
                import numpy as np
                item=max(bridge.landmarks[args.inspect_flat_colour],key=lambda item:item['area'])
                distances=np.linalg.norm(bridge.cloud-np.asarray(item['point']),axis=2)
                distances[~bridge.valid]=np.inf
                y,x=np.unravel_index(distances.argmin(),distances.shape)
                height,width=bridge.valid.shape
                action['inspection_anchor']=[x/(width-1)*1000,y/(height-1)*1000]
            if args.inspect_anchor_from_row is not None:
                all_rows=[json.loads(line) for line in (source/'trace.jsonl').read_text().splitlines()]
                selected=all_rows[args.inspect_anchor_from_row]['action']
                action['inspection_anchor']=selected['point'];action['target']=selected.get('target')
            result=bridge.execute(action)
            row=dict(observation=observation,action=action,result=result);rows.append(row)
            with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            report['model_free_active_inspection']=result
            print('INSPECTION_RESULT',json.dumps(result),flush=True)
            if args.home_after_inspection and result.get('skill_success'):
                observation=bridge.observe()
                action=dict(skill='home',arm=args.inspect_arm,observation_id=observation['observation_id'])
                result=bridge.execute(action)
                row=dict(observation=observation,action=action,result=result);rows.append(row)
                with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        from robodawn.episodic_memory import EpisodicMemory
        diagnostic_memory=EpisodicMemory()
        for row_index,retained in enumerate(recorded):
            observation=bridge.observe()
            action=dict(retained['action'],observation_id=observation['observation_id'])
            if args.reground_from_row is not None and row_index>=args.reground_from_row:
                from benchmark_clients import AuditedClient
                from robodawn.semantic_planner import ground
                if action['skill'] not in ('pick','grasp_handle','push','handover','place'):raise ValueError('Reground-prefix requires grounded contact or placement action')
                client=AuditedClient('zdtaichu',out/f'reground_calls_{row_index}')
                if args.observed_memory_on_reground:
                    from robodawn.episodic_memory import object_key
                    context=retained['observation'].get('episodic_memory',{})
                    diagnostic_memory.objects={object_key(entry['description']):entry for entry in context.get('objects',[])}
                    diagnostic_memory.events=context.get('recent_outcomes',[])
                    report['reground_memory']='recorded_same_prefix_observed_episode_identity_memory'
                request={k:action[k] for k in ('skill','arm','target','grasp_part','approach','donor','grounding_role','arm_required','relation','release','support') if k in action}
                crop=None;parent=action.get('parent_grounding')
                if parent:
                    parent=ground(client,observation,dict(skill='reach',arm=request['arm'],target=parent['target']),memory=diagnostic_memory)
                    request['camera']=parent['camera'];crop=parent['bbox']
                action=ground(client,observation,request,memory=diagnostic_memory,image_region=crop)
                if parent:action['parent_grounding']={k:parent[k] for k in ('target','bbox','point','camera')}
            if args.support_preview and action['skill']=='pick':
                target=action.get('mission_intent',{}).get('destination')
                original_id=retained['observation']['observation_id']
                for other in recorded:
                    entries=other['observation'].get('episodic_memory',{}).get('objects',[])
                    for entry in entries:
                        visual=entry.get('last_visual',{})
                        if (entry.get('description')==target and visual.get('observation_id')==original_id
                                and entry.get('identity_verification_observation_id')==original_id):
                            action['placement_preview']=dict(target=target,bbox=visual['bbox'],point=visual['point'],
                                camera=visual.get('camera','head_camera'),observation_id=observation['observation_id'])
                            report['support_preview_provenance']=dict(reference_observation_id=original_id,
                                identity_verification_observation_id=original_id,target=target)
            if args.appearance_memory and action.get('point') and action.get('bbox'):
                from robodawn.appearance_memory import repair_robot_point
                from robodawn.episodic_memory import EpisodicMemory,object_key
                from robodawn.grounding_evidence import current_robot_mask,robot_at_point
                memory=EpisodicMemory()
                context=retained['observation'].get('episodic_memory',{})
                memory.objects={object_key(e['description']):e for e in context.get('objects',[])}
                if robot_at_point(current_robot_mask(observation,bridge.rgb.shape),action['point']):
                    repaired=repair_robot_point(observation,memory,bridge.rgb,action['bbox'],action['target'])
                    if repaired is not None:
                        action['point']=repaired['point'];action['current_appearance_repair']=repaired
            result=bridge.execute(action)
            if all(k in action for k in ('target','bbox','point')):
                parent=action.get('parent_grounding')
                if parent:diagnostic_memory.remember_visual(parent,observation['observation_id'])
                diagnostic_memory.remember_visual(action,observation['observation_id'])
            diagnostic_memory.feedback(action,result)
            row=dict(observation=observation,action=action,result=result)
            rows.append(row)
            with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('PREFIX_ACTION',len(rows),action['skill'],result.get('skill_success'),result.get('failure'),flush=True)
            if result.get('success'):break
            if not result.get('skill_success') and not args.continue_on_error:break
        if args.dual_lift_after_prefix:
            observation=bridge.observe();action=dict(skill='dual_move',delta=[0,0,.12],observation_id=observation['observation_id'])
            result=bridge.execute(action)
            row=dict(observation=observation,action=action,result=result);rows.append(row)
            with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            report['diagnostic_dual_lift']=result
        if args.inspect_pick_target:
            if not args.inspect_arm:raise ValueError('Fresh diagnostic pick requires explicit inspection')
            from benchmark_clients import AuditedClient
            from robodawn.semantic_planner import ground
            observation=bridge.observe()
            action=ground(AuditedClient('zdtaichu',out/'calls'),observation,
                          dict(skill='pick',arm=args.inspect_arm,target=args.inspect_pick_target,
                               grasp_part=args.inspect_pick_part,camera=args.inspect_pick_camera or args.inspect_arm+'_camera'))
            result=bridge.execute(action)
            row=dict(observation=observation,action=action,result=result);rows.append(row)
            with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            report['fresh_model_pick_after_explicit_inspection']=result
            print('FRESH_VIEW_PICK',json.dumps(result),flush=True)
            if args.inspect_place_target and result.get('skill_success'):
                observation=bridge.observe()
                action=ground(AuditedClient('zdtaichu',out/'place_calls'),observation,
                              dict(skill='place',arm=action['arm'],target=args.inspect_place_target,
                                   support='container',relation='inside',release=True))
                result=bridge.execute(action)
                row=dict(observation=observation,action=action,result=result);rows.append(row)
                with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                report['fresh_current_grounded_container_delivery']=result
                print('FRESH_CONTAINER_DELIVERY',json.dumps(result),flush=True)
        report['official_success_after_retained_prefix']=bool(env.eval_success or env.check_success())
    except Exception as exc:report.update(error=str(exc),error_type=type(exc).__name__)
    finally:
        signal.alarm(0)
        report['actions_executed']=len(rows)
        if bridge:
            report['final_observation']=bridge.observe();bridge.close()
        env.close_env()
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('PREFIX_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
