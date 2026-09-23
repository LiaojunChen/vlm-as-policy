"""Static plan resource checks and measured-state perception recovery."""
import numpy as np
import re
from robodawn.episodic_memory import object_key
from robotwin_harness_v3 import SkillError


def validate_resources(steps,sensors):
    held={arm:state.get('remembered_object') or '<held object>' for arm,state in sensors.items() if state.get('holding')}
    acquired_in_plan=set()
    shared_grasp=None
    for index,step in enumerate(steps):
        op=step['operation'];action=step.get('action',{})
        if shared_grasp is not None and all(side in held for side in ('left','right')):
            source=step.get('source',action.get('target',''))
            if (op in ('transfer','lift','tool_contact','slide','press') and object_key(source)==shared_grasp
                    or op=='action' and action.get('skill') in ('move','rotate','present','arc','shake','place')):
                raise SkillError('One object is retained by BOTH hands. A single-arm transport/rotation would fight the other grasp. '
                                 'Use dual_move for coordinated translation, or explicitly release one hand before a single-arm operation.')
        if op=='handover':
            donor,receiver=step['arm'],step['receiver'];source=step['source']
            if receiver in held:raise SkillError('Handover needs an empty receiver')
            if donor in held and object_key(held[donor])!=object_key(source):
                raise SkillError('Handover donor holds a different source')
            held.pop(donor,None);held[receiver]=source
            acquired_in_plan.discard(donor);acquired_in_plan.add(receiver)
            continue
        if op=='bimanual_lift':
            for side in ('left','right'):
                source=step[side+'_source']
                if held.get(side) and held[side].lower()!=source.lower():
                    raise SkillError('Bimanual lift needs '+side+' free or holding its named component')
                held[side]=source
            shared_grasp=object_key(step['source'])
            continue
        if op=='action' and action.get('skill')=='dual_move':
            if not all(side in held for side in ('left','right')):
                raise SkillError('dual_move requires both arms to hold before the coordinated motion')
            continue
        arm=action.get('arm') if op=='action' else step.get('arm','auto')
        source=step.get('source')
        if arm=='auto' and op in ('slide','press') and all(side in held for side in ('left','right')):
            raise SkillError(f'Step {index}: {op} requires an EMPTY arm, but both hands are occupied; auto cannot bypass this prerequisite.')
        # Unknown auto arms cannot be simulated reliably; retain runtime checks.
        if arm not in ('left','right'):continue
        occupied=held.get(arm)
        if op=='tool_contact' and occupied and arm in acquired_in_plan:
            raise SkillError(f'Step {index}: tool_contact ALREADY performs pickup and observes its working part before pickup. DELETE the earlier lift/grasp_handle/pick step for {arm}; return the tool_contact operation alone for acquiring and using this tool.')
        if op in ('slide','press') and occupied:
            if op=='press':
                raise SkillError(f'Step {index}: {arm} holds {occupied}; PRESS requires an EMPTY arm because '
                                 'it presses directly with the gripper, not with a held object. '
                                 'If the original task requires contact USING the held object, replace the separate '
                                 'acquisition and PRESS with ONE TOOL_CONTACT: name the tool as source, the contact '
                                 'surface as destination, and its actual working face/head/tip as contact_part. '
                                 'TOOL_CONTACT includes pickup and retains the tool; name the actual grasp_part separately. '
                                 'Do not add a preceding LIFT or replace tool use with TRANSFER (which releases the object). '
                                 'If the original task instead requires direct empty-gripper contact, first complete the '
                                 'requested transfer/release or use an allowed empty arm. Preserve the original task and '
                                 'explicit arm assignments; select the correct alternative yourself.')
            raise SkillError(f'Step {index}: {arm} holds {occupied}; {op} requires an EMPTY arm. Transfer a held object, or release it before sliding/pressing.')
        if op in ('transfer','lift','tool_contact'):
            other='right' if arm=='left' else 'left'
            if held.get(other,'').lower()==source.lower():
                raise SkillError(f'Step {index}: {source} is already held by {other}. Do not independently lift the SAME object twice; use handover for passing, or ONE bimanual_lift with distinct component descriptions for simultaneous two-arm lifting.')
            if occupied and occupied.lower()!=source.lower():
                raise SkillError(f'Step {index}: {arm} already holds {occupied}; cannot acquire {source} before releasing or transferring the held object.')
            if op=='transfer':held.pop(arm,None)
            else:
                held[arm]=source
                if op=='lift':acquired_in_plan.add(arm)
                else:acquired_in_plan.discard(arm)
        if op=='action':
            skill=action.get('skill')
            if skill in ('pick','grasp_handle','home','push','press') and occupied:
                raise SkillError(f'Step {index}: {skill} needs {arm} EMPTY, but it still holds {occupied}.')
            if skill in ('place','present','shake','arc') and not occupied:
                raise SkillError(f'Step {index}: {skill} needs a held object in {arm}; first acquire it.')
            if skill in ('pick','grasp_handle'):
                held[arm]=action.get('target','<held object>');acquired_in_plan.add(arm)
            if skill=='open' or (skill=='place' and action.get('release',True)):held.pop(arm,None)
            if not all(side in held for side in ('left','right')):shared_grasp=None


def clear_view_action(obs,history):
    """Move a held free object aside, or home an empty arm, to reobserve."""
    if len(history)<2:return None
    recent=history[-2:]
    # Keep grasp and orientation unchanged. Never tug on articulated handles
    # or tool-contact actions merely because their hinge/contact is occluded.
    arm=recent[-1]['action'].get('arm')
    if (arm in ('left','right') and obs['sensors'].get(arm,{}).get('holding')
            and obs['sensors'][arm].get('remembered_object')
            and all(row['action'].get('skill')=='place' and row['action'].get('arm')==arm
                    and row['action'].get('release',True) and not row['action'].get('use_contact_part')
                    and row['result'].get('failure')=='No depth at destination'
                    and not row['result'].get('subactions') for row in recent)
            and recent[0]['action'].get('target')==recent[1]['action'].get('target')
            and not any(row['action'].get('held_view_recovery') and row['action'].get('arm')==arm for row in history)):
        pose=np.asarray(obs.get('endpose',{}).get(arm+'_endpose'),float)
        if pose.shape==(7,) and np.isfinite(pose).all():
            target=pose[:3].copy()
            target[0]=np.clip(target[0]+(-.06 if arm=='left' else .06),-.38,.38)
            target[1]=max(-.32,target[1]-.08)
            delta=target-pose[:3]
            if .02<=np.linalg.norm(delta)<=.15:
                return dict(skill='move',name='move',arm=arm,delta=delta.tolist(),observation_id=obs['observation_id'],
                            held_view_recovery=True,
                            perception_recovery='Destination depth unavailable twice; shift held object aside and reobserve')
    if any(row['result'].get('skill_success') or row['action'].get('skill') not in ('pick','grasp_handle','press','reach') for row in recent):return None
    arm=recent[-1]['action'].get('arm')
    if arm not in ('left','right') or any(row['action'].get('arm')!=arm for row in recent):return None
    if obs['sensors'].get(arm,{}).get('holding'):return None
    # Do not insert a home forever if that intervention already failed to help.
    if any(row['action'].get('perception_recovery') and row['action'].get('arm')==arm for row in history[-5:]):return None
    return dict(skill='home',name='home',arm=arm,observation_id=obs['observation_id'],
                perception_recovery='Repeated failed approach; clear the empty arm and reobserve')


def cooperative_support_action(obs,history,step,instruction,attempts,resume=False):
    """Translate an already-held support toward shared reach, keeping both grips.

    No new object is acquired and no arm assignment or final relation changes.
    Only robot proprioception and contact-verified identities are used.
    """
    if step.get('operation')!='transfer' or len(history)<2:return None
    if re.search(r'\b(stationary|without moving|keep.{0,35}still|remain.{0,20}in place)\b',
                 instruction.split('Completion requirements:')[0],re.I):return None
    source=object_key(step['source']);target=object_key(step['destination'])
    key=(source,target)
    if attempts.get(key,0)>=2:return None
    held={object_key(state.get('remembered_object','')):side for side,state in obs['sensors'].items()
          if state.get('holding') and state.get('remembered_object')}
    source_arm,support_arm=held.get(source),held.get(target)
    if source_arm not in ('left','right') or support_arm not in ('left','right') or source_arm==support_arm:return None
    acquisition=next((row for row in reversed(history) if row['action'].get('arm')==support_arm
                      and row['action'].get('skill') in ('pick','grasp_handle','handover','open','home')),None)
    if (acquisition is None or acquisition['action'].get('skill')!='pick'
            or object_key(acquisition['action'].get('target',''))!=target
            or not acquisition['result'].get('skill_success')
            or not any(part.get('name')=='lift' and part.get('motion_ok')
                       for part in acquisition['result'].get('subactions',[]))):return None
    for row in ([] if resume else history[-2:]):
        action,result=row['action'],row['result']
        if (action.get('skill')!='place' or action.get('arm')!=source_arm or object_key(action.get('target',''))!=target
                or not action.get('release',True) or result.get('subactions') or result.get('skill_success')
                or not str(result.get('failure','')).startswith('No reachable placement orientation')):return None
    source_pose=np.asarray(obs.get('endpose',{}).get(source_arm+'_endpose'),float)
    support_pose=np.asarray(obs.get('endpose',{}).get(support_arm+'_endpose'),float)
    if source_pose.shape!=(7,) or support_pose.shape!=(7,) or not np.isfinite([source_pose,support_pose]).all():return None
    midpoint=(source_pose[0]+support_pose[0])*.5
    wanted=np.clip(midpoint,.06,.15) if support_arm=='right' else np.clip(midpoint,-.15,-.06)
    dx=float(np.clip(wanted-support_pose[0],-.15,.15))
    attempts[key]=attempts.get(key,0)+1
    return dict(skill='move',name='move',arm=support_arm,delta=[dx,0.,0.],observation_id=obs['observation_id'],
                support_rendezvous=dict(source=step['source'],support=step['destination'],attempt=attempts[key]),
                recovery_reason='Both named objects held; move support into shared lateral reach before retrying original placement')
