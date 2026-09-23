"""Short-lived planning constraints learned only from measured action failures."""
import numpy as np
import re
from copy import deepcopy
from robotwin_harness_v3 import SkillError
from robodawn.episodic_memory import object_key


def repeated_unreachable_placement(history, sensors):
    recent=history[-2:]
    if len(recent)!=2:return None
    first=recent[0]['action'];side=first.get('arm')
    if side not in ('left','right'):return None
    source=sensors.get(side,{}).get('remembered_object')
    if not source or not sensors.get(side,{}).get('holding'):return None
    points=[]
    for row in recent:
        action,result=row['action'],row['result']
        if action.get('skill')!='place' or result.get('skill_success') or result.get('subactions'):return None
        if not str(result.get('failure','')).startswith('No reachable placement orientation'):return None
        if any(action.get(k,'on' if k=='relation' else None)!=first.get(k,'on' if k=='relation' else None)
               for k in ('arm','target','relation')):return None
        contact=result.get('sensors',{}).get(side,{})
        if not contact.get('holding') or object_key(contact.get('remembered_object',''))!=object_key(source):return None
        point=np.asarray((result.get('geometry') or {}).get('tcp'),float)
        if point.shape!=(3,) or not np.isfinite(point).all():return None
        points.append(point)
    if np.linalg.norm(points[0]-points[1])>.003:return None
    return dict(arm=side,target=first['target'],source=source,relation=first.get('relation','on'))


def validate_recovery_action(action,history,sensors):
    failed=repeated_unreachable_placement(history,sensors)
    if not failed or action.get('skill')!='place':return
    if (action.get('arm')==failed['arm'] and object_key(action.get('target',''))==object_key(failed['target'])
            and action.get('relation','on')==failed['relation']):
        raise SkillError('This held-source/destination placement has exhausted reachability twice with NO intervening motion. '
                         'Changing wording, approach or grasp_part cannot change the existing grasp. '
                         'Plan a physical prerequisite BEFORE retrying: move a held object, acquire/reposition a destination '
                         'with the free arm when permitted by the task, or stage and regrasp. Preserve the original goal and arm constraints.')


def validate_recovery_plan(steps,history,sensors):
    if repeated_unreachable_placement(history,sensors) is None:return
    for step in steps:
        op=step['operation'];source=step.get('source','')
        holder=next((side for side,state in sensors.items() if state.get('holding')
                     and object_key(state.get('remembered_object',''))==object_key(source)),None)
        if op=='lift' and holder and step.get('location','stay')=='stay':
            continue  # Already-held lifts compile to no motion, not recovery.
        if op=='transfer' and holder:
            validate_recovery_action(dict(skill='place',arm=holder,target=step['destination'],
                                          relation=step.get('relation','on')),history,sensors)
        elif op=='action':validate_recovery_action(step['action'],history,sensors)
        return  # A genuinely different physical first step can change feasibility.


def schedule_blocked_support_lift(steps,history,sensors,instruction):
    """Schedule an ALREADY model-requested support lift before a blocked transfer.

    Only simultaneous goals authorize this reordering. Never introduce a new
    object, arm, goal, support motion, or requested final pose.
    """
    failed=repeated_unreachable_placement(history,sensors)
    text=instruction.split('Completion requirements:')[0]
    if not failed or not re.search(r'\b(while|simultaneously|at the same time)\b',text,re.I):return steps
    if re.search(r'\b(before|after|then|first|finally)\b',text,re.I):return steps
    other='right' if failed['arm']=='left' else 'left'
    if sensors.get(other,{}).get('holding'):return steps
    transfer_index=None
    for index,step in enumerate(steps):
        if (step['operation']=='lift' and step.get('location','stay')=='stay'
                and object_key(step.get('source',''))==object_key(failed['source'])):continue
        if (step['operation']=='transfer' and object_key(step.get('source',''))==object_key(failed['source'])
                and object_key(step.get('destination',''))==object_key(failed['target'])):transfer_index=index
        break
    if transfer_index is None:return steps
    # Only an immediately following independent support lift is proven movable.
    next_index=transfer_index+1
    if next_index>=len(steps):return steps
    lift=steps[next_index]
    if (lift['operation']!='lift' or lift.get('arm')!=other
            or object_key(lift.get('source',''))!=object_key(failed['target'])):return steps
    result=deepcopy(steps);lift=result.pop(next_index)
    lift['dependency_recovery']='existing_simultaneous_support_lift_before_unreachable_transfer'
    result.insert(transfer_index,lift)
    return result
