"""Conservative normalization of model-authored compound skill prerequisites."""
from copy import deepcopy
from robodawn.episodic_memory import object_key


def schedule_simultaneous_supports(steps,instruction):
    """Honor a simultaneous support dependency before motion, not after failure.

    Only commute adjacent, independently armed model-authored operations.
    Never add an acquisition, change arms, cross another operation, or override
    explicit before/after ordering. The moved LIFT keeps its authored location.
    """
    import re
    text=instruction.split('Completion requirements:')[0]
    if not re.search(r'\b(while|simultaneously|at the same time)\b',text,re.I):return steps
    if re.search(r'\b(before|after|then|first|finally)\b',text,re.I):return steps
    result=deepcopy(steps)
    for index in range(len(result)-1):
        transfer,lift=result[index:index+2]
        if transfer.get('operation')!='transfer' or lift.get('operation')!='lift':continue
        source_arm=transfer.get('arm');support_arm=lift.get('arm')
        if source_arm not in ('left','right') or support_arm not in ('left','right') or source_arm==support_arm:continue
        if transfer.get('relation') not in ('on','inside'):continue
        if object_key(transfer['destination'])!=object_key(lift['source']):continue
        if object_key(transfer['source'])==object_key(lift['source']):continue
        lift['dependency_order']='authored_support_retained_during_other_arm_transfer'
        result[index:index+2]=[lift,transfer]
    return result


def fuse_tool_acquisition(steps):
    """A compound tool contact already acquires its tool and observes its tip.

    Fuse only an immediately preceding equivalent acquisition. Never remove
    a presentation, intervening motion, other object or other required arm.
    No task/object names or inferred goals are introduced.
    """
    result=[]
    for original in steps:
        step=deepcopy(original)
        if step.get('operation')=='tool_contact' and result:
            previous=result[-1];op=previous.get('operation');acquire=None
            if op=='lift' and previous.get('location','stay')=='stay':
                acquire=dict(target=previous['source'],arm=previous.get('arm','auto'),grasp_part=previous.get('grasp_part','body'))
            elif op=='action' and previous.get('action',{}).get('skill') in ('pick','grasp_handle'):
                acquire=previous['action']
            if acquire:
                source=object_key(step['source']);part=step.get('grasp_part','body')
                names={source,source+' '+part,part+' of '+source}
                old_arm,new_arm=acquire.get('arm','auto'),step.get('arm','auto')
                same_arm=old_arm==new_arm or 'auto' in (old_arm,new_arm)
                if (object_key(acquire.get('target','')) in names and same_arm
                        and acquire.get('grasp_part','body')==part):
                    result.pop()
                    if new_arm=='auto':step['arm']=old_arm
                    if step.get('arm') in ('left','right'):step['arm_required']=True
                    if acquire.get('approach'):step['acquisition_approach']=acquire['approach']
                    step['compiled_acquisition']=previous
        result.append(step)
    return result
