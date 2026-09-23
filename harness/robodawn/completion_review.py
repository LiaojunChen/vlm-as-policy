"""Transient episode memory for reviewing a successfully executed tool stage.

This records stage execution, not task success, support contact, or a required
release. All terminal actions remain model-authored from the original task.
"""
from robodawn.episodic_memory import object_key


def completed_contact_review(steps,index,pending,history,sensors):
    if not steps or index<len(steps) or pending or not history:return None
    row=history[-1];action=row.get('action',{});result=row.get('result',{})
    intent=action.get('mission_intent',{})
    if (result.get('skill_success') is not True or intent.get('operation')!='tool_contact'
            or action.get('skill')!='place' or action.get('use_contact_part') is not True
            or action.get('release',True) is not False):return None
    source=intent.get('source');destination=intent.get('destination');arm=action.get('arm')
    if not source or not destination or arm not in ('left','right'):return None
    if object_key(action.get('target',''))!=object_key(destination):return None
    state=sensors.get(arm,{})
    if state.get('holding') is not True or object_key(state.get('remembered_object',''))!=object_key(source):return None
    return dict(phase='terminal_state_review_after_tool_contact',source=source,destination=destination,
                arm=arm,contact_part=intent.get('contact_part'),
                recorded_status='contact_skill_executed_goal_unverified',
                no_remaining_planned_operations=True,support_contact_proven=False,
                terminal_release_required='not_inferred')


REVIEW_NOTE=(
    '\nCOMPLETION-STAGE REVIEW: The recorded non-releasing tool contact was skill-successful and no planned '
    'operations remain, but task success is unverified. Retaining a tool DURING contact is not an instruction '
    'to retain it indefinitely AFTER use. Re-read the ORIGINAL TASK and current image/state: decide whether '
    'the missing terminal correction is placement, release, or continued use. Consider a safe release when '
    'the tool is supported and no continued use or final retention is requested; preserve the grasp when '
    'the task requires continued holding, presenting, scanning, or another operation. Do not infer support '
    'from skill success alone, and do not repeat contact merely because the previous operation retained '
    'the tool. Choose the next action yourself; this memory does not prescribe an action or prove success.')
