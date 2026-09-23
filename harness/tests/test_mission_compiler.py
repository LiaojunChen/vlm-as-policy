from copy import deepcopy
import pytest
from robodawn.mission_compiler import fuse_tool_acquisition
from robodawn.mission_planner import validate_mission
from robodawn.mission_safety import validate_resources


def contact():
    return dict(operation='tool_contact',source='long tool',contact_part='working end',destination='support',
                relation='on',arm='left',grasp_part='handle')


def test_adjacent_component_grasp_fuses_into_contact_without_losing_requirements():
    pickup=dict(operation='action',action=dict(skill='grasp_handle',target='long tool handle',arm='left',grasp_part='handle',approach='top'))
    steps=[pickup,contact()];original=deepcopy(steps)
    compiled=validate_mission(dict(steps=steps))
    assert len(compiled)==1 and compiled[0]['compiled_acquisition']==pickup
    assert compiled[0]['arm']=='left' and compiled[0]['arm_required']
    assert compiled[0]['acquisition_approach']=='top'
    assert compiled[0]['contact_part']=='working end' and compiled[0]['destination']=='support'
    validate_resources(compiled,{})
    assert steps==original


@pytest.mark.parametrize('change',[dict(arm='right'),dict(source='other object'),dict(location='centre'),dict(grasp_part='body')])
def test_different_or_additional_intentions_are_not_discarded(change):
    lift=dict(operation='lift',source='long tool',arm='left',grasp_part='handle',**{})
    lift.update(change)
    assert len(fuse_tool_acquisition([lift,contact()]))==2


def test_intervening_action_and_explicit_arm_are_preserved():
    lift=dict(operation='lift',source='long tool',arm='left',grasp_part='handle')
    move=dict(operation='action',action=dict(skill='move',arm='left',delta=[0,0,.1]))
    assert len(fuse_tool_acquisition([lift,move,contact()]))==3
    tool=contact();tool['arm']='auto'
    result=fuse_tool_acquisition([lift,tool])
    assert len(result)==1 and result[0]['arm']=='left'


def support_steps():
    return [dict(operation='transfer',source='piece',destination='tray',relation='on',arm='left'),
            dict(operation='lift',source='tray',arm='right',location='stay',grasp_part='handle')]


def test_simultaneous_support_dependency_is_scheduled_before_transfer():
    from robodawn.mission_compiler import schedule_simultaneous_supports
    steps=support_steps();original=deepcopy(steps)
    result=schedule_simultaneous_supports(steps,'Lift the tray while the other arm puts the piece on it.')
    assert result[0]['operation']=='lift' and result[0]['arm']=='right' and result[0]['location']=='stay'
    assert result[1]==steps[0] and steps==original
    validate_resources(result,{'left':{'holding':False},'right':{'holding':False}})


@pytest.mark.parametrize('instruction',['Place the piece, then lift the tray.',
    'First place the piece while the tray stays on the table, then lift.',
    'Place the piece and lift the tray.'])
def test_explicit_order_or_no_simultaneity_is_never_rewritten(instruction):
    from robodawn.mission_compiler import schedule_simultaneous_supports
    steps=support_steps();assert schedule_simultaneous_supports(steps,instruction)==steps


def test_support_dependency_does_not_cross_actions_or_guess_auto_hands():
    from robodawn.mission_compiler import schedule_simultaneous_supports
    steps=support_steps();steps[1]['arm']='auto'
    assert schedule_simultaneous_supports(steps,'Hold tray while placing piece.')==steps
    steps=support_steps();steps.insert(1,dict(operation='action',action=dict(skill='wait')))
    assert schedule_simultaneous_supports(steps,'Hold tray while placing piece.')==steps


def test_ungrounded_auto_arm_is_rejected_before_execution():
    from robotwin_harness_v3 import SkillError
    with pytest.raises(SkillError,match='explicit left/right'):
        validate_mission(dict(steps=[dict(operation='action',action=dict(skill='open',arm='auto'))]))


def test_bimanual_transport_cannot_escape_resource_checks_with_auto_arm():
    from robotwin_harness_v3 import SkillError
    acquire=dict(operation='bimanual_lift',source='box',left_source='left edge',right_source='right edge')
    transfer=dict(operation='transfer',source='box',destination='support',relation='on',arm='auto')
    with pytest.raises(SkillError,match='BOTH hands'):validate_resources([acquire,transfer],{})
    with pytest.raises(SkillError,match='BOTH hands'):validate_resources([acquire,dict(transfer,operation='slide')],{})
    with pytest.raises(SkillError,match='BOTH hands'):
        validate_resources([acquire,dict(operation='action',action=dict(skill='move',arm='left',delta=[.1,0,0]))],{})
    validate_resources([acquire,dict(operation='action',action=dict(skill='dual_move',delta=[.1,0,0]))],{})
    validate_resources([acquire,dict(operation='action',action=dict(skill='open',arm='right')),dict(transfer,arm='left',source='left edge')],{})
