import json
from types import SimpleNamespace

import pytest
from robotwin_harness_v3 import SkillError
from robodawn.mission_safety import validate_resources


def test_occupied_press_explains_tool_and_empty_hand_alternatives_without_relaxation():
    steps=[dict(operation='lift',source='arbitrary implement',arm='right'),
           dict(operation='press',source='arbitrary surface',arm='right')]
    with pytest.raises(SkillError) as error:validate_resources(steps,{})
    assert 'PRESS requires an EMPTY arm' in str(error.value)
    assert 'ONE TOOL_CONTACT' in str(error.value)
    assert 'direct empty-gripper contact' in str(error.value)
    assert 'explicit arm assignments' in str(error.value)


def test_release_then_direct_press_and_opposite_empty_arm_still_valid():
    lift=dict(operation='lift',source='parcel',arm='right')
    press=dict(operation='press',source='button',arm='right')
    validate_resources([lift,dict(operation='transfer',source='parcel',arm='right'),press],{})
    validate_resources([lift,dict(press,arm='left')],{})


def test_slide_feedback_is_unchanged_and_does_not_prescribe_tool_contact():
    with pytest.raises(SkillError) as error:
        validate_resources([dict(operation='lift',source='parcel',arm='left'),
                            dict(operation='slide',source='parcel',arm='left')],{})
    assert str(error.value)=='Step 1: left holds parcel; slide requires an EMPTY arm. Transfer a held object, or release it before sliding/pressing.'


def test_original_native_planning_is_unchanged_and_repair_uses_original_budget():
    from robodawn.mission_planner import RobodawnPlanner,PROMPT
    from robodawn.semantic_planner import policy_sensor_view
    instruction='Use the implement to contact the surface with the right arm.'
    sensors={side:dict(holding=False) for side in ('left','right')}
    tool=dict(operation='tool_contact',source='implement',destination='surface',contact_part='working face',
              grasp_part='handle',arm='right',relation='on')
    replies=[[dict(operation='lift',source='implement',arm='right'),
              dict(operation='press',source='surface',arm='right')],[tool]]
    calls=[]
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            calls.append((prompt,image,kwargs))
            return SimpleNamespace(raw_text=json.dumps(dict(steps=replies.pop(0))))
    planner=RobodawnPlanner(Client(),instruction)
    expected=PROMPT+'\nTASK: '+instruction+'\nCURRENT CONTACT STATE: '+json.dumps(policy_sensor_view(sensors))+'\nEXECUTED ACTIONS: []'+planner.memory.prompt()
    planner._replan(dict(sensors=sensors,image_paths=[]),[])
    assert len(calls)==2 and calls[0][0]==expected and calls[0][1] is None
    assert calls[0][2]['response_schema'] is None and calls[1][2]['response_schema'] is not None
    assert 'ONE TOOL_CONTACT' in calls[1][0]
    assert planner.steps==[tool]
    assert all(c[2]['max_tokens']==900 and c[2]['temperature']==0 for c in calls)
