import pytest
from robotwin_harness_v3 import SkillError
from robodawn.mission_safety import validate_resources,clear_view_action


def test_plan_cannot_slide_or_press_while_still_holding():
    lift=dict(operation='lift',source='can',arm='left')
    for op in ('slide','press'):
        with pytest.raises(SkillError,match='EMPTY'):
            validate_resources([lift,dict(operation=op,source='can',arm='left')],{})
    validate_resources([lift,dict(operation='transfer',source='can',arm='left'),
                        dict(operation='press',source='button',arm='left')],{})


def test_resources_respect_current_measured_holding_and_retained_tools():
    sensors={'right':{'holding':True,'remembered_object':'stamp'}}
    validate_resources([dict(operation='tool_contact',source='stamp',arm='right')],sensors)
    with pytest.raises(SkillError,match='still holds'):
        validate_resources([dict(operation='action',action=dict(skill='home',arm='right'))],sensors)


def test_one_object_cannot_be_independently_lifted_by_both_arms():
    left=dict(operation='lift',source='vessel',arm='left')
    right=dict(operation='lift',source='vessel',arm='right')
    with pytest.raises(SkillError,match='SAME object'):
        validate_resources([left,right],{})
    validate_resources([left,{**right,'source':'another vessel'}],{})


def test_recovery_clears_only_empty_arms_and_cannot_advance_a_mission():
    from robodawn.mission_planner import RobodawnPlanner
    failed=dict(action=dict(skill='pick',arm='left'),result={'skill_success':False})
    obs=dict(observation_id=4,sensors={'left':{'holding':False}})
    action=clear_view_action(obs,[failed,failed])
    assert action['skill']=='home'
    p=RobodawnPlanner(None,'lift an object');p.steps=[dict(operation='lift')]
    p.feedback(action,{'skill_success':True})
    assert p.index==0 and p.phase=='pick' and p.failures==2
    obs['sensors']['left']['holding']=True
    assert clear_view_action(obs,[failed,failed]) is None


def test_held_object_view_recovery_preserves_grip_and_is_once_per_arm():
    from copy import deepcopy
    obs=dict(observation_id=5,sensors={'left':dict(holding=True,remembered_object='phone')},
             endpose={'left_endpose':[-.1,.02,1.,1.,0.,0.,0.]})
    failed=dict(action=dict(skill='place',arm='left',target='stand'),
                result=dict(skill_success=False,failure='No depth at destination',subactions=[]))
    history=[deepcopy(failed),deepcopy(failed)]
    a=clear_view_action(obs,history)
    assert a['skill']=='move' and a['delta'][0]<0 and a['delta'][1]<0 and a['delta'][2]==0
    assert clear_view_action(obs,[dict(action=a,result={})]+history) is None
    history[0]['action']['skill']='arc'
    assert clear_view_action(obs,history) is None
    history=[deepcopy(failed),deepcopy(failed)]
    history[0]['action']['use_contact_part']=True
    assert clear_view_action(obs,history) is None


def test_successful_held_view_shift_relocalizes_without_rewriting_pending_goal():
    from robodawn.mission_planner import RobodawnPlanner
    p=RobodawnPlanner(None,'place object on support')
    p.steps=[dict(operation='transfer',source='object',destination='support',arm='left',relation='on')]
    p.arm='left';p.phase='place';p.failures=2
    action=dict(skill='move',arm='left',held_view_recovery=True,perception_recovery='clear view')
    p.feedback(action,dict(skill_success=True,sensors={'left':dict(holding=True,remembered_object='object')}))
    assert p.index==0 and p.phase=='place' and p.arm=='left' and p.failures==0
    p.feedback(action,dict(skill_success=False))
    assert p.failures==2


def test_compiled_arc_rechecks_lost_contact_before_any_visual_query():
    from robodawn.mission_planner import RobodawnPlanner
    p=RobodawnPlanner(None,'Open a hinged panel')
    p.steps=[dict(operation='action',action=dict(skill='arc',arm='left',target='hinge',axis='x',angle=30))]
    calls=[]
    def replan(obs,history):
        calls.append(p.last_perception_failure)
        p.steps=[dict(operation='action',action=dict(skill='home',arm='left'))]
        p.index=0;p.failures=0
    p._replan=replan
    def grounding(*args,**kwargs):
        raise AssertionError('Empty-hand ARC must not request hinge geometry')
    p._ground=grounding
    obs=dict(observation_id=2,success=False,sensors={'left':dict(holding=False,remembered_object='panel edge')})
    action=p.plan(obs,[])
    assert action['skill']=='home' and len(calls)==1
    assert 'needs a held object' in calls[0]


def test_compiled_home_rechecks_current_occupancy_without_opening_hand():
    from robodawn.mission_planner import RobodawnPlanner
    p=RobodawnPlanner(None,'Retain the object')
    p.steps=[dict(operation='action',action=dict(skill='home',arm='left'))]
    calls=[]
    def replan(obs,history):
        calls.append(p.last_perception_failure)
        p.steps=[dict(operation='action',action=dict(skill='wait'))]
        p.index=0;p.failures=0
    p._replan=replan
    obs=dict(observation_id=2,success=False,sensors={'left':dict(holding=True,remembered_object='object')})
    action=p.plan(obs,[])
    assert action['skill']=='wait' and len(calls)==1 and 'EMPTY' in calls[0]
