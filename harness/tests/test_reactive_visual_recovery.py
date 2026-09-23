from copy import deepcopy
from types import SimpleNamespace

import pytest
from robotwin_harness_v3 import SkillError
from robodawn.mission_planner import RobodawnPlanner


FAILURE='Invalid policy action after repair: Grounding invalid after repair: Your contact point is on the ROBOT; target is occluded.'


def setup_planner(failure=FAILURE):
    planner=RobodawnPlanner(None,'Manipulate the requested object.')
    calls=[]
    def fail(obs,history):
        calls.append((deepcopy(obs),deepcopy(history)))
        raise SkillError(failure)
    planner.reactive=SimpleNamespace(plan=fail)
    obs=dict(success=False,observation_id=5,image_paths=['head.png','left.png','right.png'],
             sensors=dict(left=dict(holding=False),right=dict(holding=True,remembered_object='tool')))
    return planner,obs,calls


def test_postmission_occlusion_returns_inspection_without_fabricated_history_or_extra_calls():
    planner,obs,calls=setup_planner()
    planner.steps=[dict(operation='tool_contact',source='tool',destination='surface')];planner.index=1
    history=[dict(action=dict(skill='place',arm='right'),result=dict(skill_success=True))]
    previous=deepcopy(history)
    action=planner.plan(obs,history)
    assert len(calls)==1 and history==previous
    assert action['skill']=='inspect' and action['arm']=='left'
    assert action['reactive_visual_recovery'] and action['active_inspection']
    assert action['observation_id']==5 and action['perception_failure']==FAILURE
    assert 'inspection_anchor' not in action
    planner.feedback(action,dict(skill_success=True))
    assert planner.index==1 and planner.steps[0]['source']=='tool'
    with pytest.raises(SkillError,match='Grounding invalid'):
        planner.plan(dict(obs,observation_id=6),history+[dict(action=action,result=dict(skill_success=True))])
    assert len(calls)==2


def test_rejected_mission_reactive_entry_has_same_bounded_inspection():
    planner,obs,calls=setup_planner()
    def reject(*args):raise SkillError('Invalid mission after repair')
    planner._replan=reject
    action=planner.plan(obs,[])
    assert action['skill']=='inspect' and action['arm']=='left'
    assert 'Mission rejected after repair' in action['mission_recovery']
    assert len(calls)==1


@pytest.mark.parametrize('failure',['Invalid JSON syntax','No reachable placement orientation','HTTP 500 service unavailable'])
def test_nonvisual_failures_are_not_converted_to_camera_motion(failure):
    planner,obs,calls=setup_planner(failure)
    with pytest.raises(SkillError,match=failure):planner._reactive_recovery(obs,[],'reason')
    assert not planner.inspection_attempts and len(calls)==1


def test_both_occupied_or_missing_cameras_preserve_original_failure():
    for held,paths in ((True,['head','left','right']),(False,['head'])):
        planner,obs,calls=setup_planner()
        obs['sensors']['left']['holding']=held;obs['image_paths']=paths
        with pytest.raises(SkillError,match='Grounding invalid'):planner._reactive_recovery(obs,[],'reason')
        assert not planner.inspection_attempts


def test_shared_inspection_budget_is_not_reset_by_reactive_entry():
    planner,obs,calls=setup_planner();obs['sensors']['right']['holding']=False
    planner.inspection_attempts.add('left')
    assert planner._reactive_recovery(obs,[],'reason')['arm']=='right'
    with pytest.raises(SkillError):planner._reactive_recovery(obs,[],'reason')
    assert planner.inspection_attempts=={'left','right'}


def test_unknown_occupancy_is_not_treated_as_verified_empty():
    planner,obs,calls=setup_planner();obs['sensors']['left']={}
    with pytest.raises(SkillError):planner._reactive_recovery(obs,[],'reason')
    assert not planner.inspection_attempts
    obs['sensors']['right']['holding']=False
    assert planner._reactive_recovery(obs,[],'reason')['arm']=='right'
    assert obs['sensors']['left']=={}


def test_successful_reactive_action_and_ordinary_exception_behavior_unchanged():
    planner,obs,calls=setup_planner();native=dict(skill='home',arm='left')
    planner.reactive=SimpleNamespace(plan=lambda *args:dict(native))
    assert planner._reactive_recovery(obs,[],'reason')==dict(native,mission_recovery='reason')
    def fail(*args):raise RuntimeError('unexpected implementation fault')
    planner.reactive=SimpleNamespace(plan=fail)
    with pytest.raises(RuntimeError):planner._reactive_recovery(obs,[],'reason')
    assert not planner.inspection_attempts
