from types import SimpleNamespace
import pytest
from robotwin_harness_v3 import SkillError
from robodawn.mission_planner import RobodawnPlanner
from robodawn.recovery_errors import failure_kind


WRAPPER='Invalid policy action after repair: head_camera: Grounding invalid after repair: '


@pytest.mark.parametrize('inner,expected',[
    ('Model generation ended before completing JSON; return a shorter valid response.','generation_or_format'),
    ('Generation budget ended before completing schema-constrained JSON','generation_or_format'),
    ('Invalid JSON syntax','generation_or_format'),
    ('JSONDecodeError while reading visible=false','generation_or_format'),
    ('HTTP 500 service unavailable','service'),
    ('Connection timed out while grounding an occluded object','service'),
    ('No reachable empty-arm observation pose','motion'),
    ('Target is occluded','perception'),
    ('No depth at destination','perception'),
    ('Hinge endpoints lack current nonrobot above-table depth','perception'),
    ('Your contact point is on the ROBOT, not the requested scene object.','perception'),
    ('Target identity is ambiguous','perception'),
    ('Unknown operation','unknown'),
    ('','unknown'),
])
def test_inner_failure_takes_precedence_over_grounding_wrapper(inner,expected):
    assert failure_kind(WRAPPER+inner)==expected


def test_mixed_camera_fault_does_not_hide_generation_failure():
    message='right_camera: Target is occluded; head_camera: Grounding invalid after repair: Model generation ended before completing JSON'
    assert failure_kind(message)=='generation_or_format'


@pytest.mark.parametrize('entry',['reactive','direct'])
def test_recorded_wrapped_budget_failure_does_not_consume_inspection_or_move(entry):
    error=WRAPPER+'Model generation ended before completing JSON; return a shorter valid response.'
    planner=RobodawnPlanner(None,'Manipulate an object')
    calls=[]
    def fail(*args):calls.append(1);raise SkillError(error)
    planner.reactive=SimpleNamespace(plan=fail)
    obs=dict(observation_id=4,image_paths=['head','left','right'],
             sensors=dict(left=dict(holding=False),right=dict(holding=False)))
    if entry=='reactive':
        with pytest.raises(SkillError,match='Model generation ended'):
            planner._reactive_recovery(obs,[],'reason')
        assert len(calls)==1
    else:assert planner._inspection_recovery(obs,error) is None
    assert not planner.inspection_attempts


def test_real_occlusion_still_uses_one_existing_empty_arm_inspection():
    planner=RobodawnPlanner(None,'Manipulate an object')
    obs=dict(observation_id=4,image_paths=['head','left','right'],
             sensors=dict(left=dict(holding=False),right=dict(holding=True)))
    action=planner._inspection_recovery(obs,WRAPPER+'Target is occluded')
    assert action['skill']=='inspect' and action['arm']=='left'
    assert planner._inspection_recovery(obs,WRAPPER+'Target is occluded') is None
