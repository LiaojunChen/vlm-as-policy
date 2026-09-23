from copy import deepcopy
import pytest
from robotwin_harness_v3 import SkillError
from robodawn.failure_constraints import repeated_unreachable_placement,validate_recovery_plan,validate_recovery_action,schedule_blocked_support_lift


def case():
    sensors={'left':dict(holding=True,remembered_object='bread'),'right':dict(holding=False)}
    a=dict(skill='place',arm='left',target='pan',relation='on')
    r=dict(skill_success=False,failure='No reachable placement orientation; adjust held pose',
           geometry={'tcp':[.25,.1,.75]},subactions=[],sensors=sensors)
    return [dict(action=deepcopy(a),result=deepcopy(r)) for _ in range(2)],sensors,a


def test_recovery_rejects_same_failed_placement_despite_cosmetic_parameter_changes():
    h,s,a=case()
    for variant in (a,{**a,'approach':'top'},{**a,'grasp_part':'rim'}):
        with pytest.raises(SkillError,match='physical prerequisite'):validate_recovery_action(variant,h,s)


def test_failure_constraint_expires_when_state_or_destination_changes():
    h,s,a=case()
    for mutate in (lambda h:h[-1]['result'].update(subactions=[{}]),
                   lambda h:h[-1]['result']['geometry'].update(tcp=[.20,.1,.75]),
                   lambda h:h[-1]['result'].update(skill_success=True),
                   lambda h:h[-1]['result']['sensors']['left'].update(holding=False)):
        changed=deepcopy(h);mutate(changed)
        assert repeated_unreachable_placement(changed,s) is None
    validate_recovery_action({**a,'target':'clear table'},h,s)
    validate_recovery_action(dict(skill='move',arm='left',delta=[0,-.05,0]),h,s)
    assert repeated_unreachable_placement(h[:1],s) is None


def test_recovery_cannot_hide_failed_transfer_behind_noop_lift():
    h,s,a=case();transfer=dict(operation='transfer',source='bread',destination='pan',arm='auto',relation='on')
    with pytest.raises(SkillError,match='NO intervening motion'):
        validate_recovery_plan([dict(operation='lift',source='bread',arm='left',location='stay'),transfer],h,s)
    validate_recovery_plan([dict(operation='lift',source='pan',arm='right',grasp_part='handle'),transfer],h,s)
    validate_recovery_plan([dict(operation='lift',source='bread',arm='left',location='centre'),transfer],h,s)


def test_unobserved_failed_geometry_does_not_create_a_constraint():
    h,s,a=case();h[-1]['result']['geometry']={}
    assert repeated_unreachable_placement(h,s) is None


def test_simultaneous_support_lift_is_reordered_without_inventing_a_new_goal():
    h,s,a=case()
    transfer=dict(operation='transfer',source='bread',destination='pan',arm='left',relation='on')
    lift=dict(operation='lift',source='pan',arm='right',location='stay',grasp_part='handle')
    steps=[transfer,lift]
    repaired=schedule_blocked_support_lift(steps,h,s,'Lift the pan with right arm while the left places bread')
    assert repaired[0]['source']=='pan' and repaired[0]['location']=='stay'
    assert repaired[1]==transfer and steps==[transfer,lift]
    validate_recovery_plan(repaired,h,s)
    for text in ('Place bread then lift pan','First lift bread while moving pan','Put bread in pan'):
        assert schedule_blocked_support_lift(steps,h,s,text)==steps
    assert schedule_blocked_support_lift([transfer],h,s,'while moving')==[transfer]
