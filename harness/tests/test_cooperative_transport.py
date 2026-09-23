from types import SimpleNamespace
import numpy as np
import pytest
from cooperative_transport import translation_candidates,choose_support_translation
from robotwin_harness_v3 import SkillError,tcp_from_ee


def setup():
    states={'left':dict(holding=True,remembered_object='payload'),'right':dict(holding=True,remembered_object='support')}
    robot=SimpleNamespace(left_planner=SimpleNamespace(fast_preflight=False),
                          right_planner=SimpleNamespace(fast_preflight=False),
                          right_plan_path=lambda pose:dict(status='Success'))
    bridge=SimpleNamespace(depth_id=4,sensors=lambda:states,env=SimpleNamespace(robot=robot),plan_cache=[],
        held={'left':dict(bottom_offset=[0,0,-.03],grasp_quat=[1,0,0,0])},
        geometry=lambda a:dict(tcp=[.2,.05,.8]),
        endpose=lambda:{'left_endpose':[-.25,-.1,1,1,0,0,0],'right_endpose':[.25,-.1,1,1,0,0,0]})
    def source_reachable(side,pre,low):
        tcp=tcp_from_ee(low)
        return tcp[0]<.16 and tcp[1]<-.03
    bridge.plan_pair=source_reachable
    action=dict(skill='move',arm='right',delta=[-.15,0,0],support_rendezvous=dict(source='payload',support='support'),
                cooperative_destination=dict(skill='place',arm='left',target='support',observation_id=4,release=True))
    return bridge,action,states


def test_candidates_are_bounded_horizontal_and_nonzero():
    for preferred in ([-.15,0,0],[0,0,0],[.15,0,0]):
        values=translation_candidates(preferred)
        assert values and all(.025<=np.linalg.norm(v)<=.32 and v[2]==0 for v in values)


def test_search_requires_both_arms_feasible_and_returns_only_support_translation():
    bridge,action,_=setup()
    delta,audit=choose_support_translation(bridge,action)
    assert delta[0]<-.04 and delta[1]<-.08 and delta[2]==0
    assert audit['requires_fresh_destination_after_motion']
    assert audit['candidates'][-1]['source_reachable'] and audit['candidates'][-1]['support_reachable']
    assert bridge.plan_cache==[]
    assert not bridge.env.robot.left_planner.fast_preflight and not bridge.env.robot.right_planner.fast_preflight


def test_no_motion_when_support_unreachable_or_constraints_would_change():
    bridge,action,_=setup();bridge.env.robot.right_plan_path=lambda pose:dict(status='Fail')
    bridge.plan_pair=lambda *args:pytest.fail('Source cannot license an unreachable support')
    with pytest.raises(SkillError,match='No mutually reachable'):choose_support_translation(bridge,action)
    assert not bridge.env.robot.left_planner.fast_preflight
    bridge,action,_=setup();action['cooperative_destination']['facing']='left'
    with pytest.raises(SkillError,match='directed'):choose_support_translation(bridge,action)


def test_stale_and_lost_contact_never_enter_candidate_search():
    bridge,action,states=setup();bridge.geometry=lambda a:pytest.fail('Cannot ground a lost or stale grasp')
    action['cooperative_destination']['observation_id']=3
    with pytest.raises(SkillError,match='current grounding'):choose_support_translation(bridge,action)
    action['cooperative_destination']['observation_id']=4;states['right']['holding']=False
    with pytest.raises(SkillError,match='contact-verified'):choose_support_translation(bridge,action)


def test_longer_reachability_horizon_executes_only_a_bounded_stage():
    bridge,action,_=setup()
    bridge.plan_pair=lambda side,pre,low:tcp_from_ee(low)[1]<-.19
    delta,audit=choose_support_translation(bridge,action)
    assert np.linalg.norm(delta)<=.190001 and audit['staged_support_translation']
    assert np.linalg.norm(audit['planned_translation'])>.19
    assert audit['requires_fresh_destination_after_motion']
