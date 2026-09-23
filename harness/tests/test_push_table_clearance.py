"""Executor-level invariants, independent of simulator/model availability."""
from types import SimpleNamespace as S
import numpy as np
import pytest
import robot_table_clearance as clearance
from robotwin_harness_v3 import Bridge, SkillError, grasp_quat, tcp_from_ee


def setup_bridge():
    bridge=Bridge.__new__(Bridge)
    bridge.index=0;bridge.table=.74;bridge.failed={}
    planner=S(fast_preflight=False)
    bridge.env=S(robot=S(right_planner=planner),eval_success=False,check_success=lambda:False)
    bridge.endpose=lambda:dict(right_endpose=[0,0,1,*grasp_quat(0)],right_gripper=1)
    bridge.sensors=lambda:dict(right=dict(holding=False,contact_fingers=1))
    tool=S(calibration_error=lambda:0.,finger_vertices_in_ee=lambda pose,opening:np.array([[.155,0,0]]))
    bridge.table_clearance_tools={'right':tool}
    source=dict(tcp=[0.,0.,.77],top=.78,span=[.04,.04])
    bridge.geometry=lambda a:source if a['skill']=='pick' else dict(tcp=[.12,0,.745])
    bridge.plan_pair=lambda *args:True
    executed=[]
    bridge._take=lambda side,pose,grip,label,steps:executed.append((label,list(pose),grip))
    action=dict(skill='push',arm='right',target='short object',bbox=[100,100,200,200],point=[150,150],
                observation_id=1,destination=dict(target='support',bbox=[300,300,400,400],point=[350,350]))
    return bridge,action,executed,tool


def test_selected_candidate_clearance_is_used_for_contact_and_entire_slide(monkeypatch):
    bridge,action,executed,tool=setup_bridge();openings=[];queries=[]
    tool.finger_vertices_in_ee=lambda pose,opening:openings.append(opening) or np.array([[.155,0,0]])
    # Force candidate-dependent clearances to detect accidental reuse of the
    # first candidate's certificate, even though pure world-Z yaw preserves Z.
    def clear(tcp,q,fingers,table):
        queries.append(q)
        point=np.asarray(tcp).copy();point[2]+=.01*len(queries)
        return point,dict(raise_m=.01*len(queries),quaternion=list(q))
    monkeypatch.setattr(clearance,'table_clear_grasp',clear)
    bridge.plan_pair=lambda *args:len(queries)==2
    result=bridge.execute(action)
    assert result['skill_success'] and not result['success']
    assert openings==[0.]
    assert len(queries)==2
    evidence=result['geometry']['finger_table_clearance']
    assert evidence['raise_m']==.02
    for label,pose,grip in executed:
        np.testing.assert_allclose(pose[3:],queries[-1])
        expected_z=.784+(.08 if label in ('push_hover','release') else 0)
        assert tcp_from_ee(pose)[2]==pytest.approx(expected_z)
    contact=next(pose for label,pose,grip in executed if label=='push_contact')
    last_slide=[pose for label,pose,grip in executed if label=='push_slide'][-1]
    np.testing.assert_allclose(tcp_from_ee(contact)[:2],[-.04,0])
    np.testing.assert_allclose(tcp_from_ee(last_slide)[:2],[.105,0])
    assert not bridge.env.robot.right_planner.fast_preflight


def test_unclearable_candidates_are_not_planned_or_executed(monkeypatch):
    bridge,action,executed,_=setup_bridge()
    monkeypatch.setattr(clearance,'table_clear_grasp',lambda *args:None)
    bridge.plan_pair=lambda *args:pytest.fail('Invalid clearance must reject before planning')
    result=bridge.execute(action)
    assert not result['skill_success'] and not executed
    assert len(result['geometry']['push_candidates'])==5
    assert not bridge.env.robot.right_planner.fast_preflight


@pytest.mark.parametrize('missing_geometry',[True,False])
def test_missing_geometry_or_failed_calibration_prevents_motion(missing_geometry):
    bridge,action,executed,tool=setup_bridge()
    if missing_geometry:tool.finger_vertices_in_ee=lambda *args:None
    else:tool.calibration_error=lambda:.003
    result=bridge.execute(action)
    assert not result['skill_success'] and not executed


def test_planner_failure_restores_mode():
    bridge,action,executed,_=setup_bridge()
    def fail(*args):raise SkillError('planning failed')
    bridge.plan_pair=fail
    result=bridge.execute(action)
    assert not result['skill_success'] and not executed
    assert not bridge.env.robot.right_planner.fast_preflight
