from types import SimpleNamespace
import numpy as np
import pytest
from robotwin_harness_v3 import Bridge,tcp_from_ee


@pytest.mark.parametrize('skill',['reach','press'])
def test_reach_never_plans_or_executes_press_contact(skill):
    b=Bridge.__new__(Bridge);b.index=0;b.held={'left':None};b.failed={}
    b.env=SimpleNamespace(eval_success=False,check_success=lambda:False,
                          robot=SimpleNamespace(left_planner=SimpleNamespace(fast_preflight=False)))
    b.sensors=lambda:{'left':{'holding':False}}
    b.endpose=lambda:{'left_endpose':[0,0,1,1,0,0,0],'left_gripper':.6}
    point=np.array([.1,0,.8]);b.geometry=lambda a:dict(tcp=point.tolist())
    plans=[]
    def plan(side,pre,low):plans.append((pre,low));return True
    b.plan_pair=plan
    b._take=lambda side,pose,grip,label,steps:steps.append(dict(pose=pose,grip=grip,name=label))
    result=b.execute(dict(skill=skill,arm='left',target='item',bbox=[100,100,800,800],point=[500,500]))
    assert result['skill_success'] and not b.env.robot.left_planner.fast_preflight
    np.testing.assert_allclose(tcp_from_ee(plans[0][0]),point+[0,0,.07])
    actions=result['subactions']
    if skill=='reach':
        np.testing.assert_allclose(plans[0][0],plans[0][1])
        assert [a['name'] for a in actions]==['hover'] and actions[0]['grip']==.6
    else:
        np.testing.assert_allclose(tcp_from_ee(plans[0][1]),point+[0,0,-.012])
        assert [a['name'] for a in actions]==['hover','press']
        assert all(a['grip']==0 for a in actions)
