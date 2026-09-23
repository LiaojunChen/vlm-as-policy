from copy import deepcopy
from types import SimpleNamespace as S

import numpy as np
import pytest
from transforms3d.euler import euler2quat

from measured_release import measured_release_pose
from robotwin_harness_v3 import Bridge,SkillError


def fixture(side='right',release=True,lower_error=False,terminal=False):
    state=dict(left_endpose=[-.2,0,1,1,0,0,0],left_gripper=.3,
               right_endpose=[.2,0,1,1,0,0,0],right_gripper=.3)
    bridge=Bridge.__new__(Bridge);bridge.index=0;bridge.failed={};bridge.held={side:dict(height_under_tcp=.03)}
    bridge.env=S(robot=S(left_planner=S(fast_preflight=False),right_planner=S(fast_preflight=False)),
                 eval_success=False,check_success=lambda:False)
    held={side:True};executed=[];plans=[];after_lower=[]
    bridge.endpose=lambda:deepcopy(state)
    bridge.sensors=lambda:{s:dict(holding=held.get(s,False)) for s in ('left','right')}
    bridge.geometry=lambda action:dict(tcp=[.1,-.1,.76],top=.76)
    def pair(which,pre,low):
        plans.append((list(pre),list(low)));bridge.last_pair_status=dict(approach='Success',contact='Success')
        return True
    bridge.plan_pair=pair
    def take(which,pose,grip,label,steps):
        executed.append((label,list(pose),grip));state[which+'_endpose']=list(pose)
        if label=='lower':
            if lower_error:raise SkillError('lower: original tracking failure')
            state[which+'_endpose'][:3]=(np.asarray(pose[:3])+[.002,-.001,.008]).tolist()
            state[which+'_endpose'][3:]=(-euler2quat(.01,.02,0)).tolist()
            after_lower.append(deepcopy(state[which+'_endpose']))
            if terminal:bridge.env.eval_success=True
        if label=='release':held[which]=False
    bridge._take=take
    action=dict(skill='place',arm=side,target='observed support',support='object',release=release,
                bbox=[100,100,300,300],point=[200,200],observation_id=1)
    return bridge,action,executed,plans,after_lower


@pytest.mark.parametrize('side',['left','right'])
def test_open_uses_exact_measured_pose_instead_of_retrying_lower(side):
    bridge,action,executed,plans,lower=fixture(side)
    result=bridge.execute(action)
    assert result['skill_success'] and not result['success']
    assert [s[0] for s in executed]==['transfer','lower','release','retreat']
    assert executed[2][1]==lower[0] and executed[2][1]!=plans[0][1]
    assert executed[2][2]==1 and executed[3][1]==plans[0][0]
    evidence=result['geometry']['release_pose_control']
    np.testing.assert_allclose(evidence['deferred_position_correction_m'],[-.002,.001,-.008])
    assert evidence['deferred_rotation_correction_deg']>0
    assert bridge.held[side] is None


def test_retained_contact_does_not_open_or_snapshot_a_release():
    bridge,action,executed,_,_=fixture(release=False)
    result=bridge.execute(action)
    assert result['skill_success'] and [e[0] for e in executed]==['transfer','lower']
    assert 'release_pose_control' not in result['geometry'] and bridge.held['right'] is not None


def test_failed_lower_keeps_original_failure_and_never_opens():
    bridge,action,executed,_,_=fixture(lower_error=True)
    result=bridge.execute(action)
    assert not result['skill_success'] and 'original tracking failure' in result['failure']
    assert [e[0] for e in executed]==['transfer','lower'] and bridge.held['right'] is not None


def test_already_officially_successful_lower_does_not_add_release_evidence():
    bridge,action,_,_,_=fixture(terminal=True)
    result=bridge.execute(action)
    assert result['success'] and 'release_pose_control' not in result['geometry']


@pytest.mark.parametrize('pose',[None,[0,0,1],[0,0,float('nan'),1,0,0,0],[0,0,1,0,0,0,0]])
def test_missing_or_invalid_proprioception_never_falls_back_to_planned_motion(pose):
    with pytest.raises(SkillError,match='finite measured endpose'):
        measured_release_pose(dict(right_endpose=pose),'right',[0,0,1,1,0,0,0])


def test_snapshot_copies_exact_pose_without_mutating_inputs_or_quaternion_sign():
    state=dict(right_endpose=[.1,.2,.9,-1,0,0,0]);before=deepcopy(state)
    target=[.1,.2,.9,1,0,0,0];pose,evidence=measured_release_pose(state,'right',target)
    assert pose==state['right_endpose'] and pose is not state['right_endpose']
    assert state==before and target==[.1,.2,.9,1,0,0,0]
    assert evidence['deferred_rotation_correction_deg']==0
