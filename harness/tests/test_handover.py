from types import SimpleNamespace
import numpy as np
import pytest
from handover import rebase_held,require_shared_grasp,execute_handover,observed_receiver_contact
from robotwin_harness_v3 import SkillError,validate,tcp_from_ee
from robodawn.mission_planner import validate_mission,RobodawnPlanner
from robodawn.mission_safety import validate_resources
from robodawn.intent_memory import IntentMemory


def step():return dict(operation='handover',source='small object',arm='left',receiver='right',grasp_part='body')


def test_receiver_contact_stays_on_suspended_observed_component():
    x,y,z=np.meshgrid(np.linspace(-.02,.02,9),np.linspace(-.01,.01,5),np.linspace(.96,.99,9))
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]
    geometry=observed_receiver_contact(points,[0,0,.98])
    assert .96<=geometry['tcp'][2]<=.99 and geometry['bottom']>=.96
    with pytest.raises(SkillError,match='not on'):
        observed_receiver_contact(points,[.12,0,1.04])


def test_handover_is_distinct_from_support_placement_and_preserves_occupancy():
    mission=validate_mission(dict(steps=[step(),dict(operation='transfer',source='small object',
                         destination='pad',arm='right',relation='on')]))
    validate_resources(mission,dict(left=dict(holding=False),right=dict(holding=False)))
    with pytest.raises(SkillError,match='empty receiver'):
        validate_resources(mission,dict(right=dict(holding=True,remembered_object='another object')))
    with pytest.raises(SkillError,match='distinct'):
        validate_mission(dict(steps=[{**step(),'receiver':'left'}]))
    with pytest.raises(SkillError,match='not a support'):
        validate_mission(dict(steps=[dict(operation='transfer',source='small object',destination='right hand',relation='on')]))
    with pytest.raises(SkillError,match='not a placement support'):
        validate(dict(skill='place',arm='left',target='right hand',bbox=[0,0,20,20],point=[10,10]))


def test_handover_compiles_pick_offer_receive_with_current_image(monkeypatch):
    planner=RobodawnPlanner(None,'Pass the small object from left to right')
    planner.steps=[step()];planner.bound_revision=planner.revision
    obs=dict(success=False,observation_id=1,sensors={s:dict(holding=False) for s in ('left','right')})
    def ground(obs,action,**kwargs):return dict(action,bbox=[100,100,200,200],point=[150,150])
    monkeypatch.setattr(planner,'_ground',ground)
    action=planner.plan(obs,[])
    assert action['skill']=='pick' and action['arm']=='left' and action['arm_required']
    planner.feedback(action,dict(skill_success=True))
    obs['sensors']['left']=dict(holding=True,remembered_object='small object')
    action=planner.plan(obs,[]);assert action['skill']=='present'
    planner.feedback(action,dict(skill_success=True))
    obs['observation_id']=3
    action=planner.plan(obs,[])
    assert action['skill']=='handover' and action['arm']=='right' and action['donor']=='left'
    assert action['grounding_role']=='handover_receiver_contact' and action['observation_id']==3
    planner.feedback(action,dict(skill_success=False));assert planner.index==0


def test_rigid_geometry_rebased_without_moving_the_remembered_world_bottom():
    info=dict(target='small object',bottom_offset=[.03,0,-.08],working_offset=[0,.04,-.03],
              grasp_quat=[1,0,0,0],directed_axis=[1,0,0],span=[.04,.05])
    donor=[.05,-.1,1.,np.sqrt(.5),0,0,np.sqrt(.5)]
    receiver=[.08,-.12,1.03,1,0,0,0]
    result=rebase_held(info,donor,receiver)
    np.testing.assert_allclose(tcp_from_ee(receiver)+result['bottom_offset'],
                              tcp_from_ee(donor)+[0,.03,-.08])
    np.testing.assert_allclose(result['directed_axis'],[0,1,0],atol=1e-7)
    assert info['bottom_offset']==[.03,0,-.08]


def test_shared_contact_rejects_two_independently_held_objects(monkeypatch):
    bridge=SimpleNamespace(sensors=lambda:{s:dict(holding=True,opposed_contact=True) for s in ('left','right')})
    monkeypatch.setattr('handover.bilateral_tokens',lambda bridge,side:{1 if side=='left' else 2})
    with pytest.raises(SkillError,match='same anonymous body'):require_shared_grasp(bridge,'left','right')
    monkeypatch.setattr('handover.bilateral_tokens',lambda bridge,side:{1})
    require_shared_grasp(bridge,'left','right')


@pytest.mark.parametrize('contact',[False,True])
def test_donor_release_follows_verified_receiver_contact_only(monkeypatch,contact):
    poses={s+'_endpose':[0,0,1,1,0,0,0] for s in ('left','right')}
    states={'left':dict(holding=True,opposed_contact=True),'right':dict(holding=False)}
    events=[]
    def take(side,pose,grip,label,steps):
        events.append((side,label))
        if side=='right' and label=='close':states['right']=dict(holding=contact,opposed_contact=contact)
        if side=='left' and label=='release':states['left']=dict(holding=False)
    info=dict(target='small object',grasp_quat=[1,0,0,0],bottom_offset=[0,0,-.05])
    bridge=SimpleNamespace(sensors=lambda:states,held={'left':info,'right':None},endpose=lambda:poses,
             _take=take,choose_grasp=lambda a,g:(poses['right_endpose'],poses['right_endpose'],[]))
    monkeypatch.setattr('handover.bilateral_tokens',lambda bridge,side:{1})
    action=dict(arm='right',donor='left',target='small object');geometry={}
    if contact:
        execute_handover(bridge,action,geometry,[])
        assert events.index(('right','close'))<events.index(('left','release'))
        assert bridge.held['left'] is None and bridge.held['right']['target']=='small object'
        assert geometry['handover']['phase']=='receiver_retained_after_release'
    else:
        with pytest.raises(SkillError,match='uncommitted'):execute_handover(bridge,action,geometry,[])
        assert not any(side=='left' for side,label in events)
        assert events[-1]==('right','open') and bridge.held['left']==info


def test_handover_intention_not_discharged_by_pick_or_generic_success():
    memory=IntentMemory();memory.initialize([step()])
    action=dict(skill='handover',target='small object',mission_intent=step())
    memory.feedback(action,dict(skill_success=True));assert memory.pending()
    memory.feedback(action,dict(skill_success=True,geometry=dict(handover=dict(phase='receiver_retained_after_release'))))
    assert not memory.pending()
