import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from transforms3d.euler import euler2mat
from contact_geometry import working_part_reference, transformed_contact_tcp,retained_working_offset
from robotwin_harness_v3 import Bridge, SkillError, validate
from robodawn.mission_planner import RobodawnPlanner, validate_mission, validate_explicit_arm_sequence
from robodawn.mission_safety import validate_resources
from robodawn.semantic_planner import ground


def test_working_head_reference_uses_head_not_handle_position():
    cloud=np.zeros((40,60,3));valid=np.zeros((40,60),bool)
    x,y=np.meshgrid(np.linspace(.09,.13,20),np.linspace(-.02,.02,20))
    cloud[10:30,30:50,:2]=np.stack([x,y],axis=-1)
    cloud[10:30,30:50,2]=.76;valid[10:30,30:50]=True
    r=working_part_reference(cloud,valid,[450,200,900,850],[650,500],.74)
    np.testing.assert_allclose(r['point'],[.11,0,.74],atol=1e-6)
    assert r['underside_source']=='observed_support_plane'
    assert working_part_reference(cloud,np.zeros_like(valid),[0,0,1000,1000],[500,500],.74) is None


def test_contact_offset_follows_tool_rotation_and_reaches_the_target():
    offset=np.array([.12,0,-.02]);support=np.array([0,.1,.78])
    initial=euler2mat(0,.3,.2);current=euler2mat(0,.3,1.5)
    tcp=transformed_contact_tcp(support,offset,initial,current,-.004)
    np.testing.assert_allclose(tcp+current@initial.T@offset,support+[0,0,-.004])


def test_late_observed_working_part_is_registered_in_original_grasp_frame():
    initial=euler2mat(.2,.1,-.3);current=euler2mat(-.1,.3,1.4)
    tcp=np.array([.1,-.1,1.]);point=tcp+[.03,.06,-.02]
    offset=retained_working_offset(point,tcp,initial,current)
    np.testing.assert_allclose(tcp+current@initial.T@offset,point)
    with pytest.raises(ValueError,match='too far'):
        retained_working_offset(point+[1,0,0],tcp,initial,current)


def test_late_reference_is_allowed_only_for_retained_tool_contact():
    ref=dict(target='working tip',bbox=[100,100,200,200],point=[150,150],observation_id=2)
    action=dict(skill='place',arm='left',target='support',bbox=[300,300,500,500],point=[400,400],
                release=False,use_contact_part=True,contact_reference=ref)
    validate(action)
    with pytest.raises(SkillError,match='retained tool contact'):validate({**action,'release':True})


def test_tool_acquired_by_recovery_gets_a_current_working_reference(monkeypatch):
    planner=RobodawnPlanner(None,'Touch the target with a held tool')
    planner.steps=[dict(operation='tool_contact',source='tool',contact_part='head',destination='support',
                        arm='left',relation='on',grasp_part='handle')]
    planner.bound_revision=planner.revision
    requests=[]
    def ground(obs,action,**kwargs):
        requests.append(action);return dict(action,bbox=[100,100,200,200],point=[150,150],camera='head_camera')
    monkeypatch.setattr(planner,'_ground',ground)
    obs=dict(success=False,observation_id=4,sensors=dict(left=dict(holding=True,remembered_object='tool')))
    action=planner.plan(obs,[])
    assert action['contact_reference']['observation_id']==4 and not action['release']
    assert requests[-1]['target']=='head of tool'
    planner.feedback(action,dict(skill_success=False,geometry=dict(retained_working_reference={'point':[0,0,1]})))
    assert 'tool' in planner.working_references
    planner.feedback(dict(skill='pick',target='tool',mission_recovery='recover'),dict(skill_success=True))
    assert 'tool' not in planner.working_references


def test_executor_keeps_late_reference_even_if_transport_has_no_ik_solution():
    bridge=Bridge.__new__(Bridge)
    pose=[.1,-.1,1.05,1,0,0,0]
    bridge.index=0;bridge.plan_cache=[];bridge.depth_id=2;bridge.table=.74
    bridge.failed={};bridge.last_pair_status={'approach':'Fail','contact':None}
    bridge.env=SimpleNamespace(eval_success=False,check_success=lambda:False,
                              robot=SimpleNamespace(left_planner=SimpleNamespace(fast_preflight=False)))
    bridge.endpose=lambda:dict(left_endpose=pose.copy(),right_endpose=pose.copy(),left_gripper=0.,right_gripper=1.)
    bridge.sensors=lambda:dict(left=dict(holding=True,remembered_object='tool'),right=dict(holding=False))
    bridge.geometry=lambda action:dict(tcp=[0,.1,.8],top=.8)
    bridge.plan_pair=lambda *args:False
    x,y=np.meshgrid(np.linspace(.11,.13,20),np.linspace(-.01,.01,20))
    bridge.cloud=np.stack([x,y,np.full_like(x,.95)],axis=-1);bridge.valid=np.ones((20,20),bool)
    bridge.held={'left':dict(target='tool',grasp_quat=[1,0,0,0],bottom_offset=[0,0,-.1],span=[.04,.04]),'right':None}
    action=dict(skill='place',arm='left',target='support',bbox=[0,0,1000,1000],point=[500,500],support='object',
                release=False,use_contact_part=True,observation_id=2,contact_reference=dict(
                    target='tool tip',bbox=[0,0,1000,1000],point=[500,500],observation_id=2))
    result=bridge.execute(action)
    assert result['failure'].startswith('No reachable placement orientation')
    assert result['geometry']['retained_working_reference']['observation_id']==2
    np.testing.assert_allclose(bridge.held['left']['working_offset'],[-.1,.1,-.1],atol=1e-6)
    assert not result['subactions']


def test_component_crop_coordinates_map_back_to_original_sensor(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            assert image.shape[0]<240 and image.shape[1]<320
            return SimpleNamespace(raw_text=json.dumps(dict(bbox=[0,0,1000,1000],point=[500,500])))
    a=ground(Client(),{'image_paths':[str(path)],'observation_id':1},dict(skill='press',arm='left',target='tool tip'),[400,400,600,600])
    lo=np.array(a['grounding_crop_bbox_px'][:2]);hi=np.array(a['grounding_crop_bbox_px'][2:])
    np.testing.assert_allclose(a['point'],(lo+hi)*.5/[319,239]*1000)


def test_hit_plan_cannot_silently_release_the_tool():
    with pytest.raises(SkillError,match='contact/impact'):
        validate_explicit_arm_sequence([dict(operation='transfer')],'Take a tool and hit the target.')
    validate_explicit_arm_sequence([dict(operation='tool_contact')],'Take a tool and hit the target.')
    validate_explicit_arm_sequence([dict(operation='transfer')],'Move the tool onto the mat.')
    step=dict(operation='tool_contact',source='tool',destination='target',arm='left',relation='on')
    with pytest.raises(SkillError,match='contact_part'):validate_mission({'steps':[step]})
    assert validate_mission({'steps':[{**step,'contact_part':'working tip'}]})


def test_tool_contact_rejects_a_redundant_preliminary_pickup():
    with pytest.raises(SkillError,match='DELETE'):
        validate_resources([dict(operation='lift',source='tool handle',arm='left'),
                            dict(operation='tool_contact',source='tool',arm='left')],{})
    validate_resources([dict(operation='tool_contact',source='tool',arm='left')]*2,{})


def test_compiler_observes_grasp_and_working_part_before_retained_contact(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    step=dict(operation='tool_contact',source='tool',contact_part='tool head',destination='block',arm='left',relation='on',grasp_part='handle')
    replies=[{'steps':[step]},dict(bbox=[100,100,400,400],point=[250,250]),
             dict(bbox=[600,500,800,800],point=[700,650]),
             dict(description_A='tool',description_B='block',source='A',destination='B',source_grasp_part='handle')]+[dict(bbox=[100,100,400,400],point=[250,250])]*3
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    p=RobodawnPlanner(Client(),'Hit the block with the tool')
    obs=dict(success=False,image_paths=[str(path)],observation_id=1,sensors={'left':{'holding':False}})
    a=p.plan(obs,[])
    assert a['skill']=='pick' and a['contact_reference']['target']=='tool head'
    assert a['contact_reference']['observation_id']==1
    validate(a);p.feedback(a,{'skill_success':True})
    obs['sensors']['left']={'holding':True,'remembered_object':'tool'}
    a=p.plan(obs,[])
    assert a['skill']=='place' and a['release'] is False and a['use_contact_part'] is True


def test_uncoloured_pad_uses_unique_flat_observation_not_hidden_depth():
    b=Bridge.__new__(Bridge);b.depth_id=1;b.table=.74
    b.cloud=np.zeros((40,60,3));b.valid=np.zeros((40,60),bool)
    pad=dict(point=[-.2,0,.741],area=1000,yaw=0,observation_id=1)
    b.landmarks={'blue':[pad]}
    a=dict(skill='place',arm='left',target='pad',observation_id=1,bbox=[0,0,1000,1000],point=[500,500])
    assert b.geometry(a)['tcp']==pad['point']
    assert a['landmark_resolution']=='unique_observed_flat_pad'


def test_ambiguous_or_explicitly_other_coloured_pad_does_not_guess():
    b=Bridge.__new__(Bridge);b.depth_id=1;b.table=.74
    pad=dict(point=[-.2,0,.741],area=1000,yaw=0,observation_id=1)
    b.landmarks={'blue':[pad],'red':[{**pad,'point':[.2,0,.741]}]}
    b.cloud=np.zeros((40,60,3));b.valid=np.zeros((40,60),bool)
    b.cam=SimpleNamespace(get_model_matrix=lambda:np.eye(4))
    for target in ('pad','yellow pad'):
        a=dict(skill='place',arm='left',target=target,observation_id=1,bbox=[0,0,1000,1000],point=[500,500])
        with pytest.raises(SkillError):b.geometry(a)
        assert 'landmark_resolution' not in a
