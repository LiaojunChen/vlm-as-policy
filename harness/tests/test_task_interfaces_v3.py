import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from showharness.runtime import ensure_upstream
ensure_upstream()
from core.vlm.dual_roles import DualControllerAgent
from robotwin_harness_v3 import validate,SkillError,refine_colour

def test_native_show_controller_accepts_semantic_recovery_tokens():
    class Client:
        def complete_json(self,*args,**kwargs):
            value={'left':'STILL','right':'ROTATE','reasoning':'Recover held orientation'}
            return SimpleNamespace(payload={'json':value},raw_text=json.dumps(value))
    controller=DualControllerAgent(Client(),'{output_contract}','')
    decision=controller._decide_json('Rotate right held object',None,[None,None],{'left':'STILL','right':'STILL'},False)
    assert not decision.payload.get('fallback')
    assert decision.tokens['right']=='ROTATE'

def test_high_resolution_colour_grounding_preserves_normalized_coordinates(tmp_path):
    from PIL import Image
    rgb=np.zeros((480,640,3),np.uint8);rgb[200:280,300:400]=[0,255,255]
    image=tmp_path/'image.png';Image.fromarray(rgb).save(image)
    a=refine_colour({'image_paths':[str(image)]},dict(skill='place',arm='left',target='cyan pad',bbox=[400,350,650,650],point=[550,500]))
    assert abs(a['point'][0]-349.5/639*1000)<.01
    assert abs(a['point'][1]-239.5/479*1000)<.01
    validate(a)

def test_hinge_arc_rejects_unbounded_rotation():
    import pytest
    with pytest.raises(SkillError):validate(dict(skill='arc',arm='left',target='hinge',bbox=[0,0,200,200],point=[100,100],axis='z',angle=180))
    with pytest.raises(SkillError):validate(dict(skill='rotate',arm='left',axis='z',angle='not-a-number'))

def test_visible_hinge_supports_non_axis_aligned_mechanism():
    from robotwin_harness_v3 import Bridge
    b=Bridge.__new__(Bridge);b.depth_id=1;b.landmarks={};b.cloud=np.zeros((480,640,3));b.valid=np.zeros((480,640),bool)
    b.cloud[48,64]=[0,0,.8];b.cloud[431,575]=[.12,.16,.8];b.valid[48,64]=b.valid[431,575]=True
    a=dict(skill='arc',arm='left',target='lid hinge',bbox=[0,0,1000,1000],point=[500,500],hinge_line=[[100,100],[900,900]],angle=20,observation_id=1)
    validate(a);g=b.geometry(a)
    np.testing.assert_allclose(g['hinge_axis'],[.6,.8,0]);np.testing.assert_allclose(g['tcp'],[.06,.08,.8])

def test_invalid_model_plan_is_not_eligible_for_infrastructure_retry():
    from finalize_harness_v3p import normalize
    malformed=normalize(dict(status='error',success=None,error='Subgoal 1 is missing required motion'))
    assert malformed['status']=='completed' and malformed['success'] is False
    assert malformed['end_reason']=='invalid_model_plan'
    cuda=normalize(dict(status='error',success=None,error='CUDA out of memory'))
    assert cuda['status']=='error' and cuda['success'] is None
    deadline=normalize(dict(status='error',success=None,end_reason='worker_interrupted',observed_wall_s=1247,wall_budget_s=1200))
    assert deadline['status']=='completed' and deadline['success'] is False
    assert deadline['end_reason']=='wall_time_budget'

def test_target_must_be_named_and_coordinates_have_one_unambiguous_location():
    import pytest
    with pytest.raises(SkillError,match='STRING'):
        validate(dict(skill='press',arm='right',target={'bbox':[100,100,200,200],'point':[150,150]}))
    action=dict(skill='push',arm='right',target='blue cup',bbox=[100,100,200,200],point=[150,150],destination=dict(target='coaster',bbox=[300,300,400,400],point=[350,350],support='object'))
    assert validate(action)['destination']['target']=='coaster'

def test_grounding_cannot_switch_explicit_pick_arm(tmp_path):
    from PIL import Image
    from robotwin_harness_v3 import RobodawnPlanner
    path=tmp_path/'head.png';Image.new('RGB',(100,100)).save(path)
    responses=[dict(skill='pick',arm='right',target='green block',bbox=[700,300,900,500],point=[800,400]),dict(arm='left',bbox=[400,300,600,500],point=[500,400],approach='top',support='table')]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(responses.pop(0)))
    obs=dict(success=False,image_paths=[str(path)]*3,observation_id=1,endpose={},sensors={'right':{'holding':False}},table_height_m=.74)
    action=RobodawnPlanner(Client(),'Use the right arm to pick the green block.').plan(obs,[])
    assert action['arm']=='right'
    assert action['point']==[800,400]


def test_stacking_uses_visible_block_height_even_if_model_says_table():
    from robotwin_harness_v3 import Bridge
    b=Bridge.__new__(Bridge);b.depth_id=1;b.table=.74;b.landmarks={}
    ys,xs=np.meshgrid(np.linspace(-.05,.05,100),np.linspace(-.05,.05,100),indexing='ij')
    b.cloud=np.stack([xs,ys,np.full_like(xs,.74)],axis=-1)
    b.cloud[30:70,30:70,2]=.79;b.valid=np.ones((100,100),bool)
    action=dict(skill='place',arm='right',target='red block',bbox=[280,280,720,720],point=[500,500],support='table',observation_id=1)
    geometry=b.geometry(action)
    assert action['support']=='object'
    assert abs(geometry['tcp'][2]-.79)<1e-6
    assert abs(geometry['tcp'][0])<.002 and abs(geometry['tcp'][1])<.002

def test_direct_proprioception_remains_json_serializable_after_integer_gripper_commands():
    from robotwin_harness_v3 import Bridge
    bridge=Bridge.__new__(Bridge)
    bridge.env=SimpleNamespace(get_arm_pose=lambda side:[0.,0.,1.,1.,0.,0.,0.],robot=SimpleNamespace(get_left_gripper_val=lambda:np.int64(0),get_right_gripper_val=lambda:np.int64(1)))
    restored=json.loads(json.dumps(bridge.endpose()))
    assert restored['left_gripper']==0 and restored['right_gripper']==1

def test_full_observation_normalizes_numpy_gripper_state_before_model_input(tmp_path):
    from robotwin_harness_v3 import Bridge
    frame=np.zeros((4,4,3),np.uint8)
    raw={'endpose':{'left_endpose':np.array([0,0,1,1,0,0,0]),'right_endpose':np.array([0,0,1,1,0,0,0]),'left_gripper':np.int64(0),'right_gripper':np.int64(1)},'observation':{n:{'rgb':frame} for n in ('head_camera','left_camera','right_camera')}}
    position=np.zeros((4,4,4));position[:,:,2]=-1
    cam=SimpleNamespace(take_picture=lambda:None,get_picture=lambda key:position if key=='Position' else np.zeros((4,4,4)),get_model_matrix=lambda:np.eye(4),get_intrinsic_matrix=lambda:np.eye(3))
    entity=SimpleNamespace(get_links=lambda:[])
    b=Bridge.__new__(Bridge);b.directory=tmp_path;(tmp_path/'frames').mkdir();b.obs_id=0;b.landmarks={};b.instruction='test';b.cam=cam;b.sensors=lambda:{}
    b.env=SimpleNamespace(get_obs=lambda:raw,cameras=SimpleNamespace(left_camera=cam,right_camera=cam),robot=SimpleNamespace(left_entity=entity,right_entity=entity),eval_success=False,check_success=lambda:False)
    obs=json.loads(json.dumps(b.observe()))
    assert obs['endpose']['left_gripper']==0 and obs['endpose']['right_gripper']==1
