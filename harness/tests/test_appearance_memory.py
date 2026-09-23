import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from robodawn.appearance_memory import observed_signature,current_component,repair_robot_point
from robodawn.episodic_memory import EpisodicMemory
from robodawn.semantic_planner import ground
from robodawn.mission_planner import RobodawnPlanner


def image():
    rgb=np.full((100,100,3),240,np.uint8);rgb[15:85,35:65]=[40,100,60]
    return rgb


def test_signature_requires_observed_dominant_chromatic_appearance():
    rgb=image();valid=np.ones((100,100),bool)
    signature=observed_signature(rgb,valid,[300,100,700,900]);assert signature
    assert observed_signature(np.full_like(rgb,80),valid,[300,100,700,900]) is None
    assert observed_signature(rgb,np.zeros_like(valid),[300,100,700,900]) is None


def test_current_recovery_never_invents_a_pixel_through_occlusion():
    rgb=image();valid=np.ones((100,100),bool)
    signature=observed_signature(rgb,valid,[300,100,700,900])
    valid[:60]=False
    result=current_component(rgb,valid,[300,100,700,900],signature)
    x,y=np.rint(np.asarray(result['point'])*99/1000).astype(int)
    assert valid[y,x] and y>=60 and result['pixels']>=16
    assert current_component(rgb,np.zeros_like(valid),[300,100,700,900],signature) is None
    assert current_component(rgb,valid,[0,0,200,1000],signature) is None


def test_multiple_similar_components_abstain_instead_of_choosing_one():
    rgb=image();valid=np.ones((100,100),bool)
    signature=observed_signature(rgb,valid,[300,100,700,900])
    rgb[45:55]=240
    assert current_component(rgb,valid,[300,100,700,900],signature) is None


def test_disconnected_visible_parts_need_current_depth_corroborating_reference():
    rgb=image();valid=np.ones((100,100),bool)
    signature=observed_signature(rgb,valid,[300,100,700,900])
    x,y=np.meshgrid(np.linspace(-.1,.1,100),np.linspace(-.1,.1,100))
    cloud=np.stack([x,y,np.full_like(x,.8)],axis=-1)
    reference=cloud[15:85,35:65].reshape(-1,3)
    valid[45:55]=False
    result=current_component(rgb,valid,[300,100,700,900],signature,cloud,reference)
    assert result is not None
    assert current_component(rgb,valid,[300,100,700,900],signature,cloud+[.2,0,0],reference) is None


def fixture(tmp_path):
    rgb=image();valid=np.ones((100,100),bool)
    Image.fromarray(rgb).save(tmp_path/'0001_head_camera.png')
    np.savez_compressed(tmp_path/'0001_rgbd.npz',valid=valid,robot_self_mask=~valid)
    robot=np.zeros_like(valid);robot[:60]=True
    current=rgb.copy();current[robot]=80
    Image.fromarray(current).save(tmp_path/'0002_head_camera.png')
    np.savez_compressed(tmp_path/'0002_rgbd.npz',valid=~robot,robot_self_mask=robot)
    memory=EpisodicMemory();memory.objects['support']=dict(description='support',
        identity_verification_observation_id=1,last_visual=dict(bbox=[300,100,700,900],
        point=[500,500],observation_id=1,component_only=False))
    obs=dict(observation_id=2,image_paths=[str(tmp_path/'0002_head_camera.png')])
    return obs,memory,current


def test_reference_is_verified_episode_memory_not_a_historical_execution_point(tmp_path):
    obs,memory,current=fixture(tmp_path)
    result=repair_robot_point(obs,memory,current,[300,100,700,900],'support')
    assert result['point'][1]>600 and result['reference_observation_id']==1
    assert memory.objects['support']['last_visual']['point']==[500,500]
    memory.objects['support']['identity_verification_observation_id']=2
    assert repair_robot_point(obs,memory,current,[300,100,700,900],'support') is None


def test_recovery_uses_current_model_box_and_keeps_descriptor_out_of_model_prompt(tmp_path):
    obs,memory,_=fixture(tmp_path);calls=[]
    def complete(prompt,image,**kwargs):
        calls.append(prompt)
        return SimpleNamespace(raw_text=json.dumps(dict(bbox=[300,100,700,900],point=[500,500],support='object')))
    result=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='place',arm='left',target='support'),memory=memory)
    assert len(calls)==1 and result['point'][1]>600 and result['current_appearance_repair']
    assert 'hue' not in calls[0] and 'tolerance' not in calls[0]


def test_visual_replanning_receives_binding_failure_even_before_first_action(tmp_path):
    path=tmp_path/'head.png';Image.fromarray(image()).save(path);calls=[]
    def complete(prompt,image,**kwargs):
        calls.append((prompt,image));return SimpleNamespace(raw_text=json.dumps(dict(steps=[
            dict(operation='lift',source='a visible source',arm='left',location='stay')])))
    planner=RobodawnPlanner(SimpleNamespace(complete_text=complete),'Lift a visible source')
    planner.last_perception_failure='The proposed destination was the source'
    planner._replan(dict(image_paths=[str(path)],observation_id=1,sensors={}),[])
    assert calls[0][1] is not None and 'proposed destination was the source' in calls[0][0]
