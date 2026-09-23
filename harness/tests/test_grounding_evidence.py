import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from robodawn.grounding_evidence import current_robot_mask,robot_at_point,robot_hidden_image
from robodawn.semantic_planner import ground
from robodawn.mission_planner import RobodawnPlanner
from robotwin_harness_v3 import SkillError


def observation(tmp_path):
    image=tmp_path/'0001_head_camera.png';Image.new('RGB',(100,100),'white').save(image)
    mask=np.zeros((100,100),bool);mask[40:61,40:61]=True
    np.savez_compressed(tmp_path/'0001_rgbd.npz',robot_self_mask=mask)
    return dict(observation_id=1,image_paths=[str(image)]),mask


def test_only_matching_current_observation_mask_is_loaded(tmp_path):
    obs,mask=observation(tmp_path)
    np.testing.assert_array_equal(current_robot_mask(obs,(100,100,3)),mask)
    assert current_robot_mask({**obs,'observation_id':2},(100,100,3)) is None
    with pytest.raises(SkillError,match='calibration'):current_robot_mask(obs,(90,100,3))


def test_full_and_crop_pixels_use_the_same_calibrated_mask():
    mask=np.zeros((100,100),bool);mask[50,60]=True
    crop=(np.array([40,30]),np.array([80,70]),100,100)
    assert robot_at_point(mask,[60000/99,50000/99])
    assert robot_at_point(mask,[500,500],crop)
    assert not robot_at_point(mask,[0,0],crop)
    original=np.full((41,41,3),255,np.uint8)
    hidden=robot_hidden_image(original,mask,crop)
    np.testing.assert_array_equal(hidden[20,20],[150,150,150])
    np.testing.assert_array_equal(original[20,20],[255,255,255])


def test_robot_point_is_repaired_before_action_and_memory_acceptance(tmp_path):
    obs,mask=observation(tmp_path);calls=[]
    replies=[dict(bbox=[100,100,800,800],point=[500,500],support='object'),
             dict(bbox=[100,100,300,300],point=[200,200],support='object')]
    def complete(prompt,image,**kwargs):
        calls.append((prompt,image.copy()))
        return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    action=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='place',arm='left',target='support'))
    assert action['point']==[200,200] and action['robot_mask_repair']
    assert len(calls)==2 and 'on the ROBOT' in calls[1][0]
    np.testing.assert_array_equal(calls[0][1][50,50],[255,255,255])
    np.testing.assert_array_equal(calls[1][1][50,50],[150,150,150])


def test_repeated_robot_grounding_is_not_executable(tmp_path):
    obs,_=observation(tmp_path)
    client=SimpleNamespace(complete_text=lambda *a,**k:SimpleNamespace(raw_text=json.dumps(
        dict(bbox=[100,100,800,800],point=[500,500],support='object'))))
    with pytest.raises(SkillError,match='Grounding invalid after repair'):
        ground(client,obs,dict(skill='pick',arm='left',target='object'))


def test_coloured_robot_cannot_override_a_valid_model_point(tmp_path,monkeypatch):
    obs,_=observation(tmp_path)
    client=SimpleNamespace(complete_text=lambda *a,**k:SimpleNamespace(raw_text=json.dumps(
        dict(bbox=[100,100,300,300],point=[200,200],support='object'))))
    monkeypatch.setattr('robodawn.semantic_planner.refine_colour',lambda obs,a:dict(a,point=[500,500]))
    result=ground(client,obs,dict(skill='pick',arm='left',target='object'))
    assert result['point']==[200,200] and 'colour_refinement_abstained' in result


def test_failed_binding_can_request_recovery_without_fabricating_executed_history():
    planner=RobodawnPlanner(None,'Lift an object')
    planner.steps=[dict(operation='lift',source='object',arm='left')]
    planner._replan=lambda obs,history:None
    def fail(obs):raise SkillError('Target is occluded')
    planner._bind_transfer_identities=fail
    calls=[]
    def recover(obs,history):
        calls.append((obs,history));return dict(skill='home',name='home',arm='right')
    planner.reactive=SimpleNamespace(plan=recover)
    obs=dict(success=False,observation_id=1,sensors={})
    action=planner.plan(obs,[])
    assert action['skill']=='home' and action['mission_recovery'].endswith('Target is occluded')
    assert calls[0][0]['perception_failure']=='Target is occluded' and calls[0][1]==[]
    assert 'perception_failure' not in obs
