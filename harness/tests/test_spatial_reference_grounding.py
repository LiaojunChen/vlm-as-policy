import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from robodawn.grounding_evidence import admit_wrist_contact,held_destination_arm
from robodawn.semantic_planner import ground
from robodawn.mission_planner import RobodawnPlanner
from robodawn.episodic_memory import EpisodicMemory
from robotwin_harness_v3 import SkillError


def scene(tmp_path):
    paths=[]
    for camera in ('head_camera','left_camera','right_camera'):
        path=tmp_path/f'0001_{camera}.png';Image.new('RGB',(100,100),'white').save(path);paths.append(str(path))
        yy,xx=np.indices((100,100));cloud=np.dstack([xx*.001,yy*.001,np.full((100,100),.74)])
        if camera=='head_camera':cloud[30:70,30:70,2]=.8
        suffix='rgbd' if camera=='head_camera' else camera+'_rgbd'
        np.savez_compressed(tmp_path/f'0001_{suffix}.npz',world_xyz=cloud,valid=np.ones((100,100),bool),robot_self_mask=np.zeros((100,100),bool))
    return dict(image_paths=paths,observation_id=1,table_height_m=.74,inspection_cameras=['right_camera'])


def test_relative_target_depth_rejection_falls_back_before_memory_commit(tmp_path):
    obs=scene(tmp_path);memory=EpisodicMemory();calls=[]
    def complete(prompt,image,**kw):
        assert 'WHOLE spatial reference object' in prompt and 'Do not offset' in prompt
        assert not memory.objects
        calls.append(prompt)
        return SimpleNamespace(raw_text=json.dumps(dict(bbox=[300,300,700,700],point=[500,500],support='table')))
    a=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='place',arm='left',target='reference',relation='right_of'),memory=memory)
    assert len(calls)==2 and a['camera']=='head_camera' and 'right_camera' in a['rejected_grounding_views'][0]
    assert a['contact_view_evidence']['largest_component_pixels']>4


def test_on_and_floor_container_targets_do_not_inherit_reference_height_rule(tmp_path):
    from robodawn.camera_views import in_camera
    obs=in_camera(scene(tmp_path),'right_camera')
    a=dict(skill='place',arm='left',target='support',camera='right_camera',bbox=[300,300,700,700],point=[500,500])
    for relation in ('on','inside'):
        assert admit_wrist_contact(obs,dict(a,relation=relation),include_head=True) is None
    with pytest.raises(SkillError,match='No above-table'):
        admit_wrist_contact(obs,dict(a,relation='right_of'),include_head=True)


def test_mission_provides_relation_before_grounding_without_offsetting_pixels():
    p=RobodawnPlanner(None,'place an item beside another');requests=[]
    def locate(obs,a):
        requests.append(a.copy());return dict(a,bbox=[100,100,300,300],point=[200,200],support='object')
    p._ground=locate
    a=p._destination({},dict(operation='transfer',source='payload',destination='reference',relation='right_of'),'left')
    assert requests[0]['relation']=='right_of' and a['point']==[200,200] and a['support']=='table'
    p._destination({},dict(operation='transfer',source='payload',destination='support',relation='on'),'left')
    assert 'relation' not in requests[1]


def test_held_reference_is_not_misclassified_as_the_place_support():
    obs={'sensors':{'right':dict(holding=True,remembered_object='reference')}}
    a=dict(skill='place',arm='left',target='reference')
    assert held_destination_arm(obs,a)=='right'
    assert held_destination_arm(obs,dict(a,relation='right_of')) is None
