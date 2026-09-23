import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from robodawn.semantic_planner import ground
from robodawn.episodic_memory import EpisodicMemory
from robotwin_harness_v3 import SkillError


def scene(tmp_path):
    path=tmp_path/'0001_head_camera.png';Image.new('RGB',(80,80),'red').save(path)
    yy,xx=np.indices((80,80));cloud=np.dstack([xx*.001,yy*.001,np.full_like(xx,.74,dtype=float)])
    cloud[20:40,15:30,2]=.8
    np.savez_compressed(tmp_path/'0001_rgbd.npz',world_xyz=cloud,valid=np.ones((80,80),bool),robot_self_mask=np.zeros((80,80),bool))
    return dict(image_paths=[str(path)],observation_id=1,table_height_m=.74)


def test_head_table_contact_gets_one_current_depth_repair_before_memory_commit(tmp_path):
    obs=scene(tmp_path);memory=EpisodicMemory();calls=[]
    def complete(prompt,image,**kw):
        calls.append(image.copy());assert not memory.objects
        reply=dict(bbox=[650,200,900,600],point=[700,400]) if len(calls)==1 else dict(bbox=[180,230,390,530],point=[270,360])
        return SimpleNamespace(raw_text=json.dumps(reply))
    a=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='grasp_handle',arm='left',target='handle'),memory=memory)
    assert len(calls)==2 and a['depth_contact_repair']
    np.testing.assert_array_equal(calls[0][60,60],[255,0,0])
    np.testing.assert_array_equal(calls[1][60,60],[150,150,150])
    np.testing.assert_array_equal(calls[1][30,20],[255,0,0])
    assert memory.current_binding('handle',1)['point']==[270,360]


def test_head_repeated_invalid_box_abstains_without_poisoning_memory(tmp_path):
    obs=scene(tmp_path);memory=EpisodicMemory();calls=[]
    def complete(*args,**kw):
        calls.append(1);return SimpleNamespace(raw_text=json.dumps(dict(bbox=[650,200,900,600],point=[700,400])))
    with pytest.raises(SkillError,match='No above-table'):
        ground(SimpleNamespace(complete_text=complete),obs,dict(skill='pick',arm='left',target='handle'),memory=memory)
    assert len(calls)==2 and not memory.objects


def test_depth_repair_preserves_explicit_component_crop_calibration(tmp_path):
    obs=scene(tmp_path);calls=[]
    def complete(prompt,image,**kw):
        calls.append(image.shape)
        reply=dict(bbox=[750,100,950,500],point=[850,300]) if len(calls)==1 else dict(bbox=[250,350,550,850],point=[400,600])
        return SimpleNamespace(raw_text=json.dumps(reply))
    a=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='pick',arm='left',target='handle',camera='head_camera'),image_region=[100,100,450,550])
    assert len(calls)==2 and calls[0]==calls[1]
    assert a['grounding_crop_bbox_px'] and a['camera']=='head_camera'


def test_only_contact_verified_held_destinations_reject_table_background(tmp_path):
    from robodawn.grounding_evidence import admit_wrist_contact
    obs=scene(tmp_path)
    obs['sensors']={'left':{'holding':True,'remembered_object':'payload'},'right':{'holding':True,'remembered_object':'tray'}}
    action=dict(skill='place',arm='left',target='tray',camera='head_camera',bbox=[180,230,390,530],point=[270,360],support='table')
    with pytest.raises(SkillError,match='contact-verified held'):admit_wrist_contact(obs,action,include_head=True)
    assert admit_wrist_contact(obs,dict(action,support='object'),include_head=True)
    with pytest.raises(SkillError,match='contact-verified held'):
        admit_wrist_contact(obs,dict(action,bbox=[0,0,1000,1000],point=[900,900],support='container'),include_head=True)
    assert admit_wrist_contact(obs,dict(action,target='floor bin',support='container'),include_head=True) is None
    obs['sensors']['right']['holding']=False
    assert admit_wrist_contact(obs,action,include_head=True) is None
