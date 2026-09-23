import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from robodawn.camera_views import in_camera,evidence_path,observed_world_side
from robodawn.episodic_memory import EpisodicMemory,same_visual_region
from robodawn.grounding_evidence import current_robot_mask
from robodawn.semantic_planner import ground
from robodawn.mission_planner import RobodawnPlanner
from robodawn.pair_grounding import paired_image
from robotwin_harness_v3 import Bridge,SkillError


def views(tmp_path):
    paths=[]
    for camera,colour in zip(('head_camera','left_camera','right_camera'),('red','green','blue')):
        path=tmp_path/f'0002_{camera}.png';Image.new('RGB',(20,20),colour).save(path);paths.append(str(path))
        cloud=np.full((20,20,3),.2);cloud[:,:,0]=-.2
        np.savez_compressed(evidence_path(path,2,camera),valid=np.ones((20,20),bool),
                            robot_self_mask=np.zeros((20,20),bool),world_xyz=cloud)
    return dict(observation_id=2,image_paths=paths,sensors=dict(left=dict(holding=False),right=dict(holding=False)))


def action(camera='head_camera'):
    return dict(skill='reach',arm='left',target='plain item',bbox=[100,100,800,800],point=[600,500],camera=camera)


def test_camera_selection_keeps_canonical_mapping_after_reselection(tmp_path):
    obs=views(tmp_path);right=in_camera(obs,'right_camera');left=in_camera(right,'left_camera')
    assert right['image_paths'][0].endswith('right_camera.png')
    assert left['image_paths'][0].endswith('left_camera.png')
    assert in_camera(left,'head_camera')['image_paths'][0]==obs['image_paths'][0]
    assert obs['image_paths'][0].endswith('head_camera.png')
    with pytest.raises(SkillError):in_camera(obs,'actor_camera')


def test_current_wrist_mask_and_world_side_ignore_image_side(tmp_path):
    obs=views(tmp_path);wrist=in_camera(obs,'right_camera')
    assert current_robot_mask(wrist,(20,20,3)).shape==(20,20)
    assert observed_world_side(obs,action('right_camera'))=='left'  # right pixel != world right
    assert evidence_path(wrist['image_paths'][0],1,'right_camera') is None


def test_no_cross_view_or_moving_wrist_box_reuse():
    memory=EpisodicMemory();memory.remember_visual(action(),1)
    assert memory.search_region('plain item',2) is not None
    assert memory.search_region('plain item',2,camera='right_camera') is None
    memory.remember_visual(action('right_camera'),2)
    assert memory.current_binding('plain item',2,'head_camera') is None
    assert memory.current_binding('plain item',2,'right_camera')['camera']=='right_camera'
    assert memory.search_region('plain item',3,camera='right_camera') is None
    assert not same_visual_region(action(),action('right_camera'))
    assert same_visual_region(action('right_camera'),action('right_camera'))


def test_grounding_uses_only_head_until_inspection_and_tags_current_wrist(tmp_path):
    obs=views(tmp_path);calls=[]
    def complete(prompt,image,**kwargs):
        calls.append((prompt,image[0,0].tolist()))
        return SimpleNamespace(raw_text=json.dumps(dict(bbox=[100,100,800,800],point=[600,500])))
    client=SimpleNamespace(complete_text=complete)
    request=dict(skill='reach',arm='left',target='plain item')
    assert ground(client,obs,request)['camera']=='head_camera'
    wrist=ground(client,dict(obs,inspection_cameras=['right_camera']),request)
    assert wrist['camera']=='right_camera' and wrist['observation_id']==2
    assert calls[0][1]==[255,0,0] and calls[1][1]==[0,0,255]


def test_wrist_absence_falls_back_but_forced_component_view_never_switches(tmp_path):
    obs=dict(views(tmp_path),inspection_cameras=['right_camera']);calls=[]
    def complete(prompt,image,**kwargs):
        calls.append(image[0,0].tolist())
        value=dict(visible=False) if image[0,0,2] else dict(bbox=[100,100,800,800],point=[600,500])
        return SimpleNamespace(raw_text=json.dumps(value))
    client=SimpleNamespace(complete_text=complete)
    request=dict(skill='reach',arm='left',target='plain item')
    assert ground(client,obs,request)['camera']=='head_camera'
    with pytest.raises(SkillError,match='not visible'):
        ground(client,obs,dict(request,camera='right_camera'),image_region=[100,100,800,800])
    assert len(calls)==3


@pytest.mark.parametrize('fail',[False,True])
def test_geometry_routes_current_cloud_and_restores_head_even_on_error(fail):
    bridge=Bridge.__new__(Bridge)
    head=np.zeros((2,2,3));wrist=np.ones((2,2,3));valid=np.ones((2,2),bool)
    bridge.cloud=head;bridge.valid=valid;bridge.rgb=head;bridge.cam='head'
    bridge.view_data={'right_camera':dict(cloud=wrist,valid=valid,rgb=wrist,camera='right')}
    def calculate(a):
        assert bridge.cloud is wrist and bridge.cam=='right'
        if fail:raise SkillError('diagnostic failure')
        return dict(tcp=bridge.cloud[0,0].tolist())
    bridge._geometry_with_memory=calculate
    if fail:
        with pytest.raises(SkillError):bridge.geometry(action('right_camera'))
    else:assert bridge.geometry(action('right_camera'))==dict(tcp=[1,1,1],camera='right_camera')
    assert bridge.cloud is head and bridge.cam=='head'
    with pytest.raises(SkillError):bridge.geometry(action('left_camera'))


def test_pair_crops_use_actual_camera_pixels(tmp_path):
    obs=views(tmp_path)
    result=paired_image(obs['image_paths'][0],action(),action('right_camera'),obs['image_paths'][2])
    assert result.shape==(420,640,3)
    np.testing.assert_array_equal(result[270,160],[255,0,0])
    np.testing.assert_array_equal(result[270,480],[0,0,255])


def test_inspection_recovery_is_bounded_and_never_selects_occupied_arm(tmp_path):
    obs=views(tmp_path);planner=RobodawnPlanner(None,'move a visible item')
    obs['sensors']['left']['holding']=True
    assert planner._inspection_recovery(obs,'No reachable placement orientation') is None
    candidate=planner._inspection_recovery(obs,'Target not visible')
    assert candidate['skill']=='inspect' and candidate['arm']=='right'
    assert planner._inspection_recovery(obs,'Target not visible') is None
    obs['sensors']['left']['holding']=False
    assert planner._inspection_recovery(obs,'Target not visible')['arm']=='left'
    assert planner._inspection_recovery(obs,'Target not visible') is None


def test_repeated_no_contact_requests_observation_before_replanning(tmp_path):
    obs=dict(views(tmp_path),success=False);planner=RobodawnPlanner(None,'move item')
    history=[dict(action=dict(skill='pick',arm='right',target='item'),
                  result=dict(skill_success=False,failure='Empty grasp: no bilateral object contact')) for _ in range(2)]
    candidate=planner.plan(obs,history)
    assert candidate['skill']=='inspect' and candidate['observation_id']==2
    assert candidate['arm']=='right' and 'inspection_anchor' not in candidate
    assert planner.steps==[] and not planner.memory.events


def test_cross_camera_parent_crop_is_rejected_before_measuring_geometry():
    bridge=Bridge.__new__(Bridge);bridge.depth_id=2
    a=dict(action('right_camera'),observation_id=2,parent_grounding=action())
    with pytest.raises(SkillError,match='source camera'):bridge._geometry(a)


@pytest.mark.parametrize('skill',['pick','grasp_handle','push','handover'])
def test_table_only_wrist_falls_back_before_committing_memory(tmp_path,skill):
    obs=dict(views(tmp_path),table_height_m=.2,inspection_cameras=['right_camera'])
    head=evidence_path(obs['image_paths'][0],2,'head_camera')
    with np.load(head) as data:cloud=data['world_xyz'].copy()
    cloud[:,:,2]=.3
    np.savez_compressed(head,world_xyz=cloud,valid=np.ones((20,20),bool),robot_self_mask=np.zeros((20,20),bool))
    memory=EpisodicMemory();calls=[]
    def complete(prompt,image,**kwargs):
        calls.append(image[0,0].tolist())
        assert not memory.objects  # Neither a rejected view nor a partial crop commits.
        return SimpleNamespace(raw_text=json.dumps(dict(bbox=[100,100,800,800],point=[600,500])))
    request=dict(skill=skill,arm='left',target='plain item',donor='right')
    result=ground(SimpleNamespace(complete_text=complete),obs,request,memory=memory)
    assert result['camera']=='head_camera' and len(calls)==2
    assert 'No above-table' in result['rejected_grounding_views'][0]
    assert memory.current_binding('plain item',2,'head_camera') is not None
    assert memory.current_binding('plain item',2,'right_camera') is None


def test_forced_wrist_contact_fails_closed_without_memory_poisoning(tmp_path):
    obs=dict(views(tmp_path),table_height_m=.2);memory=EpisodicMemory()
    client=SimpleNamespace(complete_text=lambda *args,**kw:SimpleNamespace(raw_text=json.dumps(
        dict(bbox=[100,100,800,800],point=[600,500]))))
    with pytest.raises(SkillError,match='No above-table'):
        ground(client,obs,dict(action('right_camera'),skill='pick'),memory=memory)
    assert not memory.objects


def test_current_object_depth_admits_wrist_but_not_robot_pixels(tmp_path):
    from robodawn.grounding_evidence import admit_wrist_contact
    obs=dict(views(tmp_path),table_height_m=.1);view=in_camera(obs,'right_camera')
    request=dict(action('right_camera'),skill='pick')
    assert admit_wrist_contact(view,request)['largest_component_pixels']>=4
    path=evidence_path(view['image_paths'][0],2,'right_camera')
    with np.load(path) as data:cloud=data['world_xyz'].copy()
    np.savez_compressed(path,world_xyz=cloud,valid=np.ones((20,20),bool),robot_self_mask=np.ones((20,20),bool))
    with pytest.raises(SkillError,match='No above-table'):admit_wrist_contact(view,request)
    assert admit_wrist_contact(view,dict(request,skill='place',support='container')) is None


def test_inspection_current_anchor_has_priority_over_history_side(tmp_path):
    obs=views(tmp_path);planner=RobodawnPlanner(None,'move item')
    planner.memory.remember_visual(dict(action(),point=[200,500]),2)
    result=planner._inspection_recovery(obs,'No above-table object','right')
    assert result['arm']=='left' and result['inspection_anchor']==[200,500]
