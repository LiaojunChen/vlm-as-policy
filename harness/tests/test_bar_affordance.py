import numpy as np
from PIL import Image
from robodawn.bar_affordance import end_contact


def scene(tmp_path,clipped=False,wide=False):
    yy,xx=np.indices((121,161));x=(xx-80)*.002;y=(yy-60)*.002
    z=np.full_like(x,.74)
    mask=(np.abs(y)<(.06 if wide else .016))&(x>-.13)&(x<(.18 if clipped else .13))
    z[mask]=.77
    image=tmp_path/'0001_head_camera.png';Image.new('RGB',(161,121),'brown').save(image)
    np.savez_compressed(tmp_path/'0001_rgbd.npz',world_xyz=np.dstack([x,y,z]),valid=np.ones_like(mask),robot_self_mask=np.zeros_like(mask))
    obs=dict(observation_id=1,image_paths=[str(image)],table_height_m=.74)
    parent=dict(target='bar',camera='head_camera',observation_id=1,bbox=[0,0,1000,1000],point=[500,500])
    return obs,parent


def test_explicit_left_and_right_end_resolve_to_separated_observed_contacts(tmp_path):
    obs,parent=scene(tmp_path)
    left=end_contact(obs,parent,'left end of object','left')
    right=end_contact(obs,parent,'right end of object','right')
    assert left and right
    assert left['selected_world_point'][0]<-.08 and right['selected_world_point'][0]>.08
    assert left['world_left_to_right_axis'][0]>.99
    assert .01<left['observed_width_m']<.04
    assert left['camera']=='head_camera' and left['observation_id']==1


def test_unrelated_semantic_parts_and_broad_shapes_use_model_fallback(tmp_path):
    obs,parent=scene(tmp_path)
    assert end_contact(obs,parent,'red handle','left') is None
    assert end_contact(obs,parent,'right end','left') is None
    obs,parent=scene(tmp_path,wide=True)
    assert end_contact(obs,parent,'left end','left') is None


def test_clipped_requested_end_is_not_invented(tmp_path):
    obs,parent=scene(tmp_path,clipped=True)
    assert end_contact(obs,parent,'right end','right') is None
    assert end_contact(obs,parent,'left end','left') is not None


def test_stale_parent_and_other_gripper_proximity_abstain(tmp_path):
    obs,parent=scene(tmp_path)
    assert end_contact(obs,dict(parent,observation_id=0),'left end','left') is None
    contact=end_contact(obs,parent,'left end','left')
    from robotwin_harness_v3 import ee_from_tcp,grasp_quat
    pose=ee_from_tcp(contact['selected_world_point'],grasp_quat(0))
    assert end_contact(obs,parent,'left end','left',pose) is None
