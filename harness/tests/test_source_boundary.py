import numpy as np
from PIL import Image

from robodawn.source_boundary import clipped_source_anchor
from robodawn.mission_planner import RobodawnPlanner


def scene(tmp_path,clipped=True):
    paths=[]
    for camera in ('head_camera','left_camera','right_camera'):
        path=tmp_path/f'0001_{camera}.png';Image.new('RGB',(101,101),'white').save(path);paths.append(str(path))
    yy,xx=np.indices((101,101));cloud=np.dstack([xx*.002,yy*.002-.2,np.full((101,101),.74)])
    cloud[50:101 if clipped else 90,60:80,2]=.79
    np.savez_compressed(tmp_path/'0001_rgbd.npz',world_xyz=cloud,valid=np.ones((101,101),bool),robot_self_mask=np.zeros((101,101),bool))
    return dict(observation_id=1,image_paths=paths,table_height_m=.74,
                sensors={side:dict(holding=False) for side in ('left','right')})


def action():
    return dict(skill='pick',arm='right',target='long object',observation_id=1,
                bbox=[500,400,900,1000],point=[700,700],camera='head_camera')


def test_only_actual_current_object_foreground_at_border_requests_observation(tmp_path):
    obs=scene(tmp_path);anchor=clipped_source_anchor(obs,action())
    assert anchor and set(anchor['clipped_borders'])=={'bottom'}
    assert anchor['clipped_borders']['bottom']>=50  # depth-jump filtering removes the first column
    assert anchor['observed_component_pixels']>100
    assert 600<=anchor['image_point'][0]<=790 and 500<=anchor['image_point'][1]<=1000
    obs=scene(tmp_path,False)
    assert clipped_source_anchor(obs,action()) is None


def test_stale_other_camera_or_destination_never_reuses_head_boundary(tmp_path):
    obs=scene(tmp_path)
    for change in (dict(observation_id=0),dict(camera='right_camera'),dict(skill='place')):
        assert clipped_source_anchor(obs,dict(action(),**change)) is None


def test_parent_must_be_current_and_component_box_need_not_itself_be_clipped(tmp_path):
    obs=scene(tmp_path);parent=action();parent['target']='whole object'
    contact=dict(action(),bbox=[600,600,800,800],parent_grounding=parent)
    assert clipped_source_anchor(obs,contact)['target']=='whole object'
    parent['observation_id']=0
    assert clipped_source_anchor(obs,contact) is None


def test_inspection_is_once_per_source_and_never_moves_occupied_hand(tmp_path):
    obs=scene(tmp_path);p=RobodawnPlanner(None,'Lift an object')
    obs['sensors']['right']['holding']=True
    inspect=p._source_boundary_inspection(obs,action())
    assert inspect['skill']=='inspect' and inspect['arm']=='left'
    assert p._source_boundary_inspection(obs,action()) is None
    p=RobodawnPlanner(None,'Lift an object');obs['sensors']['left']['holding']=True
    assert p._source_boundary_inspection(obs,action()) is None


def test_successful_source_observation_does_not_advance_or_rewrite_mission(tmp_path):
    obs=scene(tmp_path);p=RobodawnPlanner(None,'Lift an object')
    p.steps=[dict(operation='lift',source='long object',arm='right',location='stay')]
    inspect=p._source_boundary_inspection(obs,action())
    p.feedback(inspect,dict(skill_success=True,sensors=obs['sensors']))
    assert p.index==0 and p.phase=='pick' and p.failures==0
    assert p.steps[0]['source']=='long object' and p.bound_revision is None


def test_second_current_camera_can_recentre_but_total_observation_budget_is_two(tmp_path):
    obs=scene(tmp_path)
    with np.load(tmp_path/'0001_rgbd.npz') as data:
        np.savez_compressed(tmp_path/'0001_right_camera_rgbd.npz',**{k:data[k] for k in data.files})
        np.savez_compressed(tmp_path/'0001_left_camera_rgbd.npz',**{k:data[k] for k in data.files})
    p=RobodawnPlanner(None,'Lift with both hands')
    assert p._source_boundary_inspection(obs,action())['inspection_camera']=='head_camera'
    second=p._source_boundary_inspection(obs,dict(action(),camera='right_camera'))
    assert second['inspection_camera']=='right_camera' and second['arm']=='right'
    assert p._source_boundary_inspection(obs,dict(action(),camera='left_camera')) is None
    assert p.source_inspection_count==2
