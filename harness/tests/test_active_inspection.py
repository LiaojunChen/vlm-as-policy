from types import SimpleNamespace
import numpy as np
import pytest
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat
from active_inspection import inspection_poses,pose_matrix,execute_inspection
from robotwin_harness_v3 import SkillError,validate


def test_camera_target_is_inverted_through_measured_rigid_robot_mount():
    ee=[.25,-.3,.94,1,0,0,0]
    mount=np.eye(4);mount[:3,:3]=euler2mat(.2,1.1,-.5);mount[:3,3]=[.04,0,.06]
    camera=pose_matrix(ee)@mount
    candidates=inspection_poses(ee,camera,.74,[.28,-.16,.76])
    assert len(candidates)==12
    for candidate in candidates:
        actual=pose_matrix(candidate['pose'])@mount
        np.testing.assert_allclose(actual,candidate['camera_matrix'],atol=1e-9)
        np.testing.assert_allclose(-actual[:3,2],[0,0,-1],atol=1e-9)
        np.testing.assert_allclose(actual[:2,3],[.28,-.16])
        assert actual[2,3]>.97


def test_inspection_cannot_move_an_occupied_arm():
    bridge=SimpleNamespace(sensors=lambda:dict(left=dict(holding=True)))
    with pytest.raises(SkillError,match='EMPTY'):execute_inspection(bridge,dict(arm='left'),[])


def test_invalid_or_stale_observation_anchor_is_rejected():
    validate(dict(skill='inspect',arm='right',inspection_anchor=[400,500]))
    with pytest.raises(SkillError):validate(dict(skill='inspect',arm='right',inspection_anchor=[-1,500]))
    with pytest.raises(ValueError,match='outside'):inspection_poses([0,0,1,1,0,0,0],np.eye(4),.74,[2,0,.74])


def test_clipped_source_prioritizes_widest_existing_pose_without_changing_optics():
    pose=[.25,-.3,.94,1,0,0,0]
    regular=inspection_poses(pose,pose_matrix(pose),.74,[.2,-.2,.78])
    wide=inspection_poses(pose,pose_matrix(pose),.74,[.2,-.2,.78],wide_search=True)
    assert len(wide)==len(regular)==12 and wide[0]['camera_height_m']==.32
    assert {x['camera_height_m'] for x in wide}=={x['camera_height_m'] for x in regular}


def test_wrist_anchor_uses_its_own_current_calibration_not_head_depth():
    pose=[.25,-.3,.94,1,0,0,0];valid=np.ones((101,101),bool)
    head=np.broadcast_to([-.2,.1,.75],(101,101,3)).copy()
    wrist=np.broadcast_to([.12,-.1,.77],(101,101,3)).copy()
    robot=SimpleNamespace(right_planner=SimpleNamespace(fast_preflight=False))
    camera=SimpleNamespace(get_model_matrix=lambda:pose_matrix(pose))
    bridge=SimpleNamespace(sensors=lambda:dict(right=dict(holding=False)),
        env=SimpleNamespace(robot=robot,cameras=SimpleNamespace(right_camera=camera)),
        endpose=lambda:dict(right_endpose=pose),table=.74,depth_id=1,valid=valid,cloud=head,
        view_data={'right_camera':dict(valid=valid,cloud=wrist)},plan_pair=lambda *args:True,
        _take=lambda *args:None)
    action=dict(skill='inspect',arm='right',observation_id=1,inspection_anchor=[500,500],inspection_camera='right_camera')
    result=execute_inspection(bridge,action,[])
    np.testing.assert_allclose(result['observed_anchor'],[.12,-.1,.77])
    assert result['anchor_camera']=='right_camera'
    with pytest.raises(SkillError,match='camera'):validate(dict(action,inspection_camera='unknown'))
