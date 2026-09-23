"""Empty-arm active perception with an EXISTING calibrated wrist camera.

Uses only robot proprioception, its own camera extrinsics and measured table
height. A requested anchor is an observed head-image pixel, never an actor pose.
The standard cameras, optics, model and task checker are untouched.
"""
import numpy as np
from transforms3d.quaternions import quat2mat,mat2quat
from transforms3d.euler import euler2mat
from robotwin_harness_v3 import SkillError,tcp_from_ee


def pose_matrix(pose):
    matrix=np.eye(4);matrix[:3,:3]=quat2mat(pose[3:]);matrix[:3,3]=pose[:3]
    return matrix


def inspection_poses(ee_pose,camera_matrix,table,anchor=None,wide_search=False):
    """Look down on the local workspace; invert measured camera-to-EE mount."""
    ee_pose=np.asarray(ee_pose,float);camera_matrix=np.asarray(camera_matrix,float)
    if ee_pose.shape!=(7,) or camera_matrix.shape!=(4,4) or not np.isfinite(ee_pose).all() or not np.isfinite(camera_matrix).all():
        raise ValueError('Invalid robot/camera calibration')
    camera_in_ee=np.linalg.inv(pose_matrix(ee_pose))@camera_matrix
    if anchor is None:
        tcp=tcp_from_ee(ee_pose);xy=tcp[:2].copy();xy[1]+= .12
    else:
        anchor=np.asarray(anchor,float)
        if anchor.shape!=(3,) or not np.isfinite(anchor).all():raise ValueError('Invalid observed inspection anchor')
        xy=anchor[:2]
    if not (-.45<xy[0]<.45 and -.4<xy[1]<.3):raise ValueError('Inspection region is outside the calibrated local workspace')
    candidates=[]
    for height in ((.32,.28,.24) if wide_search else (.28,.24,.32)):
        for yaw in (0,np.pi/2,-np.pi/2,np.pi):
            camera=np.eye(4)
            # OpenGL camera looks along -Z. +Z upwards gives a downward view.
            camera[:3,:3]=euler2mat(0,0,yaw);camera[:3,3]=[*xy,table+height]
            target=camera@np.linalg.inv(camera_in_ee)
            pose=[*target[:3,3].tolist(),*mat2quat(target[:3,:3]).tolist()]
            candidates.append(dict(pose=pose,camera_matrix=camera.tolist(),camera_height_m=height))
    return candidates


def execute_inspection(bridge,action,steps):
    side=action['arm']
    if bridge.sensors()[side].get('holding'):
        raise SkillError('Active inspection requires an EMPTY arm; retain all held objects')
    camera_name=side+'_camera';camera=getattr(bridge.env.cameras,camera_name)
    pose=bridge.endpose()[side+'_endpose'];anchor=None
    if 'inspection_anchor' in action:
        if action.get('observation_id')!=bridge.depth_id:raise SkillError('Stale inspection anchor')
        from robodawn.camera_views import CAMERAS
        anchor_camera=action.get('inspection_camera','head_camera')
        if anchor_camera not in CAMERAS:raise SkillError('Invalid inspection anchor camera')
        if anchor_camera=='head_camera':valid,cloud=bridge.valid,bridge.cloud
        else:
            view=getattr(bridge,'view_data',{}).get(anchor_camera)
            if view is None:raise SkillError('No current calibrated inspection anchor camera')
            valid,cloud=view['valid'],view['cloud']
        height,width=valid.shape
        x,y=np.rint(np.asarray(action['inspection_anchor'])*[width-1,height-1]/1000).astype(int)
        if not valid[y,x]:raise SkillError('Inspection anchor must have current non-robot depth')
        anchor=cloud[y,x].copy()
    try:candidates=inspection_poses(pose,camera.get_model_matrix(),bridge.table,anchor,
                                   wide_search=bool(action.get('source_boundary_inspection')))
    except ValueError as exc:raise SkillError(str(exc)) from exc
    planner=getattr(bridge.env.robot,side+'_planner');planner.fast_preflight=True
    chosen=None;attempts=[]
    try:
        for candidate in candidates:
            target=candidate['pose']
            rise=list(pose);rise[2]=max(pose[2],target[2])+.03
            if rise[2]>bridge.table+.5:continue
            reachable=bridge.plan_pair(side,rise,target)
            attempts.append(dict(camera_height_m=candidate['camera_height_m'],reachable=bool(reachable)))
            if reachable:chosen=(rise,target,candidate);break
    finally:planner.fast_preflight=False
    if chosen is None:raise SkillError('No reachable empty-arm observation pose')
    rise,target,candidate=chosen
    bridge._take(side,rise,1,'inspection_clearance',steps)
    bridge._take(side,target,1,'inspection_view',steps)
    return dict(source='existing_calibrated_wrist_camera_active_view',camera=camera_name,
                camera_height_m=candidate['camera_height_m'],observed_anchor=None if anchor is None else anchor.tolist(),
                anchor_camera=action.get('inspection_camera','head_camera'),
                pose_candidates=attempts)
