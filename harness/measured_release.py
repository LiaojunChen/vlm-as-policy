"""Separate opening the gripper from residual Cartesian pose correction."""
import numpy as np
from transforms3d.quaternions import quat2mat
from robotwin_harness_v3 import SkillError


def measured_release_pose(endpose,side,planned):
    measured=np.asarray(endpose.get(side+'_endpose'),float)
    if (measured.shape!=(7,) or not np.isfinite(measured).all()
            or np.linalg.norm(measured[3:])<1e-8):
        raise SkillError('No finite measured endpose for stationary release')
    planned=np.asarray(planned,float)
    angle=np.arccos(np.clip((np.trace(quat2mat(planned[3:])@quat2mat(measured[3:]).T)-1)/2,-1,1))
    # Preserve the exact measured quaternion representation, including sign,
    # so _take's established stationary path holds measured arm joints.
    return measured.tolist(),dict(mode='open_at_measured_endpose_after_accepted_lower',
        planned_pose=planned.tolist(),measured_pose=measured.tolist(),
        deferred_position_correction_m=(planned[:3]-measured[:3]).tolist(),
        deferred_rotation_correction_deg=float(np.rad2deg(angle)),
        scope='gripper_open_only_no_new_descent_or_rotation_not_payload_support_proof')
