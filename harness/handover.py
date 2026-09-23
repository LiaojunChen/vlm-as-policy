"""Contact-before-release handover using RGB-D and anonymous contact equality.

No environment object poses, meshes, asset identities or expert anchors are
queried. Contact tokens are compared only within the current sensor sample;
they are never named, persisted or sent to the planner.
"""
from copy import deepcopy
import numpy as np
from transforms3d.quaternions import quat2mat
from robotwin_harness_v3 import SkillError, tcp_from_ee, component_opening


def observed_receiver_contact(points, selected_point):
    """Fit only the visible free component, without assuming a table base."""
    points=np.asarray(points,float);point=np.asarray(selected_point,float)
    points=points[np.isfinite(points).all(1)]
    if len(points)<12 or point.shape!=(3,) or not np.isfinite(point).all():
        raise SkillError('Handover receiving component is not sufficiently observed')
    nearest=points[np.linalg.norm(points-point,axis=1).argmin()]
    if np.linalg.norm(nearest-point)>.04:
        raise SkillError('Handover contact point is not on the observed free component')
    local=points[np.linalg.norm(points-nearest,axis=1)<.035]
    if len(local)<12:raise SkillError('Handover free component is occluded; present and reobserve')
    centre=np.median(local,axis=0)
    # The source is suspended. Table-to-top height is NOT its body length.
    # Stay inside the observed vertical band instead of inventing an underside.
    low,top=np.quantile(local[:,2],[.05,.95]);centre[2]=np.clip(centre[2]-.004,low,top)
    values,vectors=np.linalg.eigh(np.cov(local[:,:2].T));major=vectors[:,-1]
    return dict(tcp=centre.tolist(),top=float(top),bottom=float(low),
                yaw=float(np.arctan2(major[1],major[0])),tall=bool(top-low>.06),
                source='current_rgbd_free_held_component',pixels=len(local))


def rebase_held(info, donor_pose, receiver_pose):
    """Express the remembered rigid geometry at the new measured grasp."""
    result=deepcopy(info)
    change=quat2mat(donor_pose[3:])@quat2mat(info['grasp_quat']).T
    translation=tcp_from_ee(donor_pose)-tcp_from_ee(receiver_pose)
    for key in ('bottom_offset','working_offset'):
        if key in info:result[key]=(translation+change@np.asarray(info[key])).tolist()
    if 'directed_axis' in info:result['directed_axis']=(change@np.asarray(info['directed_axis'])).tolist()
    result['object_rotation']=(change@np.asarray(info.get('object_rotation',np.eye(3)))).tolist()
    if 'footprint_xy' in info:
        footprint=np.asarray(info['footprint_xy'])
        result['footprint_xy']=(np.c_[footprint,np.zeros(len(footprint))]@change.T)[:,:2].tolist()
    result.update(grasp_quat=list(receiver_pose[3:]),grasp_tcp=tcp_from_ee(receiver_pose).tolist())
    if 'bottom_offset' in result:result['height_under_tcp']=max(.01,-result['bottom_offset'][2])
    return result


def bilateral_tokens(bridge, side):
    """Anonymous bodies touching BOTH fingers with a nonzero impulse."""
    fingers=[joint[0].child_link.name for joint in getattr(bridge.env.robot,side+'_gripper')]
    hits={name:set() for name in fingers}
    robot_entities={link.entity.per_scene_id for arm in ('left','right')
                    for link in getattr(bridge.env.robot,arm+'_entity').get_links()}
    for contact in bridge.env.scene.get_contacts():
        if not any(np.linalg.norm(point.impulse)>1e-7 for point in contact.points):continue
        bodies=contact.bodies
        for index,body in enumerate(bodies):
            if body.entity.name not in hits:continue
            other=bodies[1-index]
            if other.entity.per_scene_id in robot_entities or 'Static' in type(other).__name__:continue
            hits[body.entity.name].add(other.entity.per_scene_id)
    return set.intersection(*hits.values()) if hits else set()


def require_shared_grasp(bridge, donor, receiver):
    states=bridge.sensors()
    if not all(states[side].get('holding') and states[side].get('opposed_contact') is not False
               for side in (donor,receiver)):
        raise SkillError('Handover uncommitted: both hands must have opposed bilateral contact; donor retained')
    if not bilateral_tokens(bridge,donor)&bilateral_tokens(bridge,receiver):
        raise SkillError('Handover uncommitted: hands do not contact the same anonymous body; donor retained')


def execute_handover(bridge, action, geometry, steps):
    donor,receiver=action['donor'],action['arm']
    states=bridge.sensors();info=bridge.held.get(donor)
    if not states[donor].get('holding') or not info or info.get('target')!=action['target']:
        raise SkillError('Handover needs its named source held by the donor')
    if states[receiver].get('holding'):
        raise SkillError('Handover receiver must be empty before acquisition')
    pre,contact,candidates=bridge.choose_grasp({**action,'skill':'grasp_handle'},geometry)
    geometry['candidates']=candidates
    opening=component_opening(geometry)
    take=lambda pose,grip,label:bridge._take(receiver,pose,grip,label,steps)
    try:
        take(bridge.endpose()[receiver+'_endpose'],opening,'handover_open_receiver')
        take(pre,opening,'handover_approach')
        if not bridge.sensors()[donor].get('holding'):
            raise SkillError('Handover donor contact lost during receiver approach')
        take(contact,opening,'handover_contact')
        take(contact,0,'close')
        # Never release the source on the basis of IK, proximity or a closed jaw.
        require_shared_grasp(bridge,donor,receiver)
    except SkillError:
        if bridge.sensors()[donor].get('holding'):
            take(bridge.endpose()[receiver+'_endpose'],1,'open')
        raise
    poses=bridge.endpose()
    receiver_info=rebase_held(info,poses[donor+'_endpose'],poses[receiver+'_endpose'])
    geometry['handover']=dict(phase='shared_contact_verified',donor=donor,receiver=receiver)
    bridge.held[receiver]=receiver_info
    bridge._take(donor,poses[donor+'_endpose'],1,'release',steps)
    bridge.held[donor]=None
    # Separate the empty donor while the receiving hand remains stationary.
    retreat=bridge.endpose()[donor+'_endpose'].copy()
    retreat[0]+=-.08 if donor=='left' else .08
    bridge._take(donor,retreat,1,'handover_retreat',steps)
    if not bridge.sensors()[receiver].get('holding'):
        raise SkillError('Handover receiving grasp lost after donor release')
    geometry['handover']['phase']='receiver_retained_after_release'
