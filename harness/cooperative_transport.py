"""Bounded two-arm reachability search using current grounded support geometry.

Search only translations of an ALREADY held support. Move that support once,
retain both grasps, then require another observation before placing the source.
"""
import numpy as np
import math
from transforms3d.quaternions import quat2mat,mat2quat
from robotwin_harness_v3 import SkillError,ee_from_tcp,placement_rotations
from contact_geometry import transformed_contact_tcp


def translation_candidates(preferred):
    preferred=np.asarray(preferred,float)
    candidates=[]
    for dx in (float(preferred[0]),float(preferred[0])-.06,float(preferred[0])+.06,0.):
        for dy in (0.,-.05,-.10,-.15,-.20,-.25,.05):
            delta=np.array([dx,dy,0.])
            if not .025<=np.linalg.norm(delta)<=.32:continue
            if any(np.linalg.norm(delta-old)<.005 for old in candidates):continue
            candidates.append(delta)
    return sorted(candidates,key=lambda delta:float(np.linalg.norm(delta)))


def choose_support_translation(bridge,action):
    from robodawn.episodic_memory import object_key
    reference=action.get('cooperative_destination',{})
    side=action['arm'];source_side=reference.get('arm')
    if source_side not in ('left','right') or source_side==side:raise SkillError('Cooperative transfer needs distinct arms')
    if reference.get('observation_id')!=bridge.depth_id:raise SkillError('Cooperative support search requires current grounding')
    states=bridge.sensors();names=action['support_rendezvous']
    if object_key(reference.get('target',''))!=object_key(names['support']):
        raise SkillError('Cooperative destination changed the original support identity')
    for arm,name in ((side,names['support']),(source_side,names['source'])):
        state=states.get(arm,{})
        if not state.get('holding') or object_key(state.get('remembered_object',''))!=object_key(name):
            raise SkillError('Cooperative support search lost a named contact-verified grasp')
    if reference.get('skill')!='place' or not reference.get('release',True) or reference.get('use_contact_part'):
        raise SkillError('Cooperative support search only supports released transfer')
    info=bridge.held.get(source_side) or {}
    geometry=bridge.geometry(reference)
    if reference.get('facing') or info.get('desired_facing') or geometry.get('support_frame'):
        raise SkillError('Cooperative search must not relax a directed or sloped placement constraint')
    pose=bridge.endpose();support_pose=np.asarray(pose[side+'_endpose']);source_pose=pose[source_side+'_endpose']
    support=np.asarray(geometry['tcp'],float)
    if geometry.get('release_clearance'):support[2]=geometry['release_clearance']['release_height_m']
    offset=info.get('bottom_offset');records=[]
    action['cooperative_search_audit']=dict(destination_geometry=geometry,candidates=records,
                                           source='current_rgbd_two_arm_support_reachability')
    planners=[getattr(bridge.env.robot,arm+'_planner') for arm in (side,source_side)]
    previous=[getattr(planner,'fast_preflight',False) for planner in planners]
    for planner in planners:planner.fast_preflight=True
    try:
        for delta in translation_candidates(action['delta']):
            moved=support_pose.copy();moved[:3]+=delta
            # Preserve a modest gap between the two current robot wrists;
            # this is a coarse rejection, not a claim of full-scene clearance.
            if np.linalg.norm(moved[:3]-np.asarray(source_pose[:3]))<.14:continue
            support_plan=getattr(bridge.env.robot,side+'_plan_path')(moved.tolist())
            record=dict(delta=delta.tolist(),support_reachable=support_plan.get('status')=='Success',source_candidates=0)
            records.append(record)
            if not record['support_reachable']:continue
            for angle,tilt,rotation in placement_rotations(source_pose[3:],reference):
                point=transformed_contact_tcp(support+delta,offset,quat2mat(info['grasp_quat']),rotation,.005) if offset is not None else support+delta+[0,0,info.get('height_under_tcp',.025)+.008]
                # Match the existing release candidate clearance exactly;
                # preserve_tilt/facing restrictions are never relaxed here.
                point=np.asarray(point).copy()
                point[2]+=.5*max(info.get('span',[0]))*abs(math.sin(math.radians(tilt)))
                q=mat2quat(rotation).tolist()
                for hover in (.04,.02):
                    record['source_candidates']+=1
                    if bridge.plan_pair(source_side,ee_from_tcp(point+[0,0,hover],q),ee_from_tcp(point,q)):
                        record['source_reachable']=True
                        commanded=delta*min(1.,.19/float(np.linalg.norm(delta)))
                        staged=bool(np.linalg.norm(delta-commanded)>.005)
                        if staged:
                            intermediate=support_pose.copy();intermediate[:3]+=commanded
                            if getattr(bridge.env.robot,side+'_plan_path')(intermediate.tolist()).get('status')!='Success':continue
                        bridge.plan_cache=[]  # predictions are not future executable trajectories
                        return commanded,dict(source='current_rgbd_two_arm_support_reachability',
                            destination_geometry=geometry,candidates=records,selected_delta=commanded.tolist(),
                            planned_translation=delta.tolist(),staged_support_translation=staged,
                            source_yaw_delta_rad=float(angle),source_tilt_delta_deg=tilt,predicted_hover_m=hover,
                            requires_fresh_destination_after_motion=True)
    finally:
        for planner,value in zip(planners,previous):planner.fast_preflight=value
        bridge.plan_cache=[]
    raise SkillError('No mutually reachable bounded support translation; retain both grasps and replan')
