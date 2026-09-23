"""Observed working-part references for retained rigid tools."""
import numpy as np
import cv2


def working_part_reference(cloud, valid, box, point, table):
    """Locate a compact part in the model's current bbox, never an actor pose.

    For tools resting on the table, the support plane estimates the hidden
    underside. Elevated parts use their visible lower surface instead.
    """
    h,w=valid.shape;scale=np.array([w-1,h-1])/1000
    x0,y0,x1,y1=np.rint(np.asarray(box)*np.tile(scale,2)).astype(int)
    uv=np.rint(np.asarray(point)*scale).astype(int)
    mask=np.zeros((h,w),bool);mask[y0:y1+1,x0:x1+1]=True
    mask &= valid & (cloud[:,:,2]>table+.004) & (cloud[:,:,2]<table+.3)
    count,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=8]
    if not candidates:return None
    selected=min(candidates,key=lambda i:np.linalg.norm(centres[i]-uv))
    pts=cloud[labels==selected]
    if np.max(np.ptp(pts[:,:2],axis=0))>.15:return None
    xy=cv2.minAreaRect(pts[:,:2].astype(np.float32))[0]
    low=float(np.quantile(pts[:,2],.05))
    z=table if low-table<.035 else low
    return dict(point=[float(xy[0]),float(xy[1]),z],
                source='current_rgbd_working_part',pixels=len(pts),
                underside_source='observed_support_plane' if z==table else 'visible_lower_surface')


def transformed_contact_tcp(support, offset, grasp_rotation, current_rotation, clearance):
    return np.asarray(support)-current_rotation@grasp_rotation.T@np.asarray(offset)+[0,0,clearance]


def retained_working_offset(point,current_tcp,grasp_rotation,current_rotation):
    """Register a newly seen working part in the existing rigid grasp frame."""
    point=np.asarray(point,float);tcp=np.asarray(current_tcp,float)
    if point.shape!=(3,) or tcp.shape!=(3,) or not np.isfinite([point,tcp]).all():
        raise ValueError('Invalid observed working-part point')
    if np.linalg.norm(point-tcp)>.35:raise ValueError('Working part too far from the retained grasp')
    return np.asarray(grasp_rotation)@np.asarray(current_rotation).T@(point-tcp)


def measured_bar_contact_stop(geometry, actual_tcp, sensing, motion, side):
    """Allow one close at measured early contact, never force deeper motion.

    Only a high-confidence observed bar, successful IK, measured finger
    contact and a small above-target tracking stop qualify. The caller must
    still verify an opposed grasp and its retention after lifting.
    """
    if not geometry.get('handle_contact_selection'):return None
    if motion.get('name') not in ('contact','contact_refine'):return None
    if motion.get('planner_status',{}).get(side)!='Success':return None
    error=motion.get('position_error_m',float('inf'))
    if not .035<error<=.06 or sensing.get('contact_fingers',0)<1:return None
    point=np.asarray(actual_tcp,float);wanted=np.asarray(geometry.get('tcp'),float)
    if point.shape!=(3,) or wanted.shape!=(3,) or not np.isfinite([point,wanted]).all():return None
    offset=point-wanted
    if np.linalg.norm(offset[:2])>.035 or not .015<=offset[2]<=.06:return None
    return dict(source='measured_early_contact_at_observed_bar',tracking_error_m=float(error),
                requested_tcp=wanted.tolist(),measured_tcp=point.tolist(),
                contact_fingers=int(sensing['contact_fingers']))
