"""Current geometric end contacts for model-requested bimanual straight bars.

Only an explicit left-end/right-end request on a single model-bound, measured
slender horizontal component qualifies. Object identity is still visual-model
authored. Other shapes and insufficiently visible endpoints abstain.
"""
import re
import cv2
import numpy as np
from robodawn.camera_views import in_camera,evidence_path
from robodawn.grounding_evidence import current_contact_mask


def end_contact(observation,parent,target,side,other_hand=None):
    if side not in ('left','right') or not re.search(r'\b'+side+r'\s+(?:end|tip)\b',target,re.I):return None
    if (parent.get('observation_id')!=observation.get('observation_id')
            or not observation.get('image_paths')):return None
    camera=parent.get('camera','head_camera');view=in_camera(observation,camera)
    mask=current_contact_mask(view)
    if mask is None:return None
    box=np.asarray(parent.get('bbox'),float)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>1000) or np.any(box[2:]<=box[:2]):return None
    path=evidence_path(view['image_paths'][0],observation['observation_id'],camera)
    with np.load(path,allow_pickle=False) as data:cloud=data['world_xyz']
    h,w=mask.shape
    x0,y0,x1,y1=np.rint(box*[w-1,h-1,w-1,h-1]/1000).astype(int)
    region=np.zeros_like(mask);region[y0:y1+1,x0:x1+1]=True
    count,labels,stats,_=cv2.connectedComponentsWithStats((region&mask).astype(np.uint8),8)
    if count<2:return None
    label=1+int(stats[1:,cv2.CC_STAT_AREA].argmax())
    # Several unrelated regions cannot jointly invent one long object.
    if stats[label,cv2.CC_STAT_AREA]<80 or stats[label,cv2.CC_STAT_AREA]<.7*np.sum(stats[1:,cv2.CC_STAT_AREA]):return None
    ys,xs=np.where(labels==label);points=cloud[ys,xs];centre=np.median(points,axis=0)
    values,vectors=np.linalg.eigh(np.cov(points[:,:2].T));axis=vectors[:,-1]
    if values[-1]<10*max(values[0],1e-10):return None
    if axis[0]<0:axis=-axis
    lateral=np.array([-axis[1],axis[0]])
    along=(points[:,:2]-centre[:2])@axis;across=(points[:,:2]-centre[:2])@lateral
    lo,hi=np.quantile(along,[.02,.98]);length=float(hi-lo)
    width=float(np.ptp(np.quantile(across,[.02,.98])))
    # With the opposite hand already retaining the object, its occluded end
    # need not be reconstructed to grasp this separately visible terminal.
    minimum_length=.075 if other_hand is not None else .14
    if not (minimum_length<length<.45 and .008<width<.05 and np.ptp(np.quantile(points[:,2],[.02,.98]))<.065):return None
    if length*axis[0]<.055:return None  # world left/right is ambiguous for near fore-aft rods
    requested=along<lo+length*.18 if side=='left' else along>hi-length*.18
    if requested.sum()<20:return None
    # A clipped requested terminal is not an observed end. Occlusion of the
    # opposite end by its verified holder is allowed, but never extrapolated.
    if np.any((xs[requested]<=2)|(ys[requested]<=2)|(xs[requested]>=w-3)|(ys[requested]>=h-3)):return None
    desired=lo+min(.03,length*.15) if side=='left' else hi-min(.03,length*.15)
    contact_band=np.abs(along-desired)<.018
    if contact_band.sum()<20:return None
    patch=points[contact_band];px=xs[contact_band];py=ys[contact_band]
    wanted=np.median(patch,axis=0)
    # Choose an actual depth-supported pixel, not the fitted line in free space.
    chosen=int(np.argmin(np.linalg.norm(patch-wanted,axis=1)))
    if other_hand is not None:
        from robotwin_harness_v3 import tcp_from_ee
        pose=np.asarray(other_hand,float)
        if pose.shape!=(7,) or not np.isfinite(pose).all():return None
        if np.linalg.norm(patch[chosen]-tcp_from_ee(pose))<.075:return None
    low=np.array([px.min(),py.min()]);high=np.array([px.max(),py.max()])
    if np.any(high<=low):return None
    return dict(bbox=(np.r_[low,high]/[w-1,h-1,w-1,h-1]*1000).tolist(),
                point=(np.array([px[chosen],py[chosen]])/[w-1,h-1]*1000).tolist(),
                camera=camera,observation_id=observation['observation_id'],
                source='current_model_bound_slender_bar_end',world_left_to_right_axis=axis.tolist(),
                observed_length_m=length,observed_width_m=width,contact_pixels=len(patch),
                selected_world_point=patch[chosen].tolist(),requested_end=side)
