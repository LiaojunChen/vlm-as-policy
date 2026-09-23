"""Conservative current-pixel recovery from episode-local observed appearance.

Historical model boxes select a REFERENCE appearance, never current geometry.
Only a dominant visible component inside a fresh model box can supply a point.
No task names, object segmentation identities, actor state or external inputs.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from robodawn.episodic_memory import object_key


def box_mask(shape,box):
    height,width=shape[:2]
    values=np.asarray(box,float)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1000):return None
    low,high=np.rint(values.reshape(2,2)*[width-1,height-1]/1000).astype(int)
    if np.any(high<=low):return None
    mask=np.zeros((height,width),bool);mask[low[1]:high[1]+1,low[0]:high[0]+1]=True
    return mask


def hue_distance(hue,reference):
    delta=np.abs(np.asarray(hue,float)-reference)
    return np.minimum(delta,180-delta)


def observed_signature(rgb,valid,box):
    region=box_mask(rgb.shape,box)
    if region is None or valid.shape!=region.shape:return None
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    region &= valid
    coloured=region&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=25)
    hues=hsv[:,:,0][coloured]
    if len(hues)<max(30,.12*region.sum()):return None
    histogram=np.bincount(hues,minlength=180)
    support=sum(np.roll(histogram,shift) for shift in range(-10,11))
    mode=int(support.argmax());inlier=hue_distance(hues,mode)<=10
    if inlier.mean()<.6:return None
    offsets=(hues[inlier].astype(float)-mode+90)%180-90
    centre=float((mode+np.median(offsets))%180)
    return dict(hue=centre,tolerance=12.,reference_pixels=int(inlier.sum()))


def current_component(rgb,valid,box,signature,world_xyz=None,reference_points=None):
    region=box_mask(rgb.shape,box)
    if region is None or valid.shape!=region.shape:return None
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    match=(region&valid&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=25)
           &(hue_distance(hsv[:,:,0],signature['hue'])<=signature['tolerance']))
    geometry_verified=False
    if world_xyz is not None and reference_points is not None:
        if world_xyz.shape!=(*valid.shape,3) or len(reference_points)<30:return None
        points=world_xyz[match]
        if not len(points) or not np.isfinite(points).all():return None
        distance=cKDTree(reference_points).query(points)[0]
        # Current points must corroborate an actually observed reference
        # surface. A moved object or a same-colour distractor is not silently
        # assigned stale coordinates. No points are synthesized from memory.
        corroborated=distance<=.006
        if corroborated.mean()<.85:return None
        match[match]=corroborated;geometry_verified=True
    count,labels,stats,centres=cv2.connectedComponentsWithStats(match.astype(np.uint8),8)
    if count<=1:return None
    best=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());size=int(stats[best,cv2.CC_STAT_AREA])
    # Small fragments or several similarly plausible regions do not authorize
    # a contact. Do not fill holes or hallucinate hidden connecting surfaces.
    if size<16 or (not geometry_verified and size<.7*match.sum()):return None
    ys,xs=np.where(labels==best);pixels=np.c_[xs,ys]
    chosen=pixels[np.linalg.norm(pixels-centres[best],axis=1).argmin()]
    height,width=valid.shape
    return dict(point=(chosen/[width-1,height-1]*1000).tolist(),pixels=size,
                component_fraction=float(size/match.sum()),source='current_visible_component_matching_episode_appearance')


def repair_robot_point(obs,memory,rgb,current_box,target):
    """Read only observed RGB-D from this episode's verified identity memory."""
    if memory is None:return None
    entry=memory.objects.get(object_key(target),{})
    visual=entry.get('last_visual',{});reference=visual.get('observation_id');current=obs.get('observation_id')
    if (not isinstance(reference,int) or not isinstance(current,int) or reference>=current
            or visual.get('component_only',True)
            or entry.get('identity_verification_observation_id')!=reference):return None
    if visual.get('camera','head_camera')!='head_camera':return None
    path=Path(obs['image_paths'][0]);directory=path.parent
    if path.name!=f'{current:04d}_head_camera.png':return None
    reference_image=directory/f'{reference:04d}_head_camera.png'
    reference_depth=directory/f'{reference:04d}_rgbd.npz'
    current_depth=directory/f'{current:04d}_rgbd.npz'
    if not all(p.is_file() for p in (reference_image,reference_depth,current_depth)):return None
    with np.load(reference_depth,allow_pickle=False) as data:
        reference_valid=data['valid'].copy();reference_xyz=data['world_xyz'].copy() if 'world_xyz' in data else None
    with np.load(current_depth,allow_pickle=False) as data:
        current_valid=data['valid'].copy();current_xyz=data['world_xyz'].copy() if 'world_xyz' in data else None
    reference_rgb=np.asarray(Image.open(reference_image).convert('RGB'))
    signature=observed_signature(reference_rgb,reference_valid,visual['bbox'])
    if signature is None:return None
    reference_points=None
    if reference_xyz is not None:
        hsv=cv2.cvtColor(reference_rgb,cv2.COLOR_RGB2HSV)
        reference_mask=(box_mask(reference_rgb.shape,visual['bbox'])&reference_valid
                        &(hsv[:,:,1]>=45)&(hsv[:,:,2]>=25)
                        &(hue_distance(hsv[:,:,0],signature['hue'])<=signature['tolerance']))
        reference_points=reference_xyz[reference_mask]
        reference_points=reference_points[np.isfinite(reference_points).all(1)]
    result=current_component(rgb,current_valid,current_box,signature,current_xyz,reference_points)
    if result is None:return None
    result['reference_observation_id']=reference
    result['current_observation_id']=current
    result['reference_geometry_checked']=current_xyz is not None and reference_points is not None
    return result
