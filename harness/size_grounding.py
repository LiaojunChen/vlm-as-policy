"""Resolve comparative cube size using only the current calibrated RGB-D frame."""
import re
import cv2
import numpy as np


def refine_block_size(action,cloud,valid,table_height):
    description=action.get('target','').lower()
    size=re.search(r'\b(smallest|small|medium|largest|large|biggest|big)\b',description)
    if not size or not re.search(r'\b(block|cube)\b',description):return action
    if action.get('skill') not in ('pick','grasp_handle','place'):return action
    z=cloud[:,:,2]
    mask=valid&(z>table_height+.009)&(z<table_height+.12)
    mask &= (np.abs(cloud[:,:,0])<.4)&(cloud[:,:,1]>-.35)&(cloud[:,:,1]<.3)
    # Connected surfaces must also be continuous in metric depth, not just pixels.
    mask=mask.copy()
    mask[1:] &= np.linalg.norm(np.diff(cloud,axis=0),axis=2)<.02
    mask[:,1:] &= np.linalg.norm(np.diff(cloud,axis=1),axis=2)<.02
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    candidates=[]
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]<35:continue
        pts=cloud[labels==index]
        top=float(np.quantile(pts[:,2],.95)-table_height)
        (_, _),(sx,sy),_=cv2.minAreaRect(pts[:,:2].astype(np.float32))
        if not (.02<min(sx,sy)<=max(sx,sy)<.12 and .02<top<.105):continue
        if max(sx,sy)/min(sx,sy)>1.65:continue
        if not .6<np.sqrt(sx*sy)/top<1.5:continue
        yy,xx=np.where(labels==index)
        candidates.append(dict(height=top,bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                               point=[float(np.median(xx)),float(np.median(yy))]))
    if len(candidates)<2:return action
    candidates.sort(key=lambda item:item['height'])
    # Decline uncertain rankings rather than silently guessing through occlusion.
    if any(b['height']-a['height']<.006 for a,b in zip(candidates,candidates[1:])):return action
    word=size.group(1)
    if word=='medium':
        if len(candidates)!=3:return action
        selected=candidates[1]
    else:selected=candidates[0 if word in ('small','smallest') else -1]
    height,width=cloud.shape[:2]
    scale=np.array([width-1,height-1])/1000
    return {**action,'model_bbox':action.get('bbox'),'model_point':action.get('point'),
            'bbox':(np.array(selected['bbox'])/np.tile(scale,2)).tolist(),
            'point':(np.array(selected['point'])/scale).tolist(),
            'size_refinement':{'source':'current_rgbd_cube_dimensions','candidates':candidates}}
