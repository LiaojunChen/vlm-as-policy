"""Observed free-space selection for a bounded, task-independent regrasp."""
import cv2
import numpy as np
import re


def explicit_free_table(target):
    return bool(re.search(r'\b(empty|clear|free|unoccupied)\b',str(target),re.I)
                and re.search(r'\b(table|tabletop|workspace)\b',str(target),re.I))


def shared_table_support(cloud, valid, table, span, robot_origins, resolution=.005, preferred_xy=None):
    xyz=np.asarray(cloud,float);valid=np.asarray(valid,bool)&np.isfinite(xyz).all(2)
    footprint=np.asarray(span,float)
    if footprint.shape!=(2,) or not np.isfinite(footprint).all():return None
    radius=max(.06,float(np.linalg.norm(footprint)/2+.012))
    if radius>.16:return None
    origin=np.array([-.5,-.4]);shape=np.array([201,171])
    points=xyz[valid];ij=np.floor((points[:,:2]-origin)/resolution).astype(int)
    within=(ij>=0).all(1)&(ij<shape).all(1)
    points,ij=points[within],ij[within]
    if not len(points):return None
    floor=np.zeros(tuple(shape[::-1]),np.uint8);occupied=floor.copy()
    low=np.abs(points[:,2]-table)<.004
    floor[ij[low,1],ij[low,0]]=1
    high=points[:,2]>table+.008
    occupied[ij[high,1],ij[high,0]]=1
    # Only close sub-centimetre RGB-D sampling gaps; unknown regions stay out.
    floor=cv2.morphologyEx(floor,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    floor[occupied.astype(bool)]=0
    clearance=cv2.distanceTransform(floor,cv2.DIST_L2,5)*resolution
    yy,xx=np.indices(floor.shape)
    xy=np.stack([xx,yy],axis=-1)*resolution+origin+resolution/2
    nominal=(np.mean(np.asarray(robot_origins,float)[:,:2],axis=0)+[0,.20]
             if preferred_xy is None else np.asarray(preferred_xy,float))
    if nominal.shape!=(2,) or not np.isfinite(nominal).all():return None
    shared=(np.abs(xy[:,:,0]-nominal[0])<=.09)&(np.abs(xy[:,:,1]-nominal[1])<=.12)
    possible=shared&(clearance>radius)
    if not possible.any():return None
    cost=np.linalg.norm(xy-nominal,axis=2)-.1*clearance
    cost[~possible]=np.inf
    row,col=np.unravel_index(cost.argmin(),cost.shape);centre=xy[row,col]
    region=valid&(np.linalg.norm(xyz[:,:,:2]-centre,axis=2)<radius)&(np.abs(xyz[:,:,2]-table)<.004)
    ys,xs=np.where(region)
    if len(xs)<20:return None
    nearest=np.argmin(np.linalg.norm(xyz[ys,xs,:2]-centre,axis=1))
    height,width=valid.shape;scale=np.array([width-1,height-1])/1000
    image_region=dict(bbox=(np.array([xs.min(),ys.min(),xs.max(),ys.max()])/np.tile(scale,2)).tolist(),
                      point=(np.array([xs[nearest],ys[nearest]])/scale).tolist())
    return dict(tcp=[*centre.tolist(),float(table)],top=float(table),
                source='current_rgbd_shared_free_table' if preferred_xy is None else 'current_rgbd_requested_free_table',
                free_radius_m=float(clearance[row,col]),required_radius_m=radius,
                pending_release_region=image_region)
