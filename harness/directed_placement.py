"""Directed placement axes measured from model-named visible endpoints."""
import math

import numpy as np
from transforms3d.euler import euler2mat

FACING = {'left': (-1., 0.), 'right': (1., 0.),
          'front': (0., -1.), 'back': (0., 1.)}


def endpoint_reference(cloud,valid,box,point,table):
    """Use local valid depth at the selected end, not the whole bbox centroid."""
    h,w=valid.shape;scale=np.array([w-1,h-1])/1000
    pixel=np.rint(np.asarray(point)*scale).astype(int)
    x0,y0,x1,y1=np.rint(np.asarray(box)*np.tile(scale,2)).astype(int)
    yy,xx=np.indices(valid.shape)
    distance=(xx-pixel[0])**2+(yy-pixel[1])**2
    mask=(valid & np.isfinite(cloud).all(2) & (distance<=8**2)
          & (xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
          & (cloud[:,:,2]>table+.004)&(cloud[:,:,2]<table+.3))
    ys,xs=np.where(mask)
    if len(xs)<4:return None
    selected=np.argsort(distance[ys,xs])[:9]
    position=np.median(cloud[ys[selected],xs[selected]],axis=0)
    return dict(point=position.tolist(),pixels=len(selected),source='current_rgbd_named_endpoint_local_patch')


def observed_axis(start, end):
    delta = np.asarray(end, float) - np.asarray(start, float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError('Orientation endpoints need finite current RGB-D')
    length = np.linalg.norm(delta[:2])
    if not .025 <= length <= .35:
        raise ValueError('Orientation endpoints need distinct visible ends 2.5–35 cm apart')
    # Planar facing is about world Z; endpoint heights are not object tilt.
    return np.r_[delta[:2] / length, 0.]


def align_directed_axis(axis, grasp_rotation, current_rotation, facing):
    if facing not in FACING:
        raise ValueError('facing must be left/right/front/back')
    current = current_rotation @ grasp_rotation.T @ np.asarray(axis, float)
    if np.linalg.norm(current[:2]) < .25:
        raise ValueError('Directed end is nearly vertical; restore a planar held pose before placement')
    desired = FACING[facing]
    delta = math.atan2(desired[1], desired[0]) - math.atan2(current[1], current[0])
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return euler2mat(0, 0, delta) @ current_rotation, delta
