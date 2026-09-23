"""A measured free release column into a container below the table.

Does not command the hand down to an unreachable floor, infer hidden walls,
or turn an unobserved hole into a support. The caller supplies a current model
container binding; every accepted free grid cell has current RGB-D evidence.
"""
import cv2
import numpy as np


def below_table_portal(cloud,valid,box,table,held_span=(),resolution=.004):
    h,w=valid.shape;b=np.asarray(box,float)
    span=np.asarray(held_span,float)
    if (b.shape!=(4,) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1000)
            or np.any(b[2:]<=b[:2]) or span.shape!=(2,) or not np.isfinite(span).all()):return None
    if np.any(span<=0) or np.max(span)>.28:return None
    x0,y0,x1,y1=np.rint(b*[w-1,h-1,w-1,h-1]/1000).astype(int)
    selected=np.zeros((h,w),bool);selected[y0:y1+1,x0:x1+1]=True
    selected &= valid & np.isfinite(cloud).all(2)
    pts=cloud[selected]
    pts=pts[(pts[:,2]>table-.8)&(pts[:,2]<table+.3)]
    if len(pts)<100:return None
    low=pts[pts[:,2]<table-.08]
    if len(low)<100:return None
    lo,hi=np.quantile(low[:,:2],[.01,.99],axis=0)
    if np.max(hi-lo)>.65 or np.min(hi-lo)<.05:return None
    shape=np.ceil((hi-lo)/resolution).astype(int)+5;origin=lo-2*resolution
    # Include ALL current scene points in this XY region, not just the model
    # crop: a table edge/obstacle above a visible low surface blocks release.
    scene=cloud[valid & np.isfinite(cloud).all(2)]
    indices=np.rint((scene[:,:2]-origin)/resolution).astype(int)
    within=(indices>=0).all(1)&(indices<shape).all(1)
    indices,scene=indices[within],scene[within]
    top=np.full(tuple(shape[::-1]),-np.inf)
    np.maximum.at(top,(indices[:,1],indices[:,0]),scene[:,2])
    low_indices=np.rint((low[:,:2]-origin)/resolution).astype(int)
    within=(low_indices>=0).all(1)&(low_indices<shape).all(1)
    low_indices=low_indices[within]
    observed=np.zeros_like(top,dtype=np.uint8);observed[low_indices[:,1],low_indices[:,0]]=1
    free=observed.astype(bool)&np.isfinite(top)&(top<table-.06)
    # No convex hull completion or morphological filling of hidden space.
    free[[0,-1],:]=False;free[:,[0,-1]]=False
    distance=cv2.distanceTransform(free.astype(np.uint8),cv2.DIST_L2,5)*resolution
    radius=float(np.linalg.norm(span)*.5+.012)
    if distance.max()<radius:return None
    ys,xs=np.where(distance>=radius)
    xy=origin+np.c_[xs,ys]*resolution
    # Every candidate clears the entire measured payload plus margin. Prefer
    # reachable-side columns instead of unnecessarily aiming at the deepest
    # centre of the opening, which can lie outside either arm's reach.
    chosen=int(np.argmin(np.linalg.norm(xy,axis=1)));centre=xy[chosen]
    local=scene[np.linalg.norm(scene[:,:2]-centre,axis=1)<radius]
    if len(local)<30:return None
    release_z=table+.06
    if np.max(local[:,2])>=release_z-.025:return None
    return dict(tcp=[*centre.tolist(),release_z],top=release_z,
                source='current_rgbd_below_table_release_column',
                release_portal=dict(observed_low_points=len(low),clear_radius_m=float(distance[ys[chosen],xs[chosen]]),
                                    required_radius_m=radius,current_column_top_m=float(np.max(local[:,2])),
                                    release_height_m=release_z,no_hidden_cells_filled=True))
