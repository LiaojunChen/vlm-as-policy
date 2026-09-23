"""Observed free support surfaces inside a visually selected container.

Inputs contain no actor IDs or simulator object poses. All geometry is computed
from the same current RGB-D crop available to the policy.
"""
import cv2
import numpy as np


def circular_cavity_support(points, table_height):
    """Centre a genuinely observed open circular vessel, not a single pixel."""
    from component_geometry import circular_rim_footprint
    points=np.asarray(points,float)
    points=points[np.isfinite(points).all(1)]
    footprint=circular_rim_footprint(points)
    if footprint is None:return None
    centre=np.asarray(footprint['base_xy']);radius=footprint['rim_radius_m']
    inside=points[np.linalg.norm(points[:,:2]-centre,axis=1)<radius*.55]
    if len(inside)<20:return None
    floor=float(np.quantile(inside[:,2],.25));lip=float(np.quantile(points[:,2],.95))
    # A solid round surface, a hole without an observed floor, or tall contents
    # must not be interpreted as a free concave support.
    if not table_height-.005<=floor<lip-.018:return None
    if np.quantile(inside[:,2],.9)>lip+.01:return None
    if np.min(np.ptp(inside[:,:2],axis=0))<radius*.5:return None
    return dict(tcp=[*centre.tolist(),floor],top=floor,
                source='visible_rgbd_circular_cavity',rim_height_m=lip,
                rim_radius_m=radius,rim_fit_residual_m=footprint['rim_fit_residual_m'],
                observed_floor_pixels=len(inside))


def release_clearance(points, support, held):
    """Raise a release only when the item fits but opened fingers do not.

    This is a gravity insertion into the same measured clear opening, never a
    change of destination. The release is at most 18 cm above the visible floor.
    """
    free=float(support.get('free_radius_m',0))
    span=np.asarray(held.get('span',[]),float)
    if span.shape!=(2,) or not np.isfinite(span).all():return None
    item_radius=float(np.linalg.norm(span)*.5)+.001
    gripper_radius=.060
    if not (support.get('observed_clear_footprint') or item_radius<free) or not 0<free<gripper_radius:return None
    pts=np.asarray(points,float);centre=np.asarray(support['tcp'])
    local=pts[np.isfinite(pts).all(1)&(np.linalg.norm(pts[:,:2]-centre[:2],axis=1)<gripper_radius)]
    high=local[local[:,2]>centre[2]+.025]
    if len(high)<8:return None
    obstacle_top=float(np.quantile(high[:,2],.98))
    release_z=obstacle_top+.035-float(held.get('height_under_tcp',0))
    rise=release_z-centre[2]
    if not .015<rise<=.18:return None
    return dict(release_height_m=release_z,release_rise_m=rise,
                gripper_obstacle_top_m=obstacle_top,gripper_clearance_radius_m=gripper_radius,
                observed_item_radius_m=item_radius,source='observed_opening_gripper_clearance')


def container_support(points,table_height,resolution=.0025,layout=None,footprint=None):
    points=np.asarray(points,float)
    points=points[np.isfinite(points).all(1)]
    points=points[(points[:,2]>=table_height-.01)&(points[:,2]<table_height+.38)]
    if len(points)<30:
        return None
    raised=points[points[:,2]>table_height+.004]
    if len(raised)<20:
        return None
    lo,hi=np.quantile(raised[:,:2],[.01,.99],axis=0)
    if np.max(hi-lo)>.45 or np.min(hi-lo)<.025:
        return None
    origin=lo-2*resolution
    shape=np.ceil((hi-lo)/resolution).astype(int)+5
    indices=np.rint((points[:,:2]-origin)/resolution).astype(int)
    within=(indices>=0).all(1)&(indices<shape).all(1)
    indices,points=indices[within],points[within]
    hull_indices=np.rint((raised[:,:2]-origin)/resolution).astype(np.int32)
    hull=cv2.convexHull(hull_indices)
    container_mask=np.zeros(tuple(shape[::-1]),np.uint8)
    cv2.fillConvexPoly(container_mask,hull,1)
    top=np.full(tuple(shape[::-1]),-np.inf)
    np.maximum.at(top,(indices[:,1],indices[:,0]),points[:,2])
    observed=np.isfinite(top)&container_mask.astype(bool)
    heights=top[observed]
    if len(heights)<20:
        return None
    lower=heights[heights<=np.quantile(heights,.55)]
    bins,counts=np.unique(np.round(lower/.003),return_counts=True)
    floor=float(bins[counts.argmax()]*.003)
    floor_mask=(observed&(np.abs(top-floor)<.009)).astype(np.uint8)
    # Fill sampling gaps only; tall contents, walls and handles remain excluded.
    kernel=np.ones((3,3),np.uint8)
    free=cv2.morphologyEx(floor_mask,cv2.MORPH_CLOSE,kernel)
    occupied=cv2.dilate((observed&(top>floor+.018)).astype(np.uint8),kernel)
    free &= container_mask
    free &= (1-occupied)
    # The zero border also makes distance-to-boundary meaningful for tight crops.
    free[[0,-1],:]=0;free[:,[0,-1]]=0
    distances=cv2.distanceTransform(free,cv2.DIST_L2,5)*resolution
    if distances.max()<.008:
        return None
    # Directed multi-object layouts reserve parallel lanes and require the
    # entire measured footprint to fit, not merely its centre pixel.
    if layout is not None:
        from directed_placement import FACING
        slot,count,facing=layout['slot'],layout['count'],layout['facing']
        vertices=np.asarray(footprint,float)
        if vertices.ndim!=2 or vertices.shape[1]!=2 or not 3<=len(vertices)<=64 or not np.isfinite(vertices).all():return None
        if not isinstance(slot,int) or not isinstance(count,int) or not 0<=slot<count<=5 or count<2 or facing not in FACING:return None
        radius=np.ceil(np.max(np.abs(vertices),axis=0)/resolution).astype(int)+2
        if np.any(radius>shape):return None
        kernel=np.zeros(tuple((2*radius+1)[::-1]),np.uint8)
        cv2.fillConvexPoly(kernel,np.rint(vertices/resolution+radius).astype(np.int32),1)
        kernel=cv2.dilate(kernel,np.ones((3,3),np.uint8))
        feasible=cv2.erode(free,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        ys,xs=np.where(feasible)
        if not len(xs):return None
        direction=np.asarray(FACING[facing]);normal=np.array([-direction[1],direction[0]])
        # Fixed table-order convention: lower X/Y lanes are filled first.
        if normal[np.argmax(np.abs(normal))]<0:normal=-normal
        hull_world=origin+cv2.convexHull(hull_indices).reshape(-1,2)*resolution
        along=hull_world@normal
        centre_world=(lo+hi)*.5
        desired=centre_world+normal*(float(along.min()+(slot+.5)/count*np.ptp(along))-centre_world@normal)
        candidates=origin+np.c_[xs,ys]*resolution
        best=np.argmin(np.sum((candidates-desired)**2,axis=1))
    else:
        # Prefer central locations among similarly open patches.
        ys,xs=np.where(distances>=distances.max()*.94)
        centre=(shape-1)*.5
        best=np.argmin((xs-centre[0])**2+(ys-centre[1])**2)
    pixel=np.array([xs[best],ys[best]])
    xy=origin+pixel*resolution
    result=dict(tcp=[float(xy[0]),float(xy[1]),floor],top=floor,
                source='visible_rgbd_free_container_floor',
                free_radius_m=float(distances[pixel[1],pixel[0]]),
                observed_floor_pixels=int(floor_mask.sum()))
    if layout is not None:result.update(container_layout=layout,observed_clear_footprint=True,placed_footprint_xy=vertices.tolist())
    return result
