"""Local grasp geometry from a model-selected, observed RGB-D component only."""
import math
import numpy as np
import cv2


def complete_observed_circular_component(cloud, valid, table_height, seed_mask):
    """Grow a model-selected lip to its connected CURRENT-depth vessel only.

    A tight contact box need not contain the whole circular footprint. Growth
    is accepted only for a well-supported circular lip; no hidden points or
    historical object coordinates are synthesized.
    """
    xyz=np.asarray(cloud);mask=np.asarray(valid,bool)&np.isfinite(xyz).all(2)
    mask &= (xyz[:,:,2]>table_height+.004)&(xyz[:,:,2]<table_height+.38)
    mask &= (np.abs(xyz[:,:,0])<.45)&(xyz[:,:,1]>-.4)&(xyz[:,:,1]<.35)
    mask[1:] &= np.linalg.norm(np.diff(xyz,axis=0),axis=2)<=.025
    mask[:,1:] &= np.linalg.norm(np.diff(xyz,axis=1),axis=2)<=.025
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    seed=np.asarray(seed_mask,bool)&mask
    ids,overlap=np.unique(labels[seed],return_counts=True)
    if not len(ids):return None
    label=int(ids[np.argmax(overlap)])
    if label==0 or overlap.max()<12:return None
    grown=labels==label;points=xyz[grown]
    if len(points)<=int(seed.sum()) or np.max(np.ptp(points[:,:2],axis=0))>.28:return None
    fit=circular_rim_footprint(points)
    if fit is None:return None
    # A connected non-vessel extension is not part of a circular bowl body.
    radial=np.linalg.norm(points[:,:2]-np.asarray(fit['base_xy']),axis=1)
    if np.mean(radial<=fit['rim_radius_m']+.015)<.95:return None
    return grown


def circular_rim_footprint(points):
    """Return a centre only for a well-covered circular lip in observed depth."""
    pts=np.asarray(points,dtype=float)
    if len(pts)<30:return None
    band=pts[pts[:,2]>=np.quantile(pts[:,2],.95)-.009,:2]
    if len(band)<20:return None
    fit=np.linalg.lstsq(np.c_[2*band,np.ones(len(band))],np.sum(band**2,axis=1),rcond=None)[0]
    radius2=fit[2]+np.sum(fit[:2]**2)
    if not .035**2<radius2<.12**2:return None
    radius=float(np.sqrt(radius2));offset=band-fit[:2]
    residual=float(np.median(np.abs(np.linalg.norm(offset,axis=1)-radius)))
    angles=np.sort(np.arctan2(offset[:,1],offset[:,0]))
    coverage=float(2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi])))
    if residual>.003 or coverage<math.radians(200):return None
    confidence={}
    if coverage<1.5*np.pi:
        # A clipped vessel can expose >half its lip without exposing 270 deg.
        # Accept its fitted centre only when independent point subsets agree;
        # contact points themselves must still come from observed samples.
        if len(band)<80:return None
        error=np.abs(np.linalg.norm(offset,axis=1)-radius)
        if np.mean(error<.003)<.9:return None
        estimates=[]
        for i in range(5):
            sample=band[i::5]
            centre=sample.mean(0);local=sample-centre
            solution=np.linalg.lstsq(np.c_[2*local,np.ones(len(local))],np.sum(local**2,axis=1),rcond=None)[0]
            radius_squared=solution[2]+np.sum(solution[:2]**2)
            if radius_squared<=0:return None
            estimates.append(np.r_[solution[:2]+centre,np.sqrt(radius_squared)])
        estimates=np.asarray(estimates)
        centre_spread=float(np.max(np.linalg.norm(estimates[:,:2]-fit[:2],axis=1)))
        radius_spread=float(np.max(np.abs(estimates[:,2]-radius)))
        if centre_spread>.002 or radius_spread>.002:return None
        confidence=dict(partial_rim_fit=True,rim_centre_subsample_spread_m=centre_spread,
                        rim_radius_subsample_spread_m=radius_spread,rim_inlier_fraction=float(np.mean(error<.003)))
    return dict(base_xy=fit[:2].tolist(),base_source='visible_rgbd_circular_rim',**confidence,
                rim_radius_m=radius,rim_fit_residual_m=residual,rim_coverage_rad=coverage)


def near_side_rim_anchor(points, approach_origin, attempt=0):
    """Select an actually observed circular lip point on the approach side.

    The model still selects the object and rim affordance. Contact search uses
    observed geometry and the robot's own initial pose, never asset grasp tags.
    Failed attempts explore neighbouring lip sectors rather than arbitrary yaw.
    """
    pts=np.asarray(points,float)
    footprint=circular_rim_footprint(pts)
    if footprint is None:return None
    centre=np.asarray(footprint['base_xy']);radius=footprint['rim_radius_m']
    toward=np.asarray(approach_origin,float)[:2]-centre
    length=np.linalg.norm(toward)
    if not np.isfinite(length) or length<radius*1.2:return None
    angle=math.atan2(toward[1],toward[0])+math.radians((0,30,-30,60,-60)[int(attempt)%5])
    direction=np.array([math.cos(angle),math.sin(angle)])
    band=pts[pts[:,2]>=np.quantile(pts[:,2],.95)-.009]
    radial=band[:,:2]-centre;distance=np.linalg.norm(radial,axis=1)
    lip=band[(np.abs(distance-radius)<.012)&((radial@direction)/np.maximum(distance,1e-9)>.94)]
    if len(lip)<8:return None
    wanted=centre+radius*direction
    selected=lip[np.linalg.norm(lip[:,:2]-wanted,axis=1).argmin()]
    return dict(point=selected.tolist(),approach_origin=np.asarray(approach_origin).tolist(),
                sector_attempt=int(attempt),source='observed_circular_lip_approach_side')


def elevated_handle_anchor(points, selected_point, parent_centre):
    """Resolve an opening/contents point to a clearly observed raised grip bar.

    Only the model's handle component crop is searched. A compact, broad upper
    body or an unsupported extrapolation cannot qualify as a thin crossbar.
    Low protruding handles and already-on-bar points keep their normal path.
    """
    pts=np.asarray(points,float);point=np.asarray(selected_point,float)
    parent=np.asarray(parent_centre,float)
    if pts.ndim!=2 or pts.shape[1]!=3 or point.shape!=(3,) or parent.shape!=(3,):return None
    pts=pts[np.isfinite(pts).all(1)]
    if len(pts)<60 or not np.isfinite([point,parent]).all():return None
    top=float(np.quantile(pts[:,2],.95))
    if top-parent[2]<.035 or top-point[2]<.025:return None
    band=pts[pts[:,2]>=top-.008]
    if len(band)<40:return None
    centre=np.median(band[:,:2],axis=0)
    values,vectors=np.linalg.eigh(np.cov(band[:,:2].T))
    if values[-1]<4*max(values[0],1e-10):return None
    local=(band[:,:2]-centre)@vectors
    lo,hi=np.quantile(local,[.05,.95],axis=0);width,length=hi-lo
    if not (.002<=width<=.035 and .03<=length<=.20):return None
    # Require observed interior span, not just two posts defining a fake bar.
    along=local[:,1]
    counts=np.histogram(along,bins=np.linspace(lo[1],hi[1],9))[0]
    if np.count_nonzero(counts>=3)<7:return None
    middle=band[np.abs(along-(lo[1]+hi[1])*.5)<length*.15]
    if len(middle)<12:return None
    wanted=np.median(middle,axis=0)
    if np.linalg.norm(wanted[:2]-point[:2])>.12:return None
    anchor=middle[np.linalg.norm(middle-wanted,axis=1).argmin()]
    return dict(point=anchor.tolist(),source='current_observed_elevated_handle_bar',
                bar_width_m=float(width),bar_length_m=float(length),
                observed_span_bins=int(np.count_nonzero(counts>=3)),
                model_selected_point=point.tolist())


def component_grasp(points, selected_point, part, table_height, parent_centre=None):
    """Fit the local tangent of a thin horizontal rim/handle, or abstain.

    A rim's contact must be on its upper lip, even when the model's pixel
    projects onto the visible floor. Handles/edges retain the selected height;
    they must not jump to an unrelated higher part of the same connected crop.
    Whole-object dimensions remain the caller's responsibility for placement.
    """
    if part not in ('rim', 'handle', 'edge'):
        return None
    pts = np.asarray(points, dtype=float)
    point = np.asarray(selected_point, dtype=float)
    if len(pts) < 12 or not np.isfinite(point).all():
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 12:
        return None
    if part == 'rim':
        top = np.quantile(pts[:, 2], .95)
        band = pts[pts[:, 2] >= top - .009]
        if len(band) < 8:
            return None
        anchor = band[np.linalg.norm(band[:, :2] - point[:2], axis=1).argmin()]
    else:
        candidates=pts
        if part=='handle' and parent_centre is not None:
            radial=np.median(pts[:,:2],axis=0)-np.asarray(parent_centre)[:2]
            if np.linalg.norm(radial)>.035:
                values,vectors=np.linalg.eigh(np.cov(pts[:,:2].T))
                outward=vectors[:,0]
                if outward@radial<0:outward=-outward
                projection=pts[:,:2]@outward
                protruding=(values[-1]>4*max(values[0],1e-10)
                            and abs(vectors[:,-1]@radial)/np.linalg.norm(radial)>.8)
                # A long handle extending away from its parent is a solid
                # strip, not a loop's outer crossbar. Preserve both visible
                # lateral edges so the gripper centres across the whole strip.
                if not protruding and np.ptp(projection)>.012:
                    outer=pts[projection>=np.quantile(projection,.7)]
                    if len(outer)>=8:candidates=outer
        anchor = candidates[np.linalg.norm(candidates - point, axis=1).argmin()]
        band = candidates[np.abs(candidates[:, 2] - anchor[2]) < .012]
    local = band[np.linalg.norm(band[:, :2] - anchor[:2], axis=1) < .035]
    if len(local) < 8:
        return None
    centre = np.median(local[:, :2], axis=0)
    values, vectors = np.linalg.eigh(np.cov(local[:, :2].T))
    # No well-observed horizontal tangent: keep the original general fit.
    if values[-1] < .006 ** 2 or values[-1] < 2.5 * max(values[0], 1e-10):
        return None
    tangent = vectors[:, -1]
    normal = np.array([-tangent[1], tangent[0]])
    lateral = (local[:, :2] - centre) @ normal
    width = float(np.quantile(lateral, .95) - np.quantile(lateral, .05))
    if width > .035:
        return None
    # Move only across the local strip to centre the jaws; retain the model's
    # position along the component rather than sliding toward the crop centre.
    xy = anchor[:2] + normal * np.median((local[:, :2] - anchor[:2]) @ normal)
    if part=='handle' and parent_centre is not None:
        # A short visible grip bar should be held near its local midpoint,
        # rather than at the corner where a model point can touch two rails.
        xy=np.median(local[:,:2],axis=0)
    z = max(table_height + .012, float(np.median(local[:, 2])) - .007)
    result=dict(tcp=[float(xy[0]), float(xy[1]), z],
                yaw=math.atan2(tangent[1], tangent[0]), tall=False,
                source='visible_rgbd_local_component', component=part,
                component_width_m=width, component_pixels=len(local),
                component_anchor=anchor.tolist())
    if part=='rim':
        footprint=circular_rim_footprint(pts)
        if footprint is not None:result.update(footprint)
    return result
