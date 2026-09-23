"""Observed planar affordances and episode-local rigid support memory.

All geometry comes from model-bound RGB-D, never actor/asset/task metadata.
Memory requires current spatial corroboration and is invalidated on interaction.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree


def box_points(cloud,valid,box,table):
    h,w=valid.shape;b=np.asarray(box,float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1000) or np.any(b[2:]<=b[:2]):return np.empty((0,3))
    x0,y0,x1,y1=np.rint(b*[w-1,h-1,w-1,h-1]/1000).astype(int)
    pts=cloud[y0:y1+1,x0:x1+1][valid[y0:y1+1,x0:x1+1]]
    return pts[np.isfinite(pts).all(1)&(pts[:,2]>table+.004)&(pts[:,2]<table+.38)]


def flat_source(points,table):
    pts=np.asarray(points,float)
    if len(pts)<80 or not np.isfinite(pts).all():return None
    top=float(np.quantile(pts[:,2],.95))
    if not .004<top-table<.025:return None
    _,_,axes=np.linalg.svd(pts-pts.mean(0),full_matrices=False)
    if abs(axes[-1,2])<.97:return None
    local=(pts-pts.mean(0))@axes.T
    sizes=np.quantile(local,.98,axis=0)-np.quantile(local,.02,axis=0)
    if min(sizes[:2])<.035 or max(sizes[:2])>.24 or sizes[2]>.009:return None
    return dict(source='observed_low_planar_object',normal=[0.,0.,1.],
                observed_thickness_upper_m=top-table,face_size_m=sizes[:2].tolist())


def sloped_frame(points,table):
    pts=np.asarray(points,float)
    pts=pts[np.isfinite(pts).all(1)&(pts[:,2]>table+.02)]
    if len(pts)<100:return None
    pts=pts[::max(1,len(pts)//1800)]
    rng=np.random.default_rng(0);best=None
    for _ in range(160):
        a,b,c=pts[rng.choice(len(pts),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-7:continue
        normal/=length
        mask=np.abs((pts-a)@normal)<.0025
        if best is None or mask.sum()>best.sum():best=mask
    if best is None or best.sum()<max(80,len(pts)*.55):return None
    plane=pts[best];centre=plane.mean(0)
    _,_,axes=np.linalg.svd(plane-centre,full_matrices=False)
    normal=axes[-1];normal*=1 if normal[2]>=0 else -1
    if not .15<normal[2]<.92:return None  # only broad upward sloping surfaces
    residual=float(np.quantile(np.abs((plane-centre)@normal),.95))
    if residual>.0025:return None
    uphill=np.array([0.,0.,1.])-normal*normal[2];uphill/=np.linalg.norm(uphill)
    across=np.cross(normal,uphill);basis=np.column_stack([uphill,across,normal])
    local=(plane-centre)@basis
    lo,hi=np.quantile(local[:,:2],[.02,.98],axis=0);sizes=hi-lo
    if np.min(sizes)<.035 or np.max(sizes)>.3:return None
    centre=centre+basis[:,:2]@((lo+hi)*.5)
    grid=np.floor((local[:,:2]-lo)/.004).astype(int)
    coverage=len(np.unique(grid,axis=0))*.004**2/max(float(np.prod(sizes)),1e-9)
    if coverage<.55:return None
    return dict(point=centre.tolist(),normal=normal.tolist(),uphill=uphill.tolist(),
                surface_size_m=sizes.tolist(),plane_residual_m=residual,
                plane_pixels=len(plane),coverage=float(coverage))


def aligned_rotations(info,frame):
    """Match the observed broad object face to a measured sloping support."""
    from transforms3d.quaternions import quat2mat
    yaw=info['object_yaw'];axis=np.array([np.cos(yaw),np.sin(yaw),0.])
    original=np.column_stack([axis,np.cross([0,0,1],axis),[0,0,1]])
    normal=np.asarray(frame['normal']);up=np.asarray(frame['uphill'])
    for sign in (1,-1):
        desired=np.column_stack([sign*up,np.cross(normal,sign*up),normal])
        yield desired@original.T@quat2mat(info['grasp_quat'])


class SurfaceFrameMemory:
    def __init__(self):self.entries={}

    @staticmethod
    def key(target):return ' '.join(target.lower().split())

    def forget(self,target):self.entries.pop(self.key(target),None)

    def remember(self,reference,view,table,observation_id):
        if reference.get('observation_id')!=observation_id:return False
        pts=box_points(view['cloud'],view['valid'],reference['bbox'],table)
        frame=sloped_frame(pts,table)
        if frame is None:return False
        self.entries[self.key(reference['target'])]=dict(frame=frame,points=pts.copy(),observation_id=observation_id)
        return True

    def resolve(self,action,cloud,valid,table):
        entry=self.entries.get(self.key(action['target']))
        if entry is None:return None
        reference=entry['points'];current=box_points(cloud,valid,action['bbox'],table)
        current=current[(current>=reference.min(0)-.006).all(1)&(current<=reference.max(0)+.006).all(1)]
        if len(current)<60:return None
        distance=cKDTree(reference).query(current)[0];matched=current[distance<.004]
        if len(matched)<60 or np.mean(distance<.004)<.9:return None
        _,_,axes=np.linalg.svd(matched-matched.mean(0),full_matrices=False)
        spans=np.ptp((matched-matched.mean(0))@axes.T,axis=0)
        if spans[0]<.04 or spans[1]<.02:return None
        frame=deepcopy(entry['frame'])
        return dict(tcp=frame['point'],top=frame['point'][2],source='current_verified_sloped_support_memory',
                    support_frame=frame,support_frame_memory=dict(reference_observation_id=entry['observation_id'],
                    current_matched_points=len(matched),inlier_fraction=float(np.mean(distance<.004)),
                    median_error_m=float(np.median(distance))))
