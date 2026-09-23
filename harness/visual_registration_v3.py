"""Conservative RGB-D rigid tracking of a just-grasped object; no scene identities."""
import numpy as np
from scipy.spatial import cKDTree

def register_grasp(source,source_rgb,target,target_rgb,translation):
    """Return a measured transform only when colour/geometry agreement is strong."""
    if len(source)<30 or len(target)<30:return None
    source=np.asarray(source)[::max(1,len(source)//1200)]
    source_rgb=np.asarray(source_rgb)[::max(1,len(source_rgb)//1200)]/255.
    target=np.asarray(target)[::max(1,len(target)//2500)]
    target_rgb=np.asarray(target_rgb)[::max(1,len(target_rgb)//2500)]/255.
    # Colour prevents a nearby gray finger from attracting the object's correspondences.
    tree=cKDTree(np.c_[target,target_rgb*.03]);rotation=np.eye(3);offset=np.asarray(translation,float)
    for _ in range(25):
        predicted=source@rotation.T+offset
        distance,idx=tree.query(np.c_[predicted,source_rgb*.03])
        spatial=np.linalg.norm(predicted-target[idx],axis=1)
        good=(spatial<.025)&(np.linalg.norm(source_rgb-target_rgb[idx],axis=1)<.5)
        if good.sum()<30:return None
        threshold=np.quantile(distance[good],.8);good &= distance<=threshold
        a=source[good];b=target[idx[good]];ac=a.mean(0);bc=b.mean(0)
        u,_,vt=np.linalg.svd((a-ac).T@(b-bc));r=vt.T@u.T
        if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
        t=bc-r@ac
        change=np.linalg.norm(t-offset)+np.linalg.norm(r-rotation)*.01
        rotation,offset=r,t
        if change<1e-6:break
    predicted=source@rotation.T+offset
    _,idx=tree.query(np.c_[predicted,source_rgb*.03])
    error=np.linalg.norm(predicted-target[idx],axis=1)
    inliers=(error<.006)&(np.linalg.norm(source_rgb-target_rgb[idx],axis=1)<.35)
    angle=np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))
    # Reject ambiguous/occluded fits instead of manufacturing a precise pose.
    if inliers.mean()<.5 or np.median(error)>.005 or angle>np.pi/3:return None
    return {'rotation':rotation,'translation':offset,'inlier_fraction':float(inliers.mean()),'median_error_m':float(np.median(error))}
