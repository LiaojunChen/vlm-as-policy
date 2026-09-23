"""Uncertain moving-object search windows, never executable object poses.

A grasp's observed RGB-D extent is carried with measured gripper motion.
Only CURRENT foreground pixels inside that loose extent form a search crop;
the model must still identify the object and ground it in that current image.
"""
from itertools import product
import numpy as np
from transforms3d.quaternions import quat2mat
from robodawn.camera_views import camera_of, evidence_path


def broad_support_patch(cloud, region):
    """Optional current upward planar patch for support SEARCH, not a pose fit.

    Handles/rims can lie inside the same object extent, but are not broad
    support surfaces. Abstain on small/curved/non-planar current evidence.
    """
    points=cloud[region]
    if len(points)<60:return None
    sample=points[::max(1,len(points)//1800)]
    rng=np.random.default_rng(0);best=None
    for _ in range(120):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-7:continue
        normal/=length
        if abs(normal[2])<.65:continue
        inliers=np.abs((sample-a)@normal)<.0025
        if best is None or inliers.sum()>best.sum():best=inliers
    if best is None or best.sum()<max(60,len(sample)*.25):return None
    plane=sample[best];centre=plane.mean(0)
    _,_,axes=np.linalg.svd(plane-centre,full_matrices=False)
    normal=axes[-1];normal*=1 if normal[2]>0 else -1
    spans=np.ptp((plane-centre)@axes.T,axis=0)
    if normal[2]<.65 or spans[0]<.04 or spans[1]<.025 or spans[2]>.006:return None
    selected=region&(np.abs((cloud-centre)@normal)<.003)
    return selected,dict(point=centre.tolist(),normal=normal.tolist(),pixels=int(selected.sum()),
                          visible_span_m=spans[:2].tolist(),source='current_observed_upward_planar_search_patch')


class HeldSearchMemory:
    def __init__(self):self.entries={}

    def feedback(self, observation, action, result):
        from robodawn.episodic_memory import object_key
        sensors=result.get('sensors',{})
        for side in list(self.entries):
            state=sensors.get(side,{})
            if not state.get('holding') or object_key(state.get('remembered_object',''))!=self.entries[side]['target']:
                self.entries.pop(side,None)
        side=action.get('arm');state=sensors.get(side,{})
        if (action.get('skill') not in ('pick','grasp_handle') or not state.get('holding')
                or not observation or action.get('observation_id')!=observation.get('observation_id')
                or object_key(action.get('target',''))!=object_key(state.get('remembered_object',''))):return
        reference=action.get('parent_grounding',action)
        if reference is action and action.get('grasp_part','body')!='body':return
        close=next((s for s in reversed(result.get('subactions',[])) if s.get('name')=='close'),{})
        pose=np.asarray(close.get('endpose_after',{}).get(str(side)+'_endpose'),float)
        if pose.shape!=(7,) or not np.isfinite(pose).all():return
        camera=camera_of(reference)
        paths=dict(zip(('head_camera','left_camera','right_camera'),observation.get('image_paths',[])))
        if camera not in paths or 'bbox' not in reference or 'table_height_m' not in observation:return
        path=evidence_path(paths[camera],observation['observation_id'],camera)
        if path is None or not path.is_file():return
        from surface_frames import box_points
        with np.load(path,allow_pickle=False) as data:
            if 'world_xyz' not in data or 'valid' not in data:return
            pts=box_points(data['world_xyz'],data['valid'],reference['bbox'],observation['table_height_m'])
        if len(pts)<32:return
        lo,hi=np.quantile(pts,[.01,.99],axis=0)
        if max(hi-lo)>.45 or np.sort(hi-lo)[-2]<.02:return
        corners=np.array(list(product(*zip(lo,hi))))
        self.entries[side]=dict(target=object_key(action['target']),
            local_corners=(corners-pose[:3])@quat2mat(pose[3:]),
            reference_observation_id=observation['observation_id'])

    def search(self, observation, target, camera):
        from robodawn.episodic_memory import object_key
        from robodawn.grounding_evidence import current_contact_mask
        from robodawn.camera_views import in_camera
        key=object_key(target)
        held=[(side,entry) for side,entry in self.entries.items() if entry['target']==key]
        # A shared two-hand grasp is not a single rigid gripper frame.
        if len(held)!=1:return None
        side,entry=held[0];state=observation.get('sensors',{}).get(side,{})
        if not state.get('holding') or object_key(state.get('remembered_object',''))!=key:return None
        if entry['reference_observation_id']>=observation.get('observation_id',-1):return None
        pose=np.asarray(observation.get('endpose',{}).get(side+'_endpose'),float)
        if pose.shape!=(7,) or not np.isfinite(pose).all():return None
        view=in_camera(observation,camera);foreground=current_contact_mask(view)
        if foreground is None:return None
        path=evidence_path(view['image_paths'][0],observation['observation_id'],camera)
        with np.load(path,allow_pickle=False) as data:cloud=data['world_xyz']
        predicted=entry['local_corners']@quat2mat(pose[3:]).T+pose[:3]
        # This 3 cm uncertainty is for SEARCH, not registration acceptance or
        # contact execution. Slips and deformation require fresh visual binding.
        region=foreground&(cloud>=predicted.min(0)-.03).all(2)&(cloud<=predicted.max(0)+.03).all(2)
        patch=broad_support_patch(cloud,region)
        plane=None
        if patch is not None:region,plane=patch
        ys,xs=np.where(region)
        if len(xs)<32 or min(np.ptp(xs),np.ptp(ys))<4:return None
        h,w=region.shape
        result=dict(bbox=(np.array([xs.min(),ys.min(),xs.max(),ys.max()])/[w-1,h-1,w-1,h-1]*1000).tolist(),
                    status='uncertain_gripper_motion_guided_current_foreground_search',
                    reference_observation_id=entry['reference_observation_id'],
                    current_observation_id=observation['observation_id'],current_foreground_pixels=len(xs))
        if plane is not None:result['support_plane']=plane
        return result
