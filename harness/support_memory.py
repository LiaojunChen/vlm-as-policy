"""RGB-D support memory, usable only with current surface evidence."""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree


class SupportMemory:
    def __init__(self):
        self.entries={}

    @staticmethod
    def key(action):
        return (' '.join(action.get('target','').lower().split()), action.get('relation','on'))

    def forget(self, target):
        name=' '.join(str(target).lower().split())
        self.entries={k:v for k,v in self.entries.items() if k[0]!=name}

    def remember(self, action, geometry, cloud, valid, table, observation_id):
        if geometry.get('source') not in ('visible_rgbd_circular_cavity','visible_rgbd_solid_support'):
            return
        height,width=valid.shape
        x0,y0,x1,y1=np.rint(np.asarray(action['bbox'])*np.array([width-1,height-1,width-1,height-1])/1000).astype(int)
        pts=cloud[y0:y1+1,x0:x1+1][valid[y0:y1+1,x0:x1+1]]
        pts=pts[np.isfinite(pts).all(1)&(pts[:,2]>table+.004)]
        if len(pts)<60:return
        # Flat interior/table points are ambiguous under translation. Verify
        # the vessel's higher wall/lip or the support's upper surface instead.
        threshold=float(np.quantile(pts[:,2],.65))
        pts=pts[pts[:,2]>=threshold]
        if len(pts)<30 or np.min(np.ptp(pts[:,:2],axis=0))<.03:return
        pts=pts[::max(1,len(pts)//1200)].copy()
        physical={k:deepcopy(geometry[k]) for k in ('tcp','top','source','rim_height_m','rim_radius_m',
                  'rim_fit_residual_m','observed_floor_pixels','pixels') if k in geometry}
        self.entries[self.key(action)]=dict(geometry=physical,points=pts,
                                           threshold=threshold,observation_id=observation_id)

    def verify(self, action, cloud, valid):
        entry=self.entries.get(self.key(action))
        if entry is None:return None
        reference=entry['points'];lo=reference.min(0)-.008;hi=reference.max(0)+.008
        visible=valid&np.isfinite(cloud).all(2)&(cloud>=lo).all(2)&(cloud<=hi).all(2)
        visible &= cloud[:,:,2]>=entry['threshold']
        current=cloud[visible]
        if len(current)<max(30,len(reference)*.2):return None
        distance,_=cKDTree(reference).query(current)
        matched=current[distance<.004]
        if len(matched)<30 or np.mean(distance<.004)<.9 or np.median(distance)>.003:return None
        coverage=np.ptp(matched[:,:2],axis=0)/np.maximum(np.ptp(reference[:,:2],axis=0),1e-9)
        if np.min(coverage)<.5:return None
        result=deepcopy(entry['geometry'])
        result['source']='current_verified_rgbd_support_memory'
        result['support_memory']=dict(reference_observation_id=entry['observation_id'],
                                      current_matched_points=len(matched),inlier_fraction=float(np.mean(distance<.004)),
                                      median_error_m=float(np.median(distance)),visible_span_fraction=coverage.tolist())
        return result
