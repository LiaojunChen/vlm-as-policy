"""Observe a clipped, model-bound source before committing a contact.

The anchor is an actually observed current calibrated-camera foreground pixel.
No missing extent is reconstructed and no predicted object pose is executed.
"""
import cv2
import numpy as np
from robodawn.camera_views import in_camera
from robodawn.grounding_evidence import current_contact_mask


def clipped_source_anchor(observation, action):
    if (action.get('skill') not in ('pick','grasp_handle','push')
            or action.get('observation_id')!=observation.get('observation_id')):
        return None
    reference=action.get('parent_grounding',action)
    camera=reference.get('camera','head_camera')
    if camera not in ('head_camera','left_camera','right_camera'):return None
    if reference.get('observation_id')!=observation.get('observation_id'):return None
    box=np.asarray(reference.get('bbox'),float);point=np.asarray(reference.get('point'),float)
    if (box.shape!=(4,) or point.shape!=(2,) or not np.isfinite(box).all()
            or not np.isfinite(point).all() or np.any(box<0) or np.any(box>1000)
            or np.any(box[2:]<=box[:2]) or np.any(point<box[:2]) or np.any(point>box[2:])):
        return None
    foreground=current_contact_mask(in_camera(observation,camera))
    if foreground is None:return None
    h,w=foreground.shape
    x0,y0,x1,y1=np.rint(box*[w-1,h-1,w-1,h-1]/1000).astype(int)
    if x0>2 and y0>2 and x1<w-3 and y1<h-3:return None
    region=np.zeros_like(foreground);region[y0:y1+1,x0:x1+1]=True
    count,labels,stats,centres=cv2.connectedComponentsWithStats((region&foreground).astype(np.uint8),8)
    pixel=point*np.array([w-1,h-1])/1000
    candidates=[(float(np.linalg.norm(centres[i]-pixel))-min(int(stats[i,cv2.CC_STAT_AREA]),400)/100,i)
                for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=8]
    if not candidates:return None
    selected=labels==min(candidates)[1];ys,xs=np.where(selected)
    borders=dict(left=int(np.sum(xs<=2)),right=int(np.sum(xs>=w-3)),
                 top=int(np.sum(ys<=2)),bottom=int(np.sum(ys>=h-3)))
    borders={side:n for side,n in borders.items() if n>=3}
    if not borders:return None
    # Use a measured pixel near the visible component centre, not the hidden
    # continuation of the object or the model point when that is background.
    pixels=np.c_[xs,ys];centre=np.median(pixels,axis=0)
    chosen=pixels[np.linalg.norm(pixels-centre,axis=1).argmin()]
    return dict(source='current_named_source_reaches_image_boundary',
                image_point=(chosen/[w-1,h-1]*1000).tolist(),
                target=reference.get('target',action.get('target')),
                observation_id=observation['observation_id'],camera=camera,
                clipped_borders=borders,observed_component_pixels=len(xs))
