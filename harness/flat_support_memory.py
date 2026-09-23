"""Visible flat-colour boundaries, with current multiview corroboration.

No reconstruction of missing borders. Only a fully visible, rectangular
current component can replace a previously clipped support estimate.
"""
import cv2
import numpy as np

COLOURS={'red':(0,10),'green':(35,85),'blue':(100,130),
         'cyan':(80,100),'yellow':(20,35),'purple':(130,165)}


def exterior_background_fractions(image_rectangle,colour_mask,cloud,valid,table):
    """Every proposed edge needs visible table beyond it, not an occluder.

    The component's being rectangular is insufficient: a straight robot or
    object edge can leave a smaller rectangular visible fragment.
    """
    corners=cv2.boxPoints(image_rectangle);centre=corners.mean(0)
    plane=valid&np.isfinite(cloud).all(2)&(np.abs(cloud[:,:,2]-table)<.004)
    fractions=[];height,width=valid.shape
    for first,last in zip(corners,np.roll(corners,-1,axis=0)):
        edge=last-first;length2=float(np.dot(edge,edge))
        if length2<4:return [0.]*4
        normal=(first+last)/2-centre;normal-=edge*np.dot(normal,edge)/length2
        length=np.linalg.norm(normal)
        if length<1e-6:return [0.]*4
        normal/=length
        samples=(first[None,None,:]+np.linspace(.12,.88,24)[None,:,None]*edge
                 +np.arange(2,5)[:,None,None]*normal)
        pixels=np.rint(samples).astype(int).reshape(-1,2);x,y=pixels.T
        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
        visible=np.zeros(len(x),bool)
        visible[inside]=plane[y[inside],x[inside]]&~colour_mask[y[inside],x[inside]]
        fractions.append(float(visible.mean()))
    return fractions


def components(rgb,cloud,valid,table,observation_id,camera):
    h,s,v=cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV));result={}
    for colour,(lo,hi) in COLOURS.items():
        mask=(((h>=lo)&(h<=hi))|((h>=170) if colour=='red' else False))&(s>100)&(v>70)
        mask &= valid & np.isfinite(cloud).all(2)&(np.abs(cloud[:,:,2]-table)<.004)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        found=[]
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<50:continue
            ys,xs=np.where(labels==i);pts=cloud[ys,xs]
            rect=cv2.minAreaRect(pts[:,:2].astype(np.float32));centre=np.r_[rect[0],np.median(pts[:,2])]
            if not (-.4<centre[0]<.4 and -.4<centre[1]<.3):continue
            corners=cv2.boxPoints(rect);edges=np.roll(corners,-1,axis=0)-corners
            major=edges[np.linalg.norm(edges,axis=1).argmax()]
            hull=cv2.convexHull(np.c_[xs,ys].astype(np.float32))
            hull_area=cv2.contourArea(hull)
            image_rect=cv2.minAreaRect(np.c_[xs,ys].astype(np.float32))
            rect_area=float(np.prod(image_rect[1]))
            clipped=bool(xs.min()==0 or ys.min()==0 or xs.max()==mask.shape[1]-1 or ys.max()==mask.shape[0]-1)
            complete=(not clipped and min(rect[1])>.025 and max(rect[1])<.4
                      and hull_area/max(rect_area,1)>.94 and len(pts)/max(hull_area,1)>.9)
            exterior=exterior_background_fractions(image_rect,mask,cloud,valid,table) if complete else None
            complete=complete and min(exterior)>=.75
            nearest=int(np.argmin(np.linalg.norm(pts-centre,axis=1)))
            found.append(dict(point=centre.tolist(),area=len(pts),yaw=float(np.arctan2(major[1],major[0])),
                              observation_id=observation_id,camera=camera,clipped=clipped,
                              image_point=[float(xs[nearest]/(mask.shape[1]-1)*1000),float(ys[nearest]/(mask.shape[0]-1)*1000)],
                              complete_boundary=bool(complete),corners_xy=corners.tolist(),
                              exterior_background_fractions=exterior,
                              physical_area_m2=float(np.prod(rect[1]))))
        if found:result[colour]=found
    return result


def contains_points(item,points,margin=.008):
    corners=np.asarray(item['corners_xy']);axis=corners[1]-corners[0];other=corners[3]-corners[0]
    length=np.linalg.norm(axis);width=np.linalg.norm(other)
    if min(length,width)<1e-6:return np.zeros(len(points),bool)
    local=(np.asarray(points)-corners[0])@np.column_stack([axis/length,other/width])
    return (local>=-margin).all(1)&(local<=[length+margin,width+margin]).all(1)


def refresh_landmarks(landmarks,views,table,observation_id):
    observed={camera:components(view['rgb'],view['cloud'],view['valid'],table,observation_id,camera)
              for camera,view in views.items()}
    if not landmarks:
        landmarks.update(observed.get('head_camera',{}))
    for colour,items in landmarks.items():
        for item in items:
            if 'corners_xy' not in item:continue
            candidates=[]
            for by_colour in observed.values():
                for current in by_colour.get(colour,[]):
                    # A current region must corroborate the stored physical patch,
                    # not merely have the same colour elsewhere in the scene.
                    if np.linalg.norm(np.asarray(current['point'])[:2]-np.asarray(item['point'])[:2])>.06:continue
                    if not contains_points(current,item['corners_xy']).all():continue
                    if current['complete_boundary']:candidates.append(current)
            if candidates:
                best=max(candidates,key=lambda x:x['physical_area_m2'])
                if item.get('clipped') or item.get('refined_from_clipped'):
                    old_observation=item.get('initial_observation_id',item['observation_id'])
                    item.update(best,initial_observation_id=old_observation,refined_from_clipped=True)
    return landmarks


def current_support_evidence(item,views,table):
    """A saved complete boundary is usable only with current coloured support."""
    colour=item.get('colour')
    if colour not in COLOURS:return None
    counts={}
    for camera,view in views.items():
        h,s,v=cv2.split(cv2.cvtColor(view['rgb'],cv2.COLOR_RGB2HSV));lo,hi=COLOURS[colour]
        colour_mask=(((h>=lo)&(h<=hi))|((h>=170) if colour=='red' else False))&(s>100)&(v>70)
        plane=view['valid']&np.isfinite(view['cloud']).all(2)&(np.abs(view['cloud'][:,:,2]-table)<.004)
        # Visible background replacing the saved coloured patch contradicts
        # stationarity. Occluded/high/robot points are not background evidence.
        inner=contains_points(item,view['cloud'][plane,:2],margin=-.003)
        if inner.sum()>=30:
            mismatch=(~colour_mask[plane][inner]).sum()
            if mismatch>=20 and mismatch/inner.sum()>.08:return None
        mask=colour_mask&plane
        pts=view['cloud'][mask]
        if len(pts)<30:continue
        matched=pts[contains_points(item,pts[:,:2],margin=.003)]
        if len(matched)<30 or min(np.ptp(matched[:,:2],axis=0))<.02:continue
        counts[camera]=len(matched)
    return dict(current_matched_pixels=counts) if counts else None


def clipped_support_anchor(observation,target):
    """Bind a clipped support by its authored colour and current head depth."""
    import re
    from PIL import Image
    from robodawn.camera_views import in_camera,evidence_path
    if not re.search(r'\b(pad|mat|square)\b',target,re.I):return None
    colours=[c for c in COLOURS if re.search(r'\b'+c+r'\b',target,re.I)]
    if len(colours)!=1:return None
    colour=colours[0];items=observation.get('visual_landmarks',{}).get(colour,[])
    if len(items)!=1 or not items[0].get('clipped'):return None
    head=in_camera(observation,'head_camera')
    path=evidence_path(head['image_paths'][0],observation['observation_id'],'head_camera')
    if path is None or not path.is_file():return None
    with np.load(path,allow_pickle=False) as data:
        current=components(np.asarray(Image.open(head['image_paths'][0])),data['world_xyz'],data['valid'],
                           observation['table_height_m'],observation['observation_id'],'head_camera').get(colour,[])
    if len(current)!=1 or not current[0]['clipped']:return None
    if np.linalg.norm(np.asarray(current[0]['point'])-np.asarray(items[0]['point']))>.02:return None
    return current[0]
