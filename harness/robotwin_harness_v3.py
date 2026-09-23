"""RoboTwin visual RGB-D skills; no actor poses, expert contacts or task-name branches."""
from __future__ import annotations
import json, math, subprocess, re, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2mat
from transforms3d.axangles import axangle2mat

SIDES=('left','right')
RELATIVE_DIRECTIONS={'left_of':(-1,0),'right_of':(1,0),'in_front_of':(0,-1),'behind':(0,1)}
SKILLS=('pick','grasp_handle','handover','inspect','push','place','press','reach','move','dual_move','rotate','arc','open','close','home','present','shake','wait','done')
CONTEXT='''You control two Aloha robot arms in RoboTwin. Image A is head RGB, B left wrist, C right wrist.
World +X is head image right, +Y toward the back of the table, +Z upward. Coordinates in metres.
Use the arm on the same side as an object unless the task explicitly specifies an arm. Never grasp the robot.
The executor provides calibrated TCP offsets, reachable grasp orientation search and measured grasp contact.
PICK selects a graspable BODY, HANDLE, RIM or EDGE, never empty space, shadow, robot or tabletop. Short compact objects need a top grasp. For wide bowls/baskets/skillets, select a thin rim or handle, NOT the empty centre. Use grasp_part body/handle/rim/edge. It approaches,
closes and lifts 12 cm, checking bilateral finger contact. PLACE transfers the held object to the visible
specified support/container centre, lowers, opens and retreats. A successful motion alone is not a grasp.
For raising/showing objects, PICK then PRESENT each held object toward the front-centre while retaining it.
For multiple objects, finish the required transfer then HOME the empty arm before approaching the next object.
Never repeat the same failed geometry: change grasp location/arm/approach, or recover with HOME.
Available skills: pick, grasp_handle (approach and close WITHOUT automatic lift, for drawer/door/lid handles), place, press, reach, move (delta xyz, max 20 cm), rotate (axis x/y/z, angle degrees,
around TCP, max 90 deg), open, close, home, present, shake (axis x/y/z, amplitude 2-6 cm, cycles 1-4), wait, done.
ARC moves a held handle about a visible hinge: skill arc, signed angle max90 deg, and bbox/point on the hinge pivot. For a tilted/non-axis-aligned hinge supply hinge_line [[x1,y1],[x2,y2]] (two endpoints in normalized HEAD pixels); its 3-D direction is measured from depth. Otherwise supply world axis x/y/z. Positive angle follows the right-hand rule along the first-to-second endpoint. Follow a hinge with several small arcs, a drawer with small MOVE actions.
PUSH slides a short object on the table without grasping: target/bbox/point identify the object; destination is a second JSON object with target/bbox/point/support describing its desired support position. Use for sliding a cup onto a coaster or a card box toward a table edge. Do not push upright tall bottles.
Pick/grasp_handle/place/press/reach/arc need target description, a tight component/destination bbox [xmin,ymin,xmax,ymax]
and contact point [x,y] in NORMALIZED 0..1000 image A coordinates. Include the ENTIRE object's graspable body
in bbox; point lies on the graspable component. For place, identify the interior or top support, not an edge/handle.
For PLACE, target MUST name the DESTINATION (e.g. centre of table, green block, cyan pad), never the held object. Place objects precisely centred and square with the support. Both empty grippers must be open.
For other skills omit bbox/point. PICK approach: top or side. Default top for short objects, side for tall bottles.
Return one JSON object with keys skill, arm (left/right), target, bbox, point, grasp_part, approach, destination,
hinge_line, delta, axis, angle, amplitude, cycles, reason as needed. target MUST be a descriptive STRING;
bbox and point are TOP-LEVEL siblings of target, never nested inside it. For example:
{"skill":"press","arm":"right","target":"blue bell top","bbox":[100,200,300,400],"point":[200,300]}
PUSH format:
{"skill":"push","arm":"right","target":"blue cup","bbox":[100,200,300,400],"point":[200,300],"destination":{"target":"wooden coaster","bbox":[400,500,600,700],"point":[500,600],"support":"object"}}
Examples show schema only; derive coordinates from the current image. Exactly ONE action per decision. DONE only after task visibly complete.
'''

class SkillError(ValueError):pass

def parse_json(raw):
    raw=raw.strip()
    if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(raw)
    if not isinstance(value,dict):raise SkillError('Expected JSON object')
    return value

def validate(a):
    try:return _validate(a)
    except SkillError:raise
    except (ValueError,TypeError,KeyError,OverflowError) as exc:raise SkillError('Malformed action parameters: '+str(exc)) from exc

def _validate(a):
    if not isinstance(a,dict):raise SkillError('Action must be a JSON object')
    if a.get('skill') not in SKILLS:raise SkillError('Unsupported skill')
    if a['skill'] not in ('wait','done','dual_move') and a.get('arm') not in SIDES:raise SkillError('Invalid arm')
    def vector(key,n,lo,hi):
        v=np.asarray(a.get(key),dtype=float)
        if v.shape!=(n,) or not np.isfinite(v).all() or np.any(v<lo) or np.any(v>hi):raise SkillError('Invalid '+key)
        return v
    if a['skill'] in ('pick','grasp_handle','handover','push','place','press','reach','arc'):
        if not isinstance(a.get('target'),str) or not a['target'].strip():raise SkillError('target must be an object-name STRING. bbox and point must be top-level siblings of target, not nested inside target.')
        box=vector('bbox',4,0,1000);point=vector('point',2,0,1000)
        if np.any(box[2:]<=box[:2]) or np.any(point<box[:2]) or np.any(point>box[2:]):raise SkillError('Invalid bbox/point')
        if not a.get('target'):raise SkillError('Target identity required')
    if a['skill']=='handover' and (a.get('donor') not in SIDES or a['donor']==a['arm']):
        raise SkillError('Handover needs distinct donor and receiving arm')
    if a['skill']=='place' and re.fullmatch(r'(?:the )?(?:other|left|right) (?:robot )?(?:hand|arm|gripper)',a['target'].strip(),re.I):
        raise SkillError('A robot hand is not a placement support. Use contact-before-release handover.')
    if 'support' in a and a['support'] not in ('table','object','container'):raise SkillError('support must be table, object or container')
    if 'approach' in a and a['approach'] not in ('top','side'):raise SkillError('approach must be top or side')
    if 'release' in a and not isinstance(a['release'],bool):raise SkillError('release must be a JSON boolean')
    if 'location' in a and a['location'] not in ('centre','side'):raise SkillError('location must be centre or side')
    if 'relation' in a and a['relation'] not in ('on','inside','row_slot',*RELATIVE_DIRECTIONS):raise SkillError('Invalid spatial relation')
    if a.get('relation')=='row_slot':
        if type(a.get('slot')) is not int or type(a.get('slot_count')) is not int or not 0<=a['slot']<a['slot_count']<=5:raise SkillError('Invalid row layout slot')
    if a['skill'] in ('move','dual_move'):
        if np.linalg.norm(vector('delta',3,-.2,.2))>.201:raise SkillError('Delta exceeds 20 cm')
    if a['skill']=='inspect' and 'inspection_anchor' in a:vector('inspection_anchor',2,0,1000)
    if a['skill']=='inspect' and a.get('inspection_camera','head_camera') not in ('head_camera','left_camera','right_camera'):
        raise SkillError('Invalid inspection anchor camera')
    if a['skill'] in ('rotate','shake') or (a['skill']=='arc' and 'hinge_line' not in a):
        if a.get('axis') not in ('x','y','z'):raise SkillError('Invalid axis')
    if a['skill']=='arc' and 'hinge_line' in a:
        line=np.asarray(a['hinge_line'],float)
        if line.shape!=(2,2) or not np.isfinite(line).all() or np.any(line<0) or np.any(line>1000) or np.linalg.norm(line[1]-line[0])<10:raise SkillError('Invalid hinge line')
    if a['skill'] in ('rotate','arc') and not 0<abs(float(a.get('angle',0)))<=90:raise SkillError('Invalid angle')
    if 'grasp_part' in a and a['grasp_part'] not in ('body','handle','rim','edge'):raise SkillError('Invalid grasp_part')
    if a['skill']=='push':validate({**a.get('destination',{}),'skill':'place','arm':a['arm']})
    if 'contact_reference' in a:
        if a['skill']!='pick' and not (a['skill']=='place' and a.get('use_contact_part') and not a.get('release',True)):
            raise SkillError('Working-part reference belongs to pickup or retained tool contact')
        validate({**a['contact_reference'],'skill':'press','arm':a['arm']})
    if 'facing' in a:
        from directed_placement import FACING
        if a['skill']!='place' or a['facing'] not in FACING:raise SkillError('Invalid directed placement facing')
    if 'container_layout' in a:
        layout=a['container_layout']
        if (a['skill']!='place' or not isinstance(layout,dict) or type(layout.get('slot')) is not int
                or type(layout.get('count')) is not int or not 0<=layout['slot']<layout['count']<=5
                or layout['count']<2 or layout.get('facing')!=a.get('facing')):
            raise SkillError('Invalid directed container layout')
    if 'orientation_reference' in a:
        from directed_placement import FACING
        ref=a['orientation_reference']
        if a['skill']!='pick' or not isinstance(ref,dict) or ref.get('facing') not in FACING:raise SkillError('Invalid pickup orientation reference')
        for key in ('from_part','to_part'):validate({**ref[key],'skill':'press','arm':a['arm']})
    return a

def grasp_quat(yaw,pitch=math.pi/2):
    return mat2quat(euler2mat(0,pitch,yaw)).tolist()

def resolve_grasp_arm(action, geometry, instruction, sensors):
    """Resolve an unconstrained empty-arm grasp before any arm state is read."""
    side=action['arm']
    instruction=instruction.split('Completion requirements:')[0]
    constrained=action.get('arm_required') or re.search(
        r'\b(?:left|right|both|each|other)[ -]+(?:arm|hand|gripper)s?\b',instruction,re.I)
    if constrained or any(s.get('holding') for s in sensors.values()):return side
    x=geometry['tcp'][0]
    if abs(x)<.035:return side
    return 'left' if x<0 else 'right'

def prefer_side_grasp(action, geometry, table_height):
    """Low loose components need clearance from the observed support plane."""
    if geometry.get('source')=='visible_rgbd_local_component':return False
    if geometry.get('tall',False):return True
    if action.get('approach')!='side':return False
    if action['skill']=='grasp_handle':return True
    return (action.get('grasp_part') in ('handle','rim','edge')
            and geometry['tcp'][2]-table_height>.10)

def component_opening(geometry):
    """The calibrated gripper has 9 cm maximum opening; keep 24 mm clearance."""
    if geometry.get('source')!='visible_rgbd_local_component':return 1.
    width=float(geometry.get('component_width_m',.09))
    return float(np.clip((width+.024)/.09,1/3,1.))

def grasp_yaws(yaw,side_grasp):
    """Preserve preferred solutions, then cover the cylinder's full azimuth."""
    preferred=[math.pi/2,math.pi/3,2*math.pi/3,yaw,yaw+math.pi,0,math.pi] if side_grasp else [yaw,yaw+math.pi,math.pi/2,math.pi/3,2*math.pi/3,0,math.pi]
    if side_grasp:preferred += [i*math.pi/12 for i in range(24)]
    unique=[]
    for angle in preferred:
        if not any(abs((angle-old+math.pi)%(2*math.pi)-math.pi)<1e-5 for old in unique):unique.append(angle)
    return unique

def ee_from_tcp(tcp,quat):
    return (np.asarray(tcp)-quat2mat(quat)@np.array([.12,0,0])).tolist()+list(quat)

def tcp_from_ee(pose):return np.asarray(pose[:3])+quat2mat(pose[3:])@np.array([.12,0,0])

def estimate_table_height(cloud,valid,fallback=.74):
    """Estimate the dominant horizontal support solely from calibrated depth."""
    xy=cloud[:,:,:2];z=cloud[:,:,2]
    points=z[valid&(np.abs(xy[:,:,0])<.35)&(np.abs(xy[:,:,1])<.28)&(z>.50)&(z<.95)]
    if not len(points):return fallback
    values,counts=np.unique(np.round(points/.005),return_counts=True)
    return float(values[counts.argmax()]*.005)

def relative_support(points,source_span,relation,table_height):
    """Put an object beside a measured reference, with room for both footprints."""
    points=np.asarray(points)
    if len(points)<4:raise SkillError('Reference object has insufficient visible depth')
    lo,hi=np.quantile(points[:,:2],[.02,.98],axis=0)
    direction=np.asarray(RELATIVE_DIRECTIONS[relation],float)
    axis=int(np.argmax(np.abs(direction)))
    span=np.asarray(source_span,float)
    clearance=.5*(hi[axis]-lo[axis]+span[axis])+.02
    xy=(lo+hi)*.5+direction*clearance
    return [float(xy[0]),float(xy[1]),float(table_height)]

def placement_yaws(action):
    """Preserve specified planar alignment; otherwise search upright yaw freedom.

    Rotations are about world Z, so the carried object's tilt does not change.
    The original two candidates often leave a six-DOF arm unable to reach even
    a nearby support after a successful grasp at the opposite side of its arc.
    """
    if action.get('facing'):return (0.,)
    if 'align_yaw_deg' in action or action.get('preserve_yaw'):
        return (0., math.pi)
    return tuple(math.radians(d) for d in (0,180,30,-30,60,-60,90,-90,120,-120,150,-150))

def placement_rotations(quaternion,action):
    """Try upright transport first, then small tilts for released placements."""
    tilts=(0,-15,15) if action.get('release',True) and not action.get('preserve_tilt') and not action.get('facing') else (0,)
    for tilt in tilts:
        for yaw in placement_yaws(action):
            rotation=euler2mat(0,0,yaw)@quat2mat(quaternion)@euler2mat(0,math.radians(tilt),0)
            yield yaw,tilt,rotation

class Bridge:
    def __init__(self,env,instruction,directory,video=True):
        self.env,self.instruction,self.directory=env,instruction,Path(directory)
        (self.directory/'frames').mkdir(parents=True,exist_ok=True)
        self.index=0;self.obs_id=0;self.depth_id=None;self.plan_cache=[];self.landmarks={};self.held={s:None for s in SIDES};self.failed={};self.video=None
        self.initial=env.get_obs()['endpose'];self.cam=env.cameras.static_camera_list[env.cameras.head_camera_id]
        self.original_update=env._update_render;self.original_get_obs=env.get_obs;self.render_count=0;self.capturing=False
        if video:
            height,width=self.cam.get_picture('Color').shape[:2];self.video_size=(width,height)
            self.video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','20','-i','-','-vcodec','libx264','-pix_fmt','yuv420p','-crf','22',str(self.directory/'continuous.mp4')],stdin=subprocess.PIPE)
            def update():
                self.render_count+=1
                if self.render_count%25==0 and not self.capturing:
                    self.original_update();self.capture()
            env._update_render=update
            def get_obs():
                self.original_update()
                return self.original_get_obs()
            env.get_obs=get_obs
            self.capture()
    def capture(self):
        if self.video is None:return
        self.capturing=True
        try:
            self.cam.take_picture();rgb=(np.clip(self.cam.get_picture('Color')[:,:,:3],0,1)*255).astype(np.uint8)
            if rgb.shape[1::-1]!=self.video_size:raise SkillError('Camera resolution changed during recording')
            self.video.stdin.write(rgb.tobytes())
        finally:self.capturing=False
    def endpose(self):
        # Same public proprioception as Base_Task.get_obs, without rendering three cameras.
        return {key:value for side in SIDES for key,value in (
            (side+'_endpose',self.env.get_arm_pose(side)),
            (side+'_gripper',float(getattr(self.env.robot,'get_'+side+'_gripper_val')())))}
    def sensors(self):
        result={};contacts=self.env.scene.get_contacts()
        for side in SIDES:
            fingers=[j[0].child_link.name for j in getattr(self.env.robot,side+'_gripper')]
            hits={name:set() for name in fingers}
            directions={name:{} for name in fingers};has_normals=False;closing_axis=None
            for c in contacts:
                bodies=c.bodies;names=[b.entity.name for b in bodies]
                for i,n in enumerate(names):
                    if n not in hits or names[1-i].startswith(('fl_','fr_','lr_','rr_')):continue
                    if 'Static' in type(bodies[1-i]).__name__:continue
                    if not any(np.linalg.norm(p.impulse)>1e-7 for p in c.points):continue
                    body=bodies[1-i].entity.per_scene_id;hits[n].add(body)
                    for point in c.points:
                        if not hasattr(point,'normal') or np.linalg.norm(point.impulse)<=1e-7:continue
                        normal=np.asarray(point.normal,float);length=np.linalg.norm(normal)
                        if length<1e-6:continue
                        if closing_axis is None:closing_axis=quat2mat(self.env.get_arm_pose(side)[3:])[:,1]
                        projection=float(normal@closing_axis/length)*(1 if i else -1)
                        directions[n].setdefault(body,[]).append(projection);has_normals=True
            shared=set.intersection(*hits.values()) if hits else set()
            if has_normals:
                from tactile_contact import opposing_bodies
                pinched=shared&opposing_bodies(directions)
            else:pinched=shared
            entity=getattr(self.env.robot,side+'_entity');jointnames=[j.name for j in entity.get_active_joints()]
            actual=[float(entity.get_qpos()[jointnames.index(j[0].name)]) for j in getattr(self.env.robot,side+'_gripper')]
            result[side]={'holding':bool(pinched if getattr(self,'require_opposed_contacts',False) else shared),'finger_qpos_m':actual,'contact_fingers':sum(bool(v) for v in hits.values()),
                          'opposed_contact':bool(pinched) if has_normals else None,
                          'finger_normal_projection_range':{n:([min(x for values in objects.values() for x in values),max(x for values in objects.values() for x in values)] if objects else []) for n,objects in directions.items()},
                          'remembered_object':self.held[side]['target'] if self.held[side] else None}
        return result
    def observe(self):
        obs=self.env.get_obs()
        # Public simulator state can contain numpy integer gripper commands.
        # Normalize at the observation boundary before any model/log serialization.
        obs['endpose']={k:float(v) if k.endswith('_gripper') else np.asarray(v,dtype=float).tolist() for k,v in obs['endpose'].items()}
        self.obs_id+=1;self.depth_id=self.obs_id
        paths=[]
        for camera in ('head_camera','left_camera','right_camera'):
            p=self.directory/'frames'/f'{self.obs_id:04d}_{camera}.png';Image.fromarray(obs['observation'][camera]['rgb']).save(p);paths.append(str(p))
        position=self.cam.get_picture('Position')[:,:,:3].copy();m=self.cam.get_model_matrix()
        self.cloud=position@m[:3,:3].T+m[:3,3];self.valid=np.isfinite(position).all(2)&(position[:,:,2]<-.1)
        # Self filtering uses only the robot's own link IDs, never object segmentation identities.
        robot_ids={link.entity.per_scene_id for side in SIDES for link in getattr(self.env.robot,side+'_entity').get_links()}
        robot_mask=np.isin(self.cam.get_picture('Segmentation')[:,:,1],list(robot_ids))
        self.valid &= ~robot_mask
        self.rgb=obs['observation']['head_camera']['rgb']
        np.savez_compressed(self.directory/'frames'/f'{self.obs_id:04d}_rgbd.npz',world_xyz=self.cloud,valid=self.valid,robot_self_mask=robot_mask)
        self.view_data={'head_camera':dict(cloud=self.cloud,valid=self.valid,rgb=self.rgb,camera=self.cam)}
        # These are the same two wrist sensors already rendered and returned
        # by the original observer, not additional cameras or altered optics.
        for camera_name in ('left_camera','right_camera'):
            camera=getattr(self.env.cameras,camera_name)
            position=camera.get_picture('Position')[:,:,:3].copy();matrix=camera.get_model_matrix()
            cloud=position@matrix[:3,:3].T+matrix[:3,3]
            own=np.isin(camera.get_picture('Segmentation')[:,:,1],list(robot_ids))
            valid=np.isfinite(position).all(2)&(position[:,:,2]<-.1)&~own
            rgb=obs['observation'][camera_name]['rgb']
            self.view_data[camera_name]=dict(cloud=cloud,valid=valid,rgb=rgb,camera=camera)
            np.savez_compressed(self.directory/'frames'/f'{self.obs_id:04d}_{camera_name}_rgbd.npz',
                                world_xyz=cloud,valid=valid,robot_self_mask=own,cam2world_gl=matrix,
                                intrinsic_cv=camera.get_intrinsic_matrix())
        self.table=estimate_table_height(self.cloud,self.valid,getattr(self,'table',.74))
        from flat_support_memory import refresh_landmarks
        refresh_landmarks(self.landmarks,self.view_data,self.table,self.obs_id)
        self.last={'instruction':self.instruction,'image_paths':paths,'observation_id':self.obs_id,'endpose':obs['endpose'],'sensors':self.sensors(),'success':bool(self.env.eval_success or self.env.check_success()),'table_height_m':self.table,'visual_landmarks':self.landmarks}
        if getattr(self,'inspection_cameras',None):self.last['inspection_cameras']=list(self.inspection_cameras)
        return self.last
    def _remember_flat_colours(self,rgb):
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);h,s,v=cv2.split(hsv)
        for name,(lo,hi) in {'red':(0,10),'green':(35,85),'blue':(100,130),'cyan':(80,100),'yellow':(20,35),'purple':(130,165)}.items():
            colour=(((h>=lo)&(h<=hi))|((h>=170) if name=='red' else False))&(s>100)&(v>70)
            mask=colour&self.valid&(np.abs(self.cloud[:,:,2]-self.table)<.004)
            count,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
            components=[]
            for i in range(1,count):
                if stats[i,cv2.CC_STAT_AREA]<50:continue
                pts=self.cloud[labels==i];rectangle=cv2.minAreaRect(pts[:,:2].astype(np.float32));centre=np.r_[rectangle[0],np.median(pts[:,2])]
                vertices=cv2.boxPoints(rectangle);edges=np.roll(vertices,-1,axis=0)-vertices;major=edges[np.linalg.norm(edges,axis=1).argmax()]
                yaw=math.atan2(major[1],major[0])
                if not (-.4<centre[0]<.4 and -.4<centre[1]<.3):continue
                components.append({'point':centre.tolist(),'area':int(len(pts)),'yaw':yaw,'observation_id':self.obs_id})
            if components:self.landmarks[name]=components
    def geometry(self,a):
        from robodawn.camera_views import camera_of
        camera=camera_of(a)
        saved=(self.cloud,self.valid,getattr(self,'rgb',None),getattr(self,'cam',None))
        if camera!='head_camera':
            view=getattr(self,'view_data',{}).get(camera)
            if view is None:raise SkillError('No calibrated current RGB-D for '+camera)
            self.cloud,self.valid,self.rgb,self.cam=(view['cloud'],view['valid'],view['rgb'],view['camera'])
        try:
            result=self._geometry_with_memory(a)
            result['camera']=camera
            return result
        finally:self.cloud,self.valid,self.rgb,self.cam=saved

    def _geometry_with_memory(self,a):
        info=getattr(self,'held',{}).get(a.get('arm')) or {}
        if (a.get('skill')=='place' and a.get('release',True) and not a.get('use_contact_part')
                and (a.get('relation')=='inside' or a.get('support')=='container')
                and not a.get('container_layout')):
            if a.get('observation_id')!=self.depth_id:raise SkillError('Stale observation')
            from container_portal import below_table_portal
            portal=below_table_portal(self.cloud,self.valid,a['bbox'],self.table,info.get('span',[]))
            if portal is not None:return portal
        if (a.get('skill')=='place' and a.get('relation','on')=='on' and a.get('release',True)
                and not a.get('use_contact_part') and not a.get('facing') and not info.get('desired_facing')
                and info.get('observed_flat_source') and hasattr(self,'surface_frame_memory')):
            if a.get('observation_id')!=self.depth_id:raise SkillError('Stale observation')
            geometry=self.surface_frame_memory.resolve(a,self.cloud,self.valid,self.table)
            if geometry is not None:return geometry
        from support_memory import SupportMemory
        if not hasattr(self,'support_memory'):self.support_memory=SupportMemory()
        eligible=(a['skill']=='place' and a.get('relation','on') in ('on','inside')
                  and not a.get('regrasp_stage') and not a.get('container_layout'))
        try:geometry=self._geometry(a)
        except SkillError as exc:
            if not eligible or 'No depth at destination' not in str(exc):raise
            geometry=self.support_memory.verify(a,self.cloud,self.valid)
            if geometry is None:raise
        if eligible:
            if geometry.get('source') not in ('visible_rgbd_circular_cavity','visible_rgbd_solid_support'):
                verified=self.support_memory.verify(a,self.cloud,self.valid)
                if verified is not None:geometry=verified
            self.support_memory.remember(a,geometry,self.cloud,self.valid,self.table,self.obs_id if hasattr(self,'obs_id') else self.depth_id)
            if geometry.get('rim_radius_m'):a['support']='container'
        return geometry

    def _geometry(self,a):
        if a.get('observation_id')!=self.depth_id or self.depth_id is None:raise SkillError('Stale observation')
        if 'parent_grounding' in a and a['parent_grounding'].get('camera','head_camera')!=a.get('camera','head_camera'):
            raise SkillError('Parent component crop must use the source camera')
        if a['skill']=='place' and a.get('regrasp_stage'):
            from transport_planning import shared_table_support
            info=self.held.get(a['arm']) or {}
            support=shared_table_support(self.cloud,self.valid,self.table,info.get('span',[]),
                                         [self.initial[s+'_endpose'][:3] for s in SIDES])
            if support is None:raise SkillError('No observed collision-free shared table patch for a regrasp')
            a['support']='table';a['align']=False
            return support
        from size_grounding import refine_block_size
        if a['skill'] in ('pick','grasp_handle','place'):
            a.update(refine_block_size(a,self.cloud,self.valid,self.table))
        description=a['target'].lower()
        if a['skill']=='place' and a.get('relation')=='row_slot':
            height,width=self.cloud.shape[:2]
            origin=self.cam.get_model_matrix()[:3,3]
            ray=self.cloud[height//2,width//2]-origin
            if not np.isfinite(ray).all() or abs(ray[2])<1e-6:raise SkillError('Invalid calibrated table-centre ray')
            point=origin+ray*((self.table-origin[2])/ray[2])
            point[0]+=(a['slot']-(a['slot_count']-1)*.5)*.10
            a['support']='table'
            return {'tcp':point.tolist(),'top':self.table,'source':'calibrated_rgbd_table_row','slot':a['slot']}
        from transport_planning import explicit_free_table,shared_table_support
        if a['skill']=='place' and explicit_free_table(description) and a.get('relation','on')=='on':
            # A broad "empty table" box can also contain vessels. Do not
            # silently promote any such vessel into the requested destination.
            height,width=self.cloud.shape[:2]
            pixel=np.rint(np.asarray(a['point'])*[width-1,height-1]/1000).astype(int)
            origin=self.cam.get_model_matrix()[:3,3]
            ray=self.cloud[pixel[1],pixel[0]]-origin
            if not np.isfinite(ray).all() or abs(ray[2])<1e-6:raise SkillError('Invalid current free-table ray')
            hint=origin+ray*((self.table-origin[2])/ray[2])
            info=self.held.get(a['arm']) or {}
            geometry=shared_table_support(self.cloud,self.valid,self.table,info.get('span',[]),None,preferred_xy=hint[:2])
            if geometry is None:raise SkillError('No observed clear table patch fits the held footprint near the requested point')
            a['support']='table';a['align']=False
            return geometry
        colour=next((c for c in self.landmarks if re.search(r'\b'+c+r'\b',description)),None)
        if (colour is None and a['skill']=='place' and a.get('relation') not in RELATIVE_DIRECTIONS
                and re.search(r'\b(pad|mat)\b',description)
                and not re.search(r'\b(red|green|blue|cyan|yellow|purple|orange|black|white|pink|brown|gray|grey)\b',description)):
            # An uncoloured name can resolve only to ONE observed flat region.
            flat=[(c,item) for c,items in self.landmarks.items() for item in items if item['area']>=100]
            if len(flat)==1:
                colour=flat[0][0]
                a['landmark_resolution']='unique_observed_flat_pad'
        if a['skill']=='place' and a.get('relation') not in RELATIVE_DIRECTIONS and colour and (description.strip()==colour or any(w in description for w in ('pad','mat','square'))):
            item=max(self.landmarks[colour],key=lambda x:x['area']);a['support']='table';a['align_yaw_deg']=math.degrees(item['yaw'])
            if item.get('refined_from_clipped'):
                from flat_support_memory import current_support_evidence
                evidence=current_support_evidence(dict(item,colour=colour),getattr(self,'view_data',{}),self.table)
                if evidence is None:raise SkillError('Observed support boundary memory lacks current coloured depth; reobserve destination')
                return dict(tcp=item['point'],top=item['point'][2],source='current_verified_complete_flat_support',
                            landmark_observation_id=item['observation_id'],support_boundary_evidence=evidence)
            return {'tcp':item['point'],'top':item['point'][2],'source':'initial_rgbd_flat_colour','landmark_observation_id':item['observation_id']}
        height,width=self.cloud.shape[:2];scale=np.array([width-1,height-1])/1000
        if a['skill']=='arc' and 'hinge_line' in a:
            uv=np.rint(np.asarray(a['hinge_line'])*scale).astype(int)
            if not self.valid[uv[:,1],uv[:,0]].all():raise SkillError('Hinge endpoint is occluded; re-ground hinge line')
            ends=self.cloud[uv[:,1],uv[:,0]];axis=ends[1]-ends[0];length=np.linalg.norm(axis)
            if not .02<length<.5:raise SkillError('Implausible hinge length')
            return {'tcp':ends.mean(0).tolist(),'hinge_axis':(axis/length).tolist(),'source':'visible_rgbd_hinge_line'}
        box=np.asarray(a['bbox'])*np.tile(scale,2);x0,y0,x1,y1=np.rint(box).astype(int)
        pixel=np.rint(np.asarray(a['point'])*scale).astype(int)
        mask=np.zeros((height,width),np.uint8);mask[y0:y1+1,x0:x1+1]=1
        mask=mask.astype(bool)&self.valid&(self.cloud[:,:,2]>self.table+.004)&(self.cloud[:,:,2]<self.table+.38)
        mask &= (np.abs(self.cloud[:,:,0])<.45)&(self.cloud[:,:,1]>-.40)&(self.cloud[:,:,1]<.35)
        # Break connected image regions across large depth discontinuities.
        jumps=np.linalg.norm(np.diff(self.cloud,axis=0),axis=2)>.025
        mask[1:,:] &= ~jumps
        jumps=np.linalg.norm(np.diff(self.cloud,axis=1),axis=2)>.025
        mask[:,1:] &= ~jumps
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        choices=[]
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<4:continue
            pts=self.cloud[labels==i];dist=np.linalg.norm(centroids[i]-pixel)
            choices.append((dist-min(len(pts),400)/100,i))
        point=self.cloud[pixel[1],pixel[0]].copy()
        if a['skill']=='place' and a.get('relation') not in RELATIVE_DIRECTIONS and choices and not a.get('container_layout'):
            from container_geometry import circular_cavity_support
            support=circular_cavity_support(self.cloud[labels==min(choices)[1]],self.table)
            if support is not None:
                a['support']='container'
                return support
        if (a['skill']=='place' and a.get('relation','on') in ('on','inside')
                and not a.get('container_layout') and not re.search(r'\b(table|tabletop|workspace|pad|mat)\b',description)):
            from support_surfaces import observed_support_surface
            from container_geometry import release_clearance
            current=self.cloud[y0:y1+1,x0:x1+1][self.valid[y0:y1+1,x0:x1+1]]
            info=getattr(self,'held',{}).get(a['arm']) or {}
            support=observed_support_surface(current,self.table,info.get('span',[]))
            if support is not None:
                cavity=support['source']=='visible_rgbd_free_container_floor'
                a['support']='container' if cavity else 'object'
                if cavity and a.get('release',True):
                    clearance=release_clearance(current,support,info)
                    if clearance is not None:support['release_clearance']=clearance
                return support
        if a['skill']=='place' and a.get('support')=='container':
            from container_geometry import container_support,release_clearance
            crop=np.zeros((height,width),bool);crop[y0:y1+1,x0:x1+1]=True
            crop &= self.valid
            layout=a.get('container_layout');footprint=None
            if layout:
                info=self.held.get(a['arm']) or {}
                if 'directed_axis' not in info or 'footprint_xy' not in info:raise SkillError('Directed container layout requires observed source footprint and endpoints')
                from directed_placement import FACING
                axis=np.asarray(info['directed_axis']);wanted=FACING[layout['facing']]
                angle=math.atan2(wanted[1],wanted[0])-math.atan2(axis[1],axis[0])
                rotation=euler2mat(0,0,angle)[:2,:2]
                footprint=np.asarray(info['footprint_xy'])@rotation.T
            support=container_support(self.cloud[crop],self.table,layout=layout,footprint=footprint)
            if layout and support is None:raise SkillError('No observed container floor patch fits the entire directed object footprint; rearrange the contents or choose a different layout')
            if support is not None:
                if a.get('release',True):
                    clearance=release_clearance(self.cloud[crop],support,self.held.get(a['arm']) or {})
                    if clearance is not None:support['release_clearance']=clearance
                return support
        if a['skill'] in ('pick','grasp_handle'):
            if not choices:raise SkillError('No above-table object in visual bbox; re-ground target')
            i=min(choices)[1];pts=self.cloud[labels==i]
            selected_mask=labels==i;region_growth=None
            if a.get('grasp_part')=='rim':
                from component_geometry import complete_observed_circular_component,circular_rim_footprint
                if circular_rim_footprint(pts) is None:
                    grown=complete_observed_circular_component(self.cloud,self.valid,self.table,selected_mask)
                    if grown is not None:
                        region_growth=dict(source='current_connected_rgbd_circular_component',
                                           seed_pixels=int(selected_mask.sum()),grown_pixels=int(grown.sum()))
                        selected_mask=grown;pts=self.cloud[grown]
            if a.get('grasp_part') in ('handle','rim','edge') and labels[pixel[1],pixel[0]]!=i:
                # A pixel in a handle opening projects onto the table behind it.
                # Snap in image space before estimating the physical component;
                # 3-D nearest-to-background can instead select a vessel wall.
                ys,xs=np.where(labels==i)
                nearest=np.argmin((xs-pixel[0])**2+(ys-pixel[1])**2)
                point=self.cloud[ys[nearest],xs[nearest]].copy()
            self.pick_reference=(pts.copy(),self.rgb[selected_mask].copy())
            # Reject a bounding box containing a large part of the robot/workspace.
            span=np.ptp(pts[:,:2],axis=0)
            if np.max(span)>.30:raise SkillError('Object region too large; use a tighter bbox')
            xy=np.median(pts[:,:2],axis=0);top=float(np.quantile(pts[:,2],.95));low=float(np.quantile(pts[:,2],.05))
            centre_z=max(self.table+.018,top-min(.035,(top-self.table)*.45))
            # Tall upright object: side grasp around its middle; top-view sees only upper surface.
            tall=top-self.table>.10 and float(np.ptp(pts[:,2]))>.06
            if tall:
                centre_z=max(self.table+.035,self.table+(top-self.table)*.50)
            if tall:
                band=pts[(pts[:,2]>np.quantile(pts[:,2],.15))&(pts[:,2]<np.quantile(pts[:,2],.70)),:2]
                if len(band)>20:
                    fit=np.linalg.lstsq(np.c_[2*band,np.ones(len(band))],np.sum(band**2,axis=1),rcond=None)[0]
                    radius2=fit[2]+np.sum(fit[:2]**2)
                    if .012**2<radius2<.07**2:
                        residual=np.median(np.abs(np.linalg.norm(band-fit[:2],axis=1)-np.sqrt(radius2)))
                        if residual<.005 and np.linalg.norm(fit[:2]-xy)<.06:xy=fit[:2]
            covariance=np.cov(pts[:,:2].T);e,v=np.linalg.eigh(covariance);axis=v[:,np.argmax(e)]
            rectangle=cv2.minAreaRect(pts[:,:2].astype(np.float32))
            if not tall:xy=np.array(rectangle[0])
            if a.get('grasp_part') in ('handle','rim','edge'):
                nearest=pts[np.linalg.norm(pts-point,axis=1).argmin()]
                xy=nearest[:2];centre_z=float(nearest[2])-.007
            corners=cv2.boxPoints(rectangle);edges=np.roll(corners,-1,axis=0)-corners
            axis=edges[np.linalg.norm(edges,axis=1).argmax()]
            yaw=math.atan2(axis[1],axis[0]) # local Y closes across the narrow dimension
            geometry={'tcp':[float(xy[0]),float(xy[1]),centre_z],'top':top,'bottom':min(low,self.table+.01),'yaw':yaw,'span':span.tolist(),'pixels':len(pts),'tall':bool(tall),'bbox_px':[int(x0),int(y0),int(x1),int(y1)]}
            if a.get('grasp_part','body')=='body':
                from surface_frames import flat_source
                flat=flat_source(pts,self.table)
                if flat is not None:geometry['observed_flat_source']=flat
            if region_growth is not None:geometry['observed_region_growth']=region_growth
            geometry['footprint_corners_xy']=corners.tolist()
            hull=cv2.convexHull(pts[:,:2].astype(np.float32))
            geometry['footprint_hull_xy']=cv2.approxPolyDP(hull,.001,True).reshape(-1,2).tolist()
            from component_geometry import component_grasp
            parent_centre=None
            if a.get('parent_grounding') and a.get('grasp_part')=='handle':
                px0,py0,px1,py1=np.rint(np.asarray(a['parent_grounding']['bbox'])*np.tile(scale,2)).astype(int)
                parent_points=self.cloud[py0:py1+1,px0:px1+1][self.valid[py0:py1+1,px0:px1+1]]
                parent_points=parent_points[(parent_points[:,2]>self.table+.004)&(parent_points[:,2]<self.table+.38)]
                if len(parent_points)>30:parent_centre=np.median(parent_points,axis=0)
            if parent_centre is not None and a.get('grasp_part')=='handle':
                from component_geometry import elevated_handle_anchor
                selection=elevated_handle_anchor(pts,point,parent_centre)
                if selection is not None:
                    geometry['handle_contact_selection']=selection
                    point=np.asarray(selection['point'])
            if a.get('grasp_part')=='rim' and hasattr(self,'initial'):
                from component_geometry import near_side_rim_anchor
                contact_arm=resolve_grasp_arm(a,geometry,self.instruction,self.sensors())
                attempt=self.failed.get((contact_arm,a['target']),0)
                selection=near_side_rim_anchor(pts,self.initial[contact_arm+'_endpose'][:3],attempt)
                if selection is not None:
                    selection['model_selected_point']=point.tolist()
                    point=np.asarray(selection['point'])
                    geometry['rim_contact_selection']=selection
            component=component_grasp(pts,point,a.get('grasp_part'),self.table,parent_centre)
            if component is not None:geometry.update(component)
            if a.get('grounding_role')=='handover_receiver_contact':
                from handover import observed_receiver_contact
                geometry.update(observed_receiver_contact(pts,point))
            if parent_centre is not None:geometry['observed_parent_centre']=parent_centre.tolist()
            if 'orientation_reference' in a:
                reference=a['orientation_reference']
                if any(ref.get('camera','head_camera')!=a.get('camera','head_camera') for ref in (reference['from_part'],reference['to_part'])):
                    raise SkillError('Orientation endpoints must use the source camera')
                if reference.get('observation_id')!=self.depth_id:raise SkillError('Stale orientation reference')
                from directed_placement import observed_axis,endpoint_reference
                ends={}
                for key in ('from_part','to_part'):
                    ref=reference[key]
                    observed=endpoint_reference(self.cloud,self.valid,ref['bbox'],ref['point'],self.table)
                    if observed is None:raise SkillError('Orientation endpoint is not clearly observed; re-ground the named end')
                    if np.linalg.norm(np.asarray(observed['point'])[:2]-np.asarray(geometry['tcp'])[:2])>.25:
                        raise SkillError('Orientation endpoint too far from selected source; verify object identity')
                    ends[key]=observed
                try:axis=observed_axis(ends['from_part']['point'],ends['to_part']['point'])
                except ValueError as exc:raise SkillError(str(exc)) from exc
                geometry['directed_axis']=axis.tolist()
                geometry['orientation_endpoints']=ends
                geometry['desired_facing']=reference['facing']
            if 'contact_reference' in a:
                reference=a['contact_reference']
                if reference.get('camera','head_camera')!=a.get('camera','head_camera'):
                    raise SkillError('Pickup working reference must use the source camera')
                if reference.get('observation_id')!=self.depth_id:raise SkillError('Stale working-part reference')
                from contact_geometry import working_part_reference
                working=working_part_reference(self.cloud,self.valid,reference['bbox'],reference['point'],self.table)
                if working is None:raise SkillError('Working part is not clearly observed; re-ground the tool head/tip')
                if np.linalg.norm(np.asarray(working['point'])-geometry['tcp'])>.35:
                    raise SkillError('Working part too far from grasped component; check tool identity')
                geometry['working_part']=working
            return geometry
        if a['skill']=='place' and a.get('relation') in RELATIVE_DIRECTIONS:
            if not choices:raise SkillError('Reference object is not visible; re-ground before relative placement')
            pts=self.cloud[labels==min(choices)[1]]
            held=self.held.get(a['arm']) or {}
            point=relative_support(pts,held.get('span',[.05,.05]),a['relation'],self.table)
            self.relative_reference_points=pts.copy()
            if not (-.45<point[0]<.45 and -.38<point[1]<.35):raise SkillError('Relative destination lies beyond reachable table region')
            a['support']='table';a['align']=False
            return {'tcp':point,'top':point[2],'source':'visible_rgbd_spatial_relation','relation':a['relation']}
        if a['skill']=='place' and re.search(r'\b(block|cube)\b',description):
            # A named solid support cannot be flattened onto the tabletop by an
            # incorrect VLM support label. Measure its visible upper surface.
            if not choices:raise SkillError('Destination block is not visible; re-ground before stacking')
            pts=self.cloud[labels==min(choices)[1]]
            upper=pts[pts[:,2]>=np.quantile(pts[:,2],.8)]
            if len(upper)<4:raise SkillError('Insufficient visible support surface for stacking')
            centre=cv2.minAreaRect(upper[:,:2].astype(np.float32))[0]
            point=np.array([*centre,float(np.median(upper[:,2]))]);a['support']='object'
            return {'tcp':point.tolist(),'top':float(point[2]),'source':'visible_rgbd_solid_support','pixels':len(upper)}
        if a['skill']=='place' and a.get('support')=='table':
            origin=self.cam.get_model_matrix()[:3,3];ray=point-origin
            if abs(ray[2])<1e-6:raise SkillError('Invalid table ray')
            point=origin+ray*((self.table-origin[2])/ray[2])
        if not self.valid[pixel[1],pixel[0]]:raise SkillError('No depth at destination')
        if not (-.5<point[0]<.5 and -.4<point[1]<.45 and self.table-.03<point[2]<1.25):raise SkillError('Selected target outside table workspace')
        return {'tcp':point.tolist(),'top':float(point[2]),'bbox_px':[int(x0),int(y0),int(x1),int(y1)]}
    def _take(self,side,pose,grip,label,steps):
        if self.env.take_action_cnt>=self.env.step_lim:raise SkillError('Official action budget exhausted')
        if self.env.eval_success:return
        before=self.endpose();statuses={};stationary_holds=[];originals={s:getattr(self.env.robot,s+'_plan_path') for s in SIDES}
        for arm in SIDES:
            original=originals[arm]
            def capture(target,*args,original=original,arm=arm,**kwargs):
                value=None
                entity=getattr(self.env.robot,arm+'_entity');active=[j.name for j in entity.get_active_joints()]
                indices=[active.index(j.name) for j in getattr(self.env.robot,arm+'_arm_joints')]
                # Exact no-motion requests need no IK/trajectory optimization. Keep
                # the measured joints while the official loop drives the fingers.
                if np.allclose(target,before[arm+'_endpose'],atol=1e-6,rtol=0):
                    joints=np.asarray(entity.get_qpos())[indices][None,:].copy()
                    stationary_holds.append(arm)
                    statuses[arm]='Success'
                    return dict(status='Success',position=joints,velocity=np.zeros_like(joints))
                for cache_index,cached in enumerate(self.plan_cache):
                    if cached['arm']==arm and np.allclose(target,cached['pose'],atol=1e-6) and np.linalg.norm(entity.get_qpos()[indices]-cached['start_qpos'][indices])<.05:
                        value=cached['result'];self.plan_cache.pop(cache_index);break
                if value is None:value=original(target,*args,**kwargs)
                statuses[arm]=value.get('status');return value
            setattr(self.env.robot,arm+'_plan_path',capture)
        target=[]
        for arm in SIDES:target+=(list(pose) if arm==side else before[arm+'_endpose'])+[grip if arm==side else before[arm+'_gripper']]
        try:
            self.env.take_action(target,action_type='ee')
            # A zero-distance EE plan may have only one step; allow physical fingers to converge.
            if not self.env.eval_success:
                for _ in range(200 if label in ('close','release','open') else 40):
                    # Official set_gripper limits opening drive increments to 0.1.
                    # Reissue the target while settling; waiting alone leaves a
                    # one-step trajectory at only 10% open after a closed grasp.
                    for arm in SIDES:self.env.robot.set_gripper(float(grip if arm==side else before[arm+'_gripper']),arm)
                    self.env.robot._entity_qf(self.env.robot.left_entity)
                    if self.env.robot.right_entity is not self.env.robot.left_entity:self.env.robot._entity_qf(self.env.robot.right_entity)
                    self.env.scene.step();self.env._update_render()
                    if self.env.check_success():self.env.eval_success=True;break
        finally:
            for arm in SIDES:setattr(self.env.robot,arm+'_plan_path',originals[arm])
            self.depth_id=None
        after=self.endpose();error=float(np.linalg.norm(np.asarray(after[side+'_endpose'][:3])-pose[:3]));success=bool(self.env.eval_success or self.env.check_success())
        row={'name':label,'target_ee':target,'planner_status':statuses,'stationary_holds':stationary_holds,'position_error_m':error,'endpose_after':after,'motion_ok':statuses.get(side)=='Success' and error<.035,'success':success};steps.append(row)
        self.capture()
        if not row['motion_ok'] and not success and statuses.get(side)=='Success' and not label.endswith('_refine'):
            return self._take(side,pose,grip,label+'_refine',steps)
        if not row['motion_ok'] and not success:raise SkillError(f'{label}: planning/tracking failure {statuses}, error={error:.3f} m')
    def choose_grasp(self,a,g):
        planner=getattr(self.env.robot,a["arm"]+"_planner")
        planner.fast_preflight=True
        try:return self._choose_grasp(a,g)
        finally:planner.fast_preflight=False
    def _choose_grasp(self,a,g):
        side=a['arm'];tcp=np.asarray(g['tcp']);yaw=g['yaw'];candidates=[]
        from robot_table_clearance import RobotTableClearance,table_clear_grasp
        tools=getattr(self,'table_clearance_tools',{})
        if side not in tools:tools[side]=RobotTableClearance(self.env.robot,side,self.table)
        self.table_clearance_tools=tools;tool=tools[side]
        if tool.calibration_error()>.002:raise SkillError('Robot FK calibration cannot verify finger/table clearance')
        fingers=tool.finger_vertices_in_ee(self.endpose()[side+'_endpose'],component_opening(g))
        if fingers is None:raise SkillError('No calibrated robot finger collision geometry')
        collision=None
        if getattr(self,'screen_robot_mesh_collision',False):
            from robot_mesh_collision import ObservedRobotCollision
            collision=ObservedRobotCollision(self.env.robot,side,self.cloud,self.valid,self.table,opening=component_opening(g))
            g['observed_collision_screen']='current_rgbd_and_robot_collision_shapes_at_commanded_opening'
        elif getattr(self,'screen_observed_collision',False):
            from observed_collision import ObservedCollision
            collision=ObservedCollision(self.env.robot,side,self.cloud,self.valid,self.table,opening=component_opening(g))
            g['observed_collision_screen']='current_rgbd_and_robot_fk_at_commanded_opening'
        # Preferred robot yaw +/-30 degrees from forward; side approach points away from robot.
        # Measured height overrides an inappropriate VLM side-grasp request on a flat object.
        side_grasp=prefer_side_grasp(a,g,self.table)
        yaws=grasp_yaws(yaw,side_grasp)
        pitches=[0,math.pi/6,math.pi/3,math.pi/2] if side_grasp else [math.pi/2,math.pi/3,math.pi/4]
        planner=getattr(self.env.robot,side+'_plan_path')
        key=(side,a['target']);attempt=self.failed.get(key,0)%3
        for pitch in pitches:
            for angle in yaws:
                q=grasp_quat(angle,pitch);rot=quat2mat(q)
                cleared=table_clear_grasp(tcp,q,fingers,self.table)
                if cleared is None:continue
                contact_tcp,clearance=cleared
                pre=contact_tcp-rot[:,0]*.06
                grasp=ee_from_tcp(contact_tcp,q);approach=ee_from_tcp(pre,q)
                start_qpos=getattr(self.env.robot,side+'_entity').get_qpos().copy()
                result=planner(approach)
                candidates.append({'q':q,'approach_status':result.get('status')})
                print('GRASP_CANDIDATE',side,len(candidates),result.get('status'),flush=True)
                if result.get('status')!='Success':continue
                if collision is not None:
                    penetration=collision.penetration(result['position'])
                    candidates[-1]['observed_approach_penetration_m']=penetration
                    if penetration>getattr(collision,'tolerance_m',.008):continue
                # Preflight terminal grasp from the solved approach joint configuration.
                qpos=getattr(self.env.robot,side+'_entity').get_qpos().copy()
                joints=getattr(self.env.robot,side+'_arm_joints')
                active=getattr(self.env.robot,side+'_entity').get_active_joints();names=[j.name for j in active]
                for j,value in zip(joints,result['position'][-1]):qpos[names.index(j.name)]=value
                end=planner(grasp,last_qpos=qpos)
                candidates[-1]['grasp_status']=end.get('status')
                if end.get('status')=='Success':
                    if collision is not None:
                        penetration=collision.penetration(end['position'])
                        candidates[-1]['observed_contact_penetration_m']=penetration
                        if penetration>getattr(collision,'tolerance_m',.008):continue
                    if attempt>0:attempt-=1;continue
                    self.plan_cache=[{'arm':side,'pose':approach,'start_qpos':start_qpos,'result':result},
                                     {'arm':side,'pose':grasp,'start_qpos':qpos.copy(),'result':end}]
                    g['tcp']=contact_tcp.tolist();g['finger_table_clearance']=clearance
                    return approach,grasp,candidates
        raise SkillError('No reachable grasp orientation; change arm or contact location')
    def plan_pair(self,side,prepose,lowpose):
        planner=getattr(self.env.robot,side+'_plan_path');entity=getattr(self.env.robot,side+'_entity')
        start=entity.get_qpos().copy();pre=planner(prepose)
        self.last_pair_status={'approach':pre.get('status'),'contact':None}
        if pre.get('status')!='Success':return False
        qpos=start.copy();active=[j.name for j in entity.get_active_joints()]
        for j,value in zip(getattr(self.env.robot,side+'_arm_joints'),pre['position'][-1]):qpos[active.index(j.name)]=value
        low=planner(lowpose,last_qpos=qpos)
        self.last_pair_status['contact']=low.get('status')
        if low.get('status')!='Success':return False
        self.plan_cache=[dict(arm=side,pose=prepose,start_qpos=start,result=pre),dict(arm=side,pose=lowpose,start_qpos=qpos,result=low)]
        return True
    def calibrate_held(self,side,geometry):
        from visual_registration_v3 import register_grasp
        reference,colours=self.pick_reference;obs=self.observe()
        pose=obs['endpose'][side+'_endpose'];tcp=tcp_from_ee(pose)
        shift=tcp-np.array(geometry['tcp']);predicted=reference+shift
        lo,hi=predicted.min(0)-.035,predicted.max(0)+.035
        mask=self.valid&(self.cloud>lo).all(2)&(self.cloud<hi).all(2)&(self.cloud[:,:,2]>self.table+.03)
        fit=register_grasp(reference,colours,self.cloud[mask],self.rgb[mask],shift)
        info=self.held[side]
        if fit is None:
            info['visual_registration']={'accepted':False};return
        bottom=np.array([*geometry['tcp'][:2],self.table])
        measured_bottom=fit['rotation']@bottom+fit['translation']
        info.update(bottom_offset=(measured_bottom-tcp).tolist(),grasp_quat=pose[3:],object_rotation=fit['rotation'].tolist())
        if 'directed_axis' in info:
            info['directed_axis']=(fit['rotation']@np.asarray(geometry['directed_axis'])).tolist()
            footprint=np.c_[np.asarray(info['footprint_xy']),np.zeros(len(info['footprint_xy']))]
            info['footprint_xy']=(footprint@fit['rotation'].T)[:,:2].tolist()
        info['visual_registration']={k:v for k,v in fit.items() if k not in ('rotation','translation')}
        info['visual_registration']['accepted']=True
        geometry['post_grasp_registration']=info['visual_registration']
    def execute(self,a):
        steps=[];geometry=None;failure=None;self.index+=1;self.plan_cache=[]
        self.relative_reference_points=None
        try:
            validate(a);name=a['skill'];side=a.get('arm','right')
            if name=='place' and not self.sensors()[side]['holding']:
                raise SkillError('Place requires verified held object')
            if name in ('pick','grasp_handle','place','press','reach','arc'):geometry=self.geometry(a)
            if name=='handover':geometry=self.geometry({**a,'skill':'grasp_handle','grounding_role':'handover_receiver_contact'})
            if name=='push':geometry=self.geometry({**a,'skill':'pick'})
            if name in ('pick','grasp_handle'):
                resolved=resolve_grasp_arm(a,geometry,self.instruction,self.sensors())
                if resolved!=side:
                    a['requested_arm']=side
                    a['arm_resolution']='unconstrained_empty_arm_from_current_rgbd'
                    a['arm']=side=resolved
            before=self.endpose();pose=before[side+'_endpose'];grip=before[side+'_gripper']
            take=lambda p,g,label:self._take(side,p,g,label,steps)
            if name in ('pick','grasp_handle'):
                if name=='pick' and geometry.get('observed_flat_source') and a.get('placement_preview'):
                    from surface_frames import SurfaceFrameMemory
                    from robodawn.camera_views import camera_of
                    if not hasattr(self,'surface_frame_memory'):self.surface_frame_memory=SurfaceFrameMemory()
                    preview=a['placement_preview'];view=getattr(self,'view_data',{}).get(camera_of(preview))
                    if view is not None:
                        geometry['support_frame_preview_saved']=self.surface_frame_memory.remember(preview,view,self.table,self.depth_id)
                if self.sensors()[side]['holding']:raise SkillError('Already holding: place/present instead of picking again')
                pre,contact,candidates=self.choose_grasp(a,geometry);geometry['candidates']=candidates
                opening=component_opening(geometry);geometry['pregrasp_opening']=opening
                take(pose,opening,'open')
                if getattr(self,'stage_approach',True) and (geometry.get('tall') or
                        (geometry.get('source')=='visible_rgbd_local_component' and geometry['top']-self.table>.10)):
                    staged=False
                    for clearance in (.10,.06,.03):
                        high_tcp=tcp_from_ee(pre);high_tcp[2]=max(high_tcp[2],geometry['top']+clearance)
                        high=ee_from_tcp(high_tcp,pre[3:])
                        rise=pose.copy();rise[2]+=max(0,high_tcp[2]-tcp_from_ee(pose)[2])
                        if self.plan_pair(side,rise,high):
                            geometry['approach_clearance_m']=clearance
                            take(rise,opening,'clearance_rise');take(high,opening,'clearance_travel')
                            staged=True;break
                    geometry['staged_approach']=staged
                take(pre,opening,'approach')
                try:take(contact,opening,'contact')
                except SkillError:
                    if not getattr(self,'experimental_contact_stop',False):raise
                    from contact_geometry import measured_bar_contact_stop
                    actual=self.endpose()[side+'_endpose']
                    stopped=measured_bar_contact_stop(geometry,tcp_from_ee(actual),self.sensors()[side],
                                                       steps[-1] if steps else {},side)
                    if stopped is None:raise
                    geometry['measured_contact_stop']=stopped
                    contact=actual  # Stop at contact; do not command deeper penetration.
                take(contact,0,'close')
                sensing=self.sensors()[side]
                if not sensing['holding'] and not self.env.eval_success:raise SkillError('Empty grasp: no bilateral object contact')
                actual_pose=self.endpose()[side+'_endpose'];actual_tcp=tcp_from_ee(actual_pose)
                bottom=np.array([*geometry.get('base_xy',geometry['tcp'][:2]),self.table])
                self.held[side]={'target':a['target'],'grasp_tcp':geometry['tcp'],'height_under_tcp':max(.01,actual_tcp[2]-self.table),
                    'bottom_offset':(bottom-actual_tcp).tolist(),'grasp_quat':actual_pose[3:],'object_yaw':geometry['yaw'],'span':geometry['span']}
                if geometry.get('observed_flat_source'):self.held[side]['observed_flat_source']=geometry['observed_flat_source']
                if 'directed_axis' in geometry:
                    self.held[side].update(directed_axis=geometry['directed_axis'],desired_facing=geometry['desired_facing'],orientation_endpoints=geometry['orientation_endpoints'],
                        footprint_xy=(np.asarray(geometry['footprint_hull_xy'])-np.asarray(geometry.get('base_xy',geometry['tcp'][:2]))).tolist())
                if 'working_part' in geometry:
                    self.held[side]['working_offset']=(np.asarray(geometry['working_part']['point'])-actual_tcp).tolist()
                    self.held[side]['working_part']=geometry['working_part']
                if name=='grasp_handle':
                    if hasattr(self,'support_memory'):self.support_memory.forget(a.get('target',''))
                    return {'ok':True,'skill_success':True,'success':bool(self.env.eval_success or self.env.check_success()),'failure':None,'subactions':steps,'geometry':geometry,'sensors':self.sensors(),'terminal':False}
                origin=self.endpose()[side+'_endpose'].copy();lifted=None
                for distance in (.12,.08,.05):
                    candidate=origin.copy();candidate[2]+=distance
                    if getattr(self.env.robot,side+'_plan_path')(candidate).get('status')=='Success':lifted=candidate;break
                if lifted is None:raise SkillError('Holding object but lift unreachable; move toward robot then raise')
                take(lifted,0,'lift')
                if not self.sensors()[side]['holding'] and not self.env.eval_success:raise SkillError('Grasp lost during lift')
                if not self.env.eval_success and os.environ.get('ROBOTWIN_EXPERIMENTAL_ICP')=='1':self.calibrate_held(side,geometry)
            elif name=='inspect':
                from active_inspection import execute_inspection
                geometry=execute_inspection(self,a,steps)
                self.inspection_cameras=[geometry['camera']]+[c for c in getattr(self,'inspection_cameras',[]) if c!=geometry['camera']]
            elif name=='handover':
                from handover import execute_handover
                execute_handover(self,a,geometry,steps)
            elif name=='place':
                if not self.sensors()[side]['holding']:raise SkillError('Place requires verified held object')
                camera=a.get('camera','head_camera')
                observer=camera.removesuffix('_camera')
                if (geometry.get('release_portal') and observer in SIDES and observer!=side
                        and camera in getattr(self,'inspection_cameras',[])
                        and not self.sensors()[observer]['holding']):
                    self._take(observer,self.initial[observer+'_endpose'],1,'clear_observer',steps)
                    geometry['observer_clearance']=dict(arm=observer,source='empty_camera_arm_home_after_current_target_measurement')
                    if not self.sensors()[side]['holding']:raise SkillError('Held object lost while clearing observer')
                info=self.held[side] or {'height_under_tcp':.025};support=np.asarray(geometry['tcp']);q=pose[3:]
                facing=a.get('facing',info.get('desired_facing'))
                if facing:
                    if 'directed_axis' not in info:raise SkillError('Directed placement needs observed source endpoints; reobserve/regrasp with an orientation reference')
                    from directed_placement import align_directed_axis
                    try:aligned,turn=align_directed_axis(info['directed_axis'],quat2mat(info['grasp_quat']),quat2mat(q),facing)
                    except ValueError as exc:raise SkillError(str(exc)) from exc
                    q=mat2quat(aligned).tolist();a['facing']=facing
                    geometry['directed_placement']={'facing':facing,'yaw_correction_deg':math.degrees(turn),'source':'observed_endpoints_and_proprioception'}
                if geometry.get('release_clearance') and a.get('release',True):
                    support=support.copy();support[2]=geometry['release_clearance']['release_height_m']
                offset=info.get('bottom_offset');clearance=.005
                if a.get('use_contact_part'):
                    if a.get('release',True):raise SkillError('Tool contact must retain the object')
                    if 'working_offset' not in info and a.get('contact_reference'):
                        reference=a['contact_reference']
                        if reference.get('observation_id')!=self.depth_id:raise SkillError('Stale retained working-part reference')
                        from contact_geometry import working_part_reference,retained_working_offset
                        from robodawn.camera_views import camera_of
                        view=getattr(self,'view_data',{}).get(camera_of(reference))
                        if camera_of(reference)!='head_camera' and view is None:raise SkillError('No current RGB-D for working-reference camera')
                        cloud=self.cloud if view is None else view['cloud'];valid=self.valid if view is None else view['valid']
                        working=working_part_reference(cloud,valid,reference['bbox'],reference['point'],self.table)
                        if working is None:raise SkillError('Retained working part is not currently visible; change view and reobserve')
                        if 'grasp_quat' not in info:raise SkillError('Retained working part needs a verified remembered grasp frame')
                        try:offset=retained_working_offset(working['point'],tcp_from_ee(pose),quat2mat(info['grasp_quat']),quat2mat(pose[3:]))
                        except ValueError as exc:raise SkillError(str(exc)) from exc
                        info['working_offset']=offset.tolist();info['working_part']=working
                        geometry['retained_working_reference']=dict(working,observation_id=self.depth_id,
                                                                  source='current_rgbd_part_registered_to_existing_grasp')
                    if 'working_offset' not in info:raise SkillError('Held tool lacks an observed working-part reference; reobserve/regrasp before contact')
                    offset=info['working_offset'];clearance=-.004
                    geometry['contact_reference']=info['working_part']
                if 'bottom_offset' in info:
                    initial=quat2mat(info['grasp_quat']);now=quat2mat(q)
                    if not facing and a.get('support')=='table' and a.get('align',True) and (any(w in a.get('target','').lower() for w in ('pad','mat')) or a.get('colour_refinement')):
                        object_rotation=now@initial.T@np.array(info.get('object_rotation',np.eye(3)))
                        object_axis=object_rotation@np.array([math.cos(info['object_yaw']),math.sin(info['object_yaw']),0.])
                        delta=math.radians(float(a.get('align_yaw_deg',0)))-math.atan2(object_axis[1],object_axis[0])
                        delta=(delta+math.pi/2)%math.pi-math.pi/2
                        if 'object_rotation' in info:
                            desired_yaw=math.atan2(object_axis[1],object_axis[0])+delta
                            desired_rotation=euler2mat(0,0,desired_yaw-info['object_yaw'])
                            now=desired_rotation@object_rotation.T@now
                        else:now=euler2mat(0,0,delta)@now
                        q=mat2quat(now).tolist()
                    from contact_geometry import transformed_contact_tcp
                    tcp=transformed_contact_tcp(support,offset,initial,now,clearance)
                else:tcp=support+[0,0,info['height_under_tcp']+.008]
                above=tcp.copy();above[2]+=.08
                hand_clearance=None
                if a.get('relation') in RELATIVE_DIRECTIONS and a.get('release',True):
                    from placement_hand_clearance import PlacementHandClearance
                    hand_clearance=PlacementHandClearance(self.env.robot,side,pose,self.relative_reference_points)
                # Reuse a feasible pair of approach/contact trajectories rather than re-solving randomly.
                planner=getattr(self.env.robot,side+'_planner');planner.fast_preflight=True
                chosen=None
                geometry['placement_candidates']=[]
                frame=geometry.get('support_frame')
                rotations=placement_rotations(q,a)
                if geometry.get('release_portal'):
                    # The release-column footprint was measured upright.
                    # Do not tilt a tall payload into an unverified wider sweep.
                    rotations=(candidate for candidate in rotations if candidate[1]==0)
                if frame:
                    from surface_frames import aligned_rotations
                    rotations=((0.,0.,rotation) for rotation in aligned_rotations(info,frame))
                try:
                    for angle,tilt,rotated in rotations:
                        candidate_q=mat2quat(rotated).tolist()
                        candidate_tcp=transformed_contact_tcp(support,offset,quat2mat(info['grasp_quat']),rotated,clearance) if offset is not None else tcp
                        # A tilted base has a lower corner. Keep it above the
                        # support before release using the observed object span.
                        candidate_tcp=np.asarray(candidate_tcp).copy()
                        if frame:
                            candidate_tcp=transformed_contact_tcp(support,offset,quat2mat(info['grasp_quat']),rotated,0)
                            candidate_tcp+=np.asarray(frame['normal'])*clearance
                        candidate_tcp[2]+=.5*max(info.get('span',[0]))*abs(math.sin(math.radians(tilt)))
                        for hover in (.08,.04,.02):
                            candidate_above=candidate_tcp+(np.asarray(frame['normal'])*hover if frame else [0,0,hover])
                            prepose,lowpose=ee_from_tcp(candidate_above,candidate_q),ee_from_tcp(candidate_tcp,candidate_q)
                            if hand_clearance is not None:
                                screen=hand_clearance.screen(prepose,lowpose)
                                if not screen['accepted']:
                                    geometry['placement_candidates'].append(dict(yaw_delta_deg=math.degrees(angle),
                                        tilt_delta_deg=tilt,hover_m=hover,reachable=False,tcp=candidate_tcp.tolist(),
                                        hand_clearance=screen,rejected_before_planning=True))
                                    continue
                            reachable=self.plan_pair(side,prepose,lowpose)
                            geometry['placement_candidates'].append({'yaw_delta_deg':math.degrees(angle),'tilt_delta_deg':tilt,'hover_m':hover,'reachable':reachable,'tcp':np.asarray(candidate_tcp).tolist(),'pair_status':dict(self.last_pair_status)})
                            if hand_clearance is not None:geometry['placement_candidates'][-1]['hand_clearance']=screen
                            if reachable:chosen=(prepose,lowpose);break
                        if chosen:break
                finally:planner.fast_preflight=False
                if chosen is None:raise SkillError('No reachable placement orientation; choose a closer destination or adjust held pose')
                prepose,lowpose=chosen
                # Any attempted interaction can move the support. A failed
                # contact must not keep using its earlier rigid-pose memory.
                if hasattr(self,'surface_frame_memory'):self.surface_frame_memory.forget(a['target'])
                take(prepose,grip,'transfer')
                if not self.sensors()[side]['holding'] and not self.env.eval_success:
                    self.held[side]=None
                    raise SkillError('Object lost during transfer; locate it again before placing')
                take(lowpose,grip,'lower')
                if a.get('release',True):
                    release_pose=lowpose
                    if not self.env.eval_success:
                        from measured_release import measured_release_pose
                        release_pose,evidence=measured_release_pose(self.endpose(),side,lowpose)
                        geometry['release_pose_control']=evidence
                    take(release_pose,1,'release')
                    self.held[side]=None;take(prepose,1,'retreat')
            elif name in ('press','reach'):
                point=np.asarray(geometry['tcp']);hover=point+[0,0,.07]
                planner=getattr(self.env.robot,side+'_planner');planner.fast_preflight=True;chosen=None
                try:
                    for pitch in (math.pi/2,math.pi/3,math.pi/4):
                        for yaw in (math.pi/2,0,math.pi,math.pi/3,2*math.pi/3):
                            q=grasp_quat(yaw,pitch)
                            prepose=ee_from_tcp(hover,q)
                            # REACH promises a hover, not PRESS with an open grip.
                            # Do not require or execute a below-surface contact path.
                            lowpose=ee_from_tcp(point+[0,0,-.012],q) if name=='press' else prepose
                            if self.plan_pair(side,prepose,lowpose):chosen=(prepose,lowpose);break
                        if chosen:break
                finally:planner.fast_preflight=False
                if chosen is None:raise SkillError('No reachable contact orientation; change arm or contact point')
                prepose,lowpose=chosen
                take(prepose,0 if name=='press' else grip,'hover')
                if name=='press':take(lowpose,0,name)
            elif name=='push':
                if self.sensors()[side]['holding']:raise SkillError('PUSH needs an empty gripper')
                if (geometry['top']-self.table)/max(max(geometry['span']),.01)>2.2:raise SkillError('Slender object should be grasped, not pushed')
                destination={**a['destination'],'skill':'place','arm':side,'observation_id':a['observation_id']}
                goal=self.geometry(destination);centre=np.asarray(geometry['tcp']);end=np.asarray(goal['tcp'])
                direction=end[:2]-centre[:2];distance=np.linalg.norm(direction)
                if not .015<distance<.45:raise SkillError('PUSH distance must be 1.5–45 cm')
                direction/=distance;radius=float(np.dot(np.abs(direction),np.asarray(geometry['span']))*.5)
                # The push contact is also made by the closed fingers.  A nominal
                # TCP height can therefore put the actual fingertips below the
                # support plane, just as it can for a low grasp.  Compensate from
                # robot-only FK/collision geometry while preserving the horizontal
                # push path and all contact/success checks.
                raw_start=centre.copy();raw_start[:2]-=direction*(radius+.02)
                raw_start[2]=self.table+min(.03,max(.012,(geometry['top']-self.table)*.6))
                from robot_table_clearance import RobotTableClearance,table_clear_grasp
                tools=getattr(self,'table_clearance_tools',{})
                if side not in tools:tools[side]=RobotTableClearance(self.env.robot,side,self.table)
                self.table_clearance_tools=tools;tool=tools[side]
                if tool.calibration_error()>.002:
                    raise SkillError('Robot FK calibration cannot verify push finger/table clearance')
                fingers=tool.finger_vertices_in_ee(self.endpose()[side+'_endpose'],0.)
                if fingers is None:raise SkillError('No calibrated robot finger collision geometry')
                chosen=None;planner=getattr(self.env.robot,side+'_planner');planner.fast_preflight=True
                geometry['push_candidates']=[]
                try:
                    for yaw in (math.pi/2,0,math.pi,math.pi/3,2*math.pi/3):
                        q=grasp_quat(yaw)
                        cleared=table_clear_grasp(raw_start,q,fingers,self.table)
                        candidate=dict(quaternion=list(q),clearance_verified=cleared is not None,reachable=False)
                        geometry['push_candidates'].append(candidate)
                        if cleared is None:continue
                        start,clearance=cleared
                        candidate['finger_table_clearance']=clearance
                        pre=ee_from_tcp(start+[0,0,.08],q);contact=ee_from_tcp(start,q)
                        candidate['reachable']=self.plan_pair(side,pre,contact)
                        if candidate['reachable']:
                            chosen=(q,pre,contact,start,clearance);break
                finally:planner.fast_preflight=False
                if chosen is None:raise SkillError('No reachable pushing contact; use other arm or pick/place')
                q,pre,contact,start,clearance=chosen
                finish=start.copy();finish[:2]=end[:2]-direction*max(.005,radius-.005)
                geometry['finger_table_clearance']=clearance
                take(pre,0,'push_hover');take(contact,0,'push_contact');touched=False
                for alpha in np.linspace(0,1,int(np.ceil(np.linalg.norm(finish-start)/.02))+1)[1:]:
                    take(ee_from_tcp(start+alpha*(finish-start),q),0,'push_slide')
                    touched |= self.sensors()[side]['contact_fingers']>0
                    if self.env.eval_success:break
                take(ee_from_tcp(finish+[0,0,.08],q),1,'release')
                geometry['destination']=goal;geometry['measured_contact']=bool(touched)
                if not touched and not self.env.eval_success:raise SkillError('Push path completed without measured object contact')
            elif name=='home':
                if self.sensors()[side]['holding']:raise SkillError('HOME is for an empty arm; use PRESENT when holding')
                take(self.initial[side+'_endpose'],1,'home');self.held[side]=None
                if hasattr(self,'inspection_cameras'):
                    self.inspection_cameras=[camera for camera in self.inspection_cameras if camera!=side+'_camera']
            elif name=='present':
                if not self.sensors()[side]['holding']:raise SkillError('PRESENT requires held object')
                lateral=.24 if a.get('location')=='side' else .075
                tcp=[-lateral if side=='left' else lateral,-.105,1.0]
                target=ee_from_tcp(tcp,pose[3:])
                if getattr(self.env.robot,side+'_plan_path')(target).get('status')!='Success':target=ee_from_tcp(tcp,grasp_quat(math.pi/2,0))
                take(target,grip,'present')
            elif name=='move':
                if a.get('support_rendezvous') and a.get('cooperative_destination'):
                    from cooperative_transport import choose_support_translation
                    delta,geometry=choose_support_translation(self,a)
                    a['delta']=delta.tolist()
                target=pose.copy();target[:3]=(np.asarray(pose[:3])+a['delta']).tolist();take(target,grip,'move')
            elif name=='dual_move':
                from dual_motion import translate_pair
                translate_pair(self,a['delta'],steps)
            elif name=='rotate':
                angles=[0.,0.,0.];angles['xyz'.index(a['axis'])]=math.radians(float(a['angle']))
                q=mat2quat(euler2mat(*angles)@quat2mat(pose[3:]));take(ee_from_tcp(tcp_from_ee(pose),q),grip,'rotate')
            elif name=='arc':
                if not self.sensors()[side]['holding']:raise SkillError('ARC requires a grasped handle')
                pivot=np.asarray(geometry['tcp']);origin=tcp_from_ee(pose)
                if np.linalg.norm(origin-pivot)>.6:raise SkillError('Hinge too far from grasped handle')
                axis=np.asarray(geometry['hinge_axis']) if 'hinge_axis' in geometry else np.eye(3)['xyz'.index(a['axis'])]
                for degrees in np.linspace(0,float(a['angle']),max(2,int(abs(float(a['angle']))/10)+1))[1:]:
                    rotation=axangle2mat(axis,math.radians(degrees))
                    tcp=pivot+rotation@(origin-pivot);q=mat2quat(rotation@quat2mat(pose[3:]))
                    take(ee_from_tcp(tcp,q),grip,'hinge_arc')
            elif name=='shake':
                amplitude=float(a.get('amplitude',.04));cycles=int(a.get('cycles',3))
                if not .02<=amplitude<=.06 or not 1<=cycles<=4:raise SkillError('Invalid shake parameters')
                if not self.sensors()[side]['holding']:raise SkillError('SHAKE requires held object')
                for _ in range(cycles):
                    for sign in (1,-1):
                        target=pose.copy();target['xyz'.index(a['axis'])]+=sign*amplitude;take(target,grip,'shake')
            elif name in ('open','close'):
                take(pose,1 if name=='open' else 0,name)
                if name=='open':self.held[side]=None
                elif not self.sensors()[side]['holding']:raise SkillError('Empty grasp')
            elif name=='wait':take(pose,grip,'wait')
            elif name=='done':pass
        except SkillError as exc:
            failure=str(exc)
            if a.get('skill') in ('pick','grasp_handle'):
                key=(a.get('arm'),a.get('target'));self.failed[key]=self.failed.get(key,0)+1
        success=bool(self.env.eval_success or self.env.check_success())
        if steps and hasattr(self,'support_memory') and a.get('skill') in ('pick','grasp_handle','place','push','press','arc'):
            self.support_memory.forget(a.get('target',''))
        return {'ok':failure is None,'skill_success':failure is None,'success':success,'failure':failure,'subactions':steps,'geometry':geometry,'sensors':self.sensors(),'terminal':a.get('skill')=='done'}
    def close(self):
        self.env._update_render=self.original_update
        self.env.get_obs=self.original_get_obs
        if self.video:
            self.capture();self.video.stdin.close();code=self.video.wait(timeout=30);self.video=None
            if code:raise RuntimeError('ffmpeg failed')

def ground_visual(client,obs,action,instruction,keep_arm=False):
    intent={k:v for k,v in action.items() if k!='destination'} if action['skill']=='push' else action
    role='SOURCE OBJECT (not the destination)' if action['skill']=='push' else 'DESTINATION SUPPORT' if action['skill']=='place' else 'TARGET CONTACT'
    prompt=('This is ONLY the HEAD camera. All coordinates refer to this single image. '
            'Skill: '+action['skill']+'. Find the '+role+' named by the target STRING in this intent: '+json.dumps(intent)+'. Task: '+instruction+'. '
            'Return JSON with bbox [xmin,ymin,xmax,ymax] and point [x,y], normalized 0..1000, '
            'arm left/right (same image side as the target unless task explicitly specifies otherwise), '
            'approach top/side, support table/object/container. Preserve the provided target identity. For PLACE locate the destination surface, not the held object. A block/cube is support object, NEVER table. Bounding box tightly surrounds the whole graspable object body, '
            'excluding shadows/table/robot and background. For grasp_part handle/rim/edge, bbox covers that graspable component only. Preserve a hinge pivot for ARC. '
            'Output format example: {"bbox":[100,200,300,400],"point":[200,300],"arm":"left","approach":"top","support":"table"}. '
            'support must be exactly table, object or container; flat coloured pads/mats are table. approach must be top for short objects or side for upright tall objects. No prose.')
    raw=client.complete_text(prompt,np.asarray(Image.open(obs['image_paths'][0])),max_tokens=250,temperature=0).raw_text
    g=parse_json(raw);result=dict(action)
    for key in ('bbox','point','approach'):result[key]=g[key]
    result['support']=g.get('support','object')
    # The grounder cannot rename the planner's target.
    if np.linalg.norm(np.array(result['point'])-np.array(action['point']))>150:
        result['grounding_disagreement']=True
        result['bbox']=action['bbox'];result['point']=action['point']
    if not keep_arm:result['arm']=g['arm']
    result['grounding_raw']=raw
    result=refine_colour(obs,result)
    return validate(result)


def refine_colour(obs,action):
    """RGB-only colour centring for named blocks and pads; no segmentation IDs."""
    description=action['target'].lower()
    if action['skill'] not in ('place','press') and not any(word in description for word in ('block','pad','mat','square')):return action
    colours={'red':(0,10),'green':(35,85),'blue':(90,130),'cyan':(80,100),'yellow':(20,35),'purple':(130,165)}
    match=next((c for c in colours if re.search(r'\b'+c+r'\b',description)),None)
    if match is None:return action
    rgb=np.asarray(Image.open(obs['image_paths'][0]).convert('RGB'));hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    lo,hi=colours[match];h,s,v=cv2.split(hsv)
    mask=(((h>=lo)&(h<=hi))|((h>=170) if match=='red' else False))&(s>90)&(v>60)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    choices=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=20]
    if not choices:return action
    height,width=rgb.shape[:2];scale=np.array([width-1,height-1])/1000
    initial=np.array(action['point'])*scale
    i=min(choices,key=lambda k:np.linalg.norm(centres[k]-initial)-min(stats[k,cv2.CC_STAT_AREA],1000)/100.)
    x,y,w,h,_=map(int,stats[i]);u,v=centres[i]
    action=dict(action);action['model_bbox']=action['bbox'];action['model_point']=action['point']
    action.update(bbox=[x/scale[0],y/scale[1],min(x+w-1,width-1)/scale[0],min(y+h-1,height-1)/scale[1]],point=[u/scale[0],v/scale[1]],colour_refinement=match)
    return action


def __getattr__(name):
    """Resolve legacy policy imports without coupling shared code to a policy."""
    from importlib import import_module
    exports = {'RobodawnPlanner': 'robodawn.planner_v3'}
    if name in exports:
        return getattr(import_module(exports[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
