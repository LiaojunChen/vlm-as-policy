"""Semantic decisions and independent, current-frame visual grounding."""
import json
import numpy as np
from PIL import Image
from robotwin_harness_v3 import SKILLS, SkillError, validate, parse_json, refine_colour
from robodawn.schemas import ACTION_SCHEMA,GROUND_SCHEMA,HINGE_GROUND_SCHEMA

GROUNDED = ('pick', 'grasp_handle', 'handover', 'push', 'place', 'press', 'reach', 'arc')
RULES = '''Control the two Aloha arms using ONE next skill. The image is the head camera.
World axes: +x image right, +y away from the robot, +z up. Distances are metres.
Choose the arm on the object's side unless the instruction specifies an arm.
Skills and parameters:
pick: arm, target (visible object/component), grasp_part body/handle/rim/edge, approach top/side.
  Opens, grasps, verifies contact and lifts. Use top for short objects; rim/handle for bowls or pots.
grasp_handle: same fields as pick; closes WITHOUT lifting, for articulated handles and lids.
handover: arm (EMPTY receiving hand), donor (OTHER hand already holding target), target, grasp_part.
  Grasp a current visible free part of the donor's object; release donor only after shared contact verification.
  First PRESENT the donor at centre if the object is out of receiver reach. Never PLACE onto another hand.
place: arm, target (DESTINATION ONLY), support table/object/container, release true/false.
  Transfers the held object's base onto the support; normally release=true. For stamping/using a held
  tool against a surface, release=false retains it. Never name the held object as its own destination.
push: arm, target (source), destination {target, support}. Slides along the table.
press: arm, target (button/lever contact). For an EMPTY gripper only.
reach: arm, target. Moves above a visible target without grasping.
move: arm, delta [dx,dy,dz], vector length at most 0.2. Preserves orientation and grip.
dual_move: delta [dx,dy,dz], length at most 0.2. BOTH arms translate together while preserving grip; requires BOTH to hold.
rotate: arm, axis x/y/z, angle signed degrees up to 90, about the gripper.
arc: arm, target (HINGE), axis x/y/z, angle signed degrees up to 90; requires a held handle.
present: arm, location centre/side. Holds up an object at front centre or the specified arm's side.
shake: arm, axis x/y/z, amplitude 0.02..0.06, cycles 1..4; requires holding.
open, close, home: arm. HOME is for an EMPTY arm. wait: no parameters.
Use sensor holding, not command gripper state, to decide which arm carries an object.
After placement, home the empty arm if it occludes or obstructs further manipulation.
Finish all objects and all stages in the task. Repeating an unchanged failed action is not recovery:
change the actual contact/approach, move a held object to a reachable area, or home an empty arm.
Motion success does not prove task success. If a previous successful motion did not finish the task,
inspect the resulting arrangement and choose the missing correction.
Return one JSON object with skill and its parameters, plus a short reason. No bbox/point is needed:
a separate visual grounder locates the target in the CURRENT image. Do not invent coordinates.
All parameters are top-level siblings of skill, e.g.
{"skill":"pick","arm":"left","target":"named visible object","grasp_part":"body","approach":"top"}.
'''


def normalize_action(action):
    """Accept equivalent flat/nested JSON without choosing or changing actions."""
    action = dict(action)
    for key in ('parameters', 'params'):
        parameters = action.pop(key, None)
        if parameters is None:
            continue
        if not isinstance(parameters, dict):
            raise SkillError(key + ' must be an object')
        for name, value in parameters.items():
            if name in action and action[name] != value:
                raise SkillError('Conflicting nested and top-level ' + name)
            action[name] = value
    if isinstance(action.get('location'),str):
        location=action['location'].strip().lower().replace('_',' ').replace('-',' ')
        if location in ('center','centre','front center','front centre'):
            action['location']='centre'
    return action


def history_summary(history):
    rows = []
    for row in history:
        a, r = row['action'], row['result']
        item = {k: a[k] for k in ('skill', 'arm', 'target', 'delta', 'axis', 'angle', 'location', 'release') if k in a}
        item.update(ok=r.get('skill_success'), failure=r.get('failure'))
        if r.get('geometry'):
            item['measured_target_xyz'] = r['geometry'].get('tcp')
        if rows and {k: v for k, v in rows[-1].items() if k != 'repeats'} == item:
            rows[-1]['repeats'] = rows[-1].get('repeats', 1) + 1
        else:
            rows.append(item)
    return rows


def policy_sensor_view(sensors):
    """Preserve established contact inputs, keep normal audit details in logs."""
    return {arm: {k: v for k, v in state.items()
                  if k not in ('opposed_contact', 'finger_normal_projection_range')}
            for arm, state in sensors.items()}


def ground(client, obs, action, image_region=None, memory=None, distinct_from=None, _allow_memory_search=True):
    from robodawn.camera_views import candidate_views,in_camera
    from robodawn.grounding_evidence import admit_wrist_contact
    # A component crop belongs to one image, never to an alternate camera.
    if image_region is not None and 'camera' not in action:
        action=dict(action,camera=obs.get('camera','head_camera'))
    errors=[]
    for camera in candidate_views(obs,action):
        view=in_camera(obs,camera)
        other=distinct_from if distinct_from and distinct_from.get('camera','head_camera')==camera else None
        try:
            grounded=_ground_one(client,view,dict(action,camera=camera),image_region,memory,other,_allow_memory_search)
            try:evidence=admit_wrist_contact(view,grounded,include_head=True)
            except SkillError as rejected:
                if camera!='head_camera':raise
                # One corrective query in the SAME image/crop. Measured depth
                # only rejects impossible contact background; the model still
                # selects identity, component and coordinates.
                grounded=_ground_one(client,view,dict(action,camera=camera),image_region,memory,other,
                                     False,_depth_contact_repair=True)
                evidence=admit_wrist_contact(view,grounded,include_head=True)
                grounded['depth_contact_repair']=str(rejected)
            if evidence is not None:grounded['contact_view_evidence']=evidence
            if grounded['skill']=='arc':
                # Endpoint order is a visual-output convention, not authority
                # to reverse the model's signed motion about its named axis.
                if grounded.get('axis') not in ('x','y','z'):
                    raise SkillError('A grounded hinge motion needs its declared axis and signed angle')
                declared=np.eye(3)['xyz'.index(grounded['axis'])]
                alignment=float(np.asarray(evidence['axis'])@declared)
                if abs(alignment)<.25:
                    raise SkillError('Visible hinge axis is inconsistent with the declared motion axis; replan its axis and signed angle')
                if alignment<0:
                    grounded['hinge_line']=list(reversed(grounded['hinge_line']))
                    evidence['raw_endpoint_order_axis']=list(evidence['axis'])
                    evidence['axis']=(-np.asarray(evidence['axis'])).tolist()
                    evidence['endpoints_world']=list(reversed(evidence['endpoints_world']))
                grounded['hinge_axis_orientation']='aligned_to_declared_positive_world_axis'
            if errors:grounded['rejected_grounding_views']=list(errors)
            if memory is not None:memory.remember_visual(grounded,obs['observation_id'])
            return grounded
        except SkillError as exc:
            errors.append(camera+': '+str(exc))
    raise SkillError('; '.join(errors))


def _ground_one(client, obs, action, image_region=None, memory=None, distinct_from=None, _allow_memory_search=True,
                _depth_contact_repair=False):
    memory_search=None
    from robodawn.grounding_evidence import held_destination_arm
    support_holder=held_destination_arm(obs,action)
    if (image_region is None and memory is not None and _allow_memory_search
            and action.get('grounding_role')!='orientation_endpoint'):
        if support_holder:
            memory_search=memory.held_search.search(obs,action['target'],action['camera'])
        else:memory_search=memory.search_region(action['target'],obs['observation_id'],camera=action['camera'])
        if memory_search is not None:image_region=memory_search['bbox']
    role = 'destination support surface/interior' if action['skill'] == 'place' else 'hinge pivot' if action['skill'] == 'arc' else 'graspable/contact component'
    from robotwin_harness_v3 import RELATIVE_DIRECTIONS
    spatial_reference=action['skill']=='place' and action.get('relation') in RELATIVE_DIRECTIONS
    if spatial_reference:role='WHOLE spatial reference object, including all visible parts (NOT the empty space beside it)'
    if action.get('grounding_role') in ('entity_identity','collection_source_identity'):role='WHOLE physical object, including all of its visible parts'
    if action.get('grounding_role')=='handover_receiver_contact':
        role='currently visible FREE graspable part of the object held by the '+action['donor']+' hand, separated from its fingers; NOT either robot hand'
    prompt = (f'Locate the {role}: {action["target"]}. '
              f'Action={action["skill"]}, component={action.get("grasp_part", "body")}. '
              f'Use only this current {action["camera"].replace("_", "-")} image. Give a tight visible bbox and a point on that component. '
              'Coordinates are normalized 0..1000 along image width and height. '
              'Return JSON: bbox (four numbers xmin,ymin,xmax,ymax), point (two numbers x,y), '
              'approach (top or side), support (table, object or container). '
              'For a container placement select inside its opening, not the handle or wall. '
              'For a handle/rim grasp bound the thin component, not the entire vessel. '
              'Use top approach for a low tool handle lying on the table; side is for upright/elevated components. '
              'If the named target is absent or entirely occluded return {"visible":false}. No prose.')
    if action['skill']=='arc':
        held=obs.get('sensors',{}).get(action.get('arm'),{}).get('remembered_object')
        prompt+=('\nHINGE GEOMETRY: additionally return hinge_line=[[x1,y1],[x2,y2]], two distinct '
                 'currently visible solid points along the SAME rotation seam between the moving part and its base. '
                 'Both endpoints must lie in the bbox. Do not select the free edge, tabletop or robot. '
                 'A single guessed pivot cannot define the motion. Return visible=false if two points along '
                 'the actual joint cannot be seen; never infer them through an occluding gripper.')
        if held:prompt+=' The moving component currently held by the '+action['arm']+' hand is '+held+'.'
    if spatial_reference:
        prompt+=('\nSPATIAL REFERENCE: the held source will be placed '+action['relation']+' this object. '
                 'Bound the reference object itself and put the point on its visible body. Do not offset the box/point '
                 'toward the desired empty destination; the executor computes that relation from current RGB-D extents.')
    if memory is not None and action.get('grounding_role')!='orientation_endpoint':
        # Geometry must be measured in this image. Past numeric boxes are used
        # only to choose a search window, never offered as answers to copy.
        prompt += memory.prompt(action['target'],include_coordinates=False)
    if support_holder:
        prompt+=('\nCONTACT STATE CONSTRAINT: this destination is currently held by the '+support_holder+
                 ' hand. Locate its CURRENT visible supporting surface/interior, not the table beneath or behind it. '
                 'Its support is object or container, not table. Do not copy its old table location; return visible=false if hidden.')
    if distinct_from:
        subject='collection source' if action.get('grounding_role')=='collection_source_identity' else 'destination'
        prompt += ('\nIDENTITY CONSTRAINT: this '+subject+' must be a DIFFERENT object from '
                   + json.dumps({k: distinct_from[k] for k in ('target', 'bbox', 'point')})
                   + '. That source was localized in this SAME image. Locate the other requested object, not the source.')
        if action.get('grounding_role')=='collection_source_identity':
            prompt+=' These are differently named sources in one collection transfer. Do not count the already bound source again under another name; return visible=false if the other object cannot be seen.'
    image = np.asarray(Image.open(obs['image_paths'][0]))
    from robodawn.grounding_evidence import current_robot_mask,robot_at_point,robot_hidden_image
    robot_mask=current_robot_mask(obs,image.shape);robot_repair=False
    original_image=image.copy();appearance_repair=None;reacquisition_evidence=None
    if _depth_contact_repair:
        from robodawn.grounding_evidence import current_contact_mask
        foreground=current_contact_mask(obs)
        if foreground is None or not foreground.any():raise SkillError('No current above-table contact surface; change view')
        if foreground.shape!=image.shape[:2]:raise SkillError('Contact mask and current image calibration disagree')
        image=image.copy();image[~foreground]=[150,150,150]
        prompt+=('\nThe previous selected surface contradicted current depth/contact evidence. In this SAME current image, '
                 'table/background, invalid depth and robot pixels are now GREY; valid above-table object surfaces retain their original RGB. '
                 'Locate the requested component/support on the remaining observed object, not on grey background. '
                 'Keep the exact requested identity; return visible=false if its component is absent.')
    crop=None;other_point=None
    if image_region is not None:
        height,width=image.shape[:2]
        parent=np.asarray(image_region,dtype=float)*np.array([width-1,height-1,width-1,height-1])/1000
        margin=np.maximum((parent[2:]-parent[:2])*.35,12)
        lo=np.maximum(np.floor(parent[:2]-margin),0).astype(int)
        hi=np.minimum(np.ceil(parent[2:]+margin),[width-1,height-1]).astype(int)
        if np.any(hi<=lo):raise SkillError('Invalid observed component crop')
        crop=(lo,hi,width,height)
        image=image[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        if memory_search is not None and support_holder:
            prompt+=(' This is a crop of the CURRENT image around foreground near the destination\'s uncertain '
                     'gripper-motion prediction from its observed pregrasp extent. It is ONLY a search hint, not a known object pose. '
                     'Identify the named destination using its actual visible appearance, and locate its supporting surface/interior, '
                     'not the handle or gripper. Return visible=false if absent; the full current scene will then be searched. '
                     'All returned coordinates refer to THIS crop.')
            if memory_search.get('support_plane'):
                prompt+=' Current depth shows an upward broad planar patch in this crop. Select the visible supporting face, not the elevated handle or rim.'
        elif memory_search is not None:
            prompt+=(' This is a crop of the CURRENT image around the object\'s uncertain last-observed/release region. '
                     'The object name retains its INITIAL positional label, not its current location. '
                     'Find that object here using its appearance and action history; for a placed stack member select that member of the stack. '
                     'Do NOT search for the old left/right location outside this crop. If the requested object is absent, return visible=false; '
                     'the full current scene will then be searched. All returned coordinates refer to THIS crop, not to old memory boxes.')
        elif action.get('grounding_role')=='orientation_endpoint':
            prompt+=' This is a zoomed crop of the source object. Locate ONLY the named end used to determine its direction, not the whole object or its centre. Coordinates refer to this crop.'
            if action.get('other_endpoint'):
                other=action['other_endpoint']
                other_point=(np.asarray(other['point'])*np.array([width-1,height-1])/1000-lo)/(hi-lo)*1000
                prompt+=' The OTHER end ('+other['target']+') is already at '+json.dumps(other_point.tolist())+' in this crop. Locate the requested end at the OPPOSITE physical end of the same object, not that point or opening.'
        elif action['skill'] in ('pick','grasp_handle'):
            prompt+=' This image is a zoomed crop around the parent object. Locate ONLY the named graspable handle/end, NOT the whole body. Put the contact point on the solid thin bar, not in a handle opening. Coordinates refer to this crop.'
        else:
            prompt+=' This image is a zoomed crop around the parent object. Locate ONLY the named working head/tip, NOT the grasping handle or the whole tool. Coordinates refer to this crop.'
    for attempt in range(3):
        raw = '[No complete model response received]'
        try:
            schema=HINGE_GROUND_SCHEMA if action['skill']=='arc' else GROUND_SCHEMA
            raw = client.complete_text(prompt, image, max_tokens=180, temperature=0,response_schema=schema if attempt else None).raw_text
            located=parse_json(raw)
            if located.get('visible') is False:break
            box=np.asarray(located.get('bbox',located.get('bbox_2d')),float)
            point=np.asarray(located.get('point',located.get('point_2d')),float)
            if (box.shape!=(4,) or point.shape!=(2,) or not np.isfinite(box).all() or not np.isfinite(point).all()
                or np.any(box<0) or np.any(box>1000) or np.any(box[2:]<=box[:2]) or np.any(point<box[:2]) or np.any(point>box[2:])):
                raise SkillError('Need a valid normalized bbox and a point inside it')
            if located.get('support','object') not in ('table','object','container'):raise SkillError('Invalid support')
            if located.get('approach','top') not in ('top','side'):raise SkillError('Invalid approach')
            hinge_line=None
            if action['skill']=='arc':
                hinge_line=np.asarray(located.get('hinge_line'),float)
                if (hinge_line.shape!=(2,2) or not np.isfinite(hinge_line).all()
                        or np.any(hinge_line<box[:2]) or np.any(hinge_line>box[2:])
                        or np.linalg.norm(hinge_line[1]-hinge_line[0])<10):
                    raise SkillError('Need two distinct visible hinge endpoints within the current bbox')
            if distinct_from:
                from robodawn.episodic_memory import same_visual_region
                full_box=box
                if crop is not None:
                    lo,hi,width,height=crop
                    full_box=(box*np.tile((hi-lo)/1000,2)+np.tile(lo,2))/np.array([width-1,height-1,width-1,height-1])*1000
                if same_visual_region(dict(bbox=full_box,camera=action['camera']), distinct_from):
                    raise SkillError('Destination overlaps the source identity box. Identify the DIFFERENT requested object; do not relabel the source.')
            if other_point is not None and np.linalg.norm(point-other_point)<150:
                raise SkillError('The two ends cannot be the same area. Locate the opposite physical end, or return visible=false if it is not observed.')
            if robot_at_point(robot_mask,point,crop) or (hinge_line is not None and any(robot_at_point(robot_mask,p,crop) for p in hinge_line)):
                from robodawn.appearance_memory import repair_robot_point
                current_box=box
                if crop is not None:
                    low,high,width,height=crop
                    current_box=(box*np.tile((high-low)/1000,2)+np.tile(low,2))/np.array([width-1,height-1,width-1,height-1])*1000
                appearance_repair=repair_robot_point(obs,memory,original_image,current_box,action['target']) if hinge_line is None else None
                if appearance_repair is not None:
                    repaired=np.asarray(appearance_repair['point'])
                    if crop is not None:
                        low,high,width,height=crop
                        repaired=(repaired*np.array([width-1,height-1])/1000-low)/(high-low)*1000
                    located.pop('point_2d',None);located['point']=repaired.tolist()
                else:
                    image=robot_hidden_image(image,robot_mask,crop);robot_repair=True
                    raise SkillError('Your contact point is on the ROBOT, not the requested scene object. '
                                     'The calibrated robot silhouette is now neutral GREY in this same current image. '
                                     'Locate a visible part of the requested object OUTSIDE the grey robot region, '
                                     'or return visible=false if it is fully occluded. Do not guess through the robot.')
            from robodawn.reacquisition_identity import verify_reacquired_identity
            full_box=box
            if crop is not None:
                lo,hi,width,height=crop
                full_box=(box*np.tile((hi-lo)/1000,2)+np.tile(lo,2))/np.array([width-1,height-1,width-1,height-1])*1000
            reacquisition_evidence=verify_reacquired_identity(client,obs,memory,action,full_box)
            break
        except (SkillError,ValueError,TypeError) as exc:
            if attempt==2:raise SkillError('Grounding invalid after repair: '+str(exc)) from exc
            prompt+='\nInvalid grounding response: '+raw+'\nReason: '+str(exc)+'\nReturn the corrected grounding JSON for the SAME current image.'
    if located.get('visible') is False:
        if memory_search is not None:
            return _ground_one(client,obs,action,memory=memory,distinct_from=distinct_from,_allow_memory_search=False)
        raise SkillError('Target not visible in current camera image; move the occluding empty arm or choose a visible target.')
    a = dict(action)
    a['observation_id']=obs['observation_id']
    for key in ('bbox','point'):
        alias=key+'_2d'
        if alias in located:
            if key in located and located[key]!=located[alias]:
                raise SkillError('Conflicting grounding '+key)
            located[key]=located[alias]
    for key in ('bbox', 'point'):
        a[key] = located[key]
    if action['skill']=='arc':a['hinge_line']=located['hinge_line']
    if crop is not None:
        lo,hi,width,height=crop
        scale=(hi-lo)/1000
        a['bbox']=((np.asarray(a['bbox'])*np.tile(scale,2)+np.tile(lo,2))/np.array([width-1,height-1,width-1,height-1])*1000).tolist()
        a['point']=((np.asarray(a['point'])*scale+lo)/np.array([width-1,height-1])*1000).tolist()
        if 'hinge_line' in a:
            a['hinge_line']=((np.asarray(a['hinge_line'])*scale+lo)/np.array([width-1,height-1])*1000).tolist()
        a['grounding_crop_bbox_px']=[*lo.tolist(),*hi.tolist()]
    if memory_search is not None:a['memory_guided_search']=memory_search
    a['approach'] = action.get('approach', located.get('approach', 'top'))
    a['support'] = action.get('support', located.get('support', 'object'))
    a['grounding_raw'] = raw
    unrefined=dict(a)
    a = refine_colour(obs, a) if a['skill']!='arc' else a
    if robot_at_point(robot_mask,a['point']):
        a=unrefined;a['colour_refinement_abstained']='refined_point_on_current_robot_self_mask'
    if robot_repair:a['robot_mask_repair']=True
    if appearance_repair is not None:a['current_appearance_repair']=appearance_repair
    if reacquisition_evidence is not None:a['reacquisition_identity']=reacquisition_evidence
    return a


class RobodawnPlanner:
    def __init__(self, client, instruction, memory=None):
        self.client, self.instruction = client, instruction
        self.memory = memory

    def plan(self, obs, history):
        if self.memory is not None:self.memory.latest_observation=obs
        if obs['success']:
            return {'name': 'done', 'skill': 'done'}
        prompt = (RULES + '\nTASK: ' + self.instruction + '\nCURRENT ROBOT STATE: '
                  + json.dumps({k: policy_sensor_view(obs[k]) if k == 'sensors' else obs[k] for k in ('endpose', 'sensors', 'table_height_m')})
                  + '\nEXECUTED HISTORY: ' + json.dumps(history_summary(history)))
        if self.memory is not None:
            prompt += self.memory.prompt()
        if obs.get('perception_failure'):
            prompt+='\nCURRENT PERCEPTION FAILURE (no action executed): '+obs['perception_failure']+'\nChoose a safe view-clearing/recovery action using the measured arm occupancy; do not repeat an unobservable contact.'
        image = np.asarray(Image.open(obs['image_paths'][0]))
        for attempt in range(3):
            raw = '[No complete model response received]'
            try:
                raw = self.client.complete_text(prompt, image, max_tokens=350, temperature=0,response_schema=ACTION_SCHEMA if attempt else None).raw_text
                a = normalize_action(parse_json(raw))
                skill, arm = a.get('skill'), a.get('arm')
                if skill not in SKILLS:
                    raise SkillError('Unknown skill; use the available skill names.')
                if skill == 'done':
                    raise SkillError('Official task success is still false. Choose the missing correction.')
                state = obs['sensors'].get(arm, {})
                if skill=='handover':
                    donor=obs['sensors'].get(a.get('donor'),{})
                    if state.get('holding') or not donor.get('holding') or donor.get('remembered_object')!=a.get('target'):
                        raise SkillError('Handover requires an empty receiver and the named source held by its distinct donor.')
                    a['grounding_role']='handover_receiver_contact'
                if skill in ('pick', 'grasp_handle', 'home', 'push', 'press') and state.get('holding'):
                    raise SkillError('This arm holds an object. Use place, move, rotate, arc, present, shake or open.')
                if skill in ('place', 'present', 'shake', 'arc') and not state.get('holding'):
                    raise SkillError('This arm is empty. Pick/grasp_handle first, or use the occupied arm.')
                if skill == 'place' and str(a.get('target', '')).strip().lower() == str(state.get('remembered_object', '')).strip().lower():
                    raise SkillError('PLACE target names the object already held by this arm. Name the destination instead, such as an empty table patch left of another object or the top of a different block.')
                from robodawn.failure_constraints import validate_recovery_action
                validate_recovery_action(a,history,obs['sensors'])
                recent = history[-2:]
                if len(recent) == 2 and skill in ('pick', 'grasp_handle', 'place', 'present', 'press', 'home'):
                    same = all(all(r['action'].get(k) == a.get(k) for k in ('skill', 'arm', 'target', 'approach', 'grasp_part', 'location')) for r in recent)
                    if same:
                        raise SkillError('That unchanged action was already tried twice without completing the goal. Choose a DIFFERENT recovery action or contact part.')
                if skill in GROUNDED:
                    a = ground(self.client, obs, a, memory=self.memory)
                    if skill == 'push':
                        dest = a.get('destination')
                        if not isinstance(dest, dict):
                            raise SkillError('PUSH requires destination {target, support}.')
                        a['destination'] = ground(self.client, obs, dict(dest, skill='place', arm=arm), memory=self.memory)
                validate(a)
                a.update(name=skill, raw=raw, observation_id=obs['observation_id'])
                return a
            except (SkillError, ValueError, KeyError, TypeError) as exc:
                if attempt == 2:
                    raise SkillError('Invalid policy action after repair: ' + str(exc)) from exc
                prompt += '\nREJECTED: ' + raw + '\nREASON: ' + str(exc) + '\nChoose a corrected action.'
