"""VLM-authored missions compiled into tactile-gated manipulation skills.

No task names, environment objects, seeds or evaluator diagnostics enter this
module. The model owns object identity, spatial relations and mission order.
"""
import json
import re
import numpy as np
from PIL import Image
from robotwin_harness_v3 import SKILLS, SkillError, validate, parse_json
from robodawn.semantic_planner import ground, history_summary, normalize_action, policy_sensor_view, RobodawnPlanner as ReactivePlanner
from robodawn.episodic_memory import EpisodicMemory, free_region, object_key
from robodawn.mission_safety import validate_resources,clear_view_action
from robodawn.schemas import MISSION_SCHEMA

RELATIONS = ('on', 'inside', 'left_of', 'right_of', 'in_front_of', 'behind')
OPERATIONS = ('transfer', 'lift', 'handover', 'bimanual_lift', 'press', 'slide', 'tool_contact', 'action', 'arrange', 'stack')
EXAMPLE_PLACEHOLDERS={'leftmost desired object','middle desired object','rightmost desired object',
                      'object to move','support or reference object','bottom object','middle object','top object',
                      'named rear end of source','named front end of source'}


def validate_object_names(step):
    names=[step.get(k) for k in ('source','destination','left_source','right_source','contact_part')]
    names+=step.get('sources',[]) if isinstance(step.get('sources',[]),list) else []
    if isinstance(step.get('action'),dict):names.append(step['action'].get('target'))
    if isinstance(step.get('orientation'),dict):names += [step['orientation'].get(k) for k in ('from_part','to_part')]
    if any(isinstance(n,str) and n.strip().lower() in EXAMPLE_PLACEHOLDERS for n in names):
        raise SkillError('Unbound example placeholder: replace every example object with the actual task object and visible distinguishing features; do not copy the demonstration names or object count.')


def validate_directed_requirement(steps,instruction):
    # Validate an explicit end-direction requirement; never invent the named
    # ends, their locations or the orientation parameters on the model's behalf.
    directions=set(re.findall(r'\b(?:tips?|toes?)\s+(?:(?:are|should|be|point|pointing|face|facing|towards?|to|the)\s+)*(left|right|front|back)\b',instruction,re.I))
    if len(directions)!=1:return
    expected=next(iter(directions)).lower()
    transfers=[step for step in steps if step['operation']=='transfer']
    if any(step.get('orientation',{}).get('facing')!=expected for step in transfers):
        raise SkillError('The instruction explicitly requires the named tip/toe to face '+expected+'. Each transfer needs orientation with actual from_part/to_part names and facing='+expected+'. Observe both ends; do not guess a rotation angle or add unrelated operations.')
    for step in transfers:
        orientation=step['orientation']
        if (not re.search(r'\b(?:tips?|toes?|front)\b',orientation.get('to_part',''),re.I)
                or re.search(r'\b(?:tips?|toes?)\b',orientation.get('from_part',''),re.I)):
            raise SkillError('The directed vector runs FROM from_part TO to_part. The requested tip/toe must be '
                             'the to_part, not from_part; reversing the endpoints reverses the final facing. '
                             'Name the actual opposite/rear end as from_part and the requested tip/toe as to_part.')


PROMPT = '''Make an ordered manipulation plan from the task and current image.
Output JSON {"steps":[...]}. Use these operations:
ARRANGE: {"operation":"arrange","sources":["leftmost desired object","middle desired object","rightmost desired object"],"arms":["left","auto","right"]}.
  For sorting/ranking objects in a left-to-right row. sources specifies the DESIRED final order, not the current image order. The executor assigns separate aligned slots; do not send several objects to the same centre.
STACK: {"operation":"stack","sources":["bottom object","middle object","top object"],"arms":["auto","auto","auto"]}.
  sources specifies the DESIRED bottom-to-top order. Use the exact number of objects requested.
TRANSFER: {"operation":"transfer","source":"object to move","destination":"support or reference object","relation":"on","arm":"auto","grasp_part":"body"}.
  The executor picks the source, places it at the destination, releases, and homes the empty arm.
  relation is on/inside/left_of/right_of/in_front_of/behind. On means on TOP, inside means in the opening.
  For beside another object, use left_of/right_of: never use on to mean next to.
  To place in a free table region use destination="empty centre of table" or another visible table region, relation=on.
  An object cannot be its own destination. The source is moved, the reference remains stationary.
LIFT: {"operation":"lift","source":"object","arm":"auto","location":"stay","grasp_part":"body"}.
  Picks and retains the object. location stay lifts in place; centre presents at front centre; side presents on that arm's side.
BIMANUAL_LIFT: {"operation":"bimanual_lift","source":"one object","left_source":"left handle or end of that object","right_source":"right handle or end of that object","grasp_part":"handle"}.
  When BOTH arms must lift ONE object, use this operation. It grasps each component without moving the object, verifies BOTH contacts, and lifts both arms together. Do not lift one side first.
  Use separate LIFT operations only for two separate objects, one per hand.
HANDOVER: {"operation":"handover","source":"held object","arm":"left","receiver":"right","grasp_part":"body"}.
  Picks with the named donor, presents, acquires the receiver's contact on the same object, THEN releases the donor. Both arm names must differ. Never transfer onto a robot hand or use bimanual_lift for passing an object. After handover, any transfer must use the receiver.
PRESS: {"operation":"press","source":"button or contact part","arm":"auto"}.
  Tap/click/press a button directly with PRESS. The empty gripper approaches and presses automatically.
  Do not grasp a button first. There is no tap or click skill.
SLIDE: {"operation":"slide","source":"object","destination":"support or reference","relation":"on","arm":"auto"}.
TOOL_CONTACT: {"operation":"tool_contact","source":"tool","contact_part":"working head or tip of the tool","destination":"contact surface","relation":"on","arm":"auto","grasp_part":"handle"}.
  Grips the source at grasp_part, then brings contact_part onto the destination WITHOUT releasing the tool.
  Hitting/striking/stamping is TOOL_CONTACT, never TRANSFER (which drops the tool). No separate lift is needed.
  contact_part is the WORKING component, distinct from the handle held by the fingers. Name it precisely so vision can locate it before pickup.
ACTION: {"operation":"action","action":{"skill":"grasp_handle","arm":"left","target":"lid edge","grasp_part":"edge","approach":"top"}}.
  Only for an operation not covered above. Available skills: grasp_handle, move (delta xyz <=0.2 m), rotate (axis x/y/z, signed angle <=90 degrees), arc (target=hinge, axis, angle), open, close, home, present (location centre/side), shake (axis, amplitude 0.02..0.06, cycles 1..4).
  grasp_handle is ONLY for a handle or edge of an articulated door/drawer/lid, and does not lift.
  Before arc/move/rotate of an object, acquire it with grasp_handle or lift. Empty-arm moves cannot move objects.
arm must be left/right/auto; obey any explicitly assigned arms, otherwise use the arm on the object's side.
grasp_part body/handle/rim/edge. Use body for compact solid objects, a rim for a bowl, a handle for a basket/pan.
Do not add reach or preparatory moves: transfer/lift include all approach/grasp motions.
Include every requested object and stage. Preserve occupied arms needed by later stages.
No coordinates, task names or guessed hidden state. Identify objects by visible names/colours.
If a prior plan did not finish the goal, produce only the remaining corrective steps from the CURRENT image.
'''
DIRECTED_PROMPT = '''For a requested directed orientation add orientation={"from_part":"named rear end of source","to_part":"named front end of source","facing":"left"} to TRANSFER.
Name two distinct VISIBLE ends of the SAME source object. facing is left/right/front/back in the table frame (+X right, +Y back).
The executor observes these ends before pickup and aligns the from_part-to_part direction during placement.
Include this whenever the task specifies which way an end/tip/front points. Do not replace directed orientation with an arbitrary rotate angle.
'''


def _prefix_already_places(steps, child):
    """Do not expand STACK into an identical, still-valid prefix transfer.

    This is intra-plan algebra, not a claim that past execution achieved a
    relation. Any intervening operation touching either object is a barrier.
    """
    source=object_key(child['source']);destination=object_key(child['destination'])
    for previous in reversed(steps):
        if previous.get('operation')=='action':return False
        touched={object_key(previous.get(field,'')) for field in ('source','destination','left_source','right_source')}
        if not {source,destination}.intersection(touched):continue
        return (previous.get('operation')=='transfer'
                and object_key(previous['source'])==source
                and object_key(previous['destination'])==destination
                and previous.get('relation')==child.get('relation')
                and previous.get('orientation')==child.get('orientation')
                and child.get('arm','auto') in ('auto',previous.get('arm','auto')))
    return False


def prepare_acquisition_requirements(steps):
    """Schedule an authored endpoint observation before the matching LIFT.

    A direction requested by a later transfer cannot first be measured after
    pickup, when the hand may hide the ends. Only propagate across unrelated
    operations, never across another use of that object or an opaque action.
    This moves a perception prerequisite, not a physical operation or goal.
    """
    for index,step in enumerate(steps):
        if step.get('operation')!='transfer' or not step.get('orientation'):continue
        source=object_key(step['source'])
        for previous in reversed(steps[:index]):
            if previous.get('operation')=='action':break
            if object_key(previous.get('source',''))!=source:continue
            if previous.get('operation')=='lift':
                previous['acquisition_orientation']=dict(step['orientation'])
            break
    return steps


def validate_mission(value):
    steps = value.get('steps')
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        raise SkillError('Return steps: a nonempty list of at most 12 operations')
    # Some native responses serialize an operation tag followed by its object.
    # This is unambiguous only for a known tag and a matching/absent operation.
    normalized=[]
    index=0
    while index<len(steps):
        raw=steps[index]
        if isinstance(raw,str) and raw.lower() in OPERATIONS and index+1<len(steps) and isinstance(steps[index+1],dict):
            following=steps[index+1]
            if following.get('operation',raw.lower())!=raw.lower():
                raise SkillError('Conflicting operation tag and object')
            normalized.append(dict(following,operation=raw.lower()));index+=2
        else:
            normalized.append(raw);index+=1
    checked = []
    for raw in normalized:
        if not isinstance(raw, dict):
            raise SkillError('Each step must be an object')
        step = normalize_action(raw)
        validate_object_names(step)
        op = step.get('operation')
        if op in ('release','open','close','home','present','move','rotate','arc','shake','grasp_handle'):
            action={k:v for k,v in step.items() if k!='operation'}
            action['skill']='open' if op=='release' else op
            step={'operation':'action','action':action}
            op='action'
        if op not in OPERATIONS:
            raise SkillError('operation must be ' + '/'.join(OPERATIONS))
        if op in ('arrange','stack'):
            sources=step.get('sources')
            if not isinstance(sources,list) or not 2<=len(sources)<=5 or not all(isinstance(x,str) and x.strip() for x in sources):
                raise SkillError('arrange/stack needs 2..5 source descriptions in the desired final order')
            arms=step.get('arms',['auto']*len(sources))
            if not isinstance(arms,list) or len(arms)!=len(sources) or any(x not in ('left','right','auto') for x in arms):
                raise SkillError('arms must have one left/right/auto entry per source')
            if op=='stack':
                # A bottom already held by an earlier LIFT needs to be put
                # down before building on it. An untouched bottom stays put.
                held_bottom_arm=None
                for previous in checked:
                    if object_key(previous.get('source',''))!=object_key(sources[0]):continue
                    if previous['operation']=='lift':held_bottom_arm=previous.get('arm','auto')
                    elif previous['operation']=='transfer':held_bottom_arm=None
                if held_bottom_arm is not None:
                    checked.append(dict(operation='transfer',source=sources[0],destination='empty centre of table',
                                        relation='on',arm=held_bottom_arm,grasp_part=step.get('grasp_part','body')))
            for i,(source,arm) in enumerate(zip(sources,arms)):
                if op=='stack' and i==0:
                    continue
                child=dict(operation='transfer',source=source,arm=arm,grasp_part=step.get('grasp_part','body'))
                if op=='arrange':child.update(destination='empty centre of table',relation='row_slot',slot=i,slot_count=len(sources))
                else:child.update(destination='empty centre of table' if i==0 else sources[i-1],relation='on')
                if op=='stack' and _prefix_already_places(checked,child):continue
                checked.append(child)
            continue
        if op == 'action':
            if not isinstance(step.get('action'), dict):
                raise SkillError('action operation requires an action object')
            action=normalize_action(step['action'])
            if action.get('skill') not in SKILLS or action.get('skill')=='done':
                raise SkillError('Unknown action skill. Use press for tap/click; only listed skills are executable.')
            if action.get('arm') not in ('left','right','auto') and action.get('skill') not in ('wait','dual_move'):
                raise SkillError('action needs arm left/right/auto')
            if action['skill'] not in ('pick','grasp_handle','handover','push','place','press','reach','arc'):
                if action.get('arm')=='auto':
                    raise SkillError('Ungrounded ACTION '+action['skill']+' needs an explicit left/right arm; auto is resolved only from a current visual target.')
                from robotwin_harness_v3 import validate
                validate(action)
            if action.get('skill')=='grasp_handle' and re.search(r'\bbutton\b',str(action.get('target','')),re.I):
                raise SkillError('A button is a press contact, not an articulated handle. Return operation press with source=the button and the requested arm.')
            step['action']=action
        else:
            if not isinstance(step.get('source'), str) or not step['source'].strip():
                raise SkillError('source must name a visible object')
            if step.get('arm', 'auto') not in ('auto', 'left', 'right'):
                raise SkillError('arm must be auto/left/right')
            if step.get('grasp_part', 'body') not in ('body','handle','rim','edge'):
                raise SkillError('grasp_part must be body/handle/rim/edge')
        if op in ('transfer','slide','tool_contact'):
            if not isinstance(step.get('destination'), str) or not step['destination'].strip():
                raise SkillError('destination must name the support/reference object or free table region')
            if step.get('relation') not in RELATIONS:
                raise SkillError('relation must be ' + '/'.join(RELATIONS))
            if step['source'].lower().strip() == step['destination'].lower().strip():
                raise SkillError('source and destination must be different')
            if re.fullmatch(r'(?:the )?(?:other|left|right) (?:robot )?(?:hand|arm|gripper)',step['destination'].strip(),re.I):
                raise SkillError('A robot hand is not a support. Use handover with source, donor arm and distinct receiver; never release onto an empty hand.')
        if op=='handover':
            if step.get('arm') not in ('left','right') or step.get('receiver') not in ('left','right') or step['arm']==step['receiver']:
                raise SkillError('handover requires distinct explicit donor arm and receiver')
        if op=='tool_contact' and (not isinstance(step.get('contact_part'),str) or not step['contact_part'].strip()):
            raise SkillError('tool_contact needs contact_part naming the working head/tip, distinct from the grasped handle')
        if 'orientation' in step:
            from directed_placement import FACING
            orient=step['orientation']
            if op!='transfer' or not isinstance(orient,dict) or orient.get('facing') not in FACING:
                raise SkillError('orientation belongs to transfer and needs facing left/right/front/back')
            if any(not isinstance(orient.get(k),str) or not orient[k].strip() for k in ('from_part','to_part')):
                raise SkillError('orientation requires two named visible source endpoints')
            if orient['from_part'].strip().lower()==orient['to_part'].strip().lower():
                raise SkillError('orientation needs distinct from_part and to_part')
        if op == 'lift' and step.get('location', 'stay') not in ('stay','side','centre'):
            raise SkillError('lift location must be stay/side/centre')
        if op=='bimanual_lift':
            for key in ('left_source','right_source'):
                if not isinstance(step.get(key),str) or not step[key].strip():
                    raise SkillError('bimanual_lift needs distinct left_source and right_source component descriptions')
            if step['left_source'].lower()==step['right_source'].lower():
                raise SkillError('The two arms must grasp distinct components, not the same point')
        checked.append(step)
    # Final support relations must be physically consistent. Do not silently
    # choose which model-authored relation to discard when they form a cycle.
    support={}
    for step in checked:
        if step['operation']=='transfer':
            source=object_key(step['source'])
            if step['relation'] in ('on','inside') and not free_region(step['destination']):
                support[source]=object_key(step['destination'])
            else:support.pop(source,None)
    for source in support:
        seen=set();node=source
        while node in support:
            if node in seen:
                raise SkillError('Cyclic support goals: an object cannot end on/inside an object supported by itself. Choose the requested bottom-to-top order, not reciprocal transfers.')
            seen.add(node);node=support[node]
    from robodawn.mission_compiler import fuse_tool_acquisition
    return prepare_acquisition_requirements(fuse_tool_acquisition(checked))


def validate_explicit_arm_sequence(steps,instruction,complete_plan=True):
    text=instruction.split('Completion requirements:')[0]
    if complete_plan and re.search(r'\b(hit|strike)\b',text,re.I) and not any(s['operation'] in ('tool_contact','press') for s in steps):
        raise SkillError('The instruction requests contact/impact, but this plan only moves objects. Use tool_contact for a held tool (name its contact_part) or press for a direct empty-hand press; TRANSFER releases the source.')
    arms=re.findall(r'\b(left|right) arm\b',text,re.I)
    if complete_plan and len(arms)>1 and all(s['operation']=='transfer' for s in steps) and len(steps)!=len(arms):
        raise SkillError('The instruction assigns '+str(len(arms))+' object transfers, but this plan contains '+str(len(steps))+'. ARRANGE already picks and places each object; omit duplicate preliminary transfers.')
    if len(arms)>1 and len(arms)==len(steps) and all(s['operation']=='transfer' for s in steps):
        if any(s.get('arm')!=arm.lower() for s,arm in zip(steps,arms)):
            raise SkillError('The instruction explicitly assigns the arm sequence '+str(arms)+'. Preserve these assignments in order.')


def validate_explicit_comparative_order(steps,instruction,complete_plan=True):
    if not complete_plan:return
    text=instruction.split('Completion requirements:')[0].lower()
    if not re.search(r'\b(arrange|sort|rank)\b',text):return
    aliases={'smallest':'small','largest':'large','biggest':'large','big':'large'}
    def sizes(value):
        return [aliases.get(word,word) for word in re.findall(r'\b(?:small(?:est)?|medium|large(?:st)?|big(?:gest)?)\b',value.lower())]
    expected=sizes(text)
    arranged=[step for step in steps if step.get('relation')=='row_slot']
    if len(set(expected))<2 or len(expected)!=len(arranged):return
    actual=[sizes(step['source']) for step in arranged]
    if any(values!=[wanted] for values,wanted in zip(actual,expected)):
        raise SkillError('The instruction explicitly names comparative objects in this order: '+str(expected)+
                         '. Preserve EACH requested size identity in ARRANGE sources. RGB appearance/initial-position labels '
                         'must not replace the medium object with a second small object. Current RGB-D resolves comparative size.')


class RobodawnPlanner:
    def __init__(self, client, instruction):
        self.client, self.instruction = client, instruction
        self.steps = []
        self.index = 0
        self.phase = 'pick'
        self.arm = None
        self.failures = 0
        self.revision = 0
        self.memory = EpisodicMemory()
        self.reactive = ReactivePlanner(client,instruction,memory=self.memory)
        self.bound_revision = None
        self.staged_sources = set()
        self.inventory_attempted = False
        self.inventory_error = None
        self.support_rendezvous_attempts = {}
        self.support_rendezvous_pending = False
        self.last_perception_failure = None
        self.working_references = set()
        self.inspection_attempts = set()
        self.support_inspection_targets = set()
        self.support_observer_to_home = None
        self.source_inspection_targets = {}
        self.source_inspection_count = 0

    def _replan(self, obs, history):
        from robodawn.scene_inventory import needs_inventory,observe_inventory
        if not self.inventory_attempted and not history and needs_inventory(self.instruction):
            self.inventory_attempted=True
            try:observe_inventory(self.client,obs,self.instruction,self.memory)
            except SkillError as exc:self.inventory_error=str(exc)
        directed=bool(re.search(r'\b(tips?|toes?|facing|pointing)\b',self.instruction,re.I))
        prompt = (PROMPT + (DIRECTED_PROMPT if directed else '') + '\nTASK: ' + self.instruction + '\nCURRENT CONTACT STATE: '
                  + json.dumps(policy_sensor_view(obs['sensors'])) + '\nEXECUTED ACTIONS: '
                  + json.dumps(history_summary(history)) + self.memory.prompt())
        if self.last_perception_failure:
            prompt+=('\nLAST PERCEPTION/PRECONDITION FAILURE (no action executed): '+self.last_perception_failure+
                     '\nUse the CURRENT image to resolve distinct source/support identities and executable prerequisites. '
                     'Do not repeat ambiguous names or discard unexecuted original intentions.')
        # Keep the evidenced v21 semantic interface. Fresh-image identity
        # binding follows before motion; visual recovery/directed ends use RGB.
        image = np.asarray(Image.open(obs['image_paths'][0])) if history or directed or self.last_perception_failure else None
        for attempt in range(3):
            raw = '[No complete model response received]'
            try:
                raw = self.client.complete_text(prompt,image,max_tokens=900,temperature=0,response_schema=MISSION_SCHEMA if attempt else None).raw_text
                steps = validate_mission(parse_json(raw))
                from robodawn.mission_compiler import schedule_simultaneous_supports
                steps=schedule_simultaneous_supports(steps,self.instruction)
                from robodawn.failure_constraints import schedule_blocked_support_lift
                steps=schedule_blocked_support_lift(steps,history,obs['sensors'],self.instruction)
                validate_directed_requirement(steps,self.instruction)
                validate_explicit_arm_sequence(steps,self.instruction,complete_plan=not history)
                validate_explicit_comparative_order(steps,self.instruction,complete_plan=not history)
                validate_resources(steps,obs['sensors'])
                from robodawn.failure_constraints import validate_recovery_plan
                validate_recovery_plan(steps,history,obs['sensors'])
                self.memory.intentions.validate_replan(steps,obs['sensors'])
                self.memory.intentions.initialize(steps)
                self.steps = steps
                self.index, self.phase, self.arm, self.failures = 0, 'pick', None, 0
                self.revision += 1
                return
            except (SkillError, ValueError, TypeError) as exc:
                if attempt == 2:
                    raise SkillError('Invalid mission after repair: '+str(exc)) from exc
                prompt += '\nRejected plan: '+raw+'\nReason: '+str(exc)+'\nCorrect the JSON plan.'

    def _finish_step(self):
        self.index += 1
        self.phase, self.arm, self.failures = 'pick', None, 0

    def _ground(self, obs, a, image_region=None):
        # Only a current-image grounding call supplies image coordinates.
        if a['skill']=='pick' and a.get('grasp_part','body')=='body':
            preferred=self.memory.preferred_grasp_part(a['target'])
            if preferred in ('rim','handle','edge'):
                a=dict(a,grasp_part=preferred,requested_grasp_part='body',
                       grasp_part_resolution='same_model_paired_crop_affordance')
        a = ground(self.client,obs,a,image_region=image_region,memory=self.memory)
        if a.get('arm') == 'auto':
            side = 'left' if a['point'][0] < 500 else 'right'
            if a.get('camera','head_camera')!='head_camera':
                from robodawn.camera_views import observed_world_side
                side=observed_world_side(obs,a)
            other = 'right' if side == 'left' else 'left'
            if obs['sensors'].get(side,{}).get('holding') and not obs['sensors'].get(other,{}).get('holding'):
                side = other
            a['arm'] = side
        return a

    def _bind_transfer_identities(self, obs):
        """Bind source/destination while both are visible, before moving either.

        All bindings are model-produced from this image. Stored boxes inform
        later grounding but are never substituted for current geometry.
        """
        if self.bound_revision == self.revision:
            return
        bound = {};verified_pairs=set()
        for step in self.steps[self.index:]:
            if step['operation'] not in ('transfer', 'slide', 'tool_contact'):
                continue
            if free_region(step['destination']):
                continue
            source = step['source']
            # Do not compare the old source box with a held/moved object.
            held = any(s.get('holding') and object_key(s.get('remembered_object', '')) == object_key(source)
                       for s in obs['sensors'].values())
            if held:
                continue
            key = object_key(source)
            if key not in bound:
                bound[key] = self.memory.current_binding(source,obs['observation_id'])
                if bound[key] is None:
                    bound[key] = ground(self.client, obs, dict(skill='reach', arm='auto', target=source,grounding_role='entity_identity'), memory=self.memory)
            from robodawn.collection_identity import collection_collisions
            collisions=collection_collisions(self.steps[self.index:],step,bound,obs['observation_id'])
            for _ in range(3):
                if not collisions:break
                # First ordinary proposal is unchanged. Only a demonstrated
                # same-frame collision introduces this disambiguation query.
                bound[key]=ground(self.client,obs,dict(skill='reach',arm='auto',target=source,
                    grounding_role='collection_source_identity'),memory=self.memory,distinct_from=collisions[0])
                collisions=collection_collisions(self.steps[self.index:],step,bound,obs['observation_id'])
            if collisions:
                raise SkillError('Different sources for the same collection goal still bind to one visible object. '
                                 'Locate distinct objects, or use one consistent source name if the plan intentionally reuses one object.')
            destination_key=object_key(step['destination'])
            if destination_key not in bound:
                bound[destination_key] = self.memory.current_binding(step['destination'],obs['observation_id'])
                if bound[destination_key] is None:
                    bound[destination_key] = ground(self.client, obs, dict(skill='reach', arm='auto', target=step['destination'],grounding_role='entity_identity'),
                                                memory=self.memory, distinct_from=bound[key])
            from robodawn.episodic_memory import same_visual_region
            if same_visual_region(bound[key],bound[destination_key]):
                raise SkillError('Previously bound source and destination refer to the same visual object; revise their identities.')
            if (key,destination_key) not in verified_pairs:
                from robodawn.pair_grounding import verify_pair
                from robodawn.episodic_memory import same_visual_region
                first,second=verify_pair(self.client,obs,bound[key],bound[destination_key])
                if collection_collisions(self.steps[self.index:],step,bound,obs['observation_id'],candidate=first):
                    raise SkillError('Paired verification reassigned this collection source to an already bound different source; revise the identities.')
                for name,binding in ((key,first),(destination_key,second)):
                    previous=bound[name]
                    if previous.get('identity_verification_observation_id')==obs['observation_id'] and not same_visual_region(previous,binding):
                        raise SkillError('Inconsistent same-frame object identity across goal pairs; revise the object descriptions')
                # Validate BOTH roles before committing either one; a rejected
                # pair must not partially poison the shared identity memory.
                for name,binding in ((key,first),(destination_key,second)):
                    bound[name]=binding;self.memory.remember_visual(binding,obs['observation_id'])
                verified_pairs.add((key,destination_key))
        self.bound_revision = self.revision

    def _destination(self, obs, step, arm, skill='place'):
        if step.get('regrasp_stage'):
            return dict(skill=skill,arm=arm,target='visible free shared table region',support='table',
                        bbox=[0,0,1000,1000],point=[500,500],release=True,regrasp_stage=True,
                        grounding_source='current_rgbd_free_space_search')
        if step['relation']=='row_slot':
            return dict(skill=skill,arm=arm,target='empty centre of table',support='table',
                        bbox=[400,400,600,600],point=[500,500],relation='row_slot',
                        slot=step['slot'],slot_count=step['slot_count'],
                        grounding_source='calibrated_table_layout')
        request=dict(skill=skill,arm=arm,target=step['destination'])
        # Spatial relations locate the REFERENCE object's extent, not empty
        # table beside it. Make this distinction before grounding/admission.
        from robotwin_harness_v3 import RELATIVE_DIRECTIONS
        if step['relation'] in RELATIVE_DIRECTIONS:request['relation']=step['relation']
        a = self._ground(obs,request)
        if step.get('orientation'):
            a['facing']=step['orientation']['facing']
            group=[i for i,s in enumerate(self.steps) if s.get('operation')=='transfer'
                   and s.get('relation')=='inside' and s.get('destination')==step['destination']
                   and s.get('orientation',{}).get('facing')==a['facing']]
            if 2<=len(group)<=5 and self.index in group:
                a['container_layout']={'slot':group.index(self.index),'count':len(group),'facing':a['facing']}
        relation = step['relation']
        a['relation'] = relation
        if relation == 'inside':
            a['support'] = 'container'
        elif relation in RELATIONS[2:]:
            a['support'] = 'table'
        # For on, the current-image grounder distinguishes table from solid support.
        return a

    def plan(self, obs, history):
        self.memory.latest_observation=obs
        obs['episodic_memory'] = self.memory.context()
        if self.inventory_error:obs['inventory_error']=self.inventory_error
        if obs['success']:
            return dict(name='done',skill='done')
        if self.support_observer_to_home is not None:
            side=self.support_observer_to_home
            self.support_observer_to_home=None
            if not obs['sensors'].get(side,{}).get('holding'):
                return dict(skill='home',name='home',arm=side,support_observer_home=True,
                            observation_id=obs['observation_id'])
        if len(history)>=2:
            recent=history[-2:]
            perception_errors=('no depth','no above-table','empty grasp','without measured object contact')
            if (all(not r['result'].get('skill_success') and any(word in str(r['result'].get('failure','')).lower() for word in perception_errors) for r in recent)
                    and recent[0]['action'].get('target')==recent[1]['action'].get('target')):
                recovery=self._inspection_recovery(obs,str(recent[-1]['result'].get('failure','')),recent[-1]['action'].get('arm'))
                if recovery:return recovery
        recovery=clear_view_action(obs,history)
        if recovery:return recovery
        if self.index<len(self.steps):
            from robodawn.mission_safety import cooperative_support_action
            recovery=cooperative_support_action(obs,history,self.steps[self.index],self.instruction,self.support_rendezvous_attempts,
                                                resume=self.support_rendezvous_pending)
            self.support_rendezvous_pending=False
            if recovery:
                try:
                    source_arm=next(side for side,state in obs['sensors'].items() if state.get('holding')
                                    and object_key(state.get('remembered_object',''))==object_key(self.steps[self.index]['source']))
                    destination=self._destination(obs,self.steps[self.index],source_arm)
                    destination.update(release=True,observation_id=obs['observation_id'])
                    recovery['cooperative_destination']=destination
                    return recovery
                except (SkillError,ValueError,KeyError,TypeError) as exc:
                    self.last_perception_failure=str(exc)
        for attempt in range(3):
            if self.steps and self.index>=len(self.steps) and not self.memory.intentions.pending():
                return self._reactive_recovery(obs,history,'All planned operations executed; official goal still false')
            if self.index >= len(self.steps) or self.failures >= 2:
                try:self._replan(obs,history)
                except SkillError as exc:
                    # A failed high-level plan is not a terminal environment
                    # state. Let the sensor-gated single-action planner recover.
                    self.steps=[];self.index=0;self.phase='pick';self.arm=None
                    return self._reactive_recovery(obs,history,'Mission rejected after repair: '+str(exc))
            step = self.steps[self.index]
            op = step['operation']
            try:
                self._bind_transfer_identities(obs)
                inspection=self._support_boundary_inspection(obs,step)
                if inspection:return inspection
                if self.phase == 'home':
                    if obs['sensors'].get(self.arm,{}).get('holding'):
                        raise SkillError('Release did not clear the gripper')
                    a = dict(skill='home',arm=self.arm)
                elif op=='bimanual_lift':
                    missing=[]
                    for side in ('left','right'):
                        state=obs['sensors'].get(side,{})
                        if not state.get('holding'):missing.append(side)
                        elif state.get('remembered_object')!=step[side+'_source']:
                            raise SkillError(side+' holds a different object; clear it before a bimanual grasp')
                    if missing:
                        side=missing[0]
                        parent=self._ground(obs,dict(skill='reach',arm=side,target=step['source']))
                        inspection=self._source_boundary_inspection(obs,dict(parent,skill='grasp_handle'))
                        if inspection:return inspection
                        from robodawn.bar_affordance import end_contact
                        other='right' if side=='left' else 'left'
                        other_pose=obs.get('endpose',{}).get(other+'_endpose') if obs['sensors'].get(other,{}).get('holding') else None
                        measured=end_contact(obs,parent,step[side+'_source'],side,other_pose)
                        query=dict(skill='grasp_handle',arm=side,target=step[side+'_source'],camera=parent['camera'],
                                   grasp_part=step.get('grasp_part','handle'),approach='top')
                        if measured is None:a=self._ground(obs,query,image_region=parent['bbox'])
                        else:
                            a=dict(query,**{k:measured[k] for k in ('bbox','point','observation_id')},
                                   bar_end_evidence=measured)
                            self.memory.remember_visual(a,obs['observation_id'])
                        a['parent_grounding']={k:parent[k] for k in ('target','bbox','point','camera','observation_id')}
                        a['arm_required']=True
                    else:a=dict(skill='dual_move',delta=[0,0,.12])
                elif op=='handover':
                    donor,receiver=step['arm'],step['receiver']
                    donor_state=obs['sensors'].get(donor,{})
                    receiver_state=obs['sensors'].get(receiver,{})
                    if receiver_state.get('holding'):
                        raise SkillError('Handover receiver is occupied; verify or recover the previous transaction')
                    if not donor_state.get('holding'):
                        self.phase='pick'
                        a=self._ground(obs,dict(skill='pick',arm=donor,target=step['source'],
                                               grasp_part=step.get('grasp_part','body')))
                        a['arm_required']=True
                    elif donor_state.get('remembered_object')!=step['source']:
                        raise SkillError('Handover donor holds a different object')
                    elif self.phase in ('pick','offer'):
                        self.phase='offer'
                        a=dict(skill='present',arm=donor,location='centre')
                    else:
                        a=self._ground(obs,dict(skill='grasp_handle',arm=receiver,donor=donor,
                            target=step['source'],grasp_part=step.get('grasp_part','body'),
                            grounding_role='handover_receiver_contact'))
                        a.update(skill='handover',arm_required=True)
                elif op in ('transfer','lift','tool_contact'):
                    for side,state in obs['sensors'].items():
                        if state.get('holding') and state.get('remembered_object') == step['source']:
                            self.arm = side
                            if self.phase == 'pick':
                                self.phase = 'place'
                    if self.phase == 'pick':
                        query=dict(skill='pick',arm=step.get('arm','auto'),target=step['source'],grasp_part=step.get('grasp_part','body'))
                        if step.get('acquisition_approach'):query['approach']=step['acquisition_approach']
                        a = self._ground(obs,query)
                        if step.get('grasp_part')=='handle' and op!='tool_contact':
                            parent={k:a[k] for k in ('target','bbox','point','camera','observation_id')}
                            a=self._ground(obs,dict(skill='pick',arm=a['arm'],
                                target='solid graspable handle of '+step['source'],grasp_part='handle',camera=parent['camera']),image_region=parent['bbox'])
                            a['component_grounding_target']=a['target']
                            a['target']=step['source'];a['parent_grounding']=parent
                        self.arm = a['arm']
                        if step.get('arm_required'):a['arm_required']=True
                        orientation_spec=step.get('orientation') or step.get('acquisition_orientation')
                        if orientation_spec:
                            orientation={'facing':orientation_spec['facing'],'observation_id':obs['observation_id']}
                            for key in ('from_part','to_part'):
                                query=dict(skill='press',arm=self.arm,target=orientation_spec[key],grounding_role='orientation_endpoint',camera=a['camera'])
                                if key=='to_part':query['other_endpoint']=orientation['from_part']
                                endpoint=self._ground(obs,query,image_region=a.get('parent_grounding',a)['bbox'])
                                orientation[key]={k:endpoint[k] for k in ('target','bbox','point','camera')}
                                orientation[key]['grounding_crop_bbox_px']=endpoint['grounding_crop_bbox_px']
                            a['orientation_reference']=orientation
                        if op=='tool_contact':
                            reference=self._ground(obs,dict(skill='press',arm=self.arm,target=step['contact_part'],camera=a['camera']),image_region=a['bbox'])
                            a['contact_reference']={k:reference[k] for k in ('target','bbox','point','camera')}
                            a['contact_reference']['observation_id']=obs['observation_id']
                            a['contact_reference']['grounding_crop_bbox_px']=reference['grounding_crop_bbox_px']
                        if obs['sensors'].get(self.arm,{}).get('holding'):
                            raise SkillError('Required arm holds a different object')
                    elif not obs['sensors'].get(self.arm,{}).get('holding'):
                        self.phase = 'pick'
                        continue
                    elif op == 'lift':
                        if step.get('location','stay') == 'stay':
                            self._finish_step()
                            continue
                        a = dict(skill='present',arm=self.arm,location=step['location'])
                    else:
                        a = self._destination(obs,step,self.arm)
                        a['release'] = op == 'transfer'
                        if op=='tool_contact':
                            a['use_contact_part']=True
                            if object_key(step['source']) not in self.working_references:
                                reference=self._ground(obs,dict(skill='press',arm=self.arm,
                                    target=step['contact_part']+' of '+step['source'],grounding_role='retained_working_part'))
                                a['contact_reference']={k:reference[k] for k in ('target','bbox','point','camera')}
                                a['contact_reference']['observation_id']=obs['observation_id']
                elif op == 'press':
                    a = self._ground(obs,dict(skill='press',arm=step.get('arm','auto'),target=step['source']))
                elif op == 'slide':
                    a = self._ground(obs,dict(skill='push',arm=step.get('arm','auto'),target=step['source']))
                    a['destination'] = self._destination(obs,step,a['arm'])
                else:
                    a = normalize_action(step['action'])
                    # A compiled acquisition can lose contact during later
                    # observation/recovery. Recheck the actual resources before
                    # spending model calls on its dependent contact geometry.
                    validate_resources([dict(operation='action',action=a)],obs['sensors'])
                    if a.get('skill') in ('pick','grasp_handle','place','press','reach','arc'):
                        a = self._ground(obs,a)
                # Whole-source extent matters for two distinct contacts,
                # pushing a centre, and directed endpoints. A local ordinary
                # grasp need not see the entire object before acquisition.
                inspection=self._source_boundary_inspection(obs,a) if op in ('bimanual_lift','slide') or a.get('orientation_reference') else None
                if inspection:return inspection
                validate(a)
                if a.get('skill')=='pick' and op=='transfer' and step.get('relation')=='on':
                    preview=self.memory.current_binding(step['destination'],obs['observation_id'])
                    if preview is not None:a['placement_preview']=preview
                self.last_perception_failure=None
                a.update(name=a['skill'],observation_id=obs['observation_id'],
                         mission_revision=self.revision,mission_step=self.index,
                         mission_phase=self.phase,mission_intent=step)
                return a
            except (SkillError, ValueError, KeyError, TypeError) as exc:
                self.last_perception_failure=str(exc)
                recovery=self._inspection_recovery(obs,str(exc),history[-1]['action'].get('arm') if history else None)
                if recovery:return recovery
                self.failures = 2
                if attempt == 2:
                    # A failed visual binding is not a terminal environment.
                    # Expose the measured failure (not a fabricated executed
                    # action) to the existing sensor-gated recovery planner.
                    recovery_obs=dict(obs,perception_failure=str(exc))
                    return self._reactive_recovery(recovery_obs,history,'Mission grounding/precondition failed: '+str(exc))
        return self._reactive_recovery(obs,history,'Compiled operations were already satisfied by measured contact state')

    def _reactive_recovery(self,obs,history,reason):
        """Use the same bounded visual recovery at every reactive entry point.

        A rejected visual proposal did not execute an action. Return at most
        one existing sensor-gated inspection per empty arm, for execution and
        accounting by the unchanged outer loop. Do not retry the model here.
        """
        from robodawn.completion_review import completed_contact_review
        previous_review=self.memory.completion_review
        self.memory.completion_review=completed_contact_review(self.steps,self.index,self.memory.intentions.pending(),
                                                              history,obs.get('sensors',{}))
        try:
            try:a=self.reactive.plan(obs,history)
            finally:self.memory.completion_review=previous_review
        except SkillError as exc:
            self.last_perception_failure=str(exc)
            preferred=history[-1]['action'].get('arm') if history else None
            # Missing/unknown occupancy is not permission to move an arm.
            inspection_obs=dict(obs,sensors={side:state for side,state in obs.get('sensors',{}).items()
                                             if type(state.get('holding')) is bool})
            a=self._inspection_recovery(inspection_obs,str(exc),preferred)
            if a is None:raise
            a['reactive_visual_recovery']=True
        a['mission_recovery']=reason
        return a

    def _source_boundary_inspection(self,obs,action):
        if len(obs.get('image_paths',[]))!=3:return None
        from robodawn.source_boundary import clipped_source_anchor
        reference=action.get('parent_grounding',action)
        key=object_key(reference.get('target',''))
        camera=reference.get('camera','head_camera')
        seen=self.source_inspection_targets.get(key,set())
        if not key or camera in seen or self.source_inspection_count>=2:return None
        available=[side for side in ('left','right')
                   if side in obs.get('sensors',{}) and not obs['sensors'][side].get('holding')]
        if not available:return None
        anchor=clipped_source_anchor(obs,action)
        if anchor is None:return None
        preferred=camera.split('_')[0] if camera!='head_camera' else ('left' if anchor['image_point'][0]<500 else 'right')
        side=preferred if preferred in available else available[0]
        self.source_inspection_targets.setdefault(key,set()).add(camera)
        self.source_inspection_count+=1;self.inspection_attempts.add(side)
        return dict(skill='inspect',name='inspect',arm=side,observation_id=obs['observation_id'],
                    inspection_anchor=anchor['image_point'],inspection_camera=camera,active_inspection=True,
                    source_boundary_inspection=anchor,target=anchor['target'])

    def _support_boundary_inspection(self,obs,step):
        if self.phase!='pick' or step['operation'] not in ('lift','transfer'):return None
        source=object_key(step['source']);destination=None
        for pending in self.steps[self.index:]:
            if pending['operation']=='action':break
            if object_key(pending.get('source',''))!=source:continue
            if pending['operation']=='transfer' and pending.get('relation')=='on':
                destination=pending['destination'];break
            if pending['operation']!='lift':break
        if not destination or object_key(destination) in self.support_inspection_targets:return None
        available=[side for side in ('left','right') if side not in self.inspection_attempts
                   and side in obs.get('sensors',{}) and not obs['sensors'][side].get('holding')]
        if not available or len(obs.get('image_paths',[]))!=3:return None
        from flat_support_memory import clipped_support_anchor
        anchor=clipped_support_anchor(obs,destination)
        if anchor is None:return None
        preferred='left' if anchor['point'][0]<0 else 'right'
        side=preferred if preferred in available else available[0]
        self.support_inspection_targets.add(object_key(destination));self.inspection_attempts.add(side)
        return dict(skill='inspect',name='inspect',arm=side,observation_id=obs['observation_id'],
                    inspection_anchor=anchor['image_point'],active_inspection=True,
                    support_boundary_inspection=True,target=destination,
                    reason='Observe the clipped support boundary before occupying the grasping arm')

    def _inspection_recovery(self, obs, failure, preferred_arm=None):
        """At most one active view per empty arm, only on visual failures."""
        from robodawn.recovery_errors import failure_kind
        if failure_kind(failure)!='perception':return None
        available=[s for s in ('left','right') if s in obs.get('sensors',{})
                   and not obs['sensors'][s].get('holding') and s not in self.inspection_attempts]
        if not available or len(obs.get('image_paths',[]))!=3:return None
        # A fresh model-observed region may aim the camera; no actor location.
        anchor=None
        for entry in reversed(list(self.memory.objects.values())):
            visual=entry.get('last_visual',{})
            if visual.get('observation_id')==obs['observation_id'] and visual.get('camera','head_camera')=='head_camera':
                anchor=visual.get('point');break
        if anchor is not None:
            preferred='left' if anchor[0]<500 else 'right'
            if preferred in available:available.remove(preferred);available.insert(0,preferred)
        elif preferred_arm in available:
            # Action history gives a search side, never stale image coordinates.
            available.remove(preferred_arm);available.insert(0,preferred_arm)
        side=available[0];self.inspection_attempts.add(side)
        action=dict(skill='inspect',name='inspect',arm=side,observation_id=obs['observation_id'],
                    perception_recovery=True,active_inspection=True,perception_failure=failure)
        if anchor is not None:action['inspection_anchor']=anchor
        return action

    def _schedule_regrasp(self, action, result):
        """Preserve the model's goal, change the grasp topology once if needed."""
        if action.get('skill')!='place' or not str(result.get('failure','')).startswith('No reachable placement orientation'):
            return False
        if self.index>=len(self.steps):return False
        step=self.steps[self.index]
        if step['operation']!='transfer' or step.get('regrasp_stage') or step.get('arm_required'):return False
        key=object_key(step['source'])
        if key in self.staged_sources:return False
        instruction=self.instruction.split('Completion requirements:')[0]
        if re.search(r'\b(?:left|right|both|each|other)[ -]+(?:arm|hand|gripper)s?\b',instruction,re.I):return False
        side=action['arm'];other='right' if side=='left' else 'left';sensors=result.get('sensors',{})
        if not sensors.get(side,{}).get('holding') or sensors.get(other,{}).get('holding'):return False
        final=dict(step,arm=other,arm_required=True,regrasp_after_stage=True)
        staging=dict(operation='transfer',source=step['source'],destination='visible free shared table region',
                     relation='on',arm=side,grasp_part=step.get('grasp_part','body'),regrasp_stage=True)
        self.steps[self.index:self.index+1]=[staging,final]
        self.staged_sources.add(key);self.arm=side;self.phase='place';self.failures=0
        self.revision+=1;self.bound_revision=None
        return True

    def feedback(self, action, result):
        self.memory.feedback(action, result)
        if action.get('support_boundary_inspection'):
            self.support_observer_to_home=action['arm']
            self.bound_revision=None
            return
        if action.get('support_observer_home'):
            self.bound_revision=None
            return
        if action.get('active_inspection'):
            self.bound_revision=None
            self.failures=0 if result.get('skill_success') else 2
            return
        if action.get('skill')=='pick' and result.get('skill_success'):
            source=object_key(action.get('target',''))
            self.working_references.discard(source)
            if action.get('contact_reference'):self.working_references.add(source)
        if (result.get('geometry') or {}).get('retained_working_reference'):
            source=object_key(action.get('mission_intent',{}).get('source',''))
            if source:self.working_references.add(source)
        if action.get('support_rendezvous'):
            self.support_rendezvous_pending=bool(result.get('skill_success') and
                                                (result.get('geometry') or {}).get('staged_support_translation'))
            self.failures=0 if result.get('skill_success') else 2
            return
        if action.get('perception_recovery'):
            # A successful grip-preserving view shift changes only visibility.
            # Keep the pending goal/phase and re-ground it in the next frame;
            # rebuilding the entire mission can lose the existing identities.
            self.failures=0 if action.get('held_view_recovery') and result.get('skill_success') else 2
            return
        if action.get('mission_recovery'):
            return
        if not result.get('skill_success'):
            self.failures += 1
            if self.failures>=2:self._schedule_regrasp(action,result)
            return
        self.failures = 0
        if self.index >= len(self.steps):
            return
        op = self.steps[self.index]['operation']
        if op=='handover':
            if action['skill']=='pick':self.phase='offer'
            elif action['skill']=='present':self.phase='receive'
            elif action['skill']=='handover':self._finish_step()
            return
        if op=='bimanual_lift' and action['skill']=='grasp_handle':
            return
        if self.phase == 'home':
            self._finish_step()
        elif op in ('transfer','lift','tool_contact') and action['skill'] == 'pick':
            self.arm = action['arm']
            self.phase = 'place'
            if op == 'lift' and self.steps[self.index].get('location','stay') == 'stay':
                self._finish_step()
        elif op == 'transfer' and action['skill'] == 'place':
            self.phase = 'home'
        else:
            self._finish_step()
