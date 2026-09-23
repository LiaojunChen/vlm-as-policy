"""Show-Harness native dual subgoal/controller roles, ported to RoboTwin EE control."""
from pathlib import Path
import json
import copy
import numpy as np
from PIL import Image
from .runtime import UPSTREAM_ROOT, ensure_upstream

ensure_upstream()
from robotwin_bridge import SIDES, FRAME_CONTEXT, VECTORS, PolicyOutputError
from core.vlm.dual_roles import DualControllerAgent
from plugins.subgoal.dual_agent import DualSubgoalPlannerAgent
from plugins.subgoal.dual_plugin import DualSubgoalPlanner
from plugins.mem_text.plugin import MemTextPlugin

class SemanticPlanClient:
    """Constrain stage labels when the transport supports JSON-schema decoding."""
    def __init__(self,client): self.client=client

    def __getattr__(self,name): return getattr(self.client,name)

    def complete_json(self,*args,schema=None,**kwargs):
        if schema is not None:
            schema=copy.deepcopy(schema)
            for side in SIDES:
                schema['properties'][side]['items']['properties']['motion']['enum']=[
                    'GRASP','LIFT','TRANSFER','LOWER','RELEASE','RETREAT','WAIT']
        return self.client.complete_json(*args,schema=schema,**kwargs)

class FineMoveMemory(MemTextPlugin):
    def render_rules(self):
        return ('- Re-evaluate target versus fingertips after EVERY small move. '
                'Direction reversal is allowed when the current view shows overshoot.\n'
                '- Repeated moves are not evidence of progress. Use current images and measured pose.\n'
                '- A closed gripper alone does not prove a successful grasp; look for the object held between fingers.')

FINE_PLANNER = '''TASK: {task}
Create an ordered semantic stage list for each arm from the images.
The two arms execute concurrently. Respect the task's arm assignments.
The motion field MUST be exactly one of: GRASP, LIFT, TRANSFER, LOWER, RELEASE, RETREAT, WAIT.
GRASP includes approaching, aligning, lowering AND closing on an object; completion means object held.
After GRASP, LIFT before TRANSFER. LOWER the held object onto its support before RELEASE.
After RELEASE, RETREAT. Use WAIT for dependencies on the other arm.
Every active pick/place/stack track must contain GRASP and RELEASE.
Describe visible goals, not fixed movement directions. Give each stage a unique id.
Keep descriptions and completion conditions short.
Return JSON with this item schema for ALL stages in both lists:
{{"left":[{{"id":"grasp_object","target":"object name","affordance":"main body",
"motion":"GRASP","description":"Approach, align, lower and close on the object",
"completion":"Object held between closed fingers"}}],"right":[]}}
The example shows the schema only. Include all necessary stages for the actual task.
Return JSON only.
'''

FINE_CONTROLLER = '''TASK: {task}
Images: HEAD, LEFT WRIST, RIGHT WRIST.
LEFT: stage={stage_l}, target={target_l}, contact={afford_l}, gripper command={gripper_l}
Goal: {description_l}; completed only when: {completion_l}
{mem_l}
{recovery_l}
RIGHT: stage={stage_r}, target={target_r}, contact={afford_r}, gripper command={gripper_r}
Goal: {description_r}; completed only when: {completion_r}
{mem_r}
{recovery_r}

Use the calibrated HEAD pixel projection of each robot's fingertip center to locate the gripper,
including when it is outside the image. Shadows are not grippers. Find the target in the RGB images.
World directions are given in context. Wrist image axes are NOT world axes.
Compare the target to THAT ARM'S fingertip center, not to the middle of the image.
Target HEAD pixel x greater than fingertip pixel x -> MV_RIGHT; smaller -> MV_LEFT.
Choose the axis that reduces the visible error. Inspect the newest images again next step.
For GRASP: approach, align, lower to object height, THEN issue GRASP to close the fingers.
An open gripper near an object is not a completed GRASP stage. Movement never closes the fingers.
For RELEASE: align and lower the held object to its support, THEN issue RELEASE to open the fingers.
Use the clearest available view for contact; an out-of-frame or occluded second view is unavailable evidence.
Do not grasp merely because the task asks for it; require alignment and contact evidence.
DONE advances just this arm's stage; use only when its completion condition is satisfied.
WAIT means STILL until the other arm's required result is visible, then DONE.
Avoid collisions; the second arm waits until a shared destination is clear.
{mem_rules}
{output_contract}
'''

class ShowPolicy:
    def __init__(self,client,instruction,directory,frame_context=FRAME_CONTEXT,prompt_mode='legacy'):
        self.client,self.instruction,self.directory=client,instruction,Path(directory)
        self.frame_context=frame_context
        self.prompt_mode=prompt_mode
        template=(UPSTREAM_ROOT/'prompts/controller_dual.txt').read_text(encoding="utf-8")
        # The shipped wrist-relative Piper directions require a different interpreter.
        # Replace only that embodiment-specific block for the fixed-world EE adapter.
        start=template.index('DIRECTION (per arm)')
        end=template.index('COORDINATION:')
        template=template[:start]+'DIRECTION: Use the fixed world mapping in the context, with HEAD as the direction reference. Wrist images help judge contact and alignment.\n\n'+template[end:]
        if prompt_mode=='fine': template=FINE_CONTROLLER
        self.controller=DualControllerAgent(client,template,frame_context,
            mem_text_plugin=FineMoveMemory() if prompt_mode=='fine' else MemTextPlugin())
        planner_context=frame_context
        if prompt_mode=='fine':
            # Atomic direction names in planner context induced directional stage loops.
            planner_context='RoboTwin: image order is HEAD, LEFT WRIST, RIGHT WRIST. Plan object-level goals.'
        self.planner=DualSubgoalPlanner(DualSubgoalPlannerAgent(
            SemanticPlanClient(client) if prompt_mode=='fine' else client,planner_context,
            prompt_template=FINE_PLANNER if prompt_mode=='fine' else None))
        self.tracks=None
        self.indices={s:0 for s in SIDES}
        self.history={s:[] for s in SIDES}
        self.replans=0
        self.last_decision=None
        self.last_execution=None
        self.stage_steps={s:0 for s in SIDES}

    def _plan(self,images):
        for attempt in range(2 if self.prompt_mode=='fine' else 1):
            self.tracks,raw=self.planner.plan(self.instruction,*images)
            (self.directory/f'plan_{self.replans}_attempt_{attempt}.json').write_text(raw,encoding='utf-8')
            errors=[]
            if self.prompt_mode=='fine':
                manipulation=any(word in self.instruction.lower() for word in ('stack','pick','place','block'))
                for side,track in self.tracks.items():
                    motions={stage.motion for stage in track}
                    if motions & set(VECTORS): errors.append(f'{side}: movement tokens are not semantic stages')
                    if track and manipulation and not {'GRASP','RELEASE'}<=motions:
                        errors.append(f'{side}: manipulation requires explicit GRASP and RELEASE stages')
                if json.loads(raw).get('planner_fallback'): errors.append('planner returned fallback')
            if not errors:
                (self.directory/f'plan_{self.replans}.json').write_text(raw,encoding='utf-8')
                return
            self.planner.agent.common_context+='\nPrevious plan invalid: '+'; '.join(errors)+'. Generate a corrected complete plan.'
        raise PolicyOutputError('Invalid semantic plan: '+'; '.join(errors))

    def observe_execution(self,result):
        self.last_execution=result

    def decide(self,obs):
        images=[np.array(Image.open(p).convert('RGB')) for p in obs['image_paths']]
        if self.tracks is None:
            self._plan(images)
        goals={s:(self.tracks[s][self.indices[s]].to_prompt_dict() if self.indices[s]<len(self.tracks[s]) else None) for s in SIDES}
        state=obs['endpose']
        self.controller.common_context=self.frame_context+'\nMeasured EE pose (world XYZ, quaternion WXYZ) and gripper: '+json.dumps(state)
        recovery={s:'' for s in SIDES}
        if self.prompt_mode=='fine':
            compact_state={key:np.round(value,4).tolist() if isinstance(value,list) else round(value,3)
                for key,value in state.items()}
            self.controller.common_context=self.frame_context+'\nMeasured EE poses and commanded gripper aperture (not contact sensing): '+json.dumps(compact_state)
            geometry={s:{'tcp_xyz_m':g['tcp_xyz_m'],'head_projection':g['tcp_projections']['head_camera']}
                      for s,g in obs.get('robot_geometry',{}).items()}
            self.controller.common_context+='\nRobot fingertip geometry (kinematics, not object positions): '+json.dumps(geometry)
            for s in SIDES:
                recovery[s]=f'Steps in current stage: {self.stage_steps[s]}.'
                if len(self.history[s])>=8 and len(set(self.history[s][-8:]))==1:
                    recovery[s]+=' Last 8 actions identical. Recheck target relative to fingertips; change direction if the error increased.'
                if self.last_execution:
                    motion=self.last_execution.get('motion',{}).get(s,{})
                    recovery[s]+=' Last execution: '+json.dumps({
                        'actual_delta_mm':np.round(np.array(motion.get('actual_delta_m',[0,0,0]))*1000,2).tolist(),
                        'target_error_mm':round(motion.get('position_error_m',0)*1000,2),
                        'planner_status':self.last_execution.get('planner',{}).get(s,{}).get('status')})
        decision=self.controller.decide(self.instruction,goals,
            {s:'OPEN' if state[s+'_gripper']>.5 else 'CLOSED' for s in SIDES},
            {s:', '.join(self.history[s][-5:][::-1]) for s in SIDES},
            {s:self.history[s][-1] if self.history[s] else '' for s in SIDES},
            recovery,{s:None for s in SIDES},*images)
        self.last_decision=decision
        if self.client.last_error:
            raise RuntimeError('Model transport error: '+self.client.last_error)
        if self.prompt_mode=='fine' and decision.payload.get('fallback'):
            raise PolicyOutputError('Controller fallback: '+decision.raw_text)
        tokens=decision.tokens.copy()
        for s in SIDES:
            self.stage_steps[s]+=1
            if goals[s] is None: tokens[s]='STILL'
            elif tokens[s]=='DONE':
                self.indices[s]+=1;tokens[s]='STILL'
                self.stage_steps[s]=0
            self.history[s].append(tokens[s])
        done=all(self.indices[s]>=len(self.tracks[s]) for s in SIDES)
        if done and not obs['success'] and self.replans<1:
            self.tracks=None;self.indices={s:0 for s in SIDES};self.replans+=1;done=False
        return tokens,done
