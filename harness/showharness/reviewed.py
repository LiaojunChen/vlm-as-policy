"""Show-Harness adapter for the reviewed diagnostic."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reviewed_interface import CONTEXT, GROUNDED, DIRECTIONS, ContractError, parse_command, validate_action

STAGE_SKILLS={'GRASP':'pick_at','PICK':'pick_at','PLACE':'place_at','RELEASE':'place_at',
              'LIFT':'lift','REACH':'reach_at','MOVE':'reach_at','APPROACH':'reach_at',
              'TRANSFER':'reach_at','RETREAT':'lift','WAIT':'wait'}


def stage_skill(motion):
    if motion not in STAGE_SKILLS:
        raise ContractError('unsupported_stage',f'Stage {motion} has no implemented skill')
    return STAGE_SKILLS[motion]


class ReviewedShowPolicy:
    """Original planner/controller roles; explicit embodiment skill semantics."""
    def __init__(self,client,grounder,caps,instruction,directory):
        from .runtime import ensure_upstream
        ensure_upstream()
        from plugins.subgoal.dual_agent import DualSubgoalPlannerAgent
        from plugins.subgoal.dual_plugin import DualSubgoalPlanner
        from core.vlm.dual_roles import DualControllerAgent
        self.client,self.grounder,self.caps,self.instruction=client,grounder,caps,instruction
        self.directory=Path(directory);self.tracks=None;self.index=0;self.replans=0;self.history=[]
        planner_prompt=('Task: {task}\nOnly the right arm exists; left must be []. '
            'Create semantic stages using GRASP and PLACE whenever needed. GRASP is a complete '
            'approach/close/lift macro, so do not add approach/lift stages for basic picking. '
            'PLACE transfers to a visible destination, lowers and releases, then retreats. '
            'Other available stages: REACH, LIFT, RETREAT, WAIT. Never use unsupported stages. '
            'Keep target object and contact affordance explicit, including colour. '
            'Return JSON {{"left":[],"right":[{{"id":"s1","motion":"GRASP",'
            '"target":"object name","affordance":"contact part","description":"short goal",'
            '"completion":"sensor-observable completion"}}]}}. Include only stages needed for this task.')
        self.planner=DualSubgoalPlanner(DualSubgoalPlannerAgent(client,CONTEXT,planner_prompt))
        template=('Task: {task}\nLEFT unavailable: keep STILL.\n'
            'RIGHT current stage: {stage_r}; target: {target_r}; contact: {afford_r}.\n'
            'Goal: {description_r}; completion: {completion_r}.\n{recovery_r}\n'
            'EMBODIMENT SKILL MODE: at GRASP/PICK select GRASP to run the entire grounded '
            'pick-and-lift skill; at PLACE/RELEASE select RELEASE to transfer to the destination '
            'and release. At REACH/MOVE/APPROACH/TRANSFER select GRASP to approach WITHOUT closing. '
            'At LIFT/RETREAT select MV_UP for an 8 cm lift. '
            'Other movement tokens move exactly 2 cm in their world direction. '
            'STILL waits. Never select DONE to skip a stage; only verified skill feedback advances it. '
            '{output_contract}')
        self.controller=DualControllerAgent(client,template,CONTEXT)

    def decide(self,obs):
        images=[np.asarray(Image.open(p).convert('RGB')) for p in obs['image_paths']]
        if self.tracks is not None and self.index>=len(self.tracks):
            if self.replans>=1:return {'name':'done'}, {'reason':'tracks_exhausted'}
            self.tracks=None;self.replans+=1;self.index=0
        if self.tracks is None:
            tracks,raw=self.planner.plan(self.instruction,*images)
            (self.directory/f'plan_{self.replans}.json').write_text(raw,encoding='utf-8')
            if tracks['left']:raise ContractError('unavailable_arm','Planner assigned stages to nonexistent left arm')
            if json.loads(raw).get('planner_fallback'):raise ContractError('invalid_model_output','Planner fallback is not a valid plan')
            self.tracks=tracks['right']
            for s in self.tracks:stage_skill(s.motion)
            if not self.tracks:raise ContractError('invalid_model_output','Empty right track')
        stage=self.tracks[self.index].to_prompt_dict();expected=stage_skill(stage['motion'])
        self.controller.common_context=CONTEXT+'\nMeasured pose and sensing: '+json.dumps({
            'pose':obs['endpose']['right_endpose'],'gripper':obs['gripper_sensor']})
        decision=self.controller.decide(self.instruction,{'left':None,'right':stage},
            {'left':'OPEN','right':'CLOSED' if obs['gripper_sensor']['holding'] else 'OPEN'},
            {'left':'','right':str(self.history[-3:])},{'left':'','right':''},
            {'left':'','right':str(self.history[-1:])},{'left':None,'right':None},*images)
        if self.client.last_error:raise RuntimeError(self.client.last_error)
        if decision.payload.get('fallback'):raise ContractError('invalid_model_output','Controller fallback rejected')
        if decision.tokens['left']!='STILL':raise ContractError('unavailable_arm','Controller moved unavailable left arm')
        token=decision.tokens['right']
        meta={'raw':decision.raw_text,'stage':stage,'expected_skill':expected,'controller_tokens':decision.tokens}
        action={'arm':'right','stage_id':stage['id'],'controller_token':token}
        if token in ('DONE','STILL'):
            return {'name':'wait','duration':.2},dict(meta,advance_denied=token=='DONE')
        if ((expected=='pick_at' and token=='GRASP') or
            (expected=='place_at' and token=='RELEASE') or
            (expected=='reach_at' and token=='GRASP')):
            target=f"{stage['affordance']} of {stage['target']}"
            grounding=self.grounder.locate(obs['image_paths'][0],target)
            action.update(name=expected,target_description=target,pixel=grounding['pixel'],
                          grounding=grounding,observation_id=obs['observation_id'])
        elif expected=='lift' and token=='MV_UP':
            action.update(name='lift',distance=.08)
        elif token in DIRECTIONS:
            action.update(name='move_delta',delta=DIRECTIONS[token])
        elif token=='GRASP':action.update(name='close_gripper')
        elif token=='RELEASE':action.update(name='open_gripper')
        else:raise ContractError('invalid_model_output',f'Unsupported controller token {token}')
        return validate_action(action,self.caps,obs['observation_id']),meta

    def feedback(self,action,result,meta):
        self.history.append({'action':action['name'],'skill_success':result.get('skill_success'),
                             'failure':result.get('failure'),'gripper_sensor':result.get('gripper_sensor')})
        if (not meta.get('advance_denied') and action['name']==meta.get('expected_skill')
                and result.get('skill_success')):
            self.index+=1
