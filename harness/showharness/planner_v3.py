"""Native Show-Harness planning/controller roles with verified skill transitions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robotwin_harness_v3 import CONTEXT,SkillError,parse_json,validate,SIDES,refine_colour
from .runtime import ensure_upstream

ensure_upstream()

from plugins.subgoal.dual_agent import DualSubgoalPlannerAgent
from plugins.subgoal.dual_plugin import DualSubgoalPlanner
from core.vlm.dual_roles import DualControllerAgent

MOTIONS={'GRASP':'pick','PICK':'pick','PLACE':'place','RELEASE':'place','PRESS':'press',
         'REACH':'reach','APPROACH':'reach','PRESENT':'present','HOME':'home','SHAKE':'shake',
         'ROTATE':'rotate','MOVE':'move','OPEN':'open','CLOSE':'close','WAIT':'wait','LIFT':'move','GRASP_HANDLE':'grasp_handle','ARC':'arc','PUSH':'push'}
class ShowPlanner:
 def __init__(self,client,instruction,directory):
  self.client,self.instruction,self.directory=client,instruction,Path(directory)
  self.indices={s:0 for s in SIDES};self.tracks=None;self.history={s:[] for s in SIDES};self.replans=0;self.failures=0
  context=CONTEXT.split('Available skills:')[0].replace('Image A is head RGB, B left wrist, C right wrist.','Only HEAD RGB is attached.')
  prompt=('Task: {task}\nCreate short ordered stages per arm using GRASP, PLACE, PRESS, PRESENT, HOME, '
          'SHAKE, ROTATE, MOVE, GRASP_HANDLE, ARC, PUSH, OPEN, CLOSE, WAIT. GRASP includes approach, close and 12 cm lift; '
          'GRASP_HANDLE approaches and closes without lifting an appliance; ARC follows a visible door/lid hinge. '
          'PLACE includes transfer/lower/release/retreat. For lift/show/raise tasks use GRASP then PRESENT. '
          'PRESENT retains and brings the object towards the front-centre at 1 metre TCP height. '
          'Avoid redundant approach/lift/release stages. The other empty arm can stay []. '
          'For placements use HOME after release when another object remains. '
          'Return JSON {{"left":[{{"id":"l1","motion":"GRASP","target":"object name and colour",'
          '"affordance":"graspable body","description":"goal","completion":"measurable condition"}}],"right":[]}}. '
          'Use current visual scene and feedback; do not repeat completed stages.')
  self.planner=DualSubgoalPlanner(DualSubgoalPlannerAgent(client,context,prompt))
  controller=('Task: {task}\nLEFT stage {stage_l}: {description_l}, target {target_l}, contact {afford_l}. '
              'Feedback: {recovery_l}\nRIGHT stage {stage_r}: {description_r}, target {target_r}, contact {afford_r}. '
              'Feedback: {recovery_r}\nEMBODIMENT SKILL MODE: Use the current stage motion as the output token. '
              'GRASP also activates the current stage macro; RELEASE activates PLACE/OPEN. '
              'PRESS, PRESENT, HOME, SHAKE, ROTATE, MOVE, GRASP_HANDLE and ARC are supported physical macro tokens. '
              'Do not issue approach direction tokens before GRASP: the macro already approaches. '
              'Only use STILL for an absent/waiting stage. DONE cannot skip a stage. '
              'A stage advances only after verified execution.\n{output_contract}\n'
              'Use the exact supported macro token. Example for a right PRESS stage: '
              '{{"left":"STILL","right":"PRESS","reasoning":"Activate right press skill"}}. '
              'Example for a left PLACE stage: {{"left":"RELEASE","right":"STILL","reasoning":"Activate left placement"}}.')
  self.controller=DualControllerAgent(client,controller,context)
 def plan(self,obs,history):
  if obs['success']:return {'name':'done','skill':'done'}
  images=[np.asarray(Image.open(obs['image_paths'][0])),None,None]
  if self.tracks is None or all(self.indices[s]>=len(self.tracks[s]) for s in SIDES):
   self.planner.agent.common_context=CONTEXT.split('Available skills:')[0].replace('Image A is head RGB, B left wrist, C right wrist.','Only HEAD RGB is attached.')+'\nSensors and prior execution: '+json.dumps({'sensors':obs['sensors'],'history':self.history})
   self.planner.agent.common_context+='\nOnly symbolic stages here. Every target and affordance must be an object/part name STRING. Do not include coordinates or executable action parameters in a stage.'
   for attempt in range(3):
    try:
     tracks,raw=self.planner.plan(self.instruction,*images)
     (self.directory/f'plan_{self.replans}_attempt{attempt}.json').write_text(raw,encoding='utf-8')
     payload=parse_json(raw)
     if payload.get('planner_fallback'):raise SkillError('Native planner fallback rejected')
     for side in SIDES:
      for item in payload[side]:
       if not isinstance(item.get('target'),str):raise SkillError('Stage target must be an object-name STRING, never a bbox object')
      for stage in tracks[side]:
       if stage.motion not in MOTIONS:raise SkillError('Unsupported stage '+stage.motion)
     self.tracks=tracks;break
    except (ValueError,RuntimeError) as exc:
     if isinstance(exc,RuntimeError) and not any(t in str(exc) for t in ('Dual planner returned two empty tracks','Dual planner did not return JSON object','Dual planner JSON has no')):raise
     if attempt==2:raise SkillError('Invalid native model plan after repair: '+str(exc)) from exc
     self.planner.agent.common_context+='\nPrevious plan rejected: '+str(exc)+'. Return corrected symbolic stages with string names and supported motions.'
   (self.directory/f'plan_{self.replans}.json').write_text(raw,encoding='utf-8');self.replans+=1;self.indices={s:0 for s in SIDES}
  stages={s:self.tracks[s][self.indices[s]].to_prompt_dict() if self.indices[s]<len(self.tracks[s]) else None for s in SIDES}
  self.controller.common_context=CONTEXT.split('Available skills:')[0].replace('Image A is head RGB, B left wrist, C right wrist.','Only HEAD RGB is attached.')+'\nSensing: '+json.dumps(obs['sensors'])
  for attempt in range(3):
   decision=self.controller.decide(self.instruction,stages,{s:'CLOSED' if obs['sensors'][s]['holding'] else 'OPEN' for s in SIDES},
       {s:str(self.history[s][-3:]) for s in SIDES},{s:'' for s in SIDES},{s:str(self.history[s][-2:]) for s in SIDES},{s:None for s in SIDES},*images)
   if not decision.payload.get('fallback'):break
   self.controller.common_context+='\nPrevious response was rejected: '+decision.raw_text+'\nUse only the explicitly listed supported controller tokens, absent stage with STILL.'
  else:raise SkillError('Native controller fallback rejected after 3 attempts')
  active=[s for s in SIDES if stages[s] and decision.tokens[s] not in ('STILL','DONE')]
  if not active:return {'skill':'wait','name':'wait','arm':'right','controller_raw':decision.raw_text}
  side=active[0];stage=stages[side];stage_skill=MOTIONS[stage['motion']];skill=stage_skill
  if skill=='pick' and obs['sensors'][side]['holding'] and self.history[side] and 'lift' in str(self.history[side][-1].get('failure','')).lower():skill='present'
  token=decision.tokens[side]
  if token in MOTIONS and token not in ('GRASP','RELEASE'):skill=MOTIONS[token]
  if token.startswith('MV_'):
   delta={'MV_LEFT':[-.02,0,0],'MV_RIGHT':[.02,0,0],'MV_FWD':[0,.02,0],'MV_BACK':[0,-.02,0],'MV_UP':[0,0,.02],'MV_DOWN':[0,0,-.02]}[token]
   return {'name':'move','skill':'move','arm':side,'delta':delta,'controller_raw':decision.raw_text}
  a={'skill':skill,'arm':side,'target':f"{stage['affordance']} of {stage['target']}",'stage_id':stage['id'],'stage_skill':stage_skill,'controller_raw':decision.raw_text}
  if skill in ('pick','grasp_handle','push','place','press','reach','arc','move','rotate','shake'):
   prompt=CONTEXT+'\nTask: '+self.instruction+'\nNative planner current stage: '+json.dumps(stage)+'\nMeasured robot state: '+json.dumps({'endpose':obs['endpose'],'sensors':obs['sensors']})+'\nPrevious failures: '+str(self.history[side][-3:])+f'\nGround/parameterize exactly this skill: {skill}, arm: {side}. Preserve object identity. Do not select a different skill.'
   raw=self.client.complete_text(prompt,images[0],max_tokens=500,temperature=0).raw_text
   grounded=parse_json(raw)
   if grounded.get('skill')!=skill or grounded.get('arm')!=side:raise SkillError('Grounder changed selected stage/arm')
   a.update(grounded);a['raw']=raw
  a.update(name=skill,observation_id=obs['observation_id'])
  if skill in ('pick','place','press','reach'):a=refine_colour(obs,a)
  return validate(a)
 def feedback(self,a,result):
  side=a.get('arm','right');self.history[side].append({'skill':a.get('skill'),'target':a.get('target'),'ok':result.get('skill_success'),'failure':result.get('failure')})
  if a.get('stage_id') and a.get('stage_skill')==a['skill'] and result.get('skill_success'):
   self.indices[side]+=1;self.failures=0
  elif not result.get('ok'):
   self.failures+=1
   if self.failures>=2:self.tracks=None;self.failures=0
