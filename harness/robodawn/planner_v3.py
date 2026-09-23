"""Robodawn v3 visual skill planner."""
import json
import numpy as np
from PIL import Image
from robotwin_harness_v3 import CONTEXT, SkillError, validate, parse_json, ground_visual

class RobodawnPlanner:
    def __init__(self,client,instruction):self.client,self.instruction=client,instruction
    def plan(self,obs,history):
        if obs['success']:return {'name':'done','skill':'done'}
        prior=[{'action':r['action'],'result':{k:r['result'].get(k) for k in ('skill_success','failure','sensors','geometry')}} for r in history[-5:]]
        prompt=CONTEXT.replace('Image A is head RGB, B left wrist, C right wrist.','ONLY the HEAD RGB camera is attached; all coordinates and object identities refer to it.')+'\nTask: '+self.instruction+'\nMeasured robot state: '+json.dumps({k:obs[k] for k in ('endpose','sensors','table_height_m')})+'\nHistory: '+json.dumps(prior)
        images=[np.asarray(Image.open(p)) for p in obs['image_paths']]
        for attempt in range(3):
            raw=self.client.complete_text(prompt,images[0],max_tokens=550,temperature=0).raw_text
            try:
                a=validate(parse_json(raw))
                if a['skill']=='done' and not obs['success'] and attempt<2:
                    raise SkillError('Official goal not yet achieved. Check precise centring, object orientation, release and remaining objects. Choose a corrective skill instead of DONE.')
                if a['skill'] in ('pick','grasp_handle','push','place','press','reach','arc'):a=ground_visual(self.client,obs,a,self.instruction,keep_arm=True)
                state=obs['sensors'].get(a.get('arm'),{})
                if a['skill'] in ('pick','grasp_handle','home') and state.get('holding'):raise SkillError('Selected arm is already holding an object; choose place/present/move, or the other empty arm.')
                if a['skill'] in ('place','present','shake') and not state.get('holding'):raise SkillError('Selected arm is empty. Use the arm with measured holding=True, or pick first.')
                a.update(name=a['skill'],raw=raw,observation_id=obs['observation_id']);return a
            except (SkillError,ValueError,KeyError) as exc:
                if attempt==2:raise SkillError('Invalid policy action after repair: '+str(exc)) from exc
                prompt+='\nYour previous output was rejected: '+raw+'\nReason: '+str(exc)+'\nReturn a corrected single JSON action.'


