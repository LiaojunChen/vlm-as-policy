"""Robodawn planner for the reviewed diagnostic."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reviewed_interface import CONTEXT, GROUNDED, DIRECTIONS, ContractError, parse_command, validate_action

class ReviewedRobodawnPlanner:
    def __init__(self,client,grounder,caps,instruction):
        self.client,self.grounder,self.caps,self.instruction=client,grounder,caps,instruction
    def plan(self,obs,history):
        if obs['success']:return {'name':'done'}
        prior=[{'action':r['action'],'result':{k:r['result'].get(k) for k in
                 ('ok','skill_success','failure_code','failure','gripper_sensor')}} for r in history[-3:]]
        prompt=(CONTEXT+f'\nTask: {self.instruction}\nCapabilities: {json.dumps(self.caps.to_dict())}'
                f"\nMeasured robot pose: {obs['endpose']['right_endpose']}"
                f"\nGripper sensing: {obs['gripper_sensor']}\nHistory: {json.dumps(prior)}\n"
                'Choose ONE command. PICK_R: visible contact part and object name (opens, approaches, '
                'closes, lifts and verifies holding); PLACE_R: visible destination (requires holding; '
                'transfers, lowers, releases, retreats); REACH_R: visible target (approaches only); '
                'LIFT_R: distance_metres; MOVE_R: dx dy dz; POSE_R: x y z; OPEN_R; CLOSE_R; '
                'WAIT: seconds; DONE. Use PICK then PLACE for a pick-and-place task. '
                'Give target colour and name. Output exactly one line, without prose or JSON.')
        images=[np.asarray(Image.open(p).convert('RGB')) for p in obs['image_paths']]
        raw=self.client.complete_text(prompt,images[0],wrist_image=images[1:],max_tokens=160,temperature=0).raw_text
        action=parse_command(raw,self.caps)
        if action['name'] in GROUNDED:
            grounding=self.grounder.locate(obs['image_paths'][0],action['target_description'])
            action.update(pixel=grounding['pixel'],grounding=grounding,observation_id=obs['observation_id'])
        return validate_action(action,self.caps,obs['observation_id'])


