"""Robodawn primitive-action planner."""
import json
import numpy as np
from PIL import Image
from robotwin_bridge import FRAME_CONTEXT, SIDES, VECTORS, PolicyOutputError

class RobodawnPlanner:
    def __init__(self,client,frame_context=FRAME_CONTEXT): self.client,self.frame_context=client,frame_context
    def plan(self,observation,history):
        if observation['success']: return {'name':'done','reason':'environment success'}
        images=[np.array(Image.open(p).convert('RGB')) for p in observation['image_paths']]
        past=[{'action':r['action'],'result':r.get('result')} for r in history[-5:]]
        prompt=(self.frame_context+'\nGoal: '+observation['instruction']+
          '\nYou control one arm each step using Robodawn primitives. pick=CLOSE GRIPPER ONLY; place=OPEN GRIPPER ONLY; '
          'push/pull=MOVE one step of the size stated above; noop=hold; done=end episode. '
          'Move to align with an object before pick. Use no fold action on these rigid-body tasks.\n'
          'Return JSON {"name":"pick|place|push|pull|noop|done","arm":"left|right",'
          '"direction":"MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK|MV_UP|MV_DOWN","reason":"short visual reasoning"}.'+
          '\nRobot state: '+json.dumps(observation['endpose'])+'\nRecent history: '+json.dumps(past))
        try:
            response=self.client.complete_json(prompt,images[0],wrist_image=images[1:],temperature=0,max_tokens=512)
        except RuntimeError as exc:
            if not self.client.last_error:
                raise PolicyOutputError(str(exc)) from exc
            raise
        action=response.payload['json']
        if action.get('name') not in ('pick','place','push','pull','noop','done'):
            raise PolicyOutputError('unsupported action: '+json.dumps(action))
        if action.get('name') not in ('noop','done') and action.get('arm') not in SIDES:
            raise PolicyOutputError('invalid arm: '+json.dumps(action))
        if action.get('name') in ('push','pull') and action.get('direction') not in VECTORS:
            raise PolicyOutputError('invalid movement direction: '+json.dumps(action))
        return action
