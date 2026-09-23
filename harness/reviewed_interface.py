"""Versioned action contract and project adapters for the reviewed diagnostic."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image

ACTION_NAMES = ('reach_at', 'pick_at', 'place_at', 'lift', 'move_delta',
                'move_pose', 'open_gripper', 'close_gripper', 'wait', 'done')
GROUNDED = ('reach_at', 'pick_at', 'place_at')
DIRECTIONS = {'MV_LEFT':[-.02,0,0], 'MV_RIGHT':[.02,0,0],
              'MV_FWD':[0,.02,0], 'MV_BACK':[0,-.02,0],
              'MV_UP':[0,0,.02], 'MV_DOWN':[0,0,-.02]}
CONTEXT = ('MuJoCo diagnostic. Image A is HEAD top-down RGB, Image B is a fixed '
           'SIDE RGB camera, Image C is the RIGHT wrist RGB camera. Only RIGHT exists. '
           '+X is right in HEAD, +Y is up in HEAD, +Z is vertically up. Units are metres. '
           'Orientation is fixed. RGB target selection uses depth only inside the executor. '
           'Gripper contact is measured; closed fingers alone are not a successful grasp. ')


class ContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Capabilities:
    schema_version: str = 'review-1'
    arms: tuple = ('right',)
    actions: tuple = ACTION_NAMES
    image_wh: tuple = (320, 240)
    xyz_min: tuple = (-.25, -.20, .724)
    xyz_max: tuple = (.25, .20, 1.08)
    orientation_control: bool = False
    force_control: bool = False
    simultaneous_arms: bool = False
    tactile_contact: bool = True
    def to_dict(self):
        return asdict(self)


def finite_vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError('invalid_parameter', f'{label} must have {size} numbers')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ContractError('invalid_parameter', f'{label} contains nonfinite/non-numeric values')
    return np.asarray(value, dtype=float)


def validate_action(action, caps: Capabilities, observation_id=None):
    if not isinstance(action, dict) or action.get('name') not in caps.actions:
        raise ContractError('unsupported_action', f"Unsupported action: {action!r}")
    name = action['name']
    common = {'name','arm','raw','target_description','grounding','observation_id',
              'stage_id','controller_token'}
    specific = {'move_delta':{'delta'}, 'move_pose':{'xyz'}, 'lift':{'distance'},
                'wait':{'duration'}, **{n:{'pixel'} for n in GROUNDED}}
    extra = set(action) - common - specific.get(name,set())
    if extra:
        raise ContractError('unsupported_parameter', f'Unexpected action fields: {sorted(extra)}')
    if name not in ('wait','done') and action.get('arm') not in caps.arms:
        raise ContractError('unavailable_arm', f"Unavailable arm: {action.get('arm')}")
    if name in GROUNDED:
        pixel = action.get('pixel')
        if not isinstance(pixel,(list,tuple)) or len(pixel)!=2 or any(type(x) is not int for x in pixel):
            raise ContractError('invalid_pixel','Pixel must contain two integer coordinates')
        if not all(0 <= v < size for v,size in zip(pixel,caps.image_wh)):
            raise ContractError('invalid_pixel','Pixel outside camera image')
        if observation_id is None or action.get('observation_id') != observation_id:
            raise ContractError('stale_observation','Action must reference the latest observation')
    elif name in ('move_delta','move_pose'):
        vec=finite_vector(action.get('delta' if name=='move_delta' else 'xyz'),3,name)
        if name=='move_delta' and np.linalg.norm(vec)>.12:
            raise ContractError('out_of_workspace','Relative movement exceeds 12 cm')
        if name=='move_pose' and (np.any(vec< caps.xyz_min) or np.any(vec>caps.xyz_max)):
            raise ContractError('out_of_workspace','Absolute target outside workspace')
    elif name=='lift':
        distance=action.get('distance',.08)
        finite_vector([distance],1,'distance')
        if not .01<=distance<=.15:
            raise ContractError('invalid_parameter','Lift must be between 1 and 15 cm')
    elif name=='wait':
        duration=action.get('duration',.4)
        finite_vector([duration],1,'duration')
        if not .02<=duration<=2:
            raise ContractError('invalid_parameter','Wait must be between .02 and 2 seconds')
    return action


CODE_NAMES={'PICK':'pick_at','PLACE':'place_at','REACH':'reach_at','LIFT':'lift',
            'MOVE':'move_delta','POSE':'move_pose','OPEN':'open_gripper','CLOSE':'close_gripper'}


def parse_command(raw, caps):
    text=raw.strip()
    if text=='DONE':return {'name':'done','raw':raw}
    if text.startswith('WAIT'):
        match=re.fullmatch(r'WAIT(?:\s*:\s*([0-9.]+))?',text)
        if not match:raise ContractError('invalid_model_output','Invalid WAIT syntax')
        try:duration=float(match.group(1) or .4)
        except ValueError as exc:raise ContractError('invalid_model_output','Invalid WAIT duration') from exc
        return {'name':'wait','duration':duration,'raw':raw}
    match=re.fullmatch(r'([A-Z]+)_([LR])(?:\s*:\s*([^\n]+))?',text)
    if not match or match.group(1) not in CODE_NAMES:
        raise ContractError('invalid_model_output','Expected exactly one supported action line')
    code,arm,argument=match.groups()
    action={'name':CODE_NAMES[code],'arm':'right' if arm=='R' else 'left','raw':raw}
    if action['arm'] not in caps.arms:
        raise ContractError('unavailable_arm','Model requested an unavailable arm')
    if action['name'] in GROUNDED:
        if not argument or not argument.strip():
            raise ContractError('invalid_model_output','Visual action requires a target description')
        action['target_description']=argument.strip()
    elif code in ('MOVE','POSE','LIFT'):
        if argument is None:raise ContractError('invalid_model_output','Numerical parameter missing')
        try:values=[float(x) for x in argument.split()]
        except ValueError as exc:raise ContractError('invalid_model_output','Invalid numerical parameter') from exc
        if code=='LIFT':
            if len(values)!=1:raise ContractError('invalid_model_output','LIFT needs one distance')
            action['distance']=values[0]
        else:action['delta' if code=='MOVE' else 'xyz']=values
    elif argument:
        raise ContractError('invalid_model_output','Gripper primitive takes no target argument')
    return action




def __getattr__(name):
    """Resolve legacy policy imports without coupling shared code to a policy."""
    from importlib import import_module
    exports = {'ReviewedRobodawnPlanner': 'robodawn.reviewed', 'ReviewedShowPolicy': 'showharness.reviewed', 'STAGE_SKILLS': 'showharness.reviewed', 'stage_skill': 'showharness.reviewed'}
    if name in exports:
        return getattr(import_module(exports[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
