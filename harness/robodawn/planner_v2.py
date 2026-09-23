"""Grounded v2 policies; model choices and all sensor-derived targets are audited."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from robotwin_bridge import PolicyOutputError
from robotwin_grounding_v2 import FRAME_CONTEXT_V2, GridGrounder, SIDES


SKILLS = ('press_at', 'pick_at', 'place_at', 'reach_at')
ROBO_CODES = {f'{verb}_{arm}': (skill, 'left' if arm == 'L' else 'right')
              for verb, skill in [('PRESS', 'press_at'), ('PICK', 'pick_at'),
                                  ('PLACE', 'place_at'), ('REACH', 'reach_at')]
              for arm in ('L', 'R')}
ROBO_CODES['DONE'] = ('done', '')


class RobodawnGroundedPlanner:
    """One model-selected skill and visual target per high-level decision."""

    def __init__(self, client, grounder: GridGrounder, instruction: str):
        self.client = client
        self.grounder = grounder
        self.instruction = instruction

    def plan(self, observation: dict, history: list[dict]) -> dict:
        if observation['success']:
            return {'name': 'done', 'reason': 'official environment success'}
        prior = []
        for row in history[-4:]:
            action, result = row['action'], row['result']
            prior.append({
                'action': action['name'], 'arm': action.get('arm'),
                'target': action.get('target_description'),
                'ok': result.get('ok'), 'success': result.get('success'),
                'failure': result.get('failure'),
            })
        prompt = (
            f'{FRAME_CONTEXT_V2}\nTask: {self.instruction}\n'
            f'Previous actions and measured outcomes: {prior}\n'
            'Choose ONE next robot skill from PRESS_L, PRESS_R, PICK_L, PICK_R, '
            'PLACE_L, PLACE_R, REACH_L, REACH_R, DONE. '
            'PRESS touches a button/switch with a closed gripper; PICK opens, approaches, '
            'closes and lifts; PLACE approaches and releases; REACH approaches without '
            'contact. Use the correct arm from the task. DONE only when the visible '
            'goal is actually met. Give the precise visible contact part and colour. '
            'Reply in exactly one line, e.g. PRESS_R: top yellow button of blue bell. '
            'No JSON or explanation.'
        )
        image = np.asarray(Image.open(observation['image_paths'][0]).convert('RGB'))
        raw = self.client.complete_text(prompt, image, max_tokens=120,
                                        temperature=0).raw_text
        if self.client.last_error:
            raise RuntimeError('Model transport error: ' + self.client.last_error)
        match = re.search(r'\b(' + '|'.join(ROBO_CODES) + r')\b\s*[:：\-]?\s*([^\n]*)',
                          raw, re.I)
        if not match:
            raise PolicyOutputError(f'No supported skill code in {raw[:250]!r}')
        code = match.group(1).upper()
        name, side = ROBO_CODES[code]
        if name == 'done':
            return {'name': 'done', 'raw': raw}
        target = match.group(2).strip(' .;') or self.instruction
        # Limit accidental verbose explanations without inventing a target.
        target = re.split(r'\s+(?:because|since|as the)\s+', target,
                          maxsplit=1, flags=re.I)[0][:160]
        grounding = self.grounder.locate(observation['image_paths'][0], target)
        return {'name': name, 'arm': side, 'pixel': grounding['pixel'],
                'grounding': grounding, 'target_description': target,
                'raw': raw}


