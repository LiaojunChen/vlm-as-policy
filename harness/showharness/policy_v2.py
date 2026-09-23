"""Grounded v2 policies; model choices and all sensor-derived targets are audited."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from robotwin_bridge import PolicyOutputError
from robotwin_grounding_v2 import FRAME_CONTEXT_V2, GridGrounder, SIDES


from .policy_v1 import ShowPolicy


class ShowGroundedPolicy:
    """Native dual planner/controller roles with a grounded stage executor."""

    def __init__(self, client, instruction: str, directory: Path,
                 grounder: GridGrounder, *, ground_with_object: bool = False):
        self.native = ShowPolicy(client, instruction, directory,
                                 frame_context=FRAME_CONTEXT_V2)
        self.instruction = instruction
        self.grounder = grounder
        self.ground_with_object = ground_with_object
        self.attempts = {side: 0 for side in SIDES}

    @staticmethod
    def _stage_skill(motion: str, instruction: str) -> str | None:
        m = motion.upper()
        if m in ('WAIT', 'RETREAT', 'DONE', 'STOP'):
            return None
        if m in ('MOVE', 'APPROACH', 'REACH', 'ALIGN', 'PREGRASP'):
            return 'reach_at'
        if m in ('RELEASE', 'PLACE', 'DROP', 'INSERT'):
            return 'place_at'
        if m in ('PRESS', 'PUSH', 'TOUCH', 'CLICK'):
            return 'press_at'
        if m in ('GRASP', 'PICK', 'GRIP', 'PULL'):
            return 'press_at' if re.search(r'\b(?:press|click|push|touch)\b',
                                           instruction, re.I) else 'pick_at'
        return 'reach_at'

    def decide(self, observation: dict) -> tuple[list[dict], dict]:
        tokens, done = self.native.decide(observation)
        if self.native.tracks is None:
            # Native planner clears exhausted tracks to request a replan on
            # the next observation. There is no current stage to execute.
            return [], {'controller_raw': self.native.last_decision.raw_text,
                        'controller_tokens': tokens,
                        'stages': {side: None for side in SIDES},
                        'indices': self.native.indices.copy(),
                        'native_done': False, 'replan_pending': True}
        stages = {}
        for side in SIDES:
            idx = self.native.indices[side]
            track = self.native.tracks[side]
            stages[side] = track[idx].to_prompt_dict() if idx < len(track) else None
        decision = {
            'controller_raw': self.native.last_decision.raw_text,
            'controller_tokens': tokens, 'stages': stages,
            'indices': self.native.indices.copy(), 'native_done': done,
        }
        actions = []
        for side in SIDES:
            stage = stages[side]
            if stage is None or tokens[side] in ('STILL', 'DONE'):
                continue
            skill = self._stage_skill(stage['motion'], self.instruction)
            if skill is None:
                self.native.indices[side] += 1
                continue
            target = stage.get('affordance') or stage.get('target') or self.instruction
            if self.ground_with_object and stage.get('target'):
                target = f"{target} of {stage['target']}"
            grounding = self.grounder.locate(observation['image_paths'][0], target)
            actions.append({'name': skill, 'arm': side,
                            'pixel': grounding['pixel'], 'grounding': grounding,
                            'target_description': target, 'stage': stage,
                            'controller_token': tokens[side]})
        return actions, decision

    def feedback(self, action: dict, result: dict) -> None:
        side = action['arm']
        statuses = [s.get('planner_status', {}).get(side)
                    for s in result.get('subactions', [])]
        self.native.history[side].append(
            f"{action['name']}: planner={statuses}, ok={result.get('ok')}, "
            f"success={result.get('success')}, failure={result.get('failure')}")
        if result.get('ok'):
            self.native.indices[side] += 1
            self.attempts[side] = 0
        else:
            self.attempts[side] += 1
