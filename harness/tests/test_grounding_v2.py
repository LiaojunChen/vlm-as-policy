"""Contracts that distinguish the corrected evaluation loop from v1."""
import numpy as np
import pytest

from robodawn_core import run_loop
from robotwin_bridge import PolicyOutputError
from robotwin_grounding_v2 import MOVE, _cell, _refine_color


def test_failed_planner_action_is_returned_to_the_policy():
    class Robot:
        calls = 0

        def observe(self):
            return {'success': False}

        def execute(self, action):
            self.calls += 1
            return ({'ok': False, 'failure': 'planner Fail'} if self.calls == 1
                    else {'ok': True, 'success': True})

    class Planner:
        def plan(self, observation, history):
            if history:
                assert history[-1]['result']['failure'] == 'planner Fail'
            return {'name': 'press_at', 'arm': 'right'}

    robot = Robot()
    trace = run_loop(robot, robot, Planner(), max_steps=3,
                     stop_on_error=False, allowed_actions=('press_at',))
    assert len(trace) == 2
    assert trace[-1]['result']['success'] is True


def test_grid_codes_are_range_checked_and_fwd_matches_head_view():
    assert _cell('C10R7', 'C', 15, 11) == (10, 7)
    with pytest.raises(PolicyOutputError):
        _cell('C16R7', 'C', 15, 11)
    assert MOVE['MV_FWD'] == (0, 1, 0)


def test_colour_refinement_stays_near_model_selected_pixel():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:30, 20:30] = (255, 255, 0)
    image[60:70, 60:70] = (255, 255, 0)
    assert _refine_color(image, (23, 23), 'yellow', radius=10) == (24, 24)


def test_show_native_replan_gap_does_not_crash():
    from types import SimpleNamespace
    from robotwin_policies_v2 import ShowGroundedPolicy
    policy = ShowGroundedPolicy.__new__(ShowGroundedPolicy)
    policy.native = SimpleNamespace(
        decide=lambda obs: ({'left': 'STILL', 'right': 'STILL'}, False),
        tracks=None, indices={'left': 0, 'right': 0},
        last_decision=SimpleNamespace(raw_text='done'))
    actions, decision = policy.decide({})
    assert actions == [] and decision['replan_pending']


def test_show_grounding_keeps_object_context_when_enabled():
    from types import SimpleNamespace
    from robotwin_policies_v2 import ShowGroundedPolicy
    seen = []
    stage = SimpleNamespace(to_prompt_dict=lambda: {
        'motion': 'GRASP', 'target': 'red cube', 'affordance': 'top face'})
    policy = ShowGroundedPolicy.__new__(ShowGroundedPolicy)
    policy.native = SimpleNamespace(
        decide=lambda obs: ({'left': 'STILL', 'right': 'MV_DOWN'}, False),
        tracks={'left': [], 'right': [stage]}, indices={'left': 0, 'right': 0},
        last_decision=SimpleNamespace(raw_text='move'))
    policy.instruction = 'grasp the red cube'
    policy.ground_with_object = True
    def locate(path, description):
        seen.append(description)
        return {'pixel': [160, 120]}
    policy.grounder = SimpleNamespace(locate=locate)
    actions, _ = policy.decide({'image_paths': ['unused']})
    assert seen == ['top face of red cube']
    assert actions[0]['name'] == 'pick_at'
