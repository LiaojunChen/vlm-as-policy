"""Regression checks for the action boundary and semantic-plan rejection."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from robotwin_bridge import RobotWinBridge, PolicyOutputError, make_frame_context
from show_robotwin_policy import ShowPolicy
from benchmark_clients import AuditedClient


class FakeRobot:
    def left_plan_path(self, target):
        return {'status':'Success','position':[[0]]}

    def right_plan_path(self, target):
        return {'status':'Success','position':[[0]]}


class FakeEnv:
    def __init__(self):
        self.robot=FakeRobot()
        self.eval_success=False
        self.state={'left_endpose':[0.,0.,1.,1.,0.,0.,0.],'left_gripper':1.,
                    'right_endpose':[.4,0.,1.,1.,0.,0.,0.],'right_gripper':1.}

    def get_obs(self):
        return {'endpose':copy.deepcopy(self.state)}

    def check_success(self):
        return False

    def take_action(self, target, action_type):
        assert action_type=='ee'
        for i,side in enumerate(('left','right')):
            plan=getattr(self.robot,side+'_plan_path')(target[i*8:i*8+7])
            if plan['status']=='Success': self.state[side+'_endpose']=target[i*8:i*8+7]
            self.state[side+'_gripper']=target[i*8+7]


class FineControlTest(unittest.TestCase):
    def test_schema_survives_repeated_transport_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            client=AuditedClient('local',directory)
            client.structured_output=True
            schema={'type':'object','properties':{'left':{'enum':['GRASP','RELEASE']}}}
            first=client._finalize_payload({'model':'local','messages':[],'guided_json':schema})
            second=client._finalize_payload(first)
            self.assertEqual(first,second)
            self.assertEqual(second['response_format']['json_schema']['schema'],schema)
            client.structured_output=False
            self.assertEqual(client._finalize_payload(first)['response_format'],{'type':'json_object'})

    def test_small_move_preserves_orientation_and_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            env=FakeEnv()
            bridge=RobotWinBridge(env,'test',directory,step_m=.005)
            bridge.last_obs=env.get_obs()
            result=bridge.execute_tokens({'left':'MV_DOWN','right':'STILL'})
            self.assertTrue(result['ok'])
            np.testing.assert_allclose(env.state['left_endpose'],[0,0,.995,1,0,0,0])
            self.assertEqual(env.state['left_gripper'],1.)
            self.assertEqual(env.state['right_endpose'],[.4,0.,1.,1.,0.,0.,0.])
            self.assertIn('5 mm',bridge.frame_context)

    def test_failed_planner_is_reported_and_original_method_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            env=FakeEnv()
            env.robot.left_plan_path=lambda target:{'status':'Failure'}
            original=env.robot.left_plan_path
            bridge=RobotWinBridge(env,'test',directory,step_m=.002)
            bridge.last_obs=env.get_obs()
            result=bridge.execute_tokens({'left':'MV_RIGHT','right':'GRASP'})
            self.assertFalse(result['ok'])
            self.assertIs(env.robot.left_plan_path,original)
            self.assertAlmostEqual(result['motion']['left']['position_error_m'],.002)
            self.assertEqual(env.state['right_gripper'],0.)
            bridge.execute_tokens({'left':'STILL','right':'RELEASE'})
            self.assertEqual(env.state['right_gripper'],1.)

    def test_invalid_semantic_plan_retried_without_inventing_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=ShowPolicy(SimpleNamespace(),'stack blocks',directory,
                              frame_context=make_frame_context(.005),prompt_mode='fine')
            calls=[]
            def plan(*args):
                calls.append(args)
                return {'left':[SimpleNamespace(motion='MV_LEFT')],'right':[]},json.dumps({'left':[],'right':[]})
            policy.planner=SimpleNamespace(plan=plan,agent=SimpleNamespace(common_context=''))
            with self.assertRaises(PolicyOutputError): policy._plan([None,None,None])
            self.assertEqual(len(calls),2)
            self.assertFalse((Path(directory)/'plan_0.json').exists())


if __name__=='__main__':
    unittest.main()
