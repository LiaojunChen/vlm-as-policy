import unittest
from reviewed_interface import Capabilities, ContractError, validate_action, parse_command, stage_skill, ReviewedShowPolicy

class ContractTests(unittest.TestCase):
    def setUp(self):self.caps=Capabilities()
    def reject(self,action,code,obs=3):
        with self.assertRaises(ContractError) as ctx:validate_action(action,self.caps,obs)
        self.assertEqual(ctx.exception.code,code)
    def test_grounder_rejects_extra_output(self):
        from robotwin_grounding_v2 import _cell
        from robotwin_bridge import PolicyOutputError
        for text in ('choose C10R7', 'C10R7 C9R8', 'C10R7\nDONE'):
            with self.assertRaises(PolicyOutputError):_cell(text,'C',15,11)

    def test_invalid_pixels(self):
        for pixel in ([-1,120],[320,120],[12.1,1],[True,2],[12,240]):
            self.reject(dict(name='pick_at',arm='right',pixel=pixel,observation_id=3),'invalid_pixel')
    def test_observation_freshness(self):
        self.reject(dict(name='pick_at',arm='right',pixel=[100,100],observation_id=2),'stale_observation')
        self.reject(dict(name='pick_at',arm='right',pixel=[100,100],observation_id=3),'stale_observation',None)
    def test_unavailable_capabilities(self):
        self.reject(dict(name='lift',arm='left'),'unavailable_arm')
        self.reject(dict(name='rotate',arm='right'),'unsupported_action')
        self.reject(dict(name='lift',arm='right',quaternion=[1,0,0,0]),'unsupported_parameter')
    def test_no_clipping_or_nonfinite(self):
        for v in ([float('nan'),0,0],[0,float('inf'),0],[True,0,0]):
            self.reject(dict(name='move_delta',arm='right',delta=v),'invalid_parameter')
        self.reject(dict(name='move_delta',arm='right',delta=[.2,0,0]),'out_of_workspace')
        self.reject(dict(name='move_pose',arm='right',xyz=[0,0,.5]),'out_of_workspace')
        self.reject(dict(name='lift',arm='right',distance=-.1),'invalid_parameter')
        self.reject(dict(name='wait',duration=10),'invalid_parameter')
    def test_strict_parser(self):
        for raw in ('PICK_R','I choose PICK_R: red cube','PICK_R: cube\nDONE','WAIT: 1.2.3','OPEN_R: cube'):
            with self.assertRaises(ContractError):parse_command(raw,self.caps)
        self.assertEqual(parse_command('PICK_R: red cube',self.caps)['target_description'],'red cube')
        self.assertEqual(parse_command('MOVE_R: .02 0 0',self.caps)['delta'],[.02,0,0])
    def test_no_unknown_stage_fallback(self):
        for name in ('ROTATE','PUSH','PULL','INSERT'):
            with self.assertRaises(ContractError):stage_skill(name)
        self.assertEqual(stage_skill('GRASP'),'pick_at')
        self.assertEqual(stage_skill('LIFT'),'lift')
    def test_stage_needs_skill_completion(self):
        policy=ReviewedShowPolicy.__new__(ReviewedShowPolicy);policy.history=[];policy.index=0
        policy.feedback({'name':'pick_at'},{'motion_ok':True,'skill_success':False},{'expected_skill':'pick_at'})
        self.assertEqual(policy.index,0)
        policy.feedback({'name':'wait'},{'skill_success':True},{'expected_skill':'wait','advance_denied':True})
        self.assertEqual(policy.index,0)
        policy.feedback({'name':'pick_at'},{'skill_success':True},{'expected_skill':'pick_at'})
        self.assertEqual(policy.index,1)

if __name__=='__main__':unittest.main()
