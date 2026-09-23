import sys
from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robotwin_bridge import PolicyOutputError,RobotWinBridge
from robodawn_core import run_loop

class Env:
    def __init__(self):
        self.eval_success=False;self.actions=[]
        self.robot=SimpleNamespace(
            left_plan_path=lambda target: {'status':'Success','position':[]},
            right_plan_path=lambda target: {'status':'Success','position':[]})
        self.obs={'endpose':{'left_endpose':[0,0,1,1,0,0,0],'right_endpose':[1,0,1,1,0,0,0],
                             'left_gripper':1,'right_gripper':.25}}
    def take_action(self,a,action_type):
        assert action_type=='ee';self.actions.append(a)
        self.robot.left_plan_path(a[:7])
        self.robot.right_plan_path(a[8:15])
    def get_obs(self):return self.obs
    def check_success(self):return False

def test_motion_preserves_other_arm_and_orientation(tmp_path):
    env=Env();bridge=RobotWinBridge(env,'test',tmp_path);bridge.last_obs=env.obs
    bridge.execute_tokens({'left':'MV_BACK','right':'STILL'})
    a=np.array(env.actions[0]);assert len(a)==16
    np.testing.assert_allclose(a[:3],[0,.04,1]);np.testing.assert_equal(a[3:7],[1,0,0,0])
    np.testing.assert_equal(a[8:],[1,0,1,1,0,0,0,.25])

def test_gripper_without_oracle_movement(tmp_path):
    env=Env();bridge=RobotWinBridge(env,'test',tmp_path);bridge.last_obs=env.obs
    bridge.execute({'name':'pick','arm':'right'})
    assert env.actions[0][-1]==0
    assert env.actions[0][:8]==[0,0,1,1,0,0,0,1]

def test_invalid_model_token_is_a_policy_failure(tmp_path):
    env=Env();bridge=RobotWinBridge(env,'test',tmp_path);bridge.last_obs=env.obs
    with pytest.raises(PolicyOutputError,match='Invalid atomic token: Grasp'):
        bridge.execute_tokens({'left':'Grasp','right':'STILL'})
    assert env.actions==[]

def test_noop_is_optional_and_env_success_ends_loop():
    class Source:
        def observe(self):return {}
    class Planner:
        def plan(self,obs,history):return {'name':'noop' if not history else 'push'}
    class Executor:
        def execute(self,a):return {'ok':True,'success':a['name']=='push'}
    rows=[]
    trace=run_loop(Source(),Executor(),Planner(),stop_on_noop=False,on_record=rows.append)
    assert len(trace)==2 and rows==trace and trace[-1]['result']['success']
    assert len(run_loop(Source(),Executor(),Planner()))==1

def test_audited_client_writes_unicode_after_sim_changes_locale(tmp_path, monkeypatch):
    import locale
    from benchmark_clients import AuditedClient, VLMClient
    monkeypatch.setenv('VLM_ENDPOINT','https://example.invalid/v1')
    monkeypatch.setattr(VLMClient,'_post_chat',lambda *a,**k:{'choices':[{'message':{'content':'{"reason":"移動≈4cm"}'}}]})
    client=AuditedClient('test',tmp_path)
    before=locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE,'C')
        response=client.complete_json('选择动作',None)
        assert response.payload['json']['reason']=='移動≈4cm'
        assert '移動' in (tmp_path/'call_0000.response.json').read_text(encoding='utf-8')
    finally:locale.setlocale(locale.LC_CTYPE,before)
