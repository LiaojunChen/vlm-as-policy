import json
from types import SimpleNamespace

import pytest
from PIL import Image

from benchmark_clients import AuditedClient, VLMClient
from robotwin_harness_v3 import SkillError
from robodawn.mission_planner import RobodawnPlanner
from robodawn.semantic_planner import RobodawnPlanner as ReactivePlanner, ground


BUDGET = 'Generation budget ended before completing schema-constrained JSON'
ERROR = 'VLM chat completion failed: HTTP 400: ' + json.dumps({'error': {'message': BUDGET}})


@pytest.mark.parametrize('message,converted', [
    (ERROR, True),
    (ERROR.replace('400', '503'), False),
    (ERROR.replace(BUDGET, 'Unknown response schema'), False),
    ('Connection timed out', False),
    ('VLM chat completion failed: HTTP 400: malformed body', False),
])
def test_only_exact_decoder_budget_error_is_model_output(monkeypatch, tmp_path, message, converted):
    monkeypatch.setenv('VLM_JSON_SCHEMA', '1')
    client = AuditedClient('zdtaichu', tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError(message)
    monkeypatch.setattr(VLMClient, '_post_chat', fail)
    with pytest.raises(SkillError if converted else RuntimeError):
        client.complete_text('Return JSON', None, response_schema={'type': 'object'})
    # The original HTTP failure remains in the audit even when repairable.
    assert json.loads((tmp_path / 'call_0000.error.json').read_text())['error'] == message
    assert client.calls == 1


@pytest.mark.parametrize('role', ['mission', 'reactive', 'ground'])
def test_output_failure_can_be_repaired_without_reusing_previous_text(tmp_path, role):
    path = tmp_path / 'head.png'
    Image.new('RGB', (320, 240)).save(path)
    valid = {'mission': {'steps': [{'operation': 'press', 'source': 'button', 'arm': 'left'}]},
             'reactive': {'skill': 'home', 'arm': 'left'},
             'ground': {'bbox': [100, 100, 300, 300], 'point': [200, 200]}}[role]
    replies = ['{}', SkillError('Model generation ended before completing JSON'), json.dumps(valid)]
    prompts = []
    class Client:
        def complete_text(self, prompt, *args, **kwargs):
            prompts.append(prompt)
            response = replies.pop(0)
            if isinstance(response, Exception):
                raise response
            return SimpleNamespace(raw_text=response)
    obs = dict(success=False, sensors={'left': {'holding': False}}, image_paths=[str(path)],
               observation_id=1, endpose={}, table_height_m=.74)
    client = Client()
    if role == 'mission':
        planner = RobodawnPlanner(client, 'Press the button')
        planner._replan(obs, [])
        assert planner.steps[0]['source'] == 'button'
    elif role == 'reactive':
        assert ReactivePlanner(client, 'Clear the view').plan(obs, [])['skill'] == 'home'
    else:
        assert ground(client, obs, dict(skill='press', arm='left', target='button'))['point'] == [200, 200]
    assert len(prompts) == 3
    assert '[No complete model response received]' in prompts[-1]


def test_mission_transport_failure_is_not_silently_repaired(tmp_path):
    path = tmp_path / 'head.png'
    Image.new('RGB', (320, 240)).save(path)
    class Client:
        def complete_text(self, *args, **kwargs):
            raise RuntimeError('Connection refused')
    with pytest.raises(RuntimeError, match='Connection refused'):
        RobodawnPlanner(Client(), 'Lift object')._replan({'sensors': {}, 'image_paths': [str(path)]}, [])


def test_semantic_initial_plan_and_visual_recovery_keep_separate_interfaces(tmp_path):
    path = tmp_path / 'head.png'
    Image.new('RGB', (320, 240), (12, 34, 56)).save(path)
    calls=[]
    class Client:
        def complete_text(self, prompt, image, **kwargs):
            calls.append(image)
            return SimpleNamespace(raw_text=json.dumps({'steps': [dict(operation='press', source='button', arm='left')]}))
    planner=RobodawnPlanner(Client(), 'Press button');obs={'sensors': {}, 'image_paths': [str(path)]}
    planner._replan(obs, [])
    planner._replan(obs,[{'action':{'skill':'home'},'result':{'skill_success':True}}])
    assert calls[0] is None
    assert calls[1].shape == (240,320,3) and calls[1][0,0].tolist()==[12,34,56]


def test_gpu_admission_uses_selected_gpu(monkeypatch):
    import run_suite_v3
    monkeypatch.setattr(run_suite_v3.subprocess, 'check_output', lambda *a, **kw: '1700\n32000\n')
    assert run_suite_v3.gpu_free() == 1700
    assert run_suite_v3.gpu_free(1) == 32000
    assert run_suite_v3.gpu_free(2) == 0
