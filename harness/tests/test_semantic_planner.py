import json
from types import SimpleNamespace
from PIL import Image
from robodawn.semantic_planner import RobodawnPlanner, history_summary, normalize_action


def test_semantic_place_is_grounded_without_old_coordinates_and_keeps_holding_arm(tmp_path):
    path = tmp_path / 'head.png'
    Image.new('RGB', (320, 240)).save(path)
    responses = [dict(skill='place', arm='left', target='basket', support='container'),
                 dict(bbox_2d=[500, 500, 800, 800], point_2d=[650, 650], arm='right', approach='top')]
    prompts = []
    class Client:
        def complete_text(self, prompt, *args, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(raw_text=json.dumps(responses.pop(0)))
    obs = dict(success=False, image_paths=[str(path)], observation_id=2,
               endpose={}, sensors={'left': {'holding': True}}, table_height_m=.74)
    action = RobodawnPlanner(Client(), 'Put the can in the basket.').plan(obs, [])
    assert action['point'] == [650, 650]
    assert action['arm'] == 'left' and action['support'] == 'container'
    assert '100,200' not in prompts[1] and 'EXECUTED HISTORY' not in prompts[1]


def test_history_retains_early_completed_transfers_and_collapses_failure_runs():
    initial = dict(action={'skill': 'place', 'arm': 'left', 'target': 'basket', 'raw': 'unneeded'},
                   result={'skill_success': True, 'failure': None})
    failed = dict(action={'skill': 'pick', 'arm': 'right', 'target': 'can'},
                  result={'skill_success': False, 'failure': 'Empty grasp'})
    summary = history_summary([initial] + [failed] * 8)
    assert len(summary) == 2
    assert summary[0]['target'] == 'basket' and summary[0]['ok']
    assert summary[1]['repeats'] == 8 and 'raw' not in summary[0]


def test_actual_zdtaichu_nested_parameters_preserve_intent_and_reject_conflict():
    import pytest
    from robotwin_harness_v3 import SkillError
    raw={'skill':'pick','parameters':{'arm':'right','target':'dark drink bottle with logo','grasp_part':'body','approach':'side'}}
    normalized=normalize_action(raw)
    assert normalized == {'skill':'pick',**raw['parameters']}
    assert 'parameters' in raw
    with pytest.raises(SkillError,match='Conflicting'):
        normalize_action({**raw,'arm':'left'})


def test_present_location_equivalent_front_centre_labels_are_normalized():
    for location in ('center','front centre','front_center','front-centre'):
        assert normalize_action(dict(skill='present',location=location))['location']=='centre'
    assert normalize_action(dict(operation='lift',location='stay'))['location']=='stay'
