import json
from types import SimpleNamespace
import pytest
from PIL import Image
from robodawn.intent_memory import IntentMemory
from robodawn.episodic_memory import EpisodicMemory
from robodawn.mission_planner import RobodawnPlanner
from robotwin_harness_v3 import SkillError


def lift(source):
    return dict(operation='lift', source=source, arm='auto', location='stay')


def transfer(source, destination='base object'):
    return dict(operation='transfer', source=source, destination=destination, relation='on', arm='auto')


def test_failed_pick_cannot_be_forgotten_or_replaced_by_destination_only():
    memory=IntentMemory()
    memory.initialize([lift('left vessel'),lift('far vessel'),transfer('near vessel','far vessel')])
    memory.feedback(dict(skill='pick',target='far vessel'),dict(skill_success=False))
    with pytest.raises(SkillError,match='far vessel'):
        memory.validate_replan([lift('left vessel'),transfer('near vessel','far vessel')],{})
    memory.validate_replan([transfer('far vessel','near vessel'),lift('left vessel'),transfer('near vessel')],{})
    assert len(memory.pending())==3


def test_contact_is_required_for_lift_and_transfer_can_fuse_the_lift():
    memory=IntentMemory();memory.initialize([lift('vessel'),transfer('vessel')])
    memory.feedback(dict(skill='pick',target='vessel'),dict(skill_success=True,sensors={}))
    assert len(memory.pending())==2
    state={'left':dict(holding=True,remembered_object='vessel')}
    memory.feedback(dict(skill='pick',target='vessel'),dict(skill_success=True,sensors=state))
    assert len(memory.pending())==1 and memory.pending()[0]['operation']=='transfer'
    with pytest.raises(SkillError):memory.validate_replan([lift('vessel')],state)
    memory.validate_replan([transfer('vessel','other support')],state)


def test_staging_and_home_do_not_discharge_original_transfer():
    memory=IntentMemory();memory.initialize([transfer('vessel')])
    for action in [dict(skill='home',mission_intent=transfer('vessel')),
                   dict(skill='place',release=True,mission_intent={**transfer('vessel'),'regrasp_stage':True})]:
        memory.feedback(action,dict(skill_success=True))
        assert len(memory.pending())==1
    memory.feedback(dict(skill='place',release=True,mission_intent=transfer('vessel')),dict(skill_success=True))
    assert not memory.pending()
    assert memory.context()[0]['execution_status']=='skill_executed_goal_unverified'
    assert 'success' not in memory.context()[0]


def test_ledger_is_episode_local_compact_and_independent_of_spatial_coordinates():
    memory=EpisodicMemory();memory.intentions.initialize([lift('vessel')])
    assert 'original_execution_intentions' in memory.context()
    assert 'original_execution_intentions' not in memory.context('vessel')
    assert not EpisodicMemory().intentions.entries
    assert 'never goal completion' in memory.prompt()
    memory.intentions.initialize([lift('replacement')])
    assert memory.intentions.context()[0]['source']=='vessel'


def test_rejected_replan_is_transactional_and_prompts_missing_source(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    calls=[]
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            calls.append(prompt)
            return SimpleNamespace(raw_text=json.dumps({'steps':[lift('other object')]}))
    planner=RobodawnPlanner(Client(),'Lift each object')
    planner.inventory_attempted=True
    planner.steps=[lift('original object')]
    planner.memory.intentions.initialize(planner.steps)
    observation=dict(image_paths=[str(path)],observation_id=1,sensors={})
    with pytest.raises(SkillError,match='omitted unexecuted original intentions'):
        planner._replan(observation,[])
    assert planner.steps==[lift('original object')]
    assert planner.revision==0
    assert len(calls)==3 and 'original_execution_intentions' in calls[0]
    assert 'Replan omitted' in calls[1]


def test_unexecuted_lift_may_be_covered_by_current_contact_but_press_may_not():
    memory=IntentMemory();memory.initialize([lift('held object'),dict(operation='press',source='button')])
    sensors={'left':dict(holding=True,remembered_object='held object')}
    memory.validate_replan([dict(operation='press',source='button')],sensors)
    with pytest.raises(SkillError,match='button'):
        memory.validate_replan([lift('button')],sensors)
