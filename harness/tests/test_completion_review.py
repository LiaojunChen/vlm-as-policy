from copy import deepcopy
from types import SimpleNamespace
import pytest
from robotwin_harness_v3 import SkillError
from robodawn.completion_review import completed_contact_review
from robodawn.episodic_memory import EpisodicMemory
from robodawn.mission_planner import RobodawnPlanner


INTENT=dict(operation='tool_contact',source='implement',destination='surface',contact_part='working end')
ACTION=dict(skill='place',arm='right',target='surface',release=False,use_contact_part=True,mission_intent=INTENT)
HISTORY=[dict(action=ACTION,result=dict(skill_success=True))]
SENSORS=dict(right=dict(holding=True,remembered_object='implement'))


def review(**changes):
    arguments=dict(steps=[INTENT],index=1,pending=[],history=deepcopy(HISTORY),sensors=deepcopy(SENSORS))
    arguments.update(changes)
    return completed_contact_review(**arguments)


def test_execution_memory_does_not_claim_support_success_or_prescribe_release():
    value=review()
    assert value['recorded_status']=='contact_skill_executed_goal_unverified'
    assert not value['support_contact_proven'] and value['terminal_release_required']=='not_inferred'
    assert 'point' not in value and 'action' not in value


@pytest.mark.parametrize('changes',[
    dict(index=0),dict(pending=[INTENT]),dict(steps=[]),dict(history=[]),dict(sensors={}),
    dict(sensors=dict(right=dict(holding=False,remembered_object='implement'))),
    dict(sensors=dict(right=dict(holding=True,remembered_object='other object'))),
    dict(history=[dict(action=ACTION,result=dict(skill_success=False))]),
    dict(history=HISTORY+[dict(action=dict(skill='inspect',arm='left'),result=dict(skill_success=True))]),
])
def test_no_review_for_pending_failed_lost_or_intervening_stages(changes):
    assert review(**changes) is None


@pytest.mark.parametrize('change',[
    dict(release=True),dict(use_contact_part=False),dict(target='other surface'),
    dict(skill='pick'),dict(mission_intent=dict(operation='transfer',source='implement',destination='surface')),
])
def test_only_the_recorded_nonreleasing_tool_contact_qualifies(change):
    assert review(history=[dict(action=dict(ACTION,**change),result=dict(skill_success=True))]) is None


def test_review_is_transient_and_does_not_leak_to_object_grounding_or_future_native_prompts():
    planner=RobodawnPlanner(None,'Use the implement and continue to hold it after use.')
    planner.steps=[INTENT];planner.index=1;before=planner.memory.prompt();calls=[]
    def respond(*args):
        calls.append(planner.memory.prompt())
        assert 'completion_stage_review' not in planner.memory.context('implement')
        return dict(skill='wait')  # model may retain; no program-selected release
    planner.reactive=SimpleNamespace(plan=respond)
    action=planner._reactive_recovery(dict(sensors=SENSORS),HISTORY,'completed')
    assert action['skill']=='wait'
    assert 'COMPLETION-STAGE REVIEW' in calls[0]
    assert 'preserve the grasp' in calls[0]
    assert planner.memory.prompt()==before and planner.memory.completion_review is None


def test_review_clears_even_on_failed_reactive_generation():
    planner=RobodawnPlanner(None,'Use the implement.')
    planner.steps=[INTENT];planner.index=1
    def fail(*args):raise SkillError('Model generation ended before completing JSON')
    planner.reactive=SimpleNamespace(plan=fail)
    with pytest.raises(SkillError):planner._reactive_recovery(dict(sensors=SENSORS),HISTORY,'completed')
    assert planner.memory.completion_review is None


def test_context_is_copied_and_empty_initial_prompt_is_unchanged():
    memory=EpisodicMemory();assert memory.prompt()==''
    memory.completion_review=review();context=memory.context();context['completion_stage_review']['source']='changed'
    assert memory.completion_review['source']=='implement'
