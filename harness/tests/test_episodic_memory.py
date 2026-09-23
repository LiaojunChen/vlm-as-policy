import json
from types import SimpleNamespace
from PIL import Image
from robodawn.episodic_memory import EpisodicMemory, same_visual_region
from robodawn.semantic_planner import ground, policy_sensor_view
from robodawn.mission_planner import RobodawnPlanner


def visual(target='blue bowl', box=None):
    return dict(target=target, bbox=box or [100,100,300,300], point=[200,200], skill='reach', arm='left')


def test_memory_has_no_cross_episode_state_and_cannot_mutate_observations():
    m=EpisodicMemory();a=visual();m.remember_visual(a,1)
    a['bbox'][0]=999
    assert m.context()['objects'][0]['last_visual']['bbox'][0]==100
    snapshot=m.context();snapshot['objects'][0]['description']='changed'
    assert m.context()['objects'][0]['description']=='blue bowl'
    assert EpisodicMemory().context()['objects']==[]


def test_component_does_not_overwrite_identity_and_failed_motion_invalidates_position():
    m=EpisodicMemory();m.remember_visual(visual(),1)
    m.remember_visual({**visual(), 'grasp_part':'rim','bbox':[100,100,130,130]},2)
    assert not m.context()['objects'][0]['last_visual']['component_only']
    m.feedback(dict(skill='pick',target='blue bowl',arm='left'),
               dict(skill_success=False,failure='Grasp lost during lift',subactions=[{}],sensors={'left':dict(holding=False)}))
    assert m.context()['objects'][0]['state']=='may_have_moved_reobserve'


def test_verified_contact_overrides_skill_failure_and_release_is_not_goal_success():
    m=EpisodicMemory();m.remember_visual(visual(),1)
    m.feedback(dict(skill='pick',target='blue bowl',arm='left'),dict(skill_success=False,failure='lift unreachable',
               sensors={'left':dict(holding=True,remembered_object='blue bowl')}))
    assert m.context()['objects'][0]['held_by']=='left'
    m.feedback(dict(skill='open',arm='left'),dict(skill_success=True,sensors={'left':dict(holding=False)}))
    entry=m.context()['objects'][0]
    assert 'held_by' not in entry and entry['state']=='released_or_lost_reobserve'
    assert 'goal_success' not in m.context()


def test_history_is_bounded_and_repeats_do_not_erase_failure():
    m=EpisodicMemory()
    for _ in range(20):
        m.feedback(dict(skill='pick',target='bowl'),dict(skill_success=False,failure='empty grasp'))
    assert m.context()['recent_outcomes'][0]['repeats']==20
    for i in range(30):m.feedback(dict(skill='move',target=str(i)),dict(skill_success=True))
    assert len(m.events)==12 and len(m.context()['recent_outcomes'])==6


def test_distinct_region_check_does_not_equate_container_with_its_contents():
    assert same_visual_region(visual(),visual())
    assert not same_visual_region(visual(),visual(box=[50,50,600,600]))


def test_memory_is_a_hint_not_a_substitute_for_fresh_model_grounding(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    m=EpisodicMemory();m.remember_visual(visual(),1)
    calls=[]
    class Client:
        def complete_text(self,prompt,*args,**kwargs):
            calls.append(prompt)
            return SimpleNamespace(raw_text=json.dumps(dict(bbox=[500,500,700,700],point=[600,600])))
    result=ground(Client(),dict(image_paths=[str(path)],observation_id=2),visual(),memory=m)
    assert 'memory_guided_search' in result and len(calls)==1
    assert result['bbox']!=[100,100,300,300]  # Newly inferred crop coordinates, not old geometry.
    assert 'EPISODIC MEMORY' in calls[0]
    assert m.context()['objects'][0]['last_visual']['observation_id']==2


def test_identity_candidate_reuse_is_exact_frame_only_and_defensively_copied():
    m=EpisodicMemory();m.remember_visual(visual(),1)
    binding=m.current_binding('blue bowl',1)
    assert binding['bbox']==[100,100,300,300]
    binding['bbox'][0]=999
    assert m.current_binding('blue bowl',1)['bbox'][0]==100
    assert m.current_binding('blue bowl',2) is None
    m.objects['blue bowl']['held_by']='left'
    assert m.current_binding('blue bowl',1) is None


def test_same_object_destination_is_repaired_before_any_motion(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    replies=[dict(bbox=[100,100,300,300],point=[200,200]),
             dict(bbox=[600,100,800,300],point=[700,200])]
    prompts=[]
    class Client:
        def complete_text(self,prompt,*args,**kwargs):
            prompts.append(prompt);return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    a=ground(Client(),dict(image_paths=[str(path)],observation_id=1),visual('other bowl'),distinct_from=visual())
    assert a['point']==[700,200] and 'overlaps the source' in prompts[1]


def test_shared_memory_between_mission_and_recovery_and_filtered_sensor_audits():
    p=RobodawnPlanner(None,'task')
    assert p.memory is p.reactive.memory
    sensors={'left':dict(holding=True,opposed_contact=False,finger_normal_projection_range={'finger':[1,2]})}
    assert policy_sensor_view(sensors)=={'left':{'holding':True}}
    assert sensors['left']['opposed_contact'] is False
