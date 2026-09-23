from copy import deepcopy
from types import SimpleNamespace as S
import json
from PIL import Image
import pytest
from robodawn.collection_identity import collection_collisions
from robodawn.mission_planner import RobodawnPlanner


def fixture():
    steps=[dict(operation='transfer',source=name,destination='container',relation='inside',arm='auto') for name in ('red item','striped item')]
    bound={name:dict(target=name,bbox=[100,100,300,300],point=[200,200],camera='head_camera',observation_id=1) for name in ('red item','striped item')}
    return steps,bound


def test_only_same_frame_collection_source_collision_is_reported():
    steps,bound=fixture();before=deepcopy((steps,bound))
    assert collection_collisions(steps,steps[1],bound,1)==[bound['red item']]
    assert (steps,bound)==before
    assert collection_collisions(steps,steps[1],bound,2)==[]


@pytest.mark.parametrize('change',[
    lambda s,b:s[0].update(operation='lift'),lambda s,b:s[1].update(operation='slide'),
    lambda s,b:s[0].update(destination='different support'),lambda s,b:s[1].update(destination='empty table'),
    lambda s,b:s[0].update(relation='on'),lambda s,b:s[0].update(source='striped item'),
    lambda s,b:b['red item'].update(camera='left_camera'),lambda s,b:b['red item'].update(observation_id=0),
    lambda s,b:b['red item'].update(bbox=[600,100,800,300]),
])
def test_aliases_parts_different_goals_frames_views_and_objects_not_forced_apart(change):
    steps,bound=fixture();change(steps,bound)
    assert collection_collisions(steps,steps[1],bound,1)==[]


def test_collision_repaired_by_model_before_binding_second_pair(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    first=dict(bbox=[100,100,300,300],point=[200,200],support='table')
    destination=dict(bbox=[400,400,600,600],point=[500,500],support='container')
    second=dict(bbox=[700,100,900,300],point=[800,200],support='table')
    pair=dict(description_A='source',description_B='support',source='A',destination='B',source_grasp_part='body')
    responses=[first,destination,pair,first,second,pair];calls=[]
    def complete(prompt,image,**kwargs):
        calls.append(prompt);return S(raw_text=json.dumps(responses.pop(0)))
    p=RobodawnPlanner(S(complete_text=complete),'Collect the two items');p.steps=fixture()[0]
    obs=dict(observation_id=1,image_paths=[str(path)],sensors={})
    p._bind_transfer_identities(obs)
    assert len(calls)==6
    assert 'IDENTITY CONSTRAINT' not in calls[3]
    assert 'collection source must be a DIFFERENT object' in calls[4]
    assert p.memory.objects['red item']['last_visual']['point']==[200,200]
    assert p.memory.objects['striped item']['last_visual']['point']==[800,200]
    assert p.bound_revision==p.revision
