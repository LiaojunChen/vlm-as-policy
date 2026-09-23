import json
from types import SimpleNamespace
import pytest
from PIL import Image
from robotwin_harness_v3 import SkillError
from robodawn.scene_inventory import needs_inventory,validate_inventory
from robodawn.mission_planner import RobodawnPlanner,validate_mission


def item(name,box):
    return dict(description=name,bbox=box,point=[(box[0]+box[2])/2,(box[1]+box[3])/2],grasp_part='rim')


def test_collective_goals_need_instance_memory_not_extra_simulator_inputs():
    assert needs_inventory('Lift each the ceramic bowl and stack them')
    assert needs_inventory('Arrange the objects by size')
    assert not needs_inventory('Arrange large block, medium block, small block in order')
    assert needs_inventory('Put all cans in the container')
    assert needs_inventory('Raise two bottles at the same time')
    assert needs_inventory('Use both arms in parallel for the box and scanner')
    assert not needs_inventory('Use each arm to lift one end of the pot')
    assert not needs_inventory('Hit the block using the hammer')


def test_inventory_rejects_duplicate_names_duplicate_regions_and_invalid_coordinates():
    left=item('left bowl',[100,100,300,300]);right=item('right bowl',[600,100,800,300])
    assert len(validate_inventory({'objects':[left,right]}))==2
    with pytest.raises(SkillError,match='initial visible positions'):
        validate_inventory({'objects':[left,{**right,'description':'left bowl'}]})
    with pytest.raises(SkillError,match='same visible object'):
        validate_inventory({'objects':[left,{**left,'description':'another bowl'}]})
    with pytest.raises(SkillError,match='Invalid inventory box'):
        validate_inventory({'objects':[{**left,'point':[900,900]}]})


def test_inventory_runs_once_before_semantic_plan_and_names_distinct_instances(tmp_path):
    image=tmp_path/'head.png';Image.new('RGB',(320,240)).save(image)
    entries=[item('left bowl',[100,100,300,300]),item('right bowl',[600,100,800,300])]
    plan={'steps':[dict(operation='stack',sources=['left bowl','right bowl'],arms=['auto','auto'],grasp_part='rim')]}
    responses=[{'objects':entries},plan,plan];calls=[]
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            calls.append((prompt,image));return SimpleNamespace(raw_text=json.dumps(responses.pop(0)))
    p=RobodawnPlanner(Client(),'Stack the bowls');obs=dict(image_paths=[str(image)],observation_id=1,sensors={})
    p._replan(obs,[]);p._replan(obs,[])
    assert len(calls)==3 and calls[0][1] is not None and calls[1][1] is None
    assert 'left bowl' in calls[1][0] and 'right bowl' in calls[1][0]
    assert len(p.steps)==1 and p.steps[0]['source']=='right bowl'


def test_stack_puts_a_previously_lifted_bottom_down_but_not_an_unmoved_bottom():
    lift=dict(operation='lift',source='base bowl',arm='right',location='stay',grasp_part='rim')
    stack=dict(operation='stack',sources=['base bowl','upper bowl'],arms=['auto','left'],grasp_part='rim')
    steps=validate_mission({'steps':[lift,stack]})
    assert len(steps)==3 and steps[1]['source']=='base bowl' and steps[1]['destination']=='empty centre of table'
    assert steps[1]['arm']=='right' and steps[2]['destination']=='base bowl'
    steps=validate_mission({'steps':[stack]})
    assert len(steps)==1 and steps[0]['source']=='upper bowl'


def test_stack_expansion_does_not_repeat_its_own_prefix_transfer():
    prefix=dict(operation='transfer',source='middle vessel',destination='base vessel',relation='on',arm='right')
    other=dict(operation='lift',source='last vessel',arm='left',location='stay')
    stack=dict(operation='stack',sources=['base vessel','middle vessel','last vessel'])
    steps=validate_mission({'steps':[prefix,other,stack]})
    assert len(steps)==3
    assert steps[-1]['source']=='last vessel' and steps[-1]['destination']=='middle vessel'
    assert sum(s.get('source')=='middle vessel' for s in steps)==1


def test_stack_keeps_duplicate_if_support_was_moved_or_arm_is_explicitly_different():
    prefix=dict(operation='transfer',source='upper object',destination='base object',relation='on',arm='right')
    stack=dict(operation='stack',sources=['base object','upper object'])
    moved=dict(operation='transfer',source='base object',destination='empty centre of table',relation='on',arm='left')
    assert len(validate_mission({'steps':[prefix,moved,stack]}))==3
    stack['arms']=['auto','left']
    assert len(validate_mission({'steps':[prefix,stack]}))==2
