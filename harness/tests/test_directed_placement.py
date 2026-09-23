import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat

from directed_placement import observed_axis, align_directed_axis, endpoint_reference, FACING
from robotwin_harness_v3 import SkillError, placement_rotations, validate
from robodawn.mission_planner import RobodawnPlanner, validate_mission, validate_directed_requirement
from robodawn.schemas import MISSION_SCHEMA
from runtime.json_prefix import JsonPrefix


@pytest.mark.parametrize('facing', FACING)
def test_aligns_directed_end_not_undirected_major_axis(facing):
    axis=observed_axis([.06,.04,.8],[-.04,-.04,.82])
    initial=euler2mat(0,np.pi/2,.8)
    current=euler2mat(0,0,.7)@initial
    aligned,delta=align_directed_axis(axis,initial,current,facing)
    result=aligned@initial.T@axis
    np.testing.assert_allclose(result[:2],FACING[facing],atol=1e-7)
    # Facing may require >90 degrees; never reduce modulo pi.
    candidates=list(placement_rotations(mat2quat(aligned),{'facing':facing}))
    assert len(candidates)==1 and candidates[0][:2]==(0.,0)


def test_opposite_facing_really_rotates_180_degrees():
    aligned,delta=align_directed_axis([1,0,0],np.eye(3),np.eye(3),'left')
    assert abs(delta)==pytest.approx(np.pi)
    np.testing.assert_allclose(aligned@np.array([1,0,0]),[-1,0,0],atol=1e-7)


def test_degenerate_or_hidden_endpoints_do_not_create_orientation():
    for endpoint in ([.001,0,0],[1,0,0],[float('nan'),0,0]):
        with pytest.raises(ValueError):observed_axis([0,0,0],endpoint)
    with pytest.raises(ValueError,match='vertical'):
        align_directed_axis([1,0,0],np.eye(3),euler2mat(0,np.pi/2,0),'left')


def test_endpoint_uses_selected_local_depth_not_whole_object_centre():
    y,x=np.indices((40,80));cloud=np.stack([x*.003,y*.003,np.full(x.shape,.8)],axis=-1)
    valid=np.ones(x.shape,bool)
    a=endpoint_reference(cloud,valid,[0,0,1000,1000],[100,500],.74)
    b=endpoint_reference(cloud,valid,[0,0,1000,1000],[900,500],.74)
    assert a['point'][0]<.04 and b['point'][0]>.20
    valid[:]=False
    assert endpoint_reference(cloud,valid,[0,0,1000,1000],[100,500],.74) is None


def test_orientation_mission_schema_and_validation():
    step=dict(operation='transfer',source='left shoe',destination='box',relation='inside',arm='left',
              orientation=dict(from_part='heel of left shoe',to_part='toe of left shoe',facing='left'))
    assert JsonPrefix(MISSION_SCHEMA).status(json.dumps({'steps':[step]}))=='complete'
    assert validate_mission({'steps':[step]})[0]['orientation']['facing']=='left'
    with pytest.raises(SkillError,match='distinct'):
        validate_mission({'steps':[{**step,'orientation':dict(from_part='toe',to_part='toe',facing='left')} ]})
    with pytest.raises(SkillError,match='facing'):
        validate_mission({'steps':[{**step,'orientation':dict(from_part='heel',to_part='toe',facing='up')} ]})
    with pytest.raises(SkillError,match='placeholder'):
        validate_mission({'steps':[{**step,'orientation':dict(from_part='named rear end of source',to_part='toe',facing='left')} ]})


def test_explicit_direction_is_validated_not_fabricated():
    step=dict(operation='transfer',source='object',destination='box',relation='inside')
    for instruction in ('Place them in the box, tips left.','Put both inside with their toes pointing left.'):
        with pytest.raises(SkillError,match='orientation'):validate_directed_requirement([step],instruction)
        validate_directed_requirement([{**step,'orientation':dict(facing='left',from_part='rear end',to_part='toe')}],instruction)
    validate_directed_requirement([step],'Place the object in the front left corner.')
    assert 'orientation' not in step


def test_compiler_observes_two_current_endpoints_and_retains_direction(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    step=dict(operation='transfer',source='left shoe',destination='box',relation='inside',arm='left',
              orientation=dict(from_part='heel of left shoe',to_part='toe of left shoe',facing='left'))
    replies=[{'steps':[step]},dict(bbox=[100,200,400,800],point=[250,500]),
             dict(bbox=[500,300,900,900],point=[700,600]),
             dict(description_A='shoe',description_B='box',source='A',destination='B',source_grasp_part='body'),
             dict(bbox=[100,200,400,800],point=[250,500]),
             dict(bbox=[200,200,400,400],point=[300,300]),dict(bbox=[500,600,700,800],point=[600,700]),
             dict(bbox=[400,400,800,800],point=[600,600],support='container')]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    obs=dict(success=False,image_paths=[str(path)],observation_id=7,sensors={'left':{'holding':False}})
    planner=RobodawnPlanner(Client(),'Put shoe in box with toe facing left')
    action=planner.plan(obs,[]);validate(action)
    ref=action['orientation_reference']
    assert ref['observation_id']==7 and ref['facing']=='left'
    assert ref['from_part']['target']=='heel of left shoe'
    assert ref['to_part']['target']=='toe of left shoe'
    assert ref['from_part']['point']!=ref['to_part']['point']
    planner.feedback(action,dict(skill_success=True))
    obs['sensors']['left']=dict(holding=True,remembered_object='left shoe')
    assert planner.plan(obs,[])['facing']=='left'


def test_requested_end_is_the_to_part_not_a_reversed_directed_axis():
    step=dict(operation='transfer',source='long object',destination='table',relation='on',
              orientation=dict(from_part='tip of long object',to_part='rear of long object',facing='right'))
    with pytest.raises(SkillError,match='reversing the endpoints'):
        validate_directed_requirement([step],'Put the object down with its tip pointing right.')
    step['orientation'].update(from_part='rear of long object',to_part='tip of long object')
    validate_directed_requirement([step],'Put the object down with its tip pointing right.')


def test_later_direction_requirement_is_carried_to_initial_lift():
    lift=dict(operation='lift',source='long object',arm='left',location='stay')
    place=dict(operation='transfer',source='long object',destination='support',relation='on',arm='left',
               orientation=dict(from_part='rear end',to_part='tip',facing='left'))
    other=dict(operation='lift',source='unrelated object',arm='right',location='stay')
    steps=validate_mission(dict(steps=[lift,other,place]))
    assert steps[0]['acquisition_orientation']==place['orientation']
    assert 'acquisition_orientation' not in steps[1]
    assert steps[2]['orientation']==place['orientation']
    assert 'acquisition_orientation' not in lift  # no mutation of model JSON


def test_direction_dependency_does_not_cross_release_or_opaque_action():
    lift=dict(operation='lift',source='object',arm='left')
    place=dict(operation='transfer',source='object',destination='support',relation='on',
               orientation=dict(from_part='rear end',to_part='tip',facing='left'))
    release=dict(operation='transfer',source='object',destination='empty table',relation='on')
    action=dict(operation='action',action=dict(skill='rotate',arm='left',axis='z',angle=30))
    for intervening in (release,action):
        assert 'acquisition_orientation' not in validate_mission(dict(steps=[lift,intervening,place]))[0]


def test_lift_with_later_transfer_measures_endpoints_before_pick(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    lift=dict(operation='lift',source='long object',arm='left',location='stay')
    place=dict(operation='transfer',source='long object',destination='empty centre of table',relation='on',arm='left',
               orientation=dict(from_part='rear of long object',to_part='tip of long object',facing='left'))
    replies=[{'steps':[lift,place]},dict(bbox=[100,200,400,800],point=[250,500]),
             dict(bbox=[200,200,400,400],point=[300,300]),dict(bbox=[500,600,700,800],point=[600,700])]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    planner=RobodawnPlanner(Client(),'Place long object tip left')
    obs=dict(success=False,image_paths=[str(path)],observation_id=1,sensors={'left':dict(holding=False)})
    action=planner.plan(obs,[])
    assert action['skill']=='pick' and action['mission_intent']['operation']=='lift'
    assert action['orientation_reference']['to_part']['target']=='tip of long object'
    assert not replies
