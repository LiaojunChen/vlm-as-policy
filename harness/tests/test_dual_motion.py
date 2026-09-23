import json
from types import SimpleNamespace
import pytest
from PIL import Image
from robotwin_harness_v3 import validate,SkillError
from robodawn.mission_planner import RobodawnPlanner,validate_mission
from dual_motion import translate_pair


def test_bimanual_compiler_waits_for_two_real_contacts_before_lifting(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    step=dict(operation='bimanual_lift',source='vessel',left_source='left handle',right_source='right handle')
    replies=[{'steps':[step]},dict(bbox=[100,100,900,800],point=[500,400]),
             dict(bbox=[100,200,200,300],point=[150,250]),
             dict(bbox=[100,100,900,800],point=[500,400]),
             dict(bbox=[600,200,700,300],point=[650,250])]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    p=RobodawnPlanner(Client(),'Lift one vessel with both arms.')
    obs=dict(success=False,image_paths=[str(path)],observation_id=1,
             sensors={side:{'holding':False} for side in ('left','right')})
    a=p.plan(obs,[]);assert a['skill']=='grasp_handle' and a['arm']=='left'
    assert a['arm_required'] is True
    assert 'grounding_crop_bbox_px' in a and a['parent_grounding']['target']=='vessel'
    p.feedback(a,{'skill_success':True});assert p.index==0
    obs['sensors']['left']={'holding':True,'remembered_object':'left handle'}
    a=p.plan(obs,[]);assert a['skill']=='grasp_handle' and a['arm']=='right'
    p.feedback(a,{'skill_success':True});assert p.index==0
    obs['sensors']['right']={'holding':True,'remembered_object':'right handle'}
    a=p.plan(obs,[]);assert a['skill']=='dual_move' and a['delta']==[0,0,.12]
    p.feedback(a,{'skill_success':True});assert p.index==1


def test_dual_translation_is_bounded_and_never_moves_an_empty_arm():
    validate(dict(skill='dual_move',delta=[0,0,.12]))
    with pytest.raises(SkillError):validate(dict(skill='dual_move',delta=[0,0,.3]))
    b=SimpleNamespace(env=None,sensors=lambda:{'left':{'holding':True},'right':{'holding':False}})
    with pytest.raises(SkillError,match='BOTH'):translate_pair(b,[0,0,.1],[])


def test_bimanual_target_parts_must_be_distinct():
    with pytest.raises(SkillError,match='distinct'):
        validate_mission({'steps':[dict(operation='bimanual_lift',source='rod',left_source='rod',right_source='rod')]})
