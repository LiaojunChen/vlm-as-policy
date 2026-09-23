import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from robodawn.mission_planner import RobodawnPlanner,validate_mission,validate_explicit_arm_sequence
from robotwin_harness_v3 import SkillError,relative_support


def test_transfer_cannot_place_before_contact_and_clears_empty_arm(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(100,100)).save(path)
    replies=[{'steps':[{'operation':'transfer','source':'can','destination':'basket','relation':'inside','arm':'left'}]},
             {'bbox':[100,100,300,300],'point':[200,200]},
             {'bbox':[400,400,700,700],'point':[550,550]},
             dict(description_A='can',description_B='basket',source='A',destination='B',source_grasp_part='body'),
             {'bbox_2d':[100,100,300,300],'point_2d':[200,200],'support':'table'},
             {'bbox_2d':[400,400,700,700],'point_2d':[550,550],'support':'table'}]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    obs={'success':False,'image_paths':[str(path)],'observation_id':1,'sensors':{'left':{'holding':False}}}
    p=RobodawnPlanner(Client(),'put can in basket');a=p.plan(obs,[])
    assert a['skill']=='pick'
    p.feedback(a,{'skill_success':True})
    obs['sensors']['left']={'holding':True,'remembered_object':'can'}
    a=p.plan(obs,[])
    assert a['skill']=='place' and a['relation']=='inside' and a['support']=='container'
    p.feedback(a,{'skill_success':True})
    obs['sensors']['left']={'holding':False}
    a=p.plan(obs,[])
    assert a['skill']=='home' and a['arm']=='left'
    p.feedback(a,{'skill_success':True})
    assert p.index==1


def test_failed_motion_does_not_advance_mission():
    p=RobodawnPlanner(None,'transfer')
    p.steps=[{'operation':'transfer'}]
    p.feedback({'skill':'pick'},{'skill_success':False})
    assert p.index==0 and p.phase=='pick' and p.failures==1


def test_example_object_names_cannot_be_executed_as_a_real_plan():
    with pytest.raises(SkillError,match='placeholder'):
        validate_mission({'steps':[dict(operation='arrange',sources=['leftmost desired object','middle desired object'],arms=['left','right'])]})
    assert validate_mission({'steps':[dict(operation='arrange',sources=['left blue bowl','right blue bowl'],arms=['left','right'])]})


def test_handle_crop_keeps_the_source_identity_for_contact_tracking(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    replies=[{'steps':[dict(operation='lift',source='yellow basket',arm='right',grasp_part='handle',location='stay')]},
             dict(bbox=[300,200,800,800],point=[550,500]),
             dict(bbox=[300,200,700,400],point=[500,300])]
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text=json.dumps(replies.pop(0)))
    obs=dict(success=False,image_paths=[str(path)],observation_id=1,sensors={'right':{'holding':False}})
    a=RobodawnPlanner(Client(),'Lift the basket').plan(obs,[])
    assert a['target']=='yellow basket' and a['arm']=='right'
    assert a['component_grounding_target']=='solid graspable handle of yellow basket'
    assert a['parent_grounding']['target']=='yellow basket'


def test_spatial_relation_moves_to_side_and_preserves_table_height():
    xy=np.array([[-.03,-.02],[-.03,.02],[.03,-.02],[.03,.02]])
    pts=np.c_[xy,np.full(4,.80)]
    left=relative_support(pts,[.06,.04],'left_of',.74)
    right=relative_support(pts,[.06,.04],'right_of',.74)
    assert left[0]<-.075 and right[0]>.075
    assert left[1]==right[1]==0 and left[2]==right[2]==.74


def test_mission_rejects_self_destination_and_unknown_relation():
    step={'operation':'transfer','source':'red cube','destination':'red cube','relation':'on'}
    with pytest.raises(SkillError,match='different'):validate_mission({'steps':[step]})
    with pytest.raises(SkillError,match='relation'):validate_mission({'steps':[{**step,'destination':'blue cube','relation':'near'}]})


def test_arrangement_compiles_distinct_slots_in_model_selected_order():
    steps=validate_mission({'steps':[{'operation':'arrange','sources':['red','green','blue'],'arms':['left','right','left']}]})
    assert [s['source'] for s in steps]==['red','green','blue']
    assert [s['slot'] for s in steps]==[0,1,2]
    assert all(s['relation']=='row_slot' for s in steps)
    validate_explicit_arm_sequence(steps,'Using the left arm, right arm and left arm, arrange the blocks.')
    with pytest.raises(SkillError,match='arm sequence'):
        validate_explicit_arm_sequence(steps,'Using the right arm, right arm and left arm, arrange the blocks.')


def test_stack_preserves_unspecified_base_location_and_moves_only_upper_objects():
    steps=validate_mission({'steps':[dict(operation='stack',sources=['bottom bowl','middle bowl','top bowl'],
                                        arms=['auto','left','right'],grasp_part='rim')]})
    assert [(s['source'],s['destination']) for s in steps]==[('middle bowl','bottom bowl'),('top bowl','middle bowl')]
    assert [s['arm'] for s in steps]==['left','right']
    assert all(s['grasp_part']=='rim' for s in steps)


def test_support_goal_cycle_is_rejected_but_staging_can_change_final_support():
    def transfer(source,destination):
        return dict(operation='transfer',source=source,destination=destination,relation='on',arm='auto')
    with pytest.raises(SkillError,match='Cyclic support'):
        validate_mission(dict(steps=[transfer('A','B'),transfer('B','A')]))
    assert len(validate_mission(dict(steps=[transfer('A','B'),transfer('A','empty centre of table'),transfer('B','A')])))==3


def test_native_operation_tag_is_normalized_without_inventing_a_plan():
    steps=validate_mission({'steps':['LIFT',{'source':'bottle','arm':'right','location':'side'}]})
    assert steps==[{'operation':'lift','source':'bottle','arm':'right','location':'side'}]


def test_unknown_later_action_is_rejected_before_any_motion():
    steps=[{'operation':'lift','source':'handle','arm':'left'},
           {'operation':'action','action':{'skill':'tap','arm':'left'}}]
    with pytest.raises(SkillError,match='Unknown action skill'):
        validate_mission({'steps':steps})


def test_nested_action_parameters_are_validated():
    steps=validate_mission({'steps':[{'operation':'action','action':{
        'skill':'home','parameters':{'arm':'left'}}}]})
    assert steps[0]['action']=={'skill':'home','arm':'left'}


def test_button_contact_is_not_an_articulated_grasp():
    with pytest.raises(SkillError,match='press contact'):
        validate_mission({'steps':[{'operation':'action','action':{
            'skill':'grasp_handle','target':'top button','arm':'left'}}]})


def test_invalid_high_level_plan_can_fall_back_to_a_sensor_gated_action(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(100,100)).save(path)
    class Client:
        def complete_text(self,*args,**kwargs):return SimpleNamespace(raw_text='{"steps":["LIFT"]}')
    p=RobodawnPlanner(Client(),'lift the visible object')
    p.reactive=SimpleNamespace(plan=lambda obs,history:dict(skill='home',name='home',arm='left'))
    obs=dict(success=False,image_paths=[str(path)],observation_id=1,sensors={'left':{'holding':False}})
    action=p.plan(obs,[])
    assert action['skill']=='home' and 'mission_recovery' in action
    assert p.steps==[] and p.index==0


def test_explicit_transfer_sequence_rejects_redundant_initial_staging_only():
    steps=[dict(operation='transfer',arm=side) for side in ('left','right','left')]*2
    instruction='Use the left arm, right arm and left arm to arrange the three objects.'
    with pytest.raises(SkillError,match='duplicate preliminary'):
        validate_explicit_arm_sequence(steps,instruction)
    validate_explicit_arm_sequence(steps[:2],instruction,complete_plan=False)
