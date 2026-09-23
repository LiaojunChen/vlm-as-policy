import json
from types import SimpleNamespace
from PIL import Image,ImageDraw
from robodawn.pair_grounding import verify_pair,paired_image
from robodawn.episodic_memory import EpisodicMemory
from robodawn.mission_planner import RobodawnPlanner


def test_same_model_can_swap_wrong_bindings_without_rewriting_goal_names(tmp_path):
    path=tmp_path/'scene.png';Image.new('RGB',(320,240)).save(path)
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            assert image.shape==(420,320,3)
            assert 'NEVER the left/right order of the crop panels' in prompt
            return SimpleNamespace(raw_text=json.dumps(dict(description_A='flat green plate',description_B='blue bowl',
                        source='B',destination='A',source_grasp_part='rim')))
    source=dict(target='bowl',bbox=[400,500,800,900],point=[600,700])
    dest=dict(target='plate',bbox=[800,400,1000,700],point=[900,550])
    first,second=verify_pair(Client(),dict(image_paths=[str(path)],observation_id=1),source,dest)
    assert first['target']=='bowl' and first['bbox']==dest['bbox']
    assert second['target']=='plate' and second['bbox']==source['bbox']
    assert first['preferred_grasp_part']=='rim' and source['bbox']==[400,500,800,900]
    memory=EpisodicMemory();memory.remember_visual(first,1)
    assert memory.preferred_grasp_part('bowl')=='rim'


def test_full_scene_preserves_spatial_context_when_crop_order_is_reversed(tmp_path):
    path=tmp_path/'scene.png';scene=Image.new('RGB',(320,240),'white')
    draw=ImageDraw.Draw(scene)
    draw.rectangle((30,30,90,90),fill='red');draw.rectangle((220,30,280,90),fill='blue')
    scene.save(path)
    first=dict(bbox=[680,110,900,410]);second=dict(bbox=[80,110,300,410])
    result=paired_image(path,first,second)
    assert tuple(result[60,60])==(255,0,0)
    assert tuple(result[60,250])==(0,0,255)
    assert tuple(result[330,80])==(0,0,255)
    assert tuple(result[330,240])==(255,0,0)


def test_same_frame_inventory_candidates_skip_duplicate_localization_but_not_verification(tmp_path):
    path=tmp_path/'scene.png';Image.new('RGB',(320,240)).save(path);calls=[]
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            calls.append(prompt)
            assert 'TOP panel' in prompt
            return SimpleNamespace(raw_text=json.dumps(dict(description_A='blue bowl',description_B='green plate',
                                   source='A',destination='B',source_grasp_part='rim')))
    p=RobodawnPlanner(Client(),'place blue bowl onto green plate')
    p.steps=[dict(operation='transfer',source='blue bowl',destination='green plate',arm='auto',relation='on')]
    p.memory.remember_visual(dict(target='blue bowl',bbox=[100,100,300,300],point=[200,200],inventory_identity=True),1)
    p.memory.remember_visual(dict(target='green plate',bbox=[600,600,900,900],point=[750,750],inventory_identity=True),1)
    p._bind_transfer_identities(dict(image_paths=[str(path)],observation_id=1,sensors={}))
    assert len(calls)==1 and p.memory.preferred_grasp_part('blue bowl')=='rim'


def test_verified_affordance_changes_body_request_not_semantic_target(tmp_path):
    path=tmp_path/'scene.png';Image.new('RGB',(320,240)).save(path)
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            assert 'component=rim' in prompt
            return SimpleNamespace(raw_text=json.dumps(dict(bbox=[500,500,800,800],point=[650,650])))
    p=RobodawnPlanner(Client(),'place bowl')
    p.memory.remember_visual(dict(target='bowl',bbox=[500,500,800,800],point=[650,650],preferred_grasp_part='rim'),1)
    obs=dict(image_paths=[str(path)],observation_id=1,sensors={})
    a=p._ground(obs,dict(skill='pick',target='bowl',arm='right',grasp_part='body'))
    assert a['target']=='bowl' and a['grasp_part']=='rim' and a['requested_grasp_part']=='body'
