import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from robodawn.episodic_memory import EpisodicMemory
from robodawn.semantic_planner import ground


def memory():
    m=EpisodicMemory();m.remember_visual(dict(target='initial right bowl',bbox=[800,100,1000,300],point=[900,200]),1)
    m.feedback(dict(skill='place',arm='right',target='base bowl',observation_id=4,relation='on',
                    bbox=[400,300,600,600],point=[500,450],mission_intent={'source':'initial right bowl'}),
               dict(skill_success=True,sensors={'right':dict(holding=False)}))
    return m


def test_release_memory_guides_crop_but_execution_uses_fresh_model_coordinates(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            assert image.shape[1]<320 and 'INITIAL positional label' in prompt
            return SimpleNamespace(raw_text=json.dumps(dict(bbox=[200,200,800,800],point=[550,600],support='container')))
    m=memory();obs=dict(image_paths=[str(path)],observation_id=8)
    action=ground(Client(),obs,dict(skill='place',arm='right',target='initial right bowl'),memory=m)
    assert 'memory_guided_search' in action and 400<action['point'][0]<600
    crop=np.asarray(action['grounding_crop_bbox_px']);expected=(crop[:2]+(crop[2:]-crop[:2])*[.55,.60])/[319,239]*1000
    np.testing.assert_allclose(action['point'],expected)
    # A support surface is not a fresh whole-object identity. It must not
    # recursively shrink the next search region or overwrite the entity box.
    assert m.objects['initial right bowl']['last_visual']['observation_id']==1
    assert m.search_region('initial right bowl')['bbox']==[400,300,600,600]


def test_stale_release_hint_can_fall_back_to_full_current_frame(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path);widths=[]
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            widths.append(image.shape[1])
            result={'visible':False} if len(widths)==1 else dict(bbox=[100,100,300,300],point=[200,200])
            return SimpleNamespace(raw_text=json.dumps(result))
    action=ground(Client(),dict(image_paths=[str(path)],observation_id=8),
                  dict(skill='place',arm='right',target='initial right bowl'),memory=memory())
    assert widths[0]<320 and widths[1]==320 and action['point']==[200,200]
    assert 'memory_guided_search' not in action


def test_held_objects_and_never_moved_objects_do_not_use_release_search():
    m=memory();m.objects['initial right bowl']['held_by']='right'
    assert m.search_region('initial right bowl') is None
    assert m.search_region('unknown object') is None


def test_stationary_identity_is_only_a_later_current_frame_search_hint():
    m=EpisodicMemory();m.remember_visual(dict(target='bottle',bbox=[100,100,300,500],point=[200,300]),1)
    assert m.search_region('bottle',1) is None
    assert m.search_region('bottle',2)['bbox']==[100,100,300,500]
    m.objects['bottle']['state']='may_have_moved_reobserve'
    assert m.search_region('bottle',2) is None


def test_cropped_grounding_keeps_identity_but_not_old_coordinate_system(tmp_path):
    path=tmp_path/'head.png';Image.new('RGB',(320,240)).save(path)
    class Client:
        def complete_text(self,prompt,image,**kwargs):
            assert 'initial right bowl' in prompt
            assert '"last_visual":' not in prompt and '"last_release_region":' not in prompt
            assert 'Historical coordinates are intentionally omitted' in prompt
            return SimpleNamespace(raw_text=json.dumps(dict(bbox=[200,200,800,800],point=[500,500])))
    ground(Client(),dict(image_paths=[str(path)],observation_id=8),
           dict(skill='place',arm='right',target='initial right bowl'),memory=memory())
