from copy import deepcopy
import json
from types import SimpleNamespace as S
import numpy as np
from PIL import Image
import pytest
from robodawn.episodic_memory import EpisodicMemory
from robodawn.reacquisition_identity import reference_for_reacquisition,verify_reacquired_identity
from robotwin_harness_v3 import SkillError


def case(tmp_path):
    rgb=np.full((100,100,3),240,np.uint8);rgb[10:40,10:40]=[150,80,40]
    Image.fromarray(rgb).save(tmp_path/'0001_head_camera.png')
    rgb[60:90,60:90]=[30,80,150];Image.fromarray(rgb).save(tmp_path/'0004_head_camera.png')
    memory=EpisodicMemory()
    memory.objects['item']=dict(description='item',state='released_or_lost_reobserve',
        identity_verification_observation_id=1,last_visual=dict(observation_id=1,component_only=False,
            bbox=[100,100,400,400],point=[250,250],camera='head_camera'))
    memory.events=[dict(skill='pick',target='item',skill_success=True)]
    obs=dict(observation_id=4,image_paths=[str(tmp_path/'0004_head_camera.png')])
    action=dict(skill='pick',arm='left',target='item',grasp_part='body',camera='head_camera')
    return obs,memory,action


def test_reference_is_observed_verified_pregrasp_image_not_current_geometry(tmp_path):
    obs,memory,a=case(tmp_path);saved=deepcopy(memory.context())
    ref=reference_for_reacquisition(obs,memory,a)
    assert ref['observation_id']==1 and ref['image_path'].endswith('0001_head_camera.png')
    ref['visual']['bbox'][0]=999
    assert memory.context()==saved


@pytest.mark.parametrize('change',[
    lambda o,m,a:a.update(skill='place'),lambda o,m,a:a.update(grasp_part='rim'),
    lambda o,m,a:a.update(grounding_role='entity_identity'),lambda o,m,a:a.update(camera='left_camera'),
    lambda o,m,a:m.objects['item'].update(held_by='right'),
    lambda o,m,a:m.objects['item'].update(state='observed'),
    lambda o,m,a:m.objects['item'].update(identity_verification_observation_id=2),
    lambda o,m,a:m.objects['item']['last_visual'].update(component_only=True),
    lambda o,m,a:m.events.clear(),lambda o,m,a:o.update(observation_id=1),
])
def test_no_extra_call_without_complete_reacquisition_evidence(tmp_path,change):
    o,m,a=case(tmp_path);change(o,m,a)
    assert reference_for_reacquisition(o,m,a) is None
    assert verify_reacquired_identity(None,o,m,a,[600,600,900,900]) is None


@pytest.mark.parametrize('match',['different','uncertain'])
def test_conflicting_identity_rejected_without_committing_candidate(tmp_path,match):
    o,m,a=case(tmp_path);saved=deepcopy(m.context());calls=[]
    def complete(prompt,image,**kwargs):
        calls.append((prompt,image,kwargs))
        return S(raw_text=json.dumps(dict(identity_match=match,reference_description='brown rectangular object',candidate_description='blue wheeled object')))
    with pytest.raises(SkillError,match='Reacquisition identity'):
        verify_reacquired_identity(S(complete_text=complete),o,m,a,[600,600,900,900])
    assert m.context()==saved and len(calls)==1
    assert calls[0][1].shape==(320,640,3) and calls[0][2]['response_schema'] is None
    assert 'Requested identity' not in calls[0][0]
    assert np.count_nonzero(np.any(calls[0][1][:,:320]!=255,axis=2))>40000


def test_rejection_uses_existing_grounding_repair_without_changing_native_request(tmp_path):
    from robodawn.semantic_planner import ground
    o,m,a=case(tmp_path);m.search_region=lambda *args,**kwargs:None
    replies=[dict(bbox=[600,600,900,900],point=[750,750]),
             dict(identity_match='different',reference_description='brown object',candidate_description='blue object'),
             dict(bbox=[100,100,400,400],point=[250,250]),
             dict(identity_match='same',reference_description='brown object',candidate_description='brown object')]
    calls=[]
    def complete(prompt,image,**kwargs):
        calls.append((prompt,kwargs));return S(raw_text=json.dumps(replies.pop(0)))
    result=ground(S(complete_text=complete),o,a,memory=m)
    assert result['point']==[250,250] and result['reacquisition_identity']['reference_observation_id']==1
    assert len(calls)==4 and 'Reacquisition identity' not in calls[0][0]
    assert calls[0][1]['response_schema'] is None and calls[2][1]['response_schema'] is not None
    assert 'Reacquisition identity different' in calls[2][0]
    assert m.objects['item']['last_visual']['bbox']==[100,100,400,400]


def test_missing_reference_file_abstains(tmp_path):
    o,m,a=case(tmp_path);m.objects['item']['last_visual']['observation_id']=2
    m.objects['item']['identity_verification_observation_id']=2
    assert reference_for_reacquisition(o,m,a) is None
