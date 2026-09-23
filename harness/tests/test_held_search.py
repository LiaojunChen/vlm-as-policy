import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from robodawn.episodic_memory import EpisodicMemory
from robodawn.semantic_planner import ground


def observation(tmp_path,index,shift=0):
    ys,xs=np.indices((100,120))
    cloud=np.dstack([(xs-60)*.003,(ys-50)*.003,np.full_like(xs,.74,dtype=float)])
    region=(abs(cloud[:,:,0]-shift)<.05)&(abs(cloud[:,:,1])<.04)
    cloud[region,2]=.80
    path=tmp_path/f'{index:04d}_head_camera.png'
    image=np.full((100,120,3),240,np.uint8);image[region]=[60,100,150]
    Image.fromarray(image).save(path)
    np.savez_compressed(tmp_path/f'{index:04d}_rgbd.npz',world_xyz=cloud,
                        valid=np.ones(region.shape,bool),robot_self_mask=np.zeros(region.shape,bool))
    return dict(image_paths=[str(path)],observation_id=index,table_height_m=.74,
                endpose={'right_endpose':[shift,0,1,1,0,0,0]},
                sensors={'right':dict(holding=True,remembered_object='support'),
                         'left':dict(holding=True,remembered_object='payload')})


def memory(tmp_path):
    obs=observation(tmp_path,1);m=EpisodicMemory();m.latest_observation=obs
    a=dict(skill='pick',arm='right',target='support',observation_id=1,bbox=[300,300,700,700],point=[500,500])
    r=dict(skill_success=True,sensors=obs['sensors'],subactions=[dict(name='close',endpose_after=obs['endpose'])])
    m.feedback(a,r)
    return m,a,r


def test_motion_hint_searches_current_foreground_not_historical_pixels(tmp_path):
    m,_,_=memory(tmp_path);obs=observation(tmp_path,2,.12)
    region=m.held_search.search(obs,'support','head_camera')
    assert region['bbox'][0]>650 and region['current_foreground_pixels']>32
    assert region['reference_observation_id']==1 and region['current_observation_id']==2
    assert region['support_plane']['source']=='current_observed_upward_planar_search_patch'
    # Historical extent never authorizes missing/current-background evidence.
    obs['endpose']['right_endpose'][0]=-.12
    assert m.held_search.search(obs,'support','head_camera') is None


def test_search_invalidated_on_contact_loss_wrong_identity_and_release(tmp_path):
    m,_,_=memory(tmp_path);obs=observation(tmp_path,2,.12)
    obs['sensors']['right']['holding']=False
    assert m.held_search.search(obs,'support','head_camera') is None
    obs['sensors']['right'].update(holding=True,remembered_object='different support')
    assert m.held_search.search(obs,'support','head_camera') is None
    m.feedback(dict(skill='open',arm='right'),dict(sensors=obs['sensors']))
    assert not m.held_search.entries


def test_no_reference_on_failed_unclosed_or_stale_acquisition(tmp_path):
    m,a,r=memory(tmp_path);m.held_search.entries.clear()
    m.feedback(a,dict(r,subactions=[]));assert not m.held_search.entries
    m.feedback(dict(a,observation_id=2),r);assert not m.held_search.entries
    m.feedback(dict(a,grasp_part='handle'),r);assert not m.held_search.entries


def test_held_destination_crop_still_requires_current_model_point(tmp_path):
    m,_,_=memory(tmp_path);obs=observation(tmp_path,2,.12);shapes=[]
    def complete(prompt,image,**kwargs):
        shapes.append(image.shape)
        assert 'ONLY a search hint' in prompt and 'supporting surface/interior' in prompt
        return SimpleNamespace(raw_text=json.dumps(dict(bbox=[300,300,700,700],point=[500,500],support='object')))
    a=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='place',arm='left',target='support'),memory=m)
    assert shapes[0][1]<120 and a['point'][0]>650 and a['contact_view_evidence']
    assert a['memory_guided_search']['current_observation_id']==2


def test_unconfirmed_hint_falls_back_to_current_full_scene(tmp_path):
    m,_,_=memory(tmp_path);obs=observation(tmp_path,2,.12);widths=[]
    def complete(prompt,image,**kwargs):
        widths.append(image.shape[1])
        response={'visible':False} if len(widths)==1 else dict(bbox=[710,350,950,660],point=[830,500],support='object')
        return SimpleNamespace(raw_text=json.dumps(response))
    a=ground(SimpleNamespace(complete_text=complete),obs,dict(skill='place',arm='left',target='support'),memory=m)
    assert widths[0]<widths[1]==120 and 'memory_guided_search' not in a


def test_support_patch_excludes_elevated_thin_handle_and_abstains_on_rail():
    from robodawn.held_search import broad_support_patch
    ys,xs=np.indices((80,80));cloud=np.dstack([xs*.002,ys*.002,np.full((80,80),.8)])
    region=np.zeros((80,80),bool);region[15:65,15:65]=True
    cloud[5:10,:,2]=.85;region[5:10,:]=True
    mask,plane=broad_support_patch(cloud,region)
    assert mask[30,30] and not mask[7,30] and plane['pixels']==2500
    region[:]=False;region[5:10,:]=True
    assert broad_support_patch(cloud,region) is None
