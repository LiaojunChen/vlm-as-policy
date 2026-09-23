import numpy as np
from flat_support_memory import components,refresh_landmarks,current_support_evidence


def view(clipped=False,offset=0.):
    yy,xx=np.indices((80,80));cloud=np.dstack([(xx-40)*.002+offset,(yy-40)*.002,np.full_like(xx,.741,dtype=float)])
    rgb=np.full((80,80,3),255,np.uint8);rgb[15:65,25:55]=[0,255,255]
    valid=np.ones((80,80),bool)
    if clipped:return dict(rgb=rgb[:55],cloud=cloud[:55],valid=valid[:55])
    return dict(rgb=rgb,cloud=cloud,valid=valid)


def test_clipped_boundary_is_replaced_only_by_matching_complete_current_view():
    head=view(True);wrist=view();memory={}
    refresh_landmarks(memory,{'head_camera':head},.74,1)
    item=memory['cyan'][0];old_y=item['point'][1]
    assert item['clipped'] and not item['complete_boundary']
    refresh_landmarks(memory,{'head_camera':head,'left_camera':wrist},.74,2)
    assert item['complete_boundary'] and item['refined_from_clipped'] and item['camera']=='left_camera'
    assert item['point'][1]>old_y+.005 and item['observation_id']==2
    evidence=current_support_evidence(dict(item,colour='cyan'),{'head_camera':head},.74)
    assert evidence['current_matched_pixels']['head_camera']>30


def test_different_location_occluded_edges_and_missing_depth_do_not_fill_boundaries():
    head=view(True);memory={};refresh_landmarks(memory,{'head_camera':head},.74,1)
    item=memory['cyan'][0];old=dict(item)
    refresh_landmarks(memory,{'left_camera':view(offset=.2)},.74,2)
    assert item==old
    occluded=view();occluded['valid'][40:,:40]=False
    found=components(occluded['rgb'],occluded['cloud'],occluded['valid'],.74,2,'left_camera')['cyan'][0]
    assert not found['complete_boundary']
    empty=view();empty['valid'][:]=False
    assert current_support_evidence(dict(item,colour='cyan'),{'head_camera':empty},.74) is None


def test_planner_inspects_pending_support_once_then_homes_without_consuming_goal(monkeypatch):
    from robodawn.mission_planner import RobodawnPlanner
    p=RobodawnPlanner(None,'move item to cyan mat')
    p.steps=[dict(operation='lift',source='item'),dict(operation='transfer',source='item',destination='cyan mat',relation='on')]
    obs=dict(success=False,observation_id=1,image_paths=['head','left','right'],sensors={'left':{'holding':False},'right':{'holding':False}})
    monkeypatch.setattr('flat_support_memory.clipped_support_anchor',lambda *a:dict(point=[-.2,0,.74],image_point=[200,900]))
    action=p._support_boundary_inspection(obs,p.steps[0])
    assert action['arm']=='left' and action['inspection_anchor']==[200,900]
    p.feedback(action,dict(skill_success=True))
    home=p.plan(obs,[])
    assert home['skill']=='home' and home['arm']=='left'
    p.feedback(home,dict(skill_success=True))
    assert p.index==0 and p.phase=='pick'
    assert p._support_boundary_inspection(obs,p.steps[0]) is None


def test_support_inspection_never_frees_occupied_hands_or_crosses_unknown_actions(monkeypatch):
    from robodawn.mission_planner import RobodawnPlanner
    p=RobodawnPlanner(None,'move item');p.steps=[dict(operation='lift',source='item'),
        dict(operation='action',action=dict(skill='wait')),
        dict(operation='transfer',source='item',destination='cyan mat',relation='on')]
    obs=dict(sensors={'left':{'holding':False},'right':{'holding':False}},image_paths=['head','left','right'])
    def unexpected(*a):raise AssertionError('Do not inspect without a valid pending support')
    monkeypatch.setattr('flat_support_memory.clipped_support_anchor',unexpected)
    assert p._support_boundary_inspection(obs,p.steps[0]) is None
    p.steps.pop(1)
    obs['sensors']={'left':{'holding':True},'right':{'holding':True}}
    assert p._support_boundary_inspection(obs,p.steps[0]) is None


def test_visible_background_replacing_old_support_invalidates_memory():
    memory={};head=view(True);refresh_landmarks(memory,{'head_camera':head},.74,1)
    refresh_landmarks(memory,{'left_camera':view()},.74,2)
    item=dict(memory['cyan'][0],colour='cyan')
    current=view();current['rgb'][15:35,25:55]=255
    assert current_support_evidence(item,{'head_camera':current},.74) is None


def test_straight_occluding_edge_cannot_be_mistaken_for_complete_rectangle():
    for robot_occlusion in (True,False):
        current=view()
        if robot_occlusion:current['valid'][40:,:]=False
        else:current['cloud'][40:,:,2]=.83
        item=components(current['rgb'],current['cloud'],current['valid'],.74,1,'head_camera')['cyan'][0]
        assert not item['clipped']
        assert item['exterior_background_fractions'] is not None
        assert min(item['exterior_background_fractions'])<.75 and not item['complete_boundary']
