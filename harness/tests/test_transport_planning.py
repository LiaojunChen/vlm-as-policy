import numpy as np
import pytest
from transport_planning import shared_table_support
from robodawn.mission_planner import RobodawnPlanner
from robodawn.episodic_memory import EpisodicMemory


def table_scene():
    x,y=np.meshgrid(np.linspace(-.5,.5,321),np.linspace(-.4,.4,241))
    return np.stack([x,y,np.full_like(x,.74)],axis=2),np.ones(x.shape,bool)


def support(cloud,valid):
    return shared_table_support(cloud,valid,.74,[.12,.12],[[-.3,-.31,.94],[.3,-.31,.94]])


def test_shared_support_requires_observed_clear_footprint_and_avoids_obstacles():
    cloud,valid=table_scene();first=support(cloud,valid)
    assert first and abs(first['tcp'][0])<.015 and abs(first['tcp'][1]+.11)<.015
    obstruction=np.linalg.norm(cloud[:,:,:2]-np.array([0,-.11]),axis=2)<.035
    cloud[obstruction,2]=.84
    changed=support(cloud,valid)
    assert changed is None or np.linalg.norm(np.asarray(changed['tcp'])[:2]-[0,-.11])>changed['required_radius_m']+.025
    assert support(cloud,np.zeros_like(valid)) is None


def test_explicit_empty_table_never_turns_into_a_vessel_from_a_broad_bbox():
    from types import SimpleNamespace
    from robotwin_harness_v3 import Bridge
    from transport_planning import explicit_free_table
    cloud,valid=table_scene();vessel=np.linalg.norm(cloud[:,:,:2]-[.25,0],axis=2)<.065
    cloud[vessel,2]=.80
    b=Bridge.__new__(Bridge);b.cloud=cloud;b.valid=valid;b.table=.74;b.depth_id=1;b.landmarks={}
    b.held={'left':{'span':[.13,.13]}};matrix=np.eye(4);matrix[:3,3]=[0,-1,1.6]
    b.cam=SimpleNamespace(get_model_matrix=lambda:matrix)
    a=dict(skill='place',arm='left',target='empty centre of table',support='table',relation='on',
           bbox=[0,0,1000,1000],point=[500,500],observation_id=1)
    g=b.geometry(a)
    assert g['source']=='current_rgbd_requested_free_table' and a['support']=='table'
    assert abs(g['tcp'][0])<.02 and abs(g['tcp'][1])<.02
    assert 'pending_release_region' in g and g['free_radius_m']>g['required_radius_m']
    assert not explicit_free_table('bowl on the table')
    assert not explicit_free_table('empty bowl')


def planner(instruction='Nest the small bowl in the larger bowl'):
    p=RobodawnPlanner(None,instruction)
    p.steps=[dict(operation='transfer',source='small bowl',destination='large bowl',relation='inside',arm='right',grasp_part='rim')]
    p.arm='right';p.phase='place'
    return p


def failed_place():
    return dict(skill='place',arm='right',target='large bowl'),dict(skill_success=False,
        failure='No reachable placement orientation; choose a closer destination or adjust held pose',
        sensors={'right':dict(holding=True,remembered_object='small bowl'),'left':dict(holding=False)})


def test_two_failed_placements_schedule_one_regrasp_and_preserve_goal():
    p=planner();a,r=failed_place();p.feedback(a,r)
    assert len(p.steps)==1
    p.feedback(a,r)
    assert len(p.steps)==2 and p.steps[0]['regrasp_stage']
    assert p.steps[1]['destination']=='large bowl' and p.steps[1]['relation']=='inside'
    assert p.steps[1]['arm']=='left' and p.steps[1]['arm_required']
    assert p.phase=='place' and p.arm=='right' and p.failures==0
    assert not p._schedule_regrasp(a,r)


@pytest.mark.parametrize('instruction',['Use the right arm to move the bowl','Use your left hand to place the bowl','Use both arms'])
def test_regrasp_never_overrides_explicit_arm_requirements(instruction):
    p=planner(instruction);a,r=failed_place()
    assert not p._schedule_regrasp(a,r)


def test_no_regrasp_if_other_arm_is_occupied_or_grasp_is_lost():
    p=planner();a,r=failed_place();r['sensors']['left']['holding']=True
    assert not p._schedule_regrasp(a,r)
    r['sensors']['left']['holding']=False;r['sensors']['right']['holding']=False
    assert not p._schedule_regrasp(a,r)


def test_release_search_hint_remains_unverified_and_uses_observed_geometry():
    m=EpisodicMemory();m.remember_visual(dict(target='bowl',bbox=[600,400,900,700],point=[750,550]),1)
    m.feedback(dict(skill='place',arm='right',target='free table',mission_intent={'source':'bowl'}),
        dict(skill_success=True,geometry={'pending_release_region':dict(bbox=[400,500,600,700],point=[500,600])},
             sensors={'right':{'holding':False}}))
    entry=m.context()['objects'][0]
    assert entry['last_release_region']['point']==[500,600]
    assert entry['last_release_region']['status']=='expected_from_release_not_yet_visually_verified'
    assert entry['last_visual']['observation_id']==1


def test_generic_endpoint_names_do_not_pollute_cross_object_identity():
    m=EpisodicMemory();m.remember_visual(dict(target='toe',bbox=[0,0,100,100],point=[50,50],grounding_role='orientation_endpoint'),1)
    assert m.context()['objects']==[]


def test_later_placement_replaces_staging_hint_and_relative_moves_clear_it():
    m=EpisodicMemory();m.remember_visual(dict(target='bowl',bbox=[0,0,100,100],point=[50,50]),1)
    for point,relation in [([500,600],'on'),([200,300],'inside'),([800,300],'left_of')]:
        m.feedback(dict(skill='place',arm='left',target='support',bbox=[0,0,1000,1000],point=point,
                        relation=relation,mission_intent={'source':'bowl'}),dict(skill_success=True,sensors={}))
        entry=m.context()['objects'][0]
        if relation=='left_of':assert 'last_release_region' not in entry
        else:assert entry['last_release_region']['point']==point
