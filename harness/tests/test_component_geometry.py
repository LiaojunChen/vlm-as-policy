import numpy as np
import pytest
from component_geometry import component_grasp, circular_rim_footprint, near_side_rim_anchor
from component_geometry import complete_observed_circular_component
from component_geometry import elevated_handle_anchor
from robotwin_harness_v3 import resolve_grasp_arm, prefer_side_grasp, component_opening
from robodawn.mission_planner import RobodawnPlanner


def test_rim_moves_contact_from_floor_to_observed_lip_and_fits_tangent():
    angles=np.linspace(0,2*np.pi,400,endpoint=False)
    rim=np.c_[.065*np.cos(angles),.065*np.sin(angles),np.full(400,.80)]
    x,y=np.meshgrid(np.linspace(-.04,.04,20),np.linspace(-.04,.04,20))
    floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,.75)]
    g=component_grasp(np.r_[rim,floor],[.04,0,.75],'rim',.74)
    assert .788<g['tcp'][2]<.8 and g['tcp'][0]>.06
    assert abs(np.cos(g['yaw']))<.1
    assert g['component_width_m']<.012 and not g['tall']
    np.testing.assert_allclose(g['base_xy'],[0,0],atol=.001)


def test_tight_rim_contact_can_recover_only_its_observed_connected_circular_body():
    x,y=np.meshgrid(np.linspace(-.25,.25,241),np.linspace(-.15,.15,161))
    z=np.full_like(x,.74)
    for centre in (-.12,.12):
        radius=np.sqrt((x-centre)**2+y*y);inside=radius<.065
        z[inside]=.746+np.clip((radius[inside]-.04)/.025,0,1)*.044
    cloud=np.stack([x,y,z],axis=-1);valid=np.ones_like(z,bool)
    seed=(x<-.15)&(y>.02)&(z>.76)
    grown=complete_observed_circular_component(cloud,valid,.74,seed)
    assert grown is not None and grown.sum()>seed.sum()*3
    assert np.max(cloud[grown,0])<0
    assert circular_rim_footprint(cloud[grown]) is not None
    assert complete_observed_circular_component(cloud,np.zeros_like(valid),.74,seed) is None


def test_region_growth_abstains_for_non_circular_surfaces():
    x,y=np.meshgrid(np.linspace(-.08,.08,121),np.linspace(-.08,.08,121))
    cloud=np.stack([x,y,np.full_like(x,.78)],axis=-1)
    assert complete_observed_circular_component(cloud,np.ones_like(x,bool),.74,(x<-.04)) is None


def test_partial_arc_does_not_invent_a_whole_object_footprint():
    angles=np.linspace(0,np.pi/2,100)
    arc=np.c_[.06*np.cos(angles),.06*np.sin(angles),np.full(100,.8)]
    assert circular_rim_footprint(arc) is None


def test_long_partial_arc_can_have_stable_fitted_centre_without_hidden_contacts():
    rng=np.random.default_rng(7);angle=np.linspace(0,np.deg2rad(230),400)
    xy=np.c_[.065*np.cos(angle),.065*np.sin(angle)]+rng.normal(0,.0004,(400,2))
    points=np.c_[xy,np.full(400,.8)]
    fit=circular_rim_footprint(points)
    assert fit and fit['partial_rim_fit']
    np.testing.assert_allclose(fit['base_xy'],[0,0],atol=.001)
    assert fit['rim_centre_subsample_spread_m']<.002
    # No contact can be synthesized in the missing angular sector.
    assert near_side_rim_anchor(points,[0,-.3,.94]) is None


def test_short_and_non_circular_partial_surfaces_are_rejected():
    angle=np.linspace(0,np.deg2rad(180),300)
    assert circular_rim_footprint(np.c_[.065*np.cos(angle),.065*np.sin(angle),np.full(300,.8)]) is None
    angle=np.linspace(0,np.deg2rad(235),300)
    assert circular_rim_footprint(np.c_[.09*np.cos(angle),.045*np.sin(angle),np.full(300,.8)]) is None


def test_rim_contact_is_a_measured_point_on_robot_approach_side():
    angle=np.linspace(0,2*np.pi,400,endpoint=False)
    points=np.c_[.065*np.cos(angle),.065*np.sin(angle),np.full(400,.80)]
    choice=near_side_rim_anchor(points,[0,-.3,.94])
    assert choice['point'][1]<-.06 and abs(choice['point'][0])<.01
    assert np.min(np.linalg.norm(points-choice['point'],axis=1))<1e-9
    retry=near_side_rim_anchor(points,[0,-.3,.94],1)
    assert np.linalg.norm(np.asarray(retry['point'])-choice['point'])>.02
    assert retry['point'][1]<-.04


def test_rim_contact_search_abstains_for_unobserved_or_non_circular_lips():
    angle=np.linspace(0,np.pi/2,100)
    points=np.c_[.065*np.cos(angle),.065*np.sin(angle),np.full(100,.8)]
    assert near_side_rim_anchor(points,[0,-.3,.94]) is None
    angle=np.linspace(0,2*np.pi,400,endpoint=False)
    points=np.c_[.065*np.cos(angle),.065*np.sin(angle),np.full(400,.8)]
    assert near_side_rim_anchor(points,[0,0,.94]) is None


def test_handle_uses_local_tangent_not_whole_container_axis():
    x,y=np.meshgrid(np.linspace(-.12,.12,40),np.linspace(-.07,.07,30))
    body=np.c_[x.ravel(),y.ravel(),np.full(x.size,.80)]
    x,y=np.meshgrid(np.linspace(.048,.052,5),np.linspace(-.08,.08,80))
    handle=np.c_[x.ravel(),y.ravel(),np.full(x.size,.92)]
    g=component_grasp(np.r_[body,handle],[.05,0,.92],'handle',.74)
    assert abs(np.cos(g['yaw']))<.1 and abs(g['tcp'][0]-.05)<.003
    assert g['component_width_m']<.006
    assert not prefer_side_grasp(dict(skill='pick',approach='side'),g,.74)


def test_sparse_or_solid_component_abstains():
    assert component_grasp(np.zeros((3,3)),[0,0,.8],'rim',.74) is None
    assert component_grasp(np.zeros((30,3)),[0,0,.8],'body',.74) is None
    x,y=np.meshgrid(np.linspace(-.03,.03,30),np.linspace(-.03,.03,30))
    square=np.c_[x.ravel(),y.ravel(),np.full(x.size,.8)]
    assert component_grasp(square,[0,0,.8],'handle',.74) is None


def test_component_preopening_is_bounded_and_does_not_change_body_grasps():
    assert component_opening({'span':[.06,.06]})==1.
    assert component_opening(dict(source='visible_rgbd_local_component',component_width_m=.004))==pytest.approx(1/3)
    assert component_opening(dict(source='visible_rgbd_local_component',component_width_m=.03))==pytest.approx(.6)
    assert component_opening(dict(source='visible_rgbd_local_component',component_width_m=.2))==1.


def test_handle_grasp_prefers_outer_bar_away_from_observed_parent_body():
    t=np.linspace(0,1,100)
    outer=np.c_[np.full(100,.10),-.02+.04*t,np.full(100,.82)]
    top=np.c_[.05+.05*t,np.full(100,.02),np.full(100,.82)]
    bottom=np.c_[.05+.05*t,np.full(100,-.02),np.full(100,.82)]
    g=component_grasp(np.r_[outer,top,bottom],[.06,.02,.82],'handle',.74,[0,0,.8])
    assert g and g['tcp'][0]>.095 and abs(np.cos(g['yaw']))<.15


def test_protruding_straight_handle_uses_full_strip_not_one_outer_edge():
    x,y=np.meshgrid(np.linspace(-.009,.009,15),np.linspace(-.14,-.06,60))
    points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.775)]
    g=component_grasp(points,[-.009,-.10,.775],'handle',.74,[0,.02,.76])
    assert g and abs(g['tcp'][0])<.002 and g['component_width_m']>.014


def test_open_handle_point_on_contents_selects_only_observed_upper_crossbar():
    x,y=np.meshgrid(np.linspace(-.06,.06,100),np.linspace(-.006,.006,8))
    bar=np.c_[x.ravel(),y.ravel(),np.full(x.size,.92)]
    x,y=np.meshgrid(np.linspace(-.05,.05,60),np.linspace(-.04,.04,40))
    body=np.c_[x.ravel(),y.ravel(),np.full(x.size,.83)]
    points=np.r_[body,bar]
    anchor=elevated_handle_anchor(points,[0,-.025,.84],[0,0,.83])
    assert anchor is not None and anchor['point'][2]==pytest.approx(.92)
    assert min(np.linalg.norm(points-anchor['point'],axis=1))==0
    g=component_grasp(points,anchor['point'],'handle',.74,[0,0,.83])
    assert g['tcp'][2]>.90 and abs(g['tcp'][1])<.008


def test_handle_anchor_does_not_jump_to_broad_body_or_invent_a_bar_between_posts():
    x,y=np.meshgrid(np.linspace(-.05,.05,40),np.linspace(-.04,.04,40))
    body=np.c_[x.ravel(),y.ravel(),np.full(x.size,.92)]
    assert elevated_handle_anchor(body,[0,0,.82],[0,0,.80]) is None
    x,y=np.meshgrid(np.r_[np.linspace(-.06,-.04,30),np.linspace(.04,.06,30)],np.linspace(-.005,.005,8))
    posts=np.c_[x.ravel(),y.ravel(),np.full(x.size,.92)]
    assert elevated_handle_anchor(posts,[0,0,.82],[0,0,.80]) is None


def test_low_handle_or_already_selected_bar_retains_existing_path():
    x,y=np.meshgrid(np.linspace(-.06,.06,80),np.linspace(-.006,.006,8))
    bar=np.c_[x.ravel(),y.ravel(),np.full(x.size,.78)]
    assert elevated_handle_anchor(bar,[0,0,.75],[0,0,.76]) is None
    bar[:,2]=.92
    assert elevated_handle_anchor(bar,[0,0,.916],[0,0,.83]) is None


@pytest.mark.parametrize('instruction',['use the left arm','use your left hand',
    'use the right gripper','use both arms','lift with each hand','use the other arm'])
def test_explicit_arm_constraints_never_change(instruction):
    assert resolve_grasp_arm({'arm':'left'},{'tcp':[.2,0,.8]},instruction,{})=='left'


def test_arm_resolution_preserves_bimanual_roles_occupied_arms_and_midline():
    a={'arm':'left'};g={'tcp':[.2,0,.8]}
    assert resolve_grasp_arm(a,g,'lift the bowl',{})=='right'
    assert resolve_grasp_arm({**a,'arm_required':True},g,'lift the bowl',{})=='left'
    assert resolve_grasp_arm(a,g,'lift the bowl',{'right':{'holding':True}})=='left'
    assert resolve_grasp_arm(a,{'tcp':[.01,0,.8]},'lift the bowl',{})=='left'
    assert resolve_grasp_arm(a,g,'Stack the bowls.\nCompletion requirements: release both grippers at completion.',{})=='right'


def test_compiler_tracks_executed_arm_after_resolution():
    p=RobodawnPlanner(None,'place the bowl')
    p.steps=[{'operation':'transfer'}];p.arm='left'
    p.feedback({'skill':'pick','arm':'right','requested_arm':'left'},{'skill_success':True})
    assert p.arm=='right' and p.phase=='place'
