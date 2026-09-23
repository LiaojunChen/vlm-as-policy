import numpy as np
from container_geometry import container_support,circular_cavity_support,release_clearance


def test_container_support_avoids_loaded_object_and_overhead_handle():
    xs,ys=np.meshgrid(np.linspace(-.12,.12,161),np.linspace(-.10,.10,141))
    z=np.full_like(xs,.75)
    wall=(abs(xs)>.11)|(abs(ys)>.09)
    z[wall]=.82
    contents=(xs<-.03)&(ys<.04)
    z[contents]=.87
    handle=abs(ys-.05)<.006
    z[handle]=.93
    points=np.stack([xs,ys,z],axis=-1).reshape(-1,3)
    support=container_support(points,.74)
    assert support is not None
    x,y,height=support['tcp']
    assert x>-.025 and abs(y-.05)>.015
    assert abs(height-.75)<.003


def test_plain_table_does_not_invent_a_container():
    xs,ys=np.meshgrid(np.linspace(-.1,.1,30),np.linspace(-.1,.1,30))
    points=np.stack([xs,ys,np.full_like(xs,.74)],axis=-1).reshape(-1,3)
    assert container_support(points,.74) is None


def test_circular_bowl_support_centres_the_floor_not_a_sloped_wall_pixel():
    a=np.linspace(0,2*np.pi,400,endpoint=False)
    rim=np.c_[.06*np.cos(a),.06*np.sin(a),np.full(len(a),.79)]
    x,y=np.meshgrid(np.linspace(-.03,.03,25),np.linspace(-.03,.03,25))
    floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,.75)]
    g=circular_cavity_support(np.r_[rim,floor],.74)
    np.testing.assert_allclose(g['tcp'],[0,0,.75],atol=.001)
    flat=np.r_[rim,floor];flat[:,2]=.79
    assert circular_cavity_support(flat,.74) is None
    assert circular_cavity_support(rim,.74) is None


def test_release_clearance_is_for_small_items_in_a_narrow_observed_opening():
    x,y=np.meshgrid(np.linspace(-.05,.05,50),np.linspace(-.05,.05,50))
    z=np.full_like(x,.75);z[abs(x)>.04]=.91
    pts=np.stack([x,y,z],-1).reshape(-1,3)
    support=dict(tcp=[0,0,.75],free_radius_m=.038)
    held=dict(span=[.05,.05],height_under_tcp=.06)
    g=release_clearance(pts,support,held)
    assert g and .87<g['release_height_m']<.90
    assert release_clearance(pts,{**support,'free_radius_m':.08},held) is None
    assert release_clearance(pts,support,{**held,'span':[.12,.12]}) is None
    assert release_clearance(pts[:0],support,held) is None
