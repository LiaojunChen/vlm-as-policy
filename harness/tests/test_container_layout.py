import numpy as np

from container_geometry import container_support, release_clearance


def observed_box():
    x,y=np.meshgrid(np.arange(-.14,.1401,.0025),np.arange(-.12,.1201,.0025))
    z=np.full(x.shape,.75)
    z[(abs(x)>.132)|(abs(y)>.112)]=.83
    return np.c_[x.ravel(),y.ravel(),z.ravel()]


def test_parallel_lanes_require_whole_rectangular_footprint_clear():
    points=observed_box()
    footprint=[[-.09,-.035],[-.09,.035],[.09,.035],[.09,-.035]]
    first=container_support(points,.74,layout=dict(slot=0,count=2,facing='left'),footprint=footprint)
    second=container_support(points,.74,layout=dict(slot=1,count=2,facing='left'),footprint=footprint)
    assert first and second
    assert first['tcp'][1]<-.04 and second['tcp'][1]>.04
    assert first['observed_clear_footprint'] and second['observed_clear_footprint']
    assert abs(first['tcp'][0])<.02
    # An occupied first lane must remain excluded from the next placement.
    occupied=points.copy()
    mask=(abs(occupied[:,0]-first['tcp'][0])<.095)&(abs(occupied[:,1]-first['tcp'][1])<.04)
    occupied[mask,2]=.81
    again=container_support(occupied,.74,layout=dict(slot=1,count=2,facing='left'),footprint=footprint)
    assert again and again['tcp'][1]>.04


def test_large_object_or_absent_floor_cannot_be_placed_in_small_empty_pixel_patch():
    points=observed_box();layout=dict(slot=0,count=2,facing='left')
    assert container_support(points,.74,layout=layout,footprint=[[-.3,-.1],[-.3,.1],[.3,.1],[.3,-.1]]) is None
    assert container_support(points,.74,layout=layout,footprint=None) is None


def test_observed_rectangle_fit_can_raise_release_without_using_bounding_circle():
    held=dict(span=[.18,.07],height_under_tcp=.04)
    support=dict(tcp=[0,0,.75],free_radius_m=.04,observed_clear_footprint=True)
    x=np.linspace(-.05,.05,100)
    obstacle=np.c_[x,np.full(len(x),.035),np.full(len(x),.82)]
    assert release_clearance(obstacle,support,held) is not None
    assert release_clearance(obstacle,{**support,'observed_clear_footprint':False},held) is None
