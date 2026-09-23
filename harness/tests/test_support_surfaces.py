import numpy as np
from support_surfaces import observed_support_surface


def plane(z=.78,half=.06,spacing=.002):
    x,y=np.meshgrid(np.arange(-half,half+spacing/2,spacing),np.arange(-half,half+spacing/2,spacing))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]


def test_solid_support_uses_upper_plateau_and_observed_height_not_table_label():
    points=np.r_[plane(.762),plane(.753)+[-.14,0,0]]
    support=observed_support_surface(points,.74,[.06,.06])
    assert support['source']=='visible_rgbd_solid_support'
    np.testing.assert_allclose(support['tcp'],[0,0,.762],atol=.003)
    assert support['support_inference']['kind']=='observed_upper_plateau'


def test_enclosed_floor_requires_surrounding_wall_evidence():
    floor=plane(.75,half=.07)
    a=np.linspace(0,2*np.pi,300,endpoint=False)
    walls=np.c_[.075*np.cos(a),.075*np.sin(a),np.full(len(a),.84)]
    support=observed_support_surface(np.r_[floor,walls],.74,[.04,.04])
    assert support['source']=='visible_rgbd_free_container_floor'
    assert support['support_inference']['wall_sectors']>=8
    assert abs(support['tcp'][2]-.75)<.003


def test_table_plane_under_handle_is_not_a_supported_cavity():
    points=plane(.74)
    x=np.linspace(-.06,.06,150)
    bar=np.c_[x,np.zeros(len(x)),np.full(len(x),.84)]
    assert observed_support_surface(np.r_[points,bar],.74,[.04,.04]) is None


def test_narrow_rail_and_unobserved_hole_do_not_support_a_large_item():
    points=plane(.8)
    rail=points[np.abs(points[:,1])<.006]
    assert observed_support_surface(rail,.74,[.05,.05]) is None
    hole=points[np.linalg.norm(points[:,:2],axis=1)>.05]
    assert observed_support_surface(hole,.74,[.05,.05]) is None
    assert observed_support_surface(points,.74,[.2,.2]) is None
