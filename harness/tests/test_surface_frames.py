import numpy as np
from surface_frames import flat_source,sloped_frame,aligned_rotations,SurfaceFrameMemory
from transforms3d.quaternions import quat2mat


def inclined_points():
    n=np.array([0.,-.9,np.sqrt(1-.9**2)])
    up=np.array([0.,n[2],.9]);across=np.cross(n,up)
    x,y=np.meshgrid(np.linspace(-.055,.055,40),np.linspace(-.032,.032,30))
    pts=np.array([-.12,.15,.88])+x.reshape(-1,1)*up+y.reshape(-1,1)*across
    return pts,n,up


def test_general_sloping_plane_from_depth_and_orientation_alignment():
    pts,normal,up=inclined_points();frame=sloped_frame(pts,.74)
    assert frame is not None and frame['coverage']>.55
    np.testing.assert_allclose(frame['normal'],normal,atol=1e-5)
    info=dict(object_yaw=.3,grasp_quat=[1,0,0,0])
    for rotation in aligned_rotations(info,frame):
        np.testing.assert_allclose(rotation@np.array([0,0,1]),normal,atol=1e-6)
        assert np.isclose(np.linalg.det(rotation),1)
        axis=rotation@np.array([np.cos(.3),np.sin(.3),0])
        assert np.isclose(abs(axis@up),1)


def test_neither_horizontal_table_nor_tall_volume_is_a_sloping_flat_source():
    x,y=np.meshgrid(np.linspace(-.07,.07,40),np.linspace(-.03,.03,30))
    pts=np.c_[x.ravel(),y.ravel(),np.full(x.size,.752)]
    assert flat_source(pts,.74) is not None
    assert flat_source(pts+[0,0,.08],.74) is None
    assert sloped_frame(pts,.74) is None
    vertical=np.c_[x.ravel(),np.zeros(x.size),y.ravel()+.88]
    assert sloped_frame(vertical,.74) is None


def test_surface_memory_requires_current_spread_and_invalidates_on_interaction():
    pts,_,_=inclined_points();cloud=pts.reshape(30,40,3);valid=np.ones((30,40),bool)
    ref=dict(target='support',bbox=[0,0,1000,1000],observation_id=1)
    memory=SurfaceFrameMemory()
    assert memory.remember(ref,dict(cloud=cloud,valid=valid),.74,1)
    assert not memory.remember(ref,dict(cloud=cloud,valid=valid),.74,2)
    observed=valid.copy();observed[:,10:30]=False
    result=memory.resolve(ref,cloud,observed,.74)
    assert result['support_frame_memory']['current_matched_points']==600
    assert result['support_frame_memory']['reference_observation_id']==1
    assert memory.resolve(ref,cloud+[.08,0,0],valid,.74) is None
    fragment=np.zeros_like(valid);fragment[:4,:4]=True
    assert memory.resolve(ref,cloud,fragment,.74) is None
    memory.forget('support');assert memory.resolve(ref,cloud,valid,.74) is None
