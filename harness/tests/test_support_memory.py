import numpy as np
from support_memory import SupportMemory


def scene(offset=0):
    angle=np.linspace(0,2*np.pi,240,endpoint=False)
    radius=np.linspace(.03,.065,40)
    a,r=np.meshgrid(angle,radius)
    xyz=np.stack([r*np.cos(a)+offset,r*np.sin(a),.745+(r-.03)],axis=-1)
    return xyz,np.ones(a.shape,bool)


def test_static_support_memory_requires_current_matched_surface():
    cloud,valid=scene();a=dict(target='bowl',bbox=[0,0,1000,1000],relation='inside')
    g=dict(source='visible_rgbd_circular_cavity',tcp=[0,0,.745],rim_radius_m=.065)
    m=SupportMemory();m.remember(a,g,cloud,valid,.74,1)
    partial=valid.copy();partial[:,:75]=False
    verified=m.verify(a,cloud,partial)
    assert verified and verified['tcp']==[0,0,.745]
    assert verified['support_memory']['reference_observation_id']==1
    assert m.verify(a,cloud,np.zeros_like(valid)) is None
    moved,_=scene(.025)
    assert m.verify(a,moved,valid) is None
    m.forget('BOWL');assert m.verify(a,cloud,valid) is None


def test_table_guesses_are_not_promoted_to_precise_support_memory():
    cloud,valid=scene();m=SupportMemory()
    m.remember(dict(target='thing',bbox=[0,0,1000,1000]),dict(tcp=[0,0,.74]),cloud,valid,.74,1)
    assert not m.entries
