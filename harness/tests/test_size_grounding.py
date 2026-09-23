import numpy as np
from size_grounding import refine_block_size


def scene():
    cloud=np.zeros((100,160,3))
    cloud[:,:,2]=.74
    for i,edge in enumerate((.035,.05,.065)):
        xx,yy=np.meshgrid(np.linspace(0,edge,15),np.linspace(0,edge,15))
        cloud[20:35,10+i*45:25+i*45]=np.stack([xx-.2+i*.12,yy,np.full_like(xx,.74+edge)],-1)
    return cloud,np.ones((100,160),bool)


def test_metric_size_not_model_point_selects_requested_cube():
    cloud,valid=scene()
    points=[]
    for word in ('small','medium','large'):
        action=dict(skill='pick',target=word+' block',bbox=[0,0,100,100],point=[5,5])
        refined=refine_block_size(action,cloud,valid,.74)
        assert refined['model_point']==[5,5]
        points.append(refined['point'][0])
    assert points[0]<points[1]<points[2]


def test_unrelated_target_and_ambiguous_geometry_are_not_overridden():
    cloud,valid=scene()
    action=dict(skill='pick',target='small bottle',bbox=[0,0,100,100],point=[5,5])
    assert refine_block_size(action,cloud,valid,.74)==action
    cloud[:,:,2]=np.minimum(cloud[:,:,2],.775)
    action['target']='small block'
    assert refine_block_size(action,cloud,valid,.74)==action
