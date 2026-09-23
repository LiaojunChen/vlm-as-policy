from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from observed_collision import ObservedCollision


class Entity:
    def __init__(self):self.qpos=np.array([0.,.02,.02])
    def get_qpos(self):return self.qpos.copy()
    def set_qpos(self,value):self.qpos=value.copy()


def checker(entity,links):
    c=ObservedCollision.__new__(ObservedCollision)
    c.entity=entity;c.indices=[0];c.fingers=[1,2];c.links=links
    c.tree=cKDTree([[.1,0,0]])
    return c


def test_fk_screen_detects_swept_collision_and_restores_measured_joints():
    entity=Entity();initial=entity.get_qpos()
    def transform():
        pose=np.eye(4);pose[0,3]=entity.qpos[0];return pose
    link=SimpleNamespace(get_pose=lambda:SimpleNamespace(to_transformation_matrix=transform))
    c=checker(entity,[(link,np.zeros((1,3)),np.array([.02]))])
    assert c.penetration(np.array([[0.],[.1],[.2]]))==pytest.approx(.02)
    np.testing.assert_array_equal(initial,entity.get_qpos())


def test_fk_exception_also_restores_state():
    entity=Entity();initial=entity.get_qpos()
    def fail():raise RuntimeError('test FK error')
    link=SimpleNamespace(get_pose=fail)
    c=checker(entity,[(link,np.zeros((1,3)),np.array([.02]))])
    with pytest.raises(RuntimeError):c.penetration(np.array([[.1]]))
    np.testing.assert_array_equal(initial,entity.get_qpos())


def test_fk_uses_commanded_preopening_instead_of_assuming_fully_open_fingers():
    entity=Entity();initial=entity.get_qpos();seen=[]
    def transform():
        seen.append(entity.qpos[1:].copy());return np.eye(4)
    link=SimpleNamespace(get_pose=lambda:SimpleNamespace(to_transformation_matrix=transform))
    c=checker(entity,[(link,np.zeros((1,3)),np.array([.02]))]);c.finger_opening_m=.023
    c.penetration(np.array([[0.],[.2]]))
    assert all(np.allclose(values,[.023,.023]) for values in seen)
    np.testing.assert_array_equal(entity.get_qpos(),initial)
