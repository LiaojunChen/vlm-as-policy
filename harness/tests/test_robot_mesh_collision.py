from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from robot_mesh_collision import RobotPart,ObservedRobotCollision


class PhysxCollisionShapeBox:
    def get_local_pose(self):return SimpleNamespace(to_transformation_matrix=lambda:np.eye(4))
    def get_half_size(self):return [.08,.02,.01]


class Entity:
    def __init__(self):self.q=np.array([0.,.01,.01])
    def get_qpos(self):return self.q.copy()
    def set_qpos(self,q):self.q=q.copy()


class PhysxCollisionShapeCylinder(PhysxCollisionShapeBox):
    def get_radius(self):return .02
    def get_half_length(self):return .08


def test_cylinder_checks_radial_wall_and_axial_endcaps():
    part=RobotPart(PhysxCollisionShapeCylinder())
    assert part.penetration([[.07,0,0]])==pytest.approx(.01)
    assert part.penetration([[0,.015,0]])==pytest.approx(.005)
    assert part.penetration([[.081,0,0],[0,.021,0]])==0
    # Inside the AABB but outside the curved wall.
    assert part.penetration([[0,.019,.019]])==0


def test_convex_shape_detects_thin_finger_regions_outside_coarse_centre_sphere():
    part=RobotPart(PhysxCollisionShapeBox())
    assert part.penetration([[.07,0,0]])==pytest.approx(.01)
    assert part.penetration([[.09,0,0]])==0
    assert part.penetration(np.empty((0,3)))==0


def test_shape_screen_checks_swept_positions_and_restores_joints():
    entity=Entity();before=entity.get_qpos();part=RobotPart(PhysxCollisionShapeBox())
    def pose():
        result=np.eye(4);result[0,3]=entity.q[0];return result
    link=SimpleNamespace(get_pose=lambda:SimpleNamespace(to_transformation_matrix=pose))
    check=ObservedRobotCollision.__new__(ObservedRobotCollision)
    check.entity=entity;check.indices=[0];check.fingers=[1,2];check.finger_opening_m=.025
    check.tree=cKDTree([[.17,0,0]]);check.mesh_links=[(link,[part])]
    assert check.penetration(np.array([[0.],[.1],[.2]]))==pytest.approx(.01)
    np.testing.assert_array_equal(entity.get_qpos(),before)


def test_shape_query_failure_restores_measured_robot_state():
    entity=Entity();before=entity.get_qpos()
    def fail():raise RuntimeError('shape failure')
    check=ObservedRobotCollision.__new__(ObservedRobotCollision)
    check.entity=entity;check.indices=[0];check.fingers=[1,2];check.finger_opening_m=.025
    check.tree=cKDTree([[0,0,0]]);check.mesh_links=[(SimpleNamespace(get_pose=fail),[])]
    with pytest.raises(RuntimeError):check.penetration(np.array([[.2]]))
    np.testing.assert_array_equal(entity.get_qpos(),before)
