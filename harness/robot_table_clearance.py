"""Robot-only forward-kinematics audit against the measured support plane.

No scene collision objects or target geometry are inspected. Pinocchio is
queried without changing live joint state or stepping simulation.
"""
from itertools import product
import numpy as np
from transforms3d.quaternions import quat2mat


class RobotTableClearance:
    def __init__(self,robot,side,table):
        self.entity=getattr(robot,side+'_entity');self.table=float(table)
        names=[j.name for j in self.entity.get_active_joints()]
        self.arm_indices=[names.index(j.name) for j in getattr(robot,side+'_arm_joints')]
        self.finger_indices=[names.index(j[0].name) for j in getattr(robot,side+'_gripper')]
        self.finger_links={j[0].child_link.name for j in getattr(robot,side+'_gripper')}
        self.model=self.entity.create_pinocchio_model()
        prefix='fl_' if side=='left' else 'fr_'
        self.parts=[]
        for index,link in enumerate(self.entity.get_links()):
            if not link.name.startswith(prefix):continue
            for shape in link.get_collision_shapes():
                kind=type(shape).__name__
                if kind.endswith('ConvexMesh'):
                    vertices=np.asarray(shape.get_vertices(),float)*np.asarray(shape.get_scale(),float)
                elif kind.endswith('Box'):
                    vertices=np.array(list(product((-1,1),repeat=3)))*np.asarray(shape.get_half_size())
                else:continue  # Report exact supported robot shapes only; never invent vertices.
                local=shape.get_local_pose().to_transformation_matrix()
                vertices=vertices@local[:3,:3].T+local[:3,3]
                self.parts.append((index,link.name,vertices))

    def evaluate(self,positions=None,opening=None):
        qpos=self.entity.get_qpos().copy()
        if positions is not None:qpos[self.arm_indices]=np.asarray(positions)
        if opening is not None:qpos[self.finger_indices]=.045*float(opening)
        self.model.compute_forward_kinematics(qpos)
        root=self.entity.get_root_pose().to_transformation_matrix();clearance={}
        for index,name,vertices in self.parts:
            transform=root@self.model.get_link_pose(index).to_transformation_matrix()
            world=vertices@transform[:3,:3].T+transform[:3,3]
            value=float(world[:,2].min()-self.table)
            clearance[name]=min(clearance.get(name,float('inf')),value)
        fingers={k:v for k,v in clearance.items() if k in self.finger_links}
        return dict(source='robot_pinocchio_collision_vertices_vs_observed_table',
                    link_min_clearance_m=clearance,finger_min_clearance_m=fingers,
                    commanded_opening=opening,live_joint_state_changed=False)

    def calibration_error(self):
        self.model.compute_forward_kinematics(self.entity.get_qpos())
        root=self.entity.get_root_pose().to_transformation_matrix();errors=[]
        for index,link in enumerate(self.entity.get_links()):
            predicted=root@self.model.get_link_pose(index).to_transformation_matrix()
            errors.append(np.linalg.norm(predicted[:3,3]-link.get_pose().p))
        return float(max(errors))

    def finger_vertices_in_ee(self,ee_pose,opening):
        """Own finger collision geometry at the commanded opening in EE frame."""
        qpos=self.entity.get_qpos().copy();qpos[self.finger_indices]=.045*float(opening)
        self.model.compute_forward_kinematics(qpos)
        root=self.entity.get_root_pose().to_transformation_matrix()
        rotation=quat2mat(ee_pose[3:]);origin=np.asarray(ee_pose[:3]);parts=[]
        for index,name,vertices in self.parts:
            if name not in self.finger_links:continue
            transform=root@self.model.get_link_pose(index).to_transformation_matrix()
            world=vertices@transform[:3,:3].T+transform[:3,3]
            parts.append((world-origin)@rotation)
        return np.concatenate(parts) if parts else None


def table_clear_grasp(tcp,quaternion,finger_vertices,table,margin=.002,max_raise=.04):
    """Raise only enough to keep the actual open fingers above observed table.

    XY, yaw/pitch and the named contact are unchanged. This compensates the
    difference between the kinematic TCP and the real finger tips; it is not
    deeper pushing or a relaxed contact/tracking-success threshold.
    """
    from robotwin_harness_v3 import ee_from_tcp
    point=np.asarray(tcp,float);vertices=np.asarray(finger_vertices,float)
    if point.shape!=(3,) or vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices):return None
    if not np.isfinite(point).all() or not np.isfinite(vertices).all():return None
    pose=ee_from_tcp(point,quaternion)
    world=vertices@quat2mat(quaternion).T+pose[:3]
    clearance=float(world[:,2].min()-table)
    rise=max(0.,margin-clearance)
    if rise>max_raise:return None
    adjusted=point.copy();adjusted[2]+=rise
    return adjusted,dict(source='current_robot_finger_geometry_table_clearance',
                         original_tcp=point.tolist(),raise_m=rise,
                         original_min_clearance_m=clearance,minimum_clearance_m=clearance+rise)
