"""Current reference points vs own hand shapes during relative release.

No actor geometry, hidden object pose or physics mutation. The robot-only FK
expresses cooked collision shapes in the current EE frame. This screens local
descent/release, not arbitrary full-arm transfer or unobserved space.
"""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from transforms3d.quaternions import quat2mat
from robotwin_harness_v3 import SkillError


class HandPart:
    def __init__(self,name,vertices):
        self.name=name;self.vertices=np.asarray(vertices,float)
        self.lo=self.vertices.min(0);self.hi=self.vertices.max(0)
        self.planes=ConvexHull(self.vertices).equations

    def penetration(self,points):
        inside=points[(points>=self.lo).all(1)&(points<=self.hi).all(1)]
        if not len(inside):return 0.
        return float(max(0.,np.max(-np.max(inside@self.planes[:,:3].T+self.planes[:,3],axis=1))))


class PlacementHandClearance:
    tolerance_m=.003

    def __init__(self,robot,side,ee_pose,reference_points):
        points=np.asarray(reference_points,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():
            raise SkillError('Relative release requires current observed reference points')
        self.points=points.copy();self.parts=[]
        entity=getattr(robot,side+'_entity');model=entity.create_pinocchio_model()
        original=entity.get_qpos().copy();links=entity.get_links()
        names=[j.name for j in entity.get_active_joints()]
        grippers=getattr(robot,side+'_gripper')
        fingers=[names.index(j[0].name) for j in grippers]
        finger_links={j[0].child_link.name for j in grippers}
        flange=getattr(robot,side+'_ee').child_link.name
        selected=finger_links|{flange,side+'_camera'}
        root=entity.get_root_pose().to_transformation_matrix()
        model.compute_forward_kinematics(original)
        error=max(np.linalg.norm((root@model.get_link_pose(i).to_transformation_matrix())[:3,3]-link.get_pose().p)
                  for i,link in enumerate(links))
        if not np.isfinite(error) or error>.002:raise SkillError('Robot FK cannot calibrate relative release hand geometry')
        self.calibration_error_m=float(error)
        rotation=quat2mat(ee_pose[3:]);origin=np.asarray(ee_pose[:3])
        for alpha in np.linspace(0,1,5):
            qpos=original.copy();qpos[fingers]=(1-alpha)*original[fingers]+alpha*.045
            model.compute_forward_kinematics(qpos)
            for index,link in enumerate(links):
                if link.name not in selected:continue
                # Palm/camera do not change with aperture.
                if alpha>0 and link.name not in finger_links:continue
                for shape in link.get_collision_shapes():
                    kind=type(shape).__name__
                    if kind.endswith('ConvexMesh'):
                        vertices=np.asarray(shape.get_vertices(),float)*np.asarray(shape.get_scale(),float)
                    elif kind.endswith('Box'):
                        vertices=np.array(list(product((-1,1),repeat=3)))*np.asarray(shape.get_half_size())
                    else:raise SkillError('Unsupported robot hand collision shape '+kind)
                    transform=root@model.get_link_pose(index).to_transformation_matrix()@shape.get_local_pose().to_transformation_matrix()
                    local=(vertices@transform[:3,:3].T+transform[:3,3]-origin)@rotation
                    self.parts.append(HandPart(link.name,local))
        if not self.parts:raise SkillError('No calibrated robot hand collision shapes')

    def screen(self,prepose,contactpose):
        if not np.allclose(quat2mat(prepose[3:]),quat2mat(contactpose[3:]),atol=1e-7):
            raise SkillError('Relative descent screen requires a constant hand orientation')
        rotation=quat2mat(contactpose[3:]);worst=0.;link=None
        for alpha in np.linspace(0,1,5):
            origin=(1-alpha)*np.asarray(prepose[:3])+alpha*np.asarray(contactpose[:3])
            local=(self.points-origin)@rotation
            for part in self.parts:
                depth=part.penetration(local)
                if depth>worst:worst=depth;link=part.name
        return dict(source='current_reference_rgbd_vs_robot_cooked_hand_shapes',
                    reference_points=len(self.points),max_penetration_m=worst,link=link,
                    accepted=worst<=self.tolerance_m,calibration_error_m=self.calibration_error_m,
                    aperture_samples=5,descent_samples=5,
                    scope='local_constant_orientation_descent_and_release_not_full_arm_path')
