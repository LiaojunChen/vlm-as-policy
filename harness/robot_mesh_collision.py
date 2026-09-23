"""Observed-point collision checks against the robot's own collision shapes.

Only calibrated robot geometry and current RGB-D are inspected. No scene
actors, object meshes, identities or hidden collision geometry are queried.
"""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from observed_collision import ObservedCollision


class RobotPart:
    def __init__(self,shape):
        self.pose=shape.get_local_pose().to_transformation_matrix()
        kind=type(shape).__name__
        self.kind=kind
        if kind.endswith('ConvexMesh'):
            vertices=np.asarray(shape.get_vertices(),float)*np.asarray(shape.get_scale(),float)
            self.planes=ConvexHull(vertices).equations
            self.lo,self.hi=vertices.min(0),vertices.max(0)
        elif kind.endswith('Box'):
            half=np.asarray(shape.get_half_size(),float)
            vertices=np.asarray(list(product((-1,1),repeat=3)))*half
            self.planes=ConvexHull(vertices).equations
            self.lo,self.hi=-half,half
        elif kind.endswith(('Sphere','Capsule','Cylinder')):
            self.radius=float(shape.get_radius())
            self.half_length=float(shape.get_half_length()) if not kind.endswith('Sphere') else 0.
            axial_extent=self.half_length if kind.endswith('Cylinder') else self.radius+self.half_length
            self.hi=np.array([axial_extent,self.radius,self.radius]);self.lo=-self.hi
        else:raise ValueError('Unsupported robot collision shape: '+kind)
        self.centre=(self.lo+self.hi)*.5
        self.bound_radius=float(np.linalg.norm(self.hi-self.lo)*.5)

    def penetration(self,points):
        """Positive interior distance in this collision shape's local frame."""
        points=np.asarray(points,float)
        points=points[(points>=self.lo).all(1)&(points<=self.hi).all(1)]
        if not len(points):return 0.
        if hasattr(self,'planes'):
            values=points@self.planes[:,:3].T+self.planes[:,3]
            return float(max(0.,np.max(-np.max(values,axis=1))))
        if self.kind.endswith('Cylinder'):
            interior=np.minimum(self.half_length-np.abs(points[:,0]),
                                self.radius-np.linalg.norm(points[:,1:],axis=1))
            return float(max(0.,np.max(interior)))
        nearest=np.zeros_like(points);nearest[:,0]=np.clip(points[:,0],-self.half_length,self.half_length)
        return float(max(0.,np.max(self.radius-np.linalg.norm(points-nearest,axis=1))))


class ObservedRobotCollision(ObservedCollision):
    tolerance_m=.003

    def __init__(self,robot,side,cloud,valid,table,opening=1.):
        super().__init__(robot,side,cloud,valid,table,opening)
        prefix='fl_' if side=='left' else 'fr_'
        self.mesh_links=[]
        for link in self.entity.get_links():
            if not (link.name.startswith(prefix) or link.name==side+'_camera'):continue
            parts=[RobotPart(shape) for shape in link.get_collision_shapes()]
            if parts:self.mesh_links.append((link,parts))
        if not self.mesh_links:raise ValueError('Robot collision shape inventory is empty')

    def penetration(self,positions):
        if self.tree is None:return 0.
        original=self.entity.get_qpos().copy();worst=0.
        try:
            for index in np.unique(np.linspace(0,len(positions)-1,min(24,len(positions))).astype(int)):
                qpos=original.copy();qpos[self.indices]=positions[index];qpos[self.fingers]=self.finger_opening_m
                self.entity.set_qpos(qpos)
                for link,parts in self.mesh_links:
                    pose=link.get_pose().to_transformation_matrix()
                    for part in parts:
                        world=pose@part.pose
                        centre=world[:3,:3]@part.centre+world[:3,3]
                        indices=self.tree.query_ball_point(centre,part.bound_radius)
                        if not indices:continue
                        local=(self.tree.data[indices]-world[:3,3])@world[:3,:3]
                        worst=max(worst,part.penetration(local))
        finally:self.entity.set_qpos(original)
        return worst
