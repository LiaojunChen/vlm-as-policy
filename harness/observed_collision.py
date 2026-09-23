"""Screen robot-only FK trajectories against the current self-masked depth cloud.

No environment actors, object poses, or collision identities are inspected.
FK queries temporarily set only robot joints, never step physics, and restore
the measured configuration in a finally block.
"""
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree


class ObservedCollision:
    def __init__(self,robot,side,cloud,valid,table,opening=1.):
        if not np.isfinite(opening) or not 0<=opening<=1:raise ValueError('Invalid gripper opening')
        self.finger_opening_m=.045*float(opening)
        self.entity=getattr(robot,side+'_entity')
        names=[joint.name for joint in self.entity.get_active_joints()]
        self.indices=[names.index(j.name) for j in getattr(robot,side+'_arm_joints')]
        self.fingers=[names.index(j[0].name) for j in getattr(robot,side+'_gripper')]
        planner=getattr(robot,side+'_planner')
        config=yaml.safe_load(Path(planner.yml_path).read_text())
        spheres=config['robot_cfg']['kinematics']['collision_spheres']
        sphere_data=yaml.safe_load(Path(spheres).read_text())['collision_spheres']
        prefix='fl_' if side=='left' else 'fr_'
        self.links=[]
        for link in self.entity.get_links():
            if not (link.name.startswith(prefix) or link.name==side+'_camera'):continue
            items=sphere_data.get(link.name,[])
            if items:self.links.append((link,np.array([s['center'] for s in items]),np.array([s['radius'] for s in items])))
        mask=valid&(cloud[:,:,2]>table+.012)&(cloud[:,:,2]<table+.4)
        mask &= (np.abs(cloud[:,:,0])<.5)&(cloud[:,:,1]>-.4)&(cloud[:,:,1]<.35)
        points=cloud[mask]
        if len(points)>12000:points=points[::int(np.ceil(len(points)/12000))]
        self.tree=cKDTree(points) if len(points) else None

    def penetration(self,positions):
        if self.tree is None:return 0.
        original=self.entity.get_qpos().copy();worst=0.
        try:
            for index in np.unique(np.linspace(0,len(positions)-1,min(24,len(positions))).astype(int)):
                qpos=original.copy();qpos[self.indices]=positions[index]
                qpos[self.fingers]=getattr(self,'finger_opening_m',.045)
                self.entity.set_qpos(qpos)
                for link,centres,radii in self.links:
                    pose=link.get_pose().to_transformation_matrix()
                    world=centres@pose[:3,:3].T+pose[:3,3]
                    distance=self.tree.query(world)[0]
                    worst=max(worst,float(np.max(radii-distance)))
        finally:self.entity.set_qpos(original)
        return worst
