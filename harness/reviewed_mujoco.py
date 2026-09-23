"""Sensor-based skill completion and strict action validation in the copied scene."""
from __future__ import annotations
import numpy as np
from PIL import Image
from simple_mujoco_grasp import SimpleGrasp
from reviewed_interface import Capabilities, ContractError, finite_vector, validate_action

INSTRUCTIONS={
    'lift':'Use only the RIGHT gripper to grasp the red cube and lift it clear of the table.',
    'pick_place':'Use only the RIGHT gripper to pick up the red cube, place it in the centre of the green pad, and release it on the table.'}


class ReviewedGrasp(SimpleGrasp):
    def __init__(self,seed,directory,task='lift'):
        if task not in INSTRUCTIONS:raise ValueError(task)
        self.task=task;self.caps=Capabilities();self.obs_counter=0
        self.depth_observation_id=None;self.closed=False
        try:super().__init__(seed,directory)
        except Exception:
            self.close()
            raise

    def gripper_sensor(self):
        """Tactile attachment evidence; does not expose target identity or pose."""
        bodies={g:set() for g in self.finger_geoms}
        for contact in self.data.contact[:self.data.ncon]:
            for finger,other in [(int(contact.geom1),int(contact.geom2)),
                                 (int(contact.geom2),int(contact.geom1))]:
                if finger not in bodies or other in self.finger_geoms:continue
                body=int(self.model.geom_bodyid[other])
                if body and self.model.body_dofnum[body]==6:
                    bodies[finger].add(body)
        shared=set.intersection(*bodies.values())
        aperture=sum(float(self.data.joint(n).qpos[0]) for n in ('finger_a_slide','finger_b_slide'))
        return {'holding':bool(shared),'aperture_m':aperture,
                'two_finger_contact':bool(shared)}

    def _score(self):
        # Target truth is private to scoring; the policy receives only sensors.
        if self.task=='lift':return super()._score()
        position=self.data.body('cube').xpos.copy()
        self.max_lift_m=max(self.max_lift_m,float(position[2]-self.initial_cube_z))
        velocity=self.data.joint('cube_joint').qvel[:3]
        settled=(np.linalg.norm(position[:2]-[.14,.095])<=.025 and
                 abs(position[2]-.720)<.008 and np.linalg.norm(velocity)<.03 and
                 not self.gripper_sensor()['holding'] and self.max_lift_m>=.06)
        self.hold_s=self.hold_s+float(self.model.opt.timestep) if settled else 0.
        if self.hold_s>=.4:self.success=True

    def observe(self):
        obs_id=self.obs_counter;self.obs_counter+=1
        paths=[]
        for name in ('head','side','wrist'):
            path=self.directory/'frames'/f'obs_{obs_id:05d}_{name}.png'
            Image.fromarray(self.rgb(name)).save(path);paths.append(str(path))
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data,camera='head')
            self.depth=self.renderer.render().copy()
        finally:self.renderer.disable_depth_rendering()
        self.depth_observation_id=obs_id
        sensor=self.gripper_sensor()
        return {'instruction':INSTRUCTIONS[self.task],'image_paths':paths,
                'camera_roles':['head_top_down','side_fixed','right_wrist'],
                'observation_id':obs_id,'simulation_time_s':float(self.data.time),
                'capabilities':self.caps.to_dict(),'gripper_sensor':sensor,
                'endpose':{'right_endpose':self.data.body('right').xpos.tolist()+[1,0,0,0],
                           'right_gripper':sensor['aperture_m']/.08,
                           'left_endpose':[-.3,-.3,1.,1,0,0,0],'left_gripper':1.},
                'step':self.index,'success':self.success}

    def pixel_world(self,pixel):
        validate_action({'name':'reach_at','arm':'right','pixel':pixel,
                         'observation_id':self.depth_observation_id},self.caps,self.depth_observation_id)
        u,v=pixel;depth=float(self.depth[v,u])
        if not np.isfinite(depth) or not .02<depth<2.:
            raise ContractError('invalid_depth','No valid depth for selected pixel')
        point=super().pixel_world(pixel)
        if (not np.isfinite(point).all() or not -.25<=point[0]<=.25 or
            not -.20<=point[1]<=.20 or not .695<=point[2]<=1.10):
            raise ContractError('out_of_workspace','Selected surface is outside the calibrated workspace')
        return point

    def _go(self,steps,label,xyz=None,opening=None,duration=.65):
        if xyz is not None:
            values=finite_vector(list(xyz),3,'target')
            if np.any(values<self.caps.xyz_min) or np.any(values>self.caps.xyz_max):
                raise ContractError('out_of_workspace',f'{label} target outside workspace')
        row=self.move(xyz,opening,duration,label)
        steps.append(row)
        if not row['motion_ok']:
            raise ContractError('motion_failed',f'{label}: {row["position_error_m"]:.4f} m tracking error')
        return row

    def execute(self,action):
        steps=[];self.index+=1;point=None;failure=None;code=None
        try:
            validate_action(action,self.caps,self.depth_observation_id)
            name=action['name'];current=self.data.body('right').xpos.copy()
            if name in ('reach_at','pick_at','place_at'):point=self.pixel_world(action['pixel'])
            if name=='pick_at':
                if self.gripper_sensor()['holding']:
                    raise ContractError('already_holding','Place or release held object before another pick')
                self._go(steps,'open',opening=1)
                self._go(steps,'hover',[*point[:2],max(.85,point[2]+.11)])
                self._go(steps,'descend',[*point[:2],max(.724,point[2]-.015)])
                self._go(steps,'close',opening=0,duration=.8)
                if not self.gripper_sensor()['holding']:
                    raise ContractError('empty_grasp','No object detected between both fingers')
                self._go(steps,'lift',[*point[:2],.90],duration=1.2)
                self._go(steps,'hold',duration=.6)
                if not self.gripper_sensor()['holding']:
                    raise ContractError('grasp_lost','Object lost during lift')
            elif name=='place_at':
                if not self.gripper_sensor()['holding']:
                    raise ContractError('not_holding','PLACE requires a verified held object')
                self._go(steps,'transfer',[*point[:2],max(.85,point[2]+.15)])
                if not self.gripper_sensor()['holding']:
                    raise ContractError('grasp_lost','Object lost during transfer')
                self._go(steps,'lower',[*point[:2],max(.724,point[2]+.024)])
                self._go(steps,'release',opening=1,duration=.8)
                self._go(steps,'retreat',[*point[:2],.90])
                self._go(steps,'settle',duration=.6)
                if self.gripper_sensor()['holding']:
                    raise ContractError('release_failed','Contact persists after release')
            elif name=='reach_at':self._go(steps,'reach',[*point[:2],max(.85,point[2]+.11)])
            elif name=='move_pose':self._go(steps,'move_pose',action['xyz'])
            elif name=='move_delta':self._go(steps,'move_delta',current+np.asarray(action['delta']))
            elif name=='lift':self._go(steps,'lift',current+[0,0,action.get('distance',.08)])
            elif name=='open_gripper':self._go(steps,'open',opening=1)
            elif name=='close_gripper':
                self._go(steps,'close',opening=0,duration=.8)
                if not self.gripper_sensor()['holding']:
                    raise ContractError('empty_grasp','Closed gripper has no bilateral object contact')
            elif name=='wait':self._go(steps,'wait',duration=action.get('duration',.4))
            elif name=='done':pass
        except ContractError as exc:
            failure=str(exc);code=exc.code
        finally:
            if steps:self.depth_observation_id=None
        return {'ok':failure is None,'skill_success':failure is None,
                'motion_ok':bool(steps) and all(s['motion_ok'] for s in steps),
                'success':self.success,'failure':failure,'failure_code':code,
                'subactions':steps,'gripper_sensor':self.gripper_sensor(),
                'world_point_from_depth':point.tolist() if point is not None else None,
                'terminal':action.get('name')=='done'}

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if hasattr(self,'video'):
                self._video_frame()
                self.video.stdin.close()
                code=self.video.wait(timeout=30)
                if code:raise RuntimeError(f'ffmpeg exited {code}')
        finally:
            if hasattr(self,'renderer'):self.renderer.close()
