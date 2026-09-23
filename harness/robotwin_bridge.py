"""RoboTwin EE action grounding. No object state or expert action reaches a policy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image

SIDES=('left','right')
class PolicyOutputError(ValueError):
    """A real model answered, but its proposed action violates the policy contract."""
    pass
VECTORS={'MV_LEFT':(-1,0,0),'MV_RIGHT':(1,0,0),'MV_FWD':(0,-1,0),
         'MV_BACK':(0,1,0),'MV_UP':(0,0,1),'MV_DOWN':(0,0,-1)}
def make_frame_context(step_m=.04):
    if not np.isfinite(step_m) or not 0 < step_m <= .1:
        raise ValueError('step_m must be finite and in (0, 0.1] metres')
    return f'''RoboTwin simulation: image order is HEAD, LEFT WRIST, RIGHT WRIST.
Robot world coordinates in metres: +X is right in the head image, +Y is away from the camera (toward image top), +Z is up.
Actions use a FIXED WORLD frame, regardless of wrist orientation: MV_LEFT=-X, MV_RIGHT=+X, MV_FWD=-Y, MV_BACK=+Y, MV_UP=+Z, MV_DOWN=-Z.
A move is {step_m:g} m ({step_m * 1000:g} mm). Gripper 0=closed and 1=open. GRASP closes, RELEASE opens. STILL holds.
There are no object coordinates available. Ground each decision in the current RGB views and measured robot pose.
The table surface is at world z=0.74 m. End-effector pose is the RoboTwin EE reference, not the fingertip.
'''

FRAME_CONTEXT = make_frame_context()

class RobotWinBridge:
    def __init__(self, env, instruction, directory, step_m=.04, defer_render=False, robot_geometry=False):
        self.env,self.instruction,self.directory = env,instruction,Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step_m=step_m
        self.frame_context=make_frame_context(step_m)
        self.robot_geometry=robot_geometry
        self.defer_render=defer_render
        self.index=0
        self.last_obs=None

    def observe(self):
        obs=self.last_obs if self.last_obs is not None else self.env.get_obs()
        self.last_obs=obs
        paths=[]
        for camera in ('head_camera','left_camera','right_camera'):
            path=self.directory/f'{self.index:04d}_{camera}.png'
            Image.fromarray(obs['observation'][camera]['rgb']).save(path)
            paths.append(str(path))
        result={'instruction':self.instruction,'image_paths':paths,'endpose':obs['endpose'],
                'step':self.index,'success':bool(self.env.eval_success or self.env.check_success())}
        if self.robot_geometry:
            # Only robot kinematics and camera calibration; never object state.
            cameras=self.env.cameras.get_config()
            geometry={}
            for side in SIDES:
                tcp=getattr(self.env.robot, f'get_{side}_tcp_pose')()
                projections={}
                for camera, config in cameras.items():
                    if camera not in ('head_camera','left_camera','right_camera'): continue
                    cam=np.asarray(config['extrinsic_cv']) @ np.r_[tcp[:3],1.]
                    pixel=np.asarray(config['intrinsic_cv']) @ cam[:3]
                    h,w=obs['observation'][camera]['rgb'].shape[:2]
                    uv=np.round(pixel[:2]/pixel[2],1).tolist() if cam[2]>0 else None
                    projections[camera]={'pixel_xy':uv,'image_wh':[w,h],
                        'in_frame':bool(uv is not None and 0<=uv[0]<w and 0<=uv[1]<h)}
                geometry[side]={'tcp_xyz_m':np.round(tcp[:3],4).tolist(),'tcp_projections':projections}
            result['robot_geometry']=geometry
        return result

    def execute_tokens(self,tokens):
        pose=self.last_obs['endpose']
        target=[]
        for side in SIDES:
            ee=np.asarray(pose[side+'_endpose'], dtype=float).copy()
            token=tokens[side]
            gripper=float(pose[side+'_gripper'])
            if token in VECTORS: ee[:3]+=np.asarray(VECTORS[token])*self.step_m
            elif token=='GRASP': gripper=0.
            elif token=='RELEASE': gripper=1.
            elif token not in ('STILL','DONE'): raise PolicyOutputError('Invalid atomic token: '+str(token))
            if not np.isfinite(ee).all(): raise ValueError('Nonfinite end pose')
            target.extend(ee.tolist()+[gripper])
        import time
        started=time.monotonic()
        plan_status={}
        original_planners={side:getattr(self.env.robot,f'{side}_plan_path') for side in SIDES}
        def audited_plan(side):
            def call(*args,**kwargs):
                plan=original_planners[side](*args,**kwargs)
                plan_status[side]={'status':str(plan.get('status')),
                    'trajectory_points':len(plan.get('position',[]))}
                return plan
            return call
        original_update=self.env._update_render if self.defer_render else None
        if self.defer_render:
            if self.env.render_freq or self.env.crazy_random_light or self.env.eval_video_path:
                raise ValueError('Deferred render is only supported for headless clean scenes without per-frame video')
            self.env._update_render=lambda: None
        try:
            for side in SIDES: setattr(self.env.robot,f'{side}_plan_path',audited_plan(side))
            self.env.take_action(target,action_type='ee')
        finally:
            for side in SIDES: setattr(self.env.robot,f'{side}_plan_path',original_planners[side])
            if original_update is not None:self.env._update_render=original_update
        action_s=time.monotonic()-started
        self.index+=1
        after=self.env.get_obs();self.last_obs=after
        measured={}
        for i,side in enumerate(SIDES):
            before=np.asarray(pose[side+'_endpose'][:3])
            actual=np.asarray(after['endpose'][side+'_endpose'][:3])
            desired=np.asarray(target[8*i:8*i+3])
            measured[side]={'requested_delta_m':(desired-before).tolist(),
                'actual_delta_m':(actual-before).tolist(),
                'position_error_m':float(np.linalg.norm(actual-desired))}
        return {'ok':set(plan_status)==set(SIDES) and all(v['status']=='Success' for v in plan_status.values()),
                'target_ee':target,'endpose_after':after['endpose'],'action_elapsed_s':action_s,
                'motion':measured,'planner':plan_status,'step_m':self.step_m,
                'success':bool(self.env.eval_success or self.env.check_success())}

    def execute(self,action):
        # Robodawn primitives are grounded without reading object poses.
        side=action.get('arm','left')
        if side not in SIDES: raise ValueError('arm must be left or right')
        tokens={s:'STILL' for s in SIDES}
        if action['name']=='pick': tokens[side]='GRASP'
        elif action['name']=='place': tokens[side]='RELEASE'
        elif action['name'] in ('push','pull'):
            direction=action.get('direction')
            if direction not in VECTORS: raise ValueError('push/pull requires a valid direction')
            tokens[side]=direction
        elif action['name'] not in ('noop','done'): raise ValueError('fold is unsupported for rigid-body RoboTwin tasks')
        return self.execute_tokens(tokens)



def __getattr__(name):
    """Resolve legacy policy imports without coupling shared code to a policy."""
    from importlib import import_module
    exports = {'RobodawnPlanner': 'robodawn.planner_v1'}
    if name in exports:
        return getattr(import_module(exports[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
