"""Small contact-only MuJoCo grasp diagnostic with the existing VLM project loops."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
INSTRUCTION = 'Use only the RIGHT gripper to grasp the red cube and lift it clear of the table. The left arm is unavailable.'


class SimpleGrasp:
    def __init__(self, seed, directory):
        self.directory = Path(directory)
        (self.directory / 'frames').mkdir(parents=True, exist_ok=True)
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'simple_mujoco/grasp.xml'))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.index = self.physics_steps = 0
        self.success = False
        self.hold_s = self.max_lift_m = 0.
        self.cube_geom = self.model.geom('cube_geom').id
        self.finger_geoms = {self.model.geom(n).id for n in ('finger_a_geom', 'finger_b_geom')}
        rng = np.random.default_rng(seed)
        # Ground truth is used only for reset, scoring and labelled control tests.
        cube_q = self.model.joint('cube_joint').qposadr[0]
        self.data.qpos[cube_q:cube_q+3] = [rng.uniform(-.10, .10), rng.uniform(-.07, .07), .721]
        self.data.qpos[cube_q+3:cube_q+7] = [1, 0, 0, 0]
        for name, value in zip(('right_x','right_y','right_z','finger_a_slide','finger_b_slide'),
                                (.20, -.15, .96, .04, .04)):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.data.ctrl[:] = [.20, -.15, .96, .04, .04]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(150):
            mujoco.mj_step(self.model, self.data)
        self.initial_cube_z = float(self.data.body('cube').xpos[2])
        self.video = subprocess.Popen([
            'ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
            '-video_size','320x240','-framerate','20','-i','-','-vcodec','libx264',
            '-pix_fmt','yuv420p','-crf','20',str(self.directory / 'continuous.mp4')],
            stdin=subprocess.PIPE)
        self._video_frame()

    def rgb(self, camera):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _video_frame(self):
        self.video.stdin.write(self.rgb('side').tobytes())

    def _score(self):
        contacts = set()
        for c in self.data.contact[:self.data.ncon]:
            if c.geom1 == self.cube_geom and c.geom2 in self.finger_geoms:
                contacts.add(int(c.geom2))
            elif c.geom2 == self.cube_geom and c.geom1 in self.finger_geoms:
                contacts.add(int(c.geom1))
        lift = float(self.data.body('cube').xpos[2] - self.initial_cube_z)
        self.max_lift_m = max(self.max_lift_m, lift)
        if lift >= .06 and len(contacts) == 2:
            self.hold_s += float(self.model.opt.timestep)
        else:
            self.hold_s = 0.
        if self.hold_s >= .4:
            self.success = True

    def move(self, xyz=None, opening=None, duration=.65, label='move'):
        before = self.data.body('right').xpos.copy()
        old = self.data.ctrl.copy()
        target = old.copy()
        if xyz is not None:
            target[:3] = np.clip(xyz, [-.25,-.20,.724], [.25,.20,1.08])
        if opening is not None:
            target[3:] = np.clip(opening, 0, 1) * .04
        steps = max(1, int(duration / self.model.opt.timestep))
        for i in range(steps):
            # Position actuators drive dynamic bodies; no object teleport/weld.
            alpha = min(1., (i + 1) / max(1, steps * .70))
            self.data.ctrl[:] = old + alpha * (target - old)
            mujoco.mj_step(self.model, self.data)
            self._score()
            self.physics_steps += 1
            if self.physics_steps % 25 == 0:
                self._video_frame()
        after = self.data.body('right').xpos.copy()
        return {'name': label, 'target_xyz': target[:3].tolist(),
                'actual_xyz': after.tolist(), 'actual_translation_m': float(np.linalg.norm(after-before)),
                'position_error_m': float(np.linalg.norm(after-target[:3])),
                'motion_ok': bool(np.linalg.norm(after-target[:3]) < .012),
                'success': self.success}

    def observe(self):
        paths = []
        for name in ('head','side','wrist'):
            path = self.directory / 'frames' / f'{self.index:04d}_{name}.png'
            Image.fromarray(self.rgb(name)).save(path)
            paths.append(str(path))
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera='head')
        self.depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        opening = float(np.mean([self.data.joint(n).qpos[0] for n in
                                 ('finger_a_slide','finger_b_slide')]) / .04)
        return {'instruction': INSTRUCTION, 'image_paths': paths,
                'endpose': {'right_endpose': self.data.body('right').xpos.tolist()+[1,0,0,0],
                            'right_gripper': opening,
                            'left_endpose': [-.3,-.3,1.,1,0,0,0], 'left_gripper': 1.},
                'step': self.index, 'success': self.success}

    def pixel_world(self, pixel):
        u,v = [int(x) for x in pixel]
        depth = float(self.depth[v,u])
        camera = self.model.camera('head').id
        focal = 120. / np.tan(np.deg2rad(self.model.cam_fovy[camera]) / 2.)
        local = np.array([(u-159.5)*depth/focal, -(v-119.5)*depth/focal, -depth])
        return self.data.cam_xpos[camera] + self.data.cam_xmat[camera].reshape(3,3) @ local

    def execute(self, action):
        self.index += 1
        if action.get('arm') != 'right':
            return {'ok': False, 'success': False, 'failure': 'Only right gripper exists', 'subactions': []}
        point = self.pixel_world(action['pixel'])
        name = action['name']
        steps = []
        if name == 'pick_at':
            steps.append(self.move(opening=1, label='open'))
            steps.append(self.move([point[0],point[1],max(.85,point[2]+.11)], label='hover'))
            steps.append(self.move([point[0],point[1],max(.724,point[2]-.015)], label='descend'))
            steps.append(self.move(opening=0, duration=.8, label='close'))
            steps.append(self.move([point[0],point[1],.90], duration=1.2, label='lift'))
            steps.append(self.move(duration=.6, label='hold'))
        elif name == 'reach_at':
            steps.append(self.move([point[0],point[1],max(.85,point[2]+.11)], label='reach'))
        elif name == 'place_at':
            steps.append(self.move([point[0],point[1],.76], label='place'))
            steps.append(self.move(opening=1, label='release'))
        elif name == 'press_at':
            steps.append(self.move([point[0],point[1],max(.724,point[2])], opening=0,label='press'))
        else:
            raise ValueError(f'Unknown skill {name}')
        ok = all(s['motion_ok'] for s in steps)
        return {'ok': ok, 'success': self.success, 'subactions': steps,
                'world_point_from_depth': point.tolist(),
                'failure': None if ok else 'Position actuator tracking error'}

    def close(self):
        self._video_frame()
        self.video.stdin.close()
        self.video.wait(timeout=30)
        self.renderer.close()


def run(project, seed, out, max_steps, ground_with_object=False):
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = SimpleGrasp(seed, out)
    client = None
    result = {'project': project, 'seed': seed, 'task': 'simple_contact_grasp',
              'model': 'qwen3-vl-plus' if project != 'physics_control' else None,
              'status': 'running', 'success': None, 'simulator': mujoco.__version__,
              'ground_with_object': ground_with_object,
              'video': str(out/'continuous.mp4')}
    def record(row):
        with (out/'trace.jsonl').open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    try:
        if project == 'physics_control':
            # Labelled oracle: validates physics only, excluded from VLM scores.
            point = env.data.body('cube').xpos.copy()
            for xyz, opening, label in [(None,1,'open'),([*point[:2],.86],1,'hover'),
                                       ([*point[:2],.724],1,'descend'),(None,0,'close'),
                                       ([*point[:2],.90],0,'lift'),(None,0,'hold')]:
                record({'action':label,'result':env.move(xyz,opening,1.,label)})
        else:
            from benchmark_clients import AuditedClient
            from robotwin_grounding_v2 import GridGrounder
            from robodawn.planner_v2 import RobodawnGroundedPlanner, SKILLS
            from showharness.policy_v2 import ShowGroundedPolicy
            from robodawn.core import run_loop
            client = AuditedClient('qwen3-vl-plus',out/'calls')
            grounder = GridGrounder(client,out/'grounding')
            if project == 'robodawn':
                run_loop(env,env,RobodawnGroundedPlanner(client,grounder,INSTRUCTION),
                         max_steps=max_steps,stop_on_error=False,
                         allowed_actions=(*SKILLS,'done'),on_record=record)
            else:
                policy = ShowGroundedPolicy(client,INSTRUCTION,out,grounder,
                                           ground_with_object=ground_with_object)
                for step in range(max_steps):
                    obs = env.observe()
                    if obs['success']:break
                    actions,decision = policy.decide(obs)
                    if not actions:
                        record({'step':step,'observation':obs,'action':{'name':'wait'},
                                'result':{'success':env.success},'model_decision':decision})
                    for action in actions:
                        outcome = env.execute(action)
                        policy.feedback(action,outcome)
                        record({'step':step,'observation':obs,'action':action,
                                'result':outcome,'model_decision':decision})
                        if outcome['success']:break
                    if decision['native_done'] or env.success:break
        env.observe()
        result.update(status='completed',success=env.success,
                      max_lift_m=env.max_lift_m,final_hold_s=env.hold_s,
                      model_calls=client.calls if client else 0,skills_executed=env.index)
    except Exception as e:
        result.update(status='error',error=str(e),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        env.close()
        result['elapsed_s']=time.monotonic()-started
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',choices=['physics_control','robodawn','show_harness'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--max-steps',type=int,default=5)
    p.add_argument('--ground-with-object',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results/simple_mujoco_v1')
    a=p.parse_args()
    for seed in a.seeds:
        target=a.output.resolve()/a.project/f'seed_{seed}'
        if (target/'result.json').exists():
            raise RuntimeError(f'Refusing to overwrite {target}')
        run(a.project,seed,target,a.max_steps,a.ground_with_object)
