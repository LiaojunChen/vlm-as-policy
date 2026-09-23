"""Render-only integration check: actual robot self mask, no motion planner/API calls."""
import json,os,sys,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
from eval_policy_xpolicylab import load_task_args,class_decorator
from envs.robot.robot import Robot
from robotwin_harness_v3 import Bridge
from envs.robot.planner import CuroboPlanner
from types import SimpleNamespace
def skip_arm_planner(self,scene=None):
 self.communication_flag=False
 self.left_planner=self.right_planner=SimpleNamespace(plan_grippers=lambda a,b:CuroboPlanner.plan_grippers(None,a,b))
Robot.set_planner=skip_arm_planner
parser=argparse.ArgumentParser();parser.add_argument('--wide',action='store_true');parser.add_argument('--hold',action='store_true');args=parser.parse_args()
task='place_empty_cup' if args.wide else 'click_bell'
config,_=load_task_args(dict(task_name=task,policy_name='probe',task_config='demo_clean'))
config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
config['camera']['head_camera_type']='Wide_D435' if args.wide else 'Large_D435'
for camera in config['left_embodiment_config']['static_camera_list']:
 if camera['name']=='head_camera':camera['type']=config['camera']['head_camera_type']
out=ROOT/('results/robot_self_mask_probe_wide' if args.wide else 'results/robot_self_mask_probe');out.mkdir(parents=True,exist_ok=True)
env=class_decorator(task);env.setup_demo(now_ep_num=0,seed=100000,is_test=True,**config)
bridge=Bridge(env,'Inspect sensors only',out,video=False)
try:
 obs=bridge.observe();data=np.load(out/'frames/0001_rgbd.npz');mask=data['robot_self_mask']
 direct=bridge.endpose()
 for key in obs['endpose']:np.testing.assert_allclose(direct[key],obs['endpose'][key],atol=1e-8)
 assert mask.shape==((720,960) if args.wide else (480,640))
 assert 100<int(mask.sum())<mask.size*.5
 assert not np.any(data['valid']&mask)
 assert int(data['valid'].sum())>10000
 result=dict(passed=True,resolution=list(mask.shape[::-1]),robot_pixels=int(mask.sum()),valid_scene_pixels=int(data['valid'].sum()),model_calls=0,robot_motion=False,proprioception_matches_official_observation=True)
 if args.hold:
  env._update_render=lambda:None  # Physics-only actuator test after the render check.
  steps=[];before=bridge.endpose();before_joints=env.robot.left_entity.get_qpos().copy()
  bridge._take('left',before['left_endpose'],0,'close',steps)
  closed=bridge.sensors()['left']['finger_qpos_m']
  bridge._take('left',bridge.endpose()['left_endpose'],1,'open',steps)
  opened=bridge.sensors()['left']['finger_qpos_m']
  assert all(set(step['stationary_holds'])=={'left','right'} for step in steps)
  assert max(closed)<.003 and min(opened)>.04,(closed,opened)
  for side in ('left','right'):
   assert np.linalg.norm(np.asarray(bridge.endpose()[side+'_endpose'][:3])-before[side+'_endpose'][:3])<.005
  result.update(stationary_hold_passed=True,gripper_motion=True,closed_finger_qpos=closed,opened_finger_qpos=opened,official_actions=env.take_action_cnt)
  (out/'hold_trace.json').write_text(json.dumps(steps,indent=2))
 (out/'validation.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:bridge.close();env.close_env()
