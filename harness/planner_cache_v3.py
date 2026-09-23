"""Reuse scene-independent robot planners in a sequential worker; reset their RNG per episode."""
import os
_CACHE={}
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED or os.environ.get('ROBOTWIN_REUSE_PLANNERS')!='1':return
 import envs.robot.robot as robot_module
 original=robot_module.CuroboPlanner
 def cached(origin,active,all_joints,yml_path=None):
  key=(yml_path,tuple(active),tuple(all_joints),tuple(origin.p),tuple(origin.q))
  if key not in _CACHE:_CACHE[key]=original(origin,active,all_joints,yml_path=yml_path)
  planner=_CACHE[key];planner.fast_preflight=False;planner.motion_gen.reset(reset_seed=True)
  return planner
 robot_module.CuroboPlanner=cached;_INSTALLED=True
