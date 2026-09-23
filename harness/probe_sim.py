import os,sys,time,json,faulthandler
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root/'RoboTwin');sys.path.insert(0,str(root/'RoboTwin'/'scripts'))
faulthandler.dump_traceback_later(120,repeat=True)
from eval_policy_xpolicylab import load_task_args,class_decorator
name=sys.argv[1] if len(sys.argv)>1 else 'stack_blocks_two'
args,_=load_task_args(dict(task_name=name,policy_name='probe',task_config='demo_clean'))
args.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
for seed in range(100000,100010):
 t=time.monotonic();env=class_decorator(name)
 print('SETUP',name,seed,flush=True)
 try:
  env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
  print('SETUP_OK',time.monotonic()-t,flush=True)
  obs=env.get_obs()
  print('OBS', {k:str(type(v)) for k,v in obs.items()},flush=True)
  print('EE',obs.get('endpose'),flush=True)
  from PIL import Image
  Image.fromarray(obs['observation']['head_camera']['rgb']).save(root/'results/20260916'/f'{name}.png')
  print('EXPERT_START',flush=True);env.play_once()
  print('EXPERT_DONE',env.plan_success,env.check_success(),time.monotonic()-t,flush=True)
  env.close_env();break
 except Exception as e:
  print('ERROR',type(e).__name__,str(e),time.monotonic()-t,flush=True)
  try: env.close_env()
  except: pass
