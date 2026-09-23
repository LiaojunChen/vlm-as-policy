"""Run isolated v3 skills against untouched official RoboTwin scoring."""
import argparse,json,os,sys,time,traceback,faulthandler
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class EpisodeBudgetExceeded(Exception):pass

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--project',default='robodawn',choices=['robodawn','show_harness']);p.add_argument('--seed',type=int);p.add_argument('--seed-manifest',type=Path);p.add_argument('--max-steps',type=int,default=12);p.add_argument('--head-resolution',type=int,choices=[320,640],default=320);p.add_argument('--wide-head',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv);faulthandler.dump_traceback_later(90,repeat=True);out=a.output.resolve()
 if not out.is_relative_to(ROOT):raise ValueError('Outputs must belong to isolated v3 copy')
 out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise ValueError('Refusing to overwrite existing episode')
 seed=json.loads((a.seed_manifest or ROOT/'seeds'/a.task/'result.json').read_text());seed_id=a.seed if a.seed is not None else seed['seed']
 if seed_id!=seed['seed']:raise ValueError('A new seed requires its own instruction/validation manifest')
 model_name=os.environ.get('VLM_MODEL','qwen3-vl-plus')
 result=dict(task=a.task,project=a.project,model=model_name,seed=seed_id,status='starting',success=None,instruction=seed['instruction'],protocol='v3_rgbd_geometry_tactile',max_policy_decisions=a.max_steps,seed_validation=seed['status'] if seed_id==seed['seed'] else 'not_expert_validated',started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 result['policy_memory']='episode_local_visual_identity_and_execution_feedback_no_extra_sensors'
 result['policy_recovery']='bounded_shared_workspace_regrasp_and_current_verified_rgbd_support_memory'
 result['policy_perception']='same_frame_contextual_pair_verification_and_collective_scene_inventory'
 result['policy_adaptive_recovery']='measured_failure_constraints_simultaneous_dependency_scheduling_and_bounded_held_view_motion'
 result['policy_memory_search']='current_frame_release_guided_crop_with_full_frame_fallback_no_stale_execution_coordinates'
 result['policy_observed_execution']='same_frame_identity_candidate_reuse_stationary_search_explicit_free_table_and_protruding_handle_geometry'
 dump(out/'result.json',result);started=time.monotonic();env=bridge=client=None;rows=[]
 os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1';os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
 try:
  from eval_policy_xpolicylab import load_task_args,class_decorator
  from benchmark_clients import AuditedClient
  from planner_cache_v3 import install
  install()
  from robotwin_harness_v3 import Bridge,SkillError,SKILLS
  from robodawn.mission_planner import RobodawnPlanner
  from task_contracts_v3 import contract
  from robodawn.core import run_loop
  config,_=load_task_args(dict(task_name=a.task,policy_name=a.project,task_config='demo_clean'));config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
  if a.head_resolution==640 or a.wide_head:
   import copy
   profile='Wide_D435' if a.wide_head else 'Large_D435'
   config=copy.deepcopy(config);config['camera']['head_camera_type']=profile
   for camera in config['left_embodiment_config']['static_camera_list']:
    if camera['name']=='head_camera':camera['type']=profile
  result.update(head_resolution=960 if a.wide_head else a.head_resolution,head_fov_deg=60 if a.wide_head else 37,protocol='v3_task_contracts_rgbd_selfmask'+('_wide_head' if a.wide_head else ''))
  env=class_decorator(a.task);env.setup_demo(now_ep_num=0,seed=seed_id,is_test=True,**config)
  goal_contract=contract(a.task);policy_instruction=seed['instruction']+'\nCompletion requirements: '+goal_contract
  result['task_contract']=goal_contract
  bridge=Bridge(env,policy_instruction,out);client=AuditedClient(model_name,out/'calls')
  if a.project=='show_harness':
   from showharness.planner_v3 import ShowPlanner
   planner=ShowPlanner(client,policy_instruction,out)
  else:planner=RobodawnPlanner(client,policy_instruction)
  result.update(status='running',official_step_limit=env.step_lim);dump(out/'result.json',result)
  def record(row):
   rows.append(row)
   if hasattr(planner,'feedback'):planner.feedback(row['action'],row['result'])
   with (out/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   result.update(policy_decisions=len(rows),model_calls=client.calls,last_failure=row['result'].get('failure'),last_success=row['result'].get('success'))
   dump(out/'result.json',result);print('V3_ACTION',len(rows),row['action']['name'],row['result'].get('ok'),row['result'].get('success'),row['result'].get('failure'),flush=True)
  run_loop(bridge,bridge,planner,max_steps=a.max_steps,stop_on_error=False,on_record=record,allowed_actions=SKILLS)
  result.update(status='completed',success=bool(env.eval_success or env.check_success()),end_reason='environment_success' if env.eval_success or env.check_success() else 'policy_terminated' if rows and rows[-1]['action']['name']=='done' else 'decision_budget')
 except Exception as exc:
  result.update(status='completed' if type(exc).__name__ in ('SkillError','JSONDecodeError') else 'error',success=False if type(exc).__name__ in ('SkillError','JSONDecodeError') else None,error=str(exc),error_type=type(exc).__name__,end_reason='invalid_model_action' if type(exc).__name__ in ('SkillError','JSONDecodeError') else 'infrastructure_error');(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
  if isinstance(exc,(TimeoutError,EpisodeBudgetExceeded)):result.update(status='completed',success=False,end_reason='wall_time_budget')
 finally:
  if env:
   diagnostics={}
   for name,actor in vars(env).items():
    if type(actor).__name__=='Actor':
     try:
      pose=actor.get_pose();diagnostics[name]={'position':pose.p.tolist(),'quaternion':pose.q.tolist()}
     except Exception as diagnostic_error:diagnostics[name]={'error':str(diagnostic_error)}
   dump(out/'post_episode_evaluator_diagnostics.json',{'policy_input':False,'captured_after_policy_finished':True,'actors':diagnostics})
  if bridge:
   try:result['final_observation']=bridge.observe();bridge.close()
   except Exception as exc:result['cleanup_error']=str(exc)
  if env:
   try:env.close_env()
   except Exception as exc:result['env_cleanup_error']=str(exc)
  result.update(elapsed_s=time.monotonic()-started,model_calls=client.calls if client else 0,policy_decisions=len(rows));dump(out/'result.json',result)
  print('V3_RESULT',json.dumps({k:result.get(k) for k in ('task','project','seed','status','success','error','elapsed_s','policy_decisions')}),flush=True)
  faulthandler.cancel_dump_traceback_later()
if __name__=='__main__':main()
