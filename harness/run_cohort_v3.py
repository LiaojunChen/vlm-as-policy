"""Frozen, resumable model cohort with persistent planners and bounded episodes."""
import argparse,csv,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from source_layout import evaluation_sources
from run_suite_v3 import read,dump,gpu_free
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'robodawn_robotwin/.venv-robotwin310/bin/python'
TERMINAL=('completed','error')

def write_summary(out,manifest):
 rows=[{**j,'status':'pending',**read(out/j['project']/j['task']/'result.json')} for j in manifest['jobs']]
 data={'requested':len(rows),'completed':sum(r['status']=='completed' for r in rows),'successes':sum(r.get('success') is True for r in rows),'errors':sum(r['status']=='error' for r in rows),'episodes':rows,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 dump(out/'summary.json',data)
 with (out/'results.csv').open('w',newline='') as f:
  columns=['task','project','model','seed','status','success','policy_decisions','model_calls','end_reason','error','elapsed_s']
  writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 lines=[f"# {manifest['model_checkpoint']} + RoboDawn v3 RoboTwin evaluation",'',f"Updated: {data['updated_utc']}",'',
        'Protocol: demo_clean; official success checker; RGB-D + robot proprioception/contact with episode-local visual identity/action memory; no extra sensors, actor poses or expert actions in policy.',
        f"Budget: {manifest['max_steps']} policy decisions and {manifest['timeout_s']} seconds per episode.",
        f"Head camera: {manifest.get('head_resolution',320)} px wide; {'60-degree experimental wider view' if manifest.get('wide_head') else '37-degree original field of view'}.",
        'Development seeds: same-seed comparison, not a held-out generalization score. All attempts retained.',
        '', '| Project | Success | Completed | Errors | Requested |','|---|---:|---:|---:|---:|']
 for project in manifest['projects']:
  subset=[r for r in rows if r['project']==project]
  lines.append(f"| {project} | {sum(r.get('success') is True for r in subset)} | {sum(r['status']=='completed' for r in subset)} | {sum(r['status']=='error' for r in subset)} | {len(subset)} |")
 lines+=['','| Task | Project | Status | Success | Evidence |','|---|---|---|---|---|']
 for r in rows:
  path=f"{r['project']}/{r['task']}"
  lines.append(f"| {r['task']} | {r['project']} | {r['status']} | {r.get('success')} | [Result]({path}/result.json) · [Trace]({path}/trace.jsonl) · [Video]({path}/continuous.mp4) |")
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 return data

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+');p.add_argument('--projects',nargs='+',default=['robodawn','show_harness']);p.add_argument('--max-steps',type=int,default=30);p.add_argument('--workers',type=int,default=3);p.add_argument('--timeout',type=int,default=1200);p.add_argument('--head-resolution',type=int,choices=[320,640],default=320);p.add_argument('--wide-head',action='store_true');p.add_argument('--gpu-ids',type=int,nargs='+',default=[0]);p.add_argument('--base-port',type=int,default=18050);p.add_argument('--endpoint');p.add_argument('--endpoints',nargs='+');p.add_argument('--model',default='zdtaichu');p.add_argument('--model-checkpoint',default='phyRSI/ZDTaichu5.0-9B');a=p.parse_args();out=a.output.resolve();a.workers=min(a.workers,len(a.gpu_ids))
 if not out.is_relative_to(ROOT):raise ValueError('Cohort must be in the isolated copy')
 out.mkdir(parents=True,exist_ok=True);(out/'claims').mkdir(exist_ok=True)
 tasks=a.tasks or sorted(d.name for d in (ROOT/'seeds').iterdir())
 endpoints=a.endpoints or ([a.endpoint] if a.endpoint else [])
 manifest=dict(jobs=[dict(task=t,project=s) for t in tasks for s in a.projects],projects=a.projects,model=a.model,model_checkpoint=a.model_checkpoint,enable_thinking=False,dtype='bfloat16',temperature=0,max_steps=a.max_steps,timeout_s=a.timeout,head_resolution=960 if a.wide_head else a.head_resolution,wide_head=a.wide_head,worker_gpu_ids=a.gpu_ids,model_endpoints=endpoints or 'per_gpu_local_endpoint')
 manifest['json_schema']=os.environ.get('VLM_JSON_SCHEMA')=='1'
 manifest['policy_memory']='episode_local_visual_identity_and_execution_feedback_no_extra_sensors'
 manifest['policy_recovery']='bounded_shared_workspace_regrasp_and_current_verified_rgbd_support_memory'
 manifest['policy_perception']='same_frame_contextual_pair_verification_and_collective_scene_inventory'
 manifest['policy_adaptive_recovery']='measured_failure_constraints_simultaneous_dependency_scheduling_and_bounded_held_view_motion'
 manifest['policy_memory_search']='current_frame_release_guided_crop_with_full_frame_fallback_no_stale_execution_coordinates'
 manifest['policy_observed_execution']='same_frame_identity_candidate_reuse_stationary_search_explicit_free_table_and_protruding_handle_geometry'
 manifest['json_schema_mode']='repair_only' if manifest['json_schema'] else 'disabled'
 if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest,'Cannot change a frozen cohort'
 else:dump(out/'manifest.json',manifest)
 files=evaluation_sources(ROOT)
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 if (out/'code_hashes.json').exists():assert read(out/'code_hashes.json')==hashes,'Source changed after freeze'
 else:dump(out/'code_hashes.json',hashes)
 children={};launches=0
 while True:
  # Claims survive restart. Interrupted attempts stay in the denominator and retain their evidence.
  for claim in (out/'claims').glob('*.json'):
   info=read(claim);j=info.get('job',{});pid=info.get('pid',0)
   if not j:continue
   result_path=out/j['project']/j['task']/'result.json';r=read(result_path)
   if r.get('status') in TERMINAL:continue
   try:
    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    live='cohort_worker_v3.py' in command and str(out) in command
   except FileNotFoundError:live=False
   timed_out=live and time.time()-info['started']>a.timeout+45
   if timed_out:
    os.killpg(pid,signal.SIGTERM);live=False
   if not live:
    r.update(**j,status='error',success=None,end_reason='worker_interrupted',error='Worker exited or exceeded hard timeout; evidence retained')
    if timed_out:r.update(status='completed',success=False,end_reason='wall_time_budget')
    dump(result_path,r)
  for pid,(proc,log,gpu,slot) in list(children.items()):
   if proc.poll() is not None:log.close();del children[pid]
  data=write_summary(out,manifest)
  if data['completed']+data['errors']==data['requested']:break
  unclaimed=sum(not (out/'claims'/f"{j['project']}__{j['task']}.json").exists() and not (out/j['project']/j['task']/'result.json').exists() for j in manifest['jobs'])
  while unclaimed and len(children)<a.workers:
   used_slots={c[3] for c in children.values()}
   slot=next((i for i,g in enumerate(a.gpu_ids) if i not in used_slots and gpu_free(g)>8000),None)
   if slot is None:break
   gpu=a.gpu_ids[slot]
   launches+=1;log=(out/f'worker_{launches:03d}.log').open('a')
   endpoint=endpoints[slot%len(endpoints)] if endpoints else f'http://127.0.0.1:{a.base_port+gpu}/v1'
   worker_env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'VLM_MODEL':a.model,'VLM_ENDPOINT':endpoint}
   proc=subprocess.Popen([str(PYTHON),'-u',str(ROOT/'cohort_worker_v3.py'),'--cohort',str(out)],cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   children[proc.pid]=(proc,log,gpu,slot);unclaimed-=1;time.sleep(2)
  print('COHORT',data['completed'],data['successes'],data['errors'],data['requested'],flush=True)
  time.sleep(15)
 for proc,log,gpu,slot in children.values():
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:proc.terminate()
  log.close()
 print('COHORT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
