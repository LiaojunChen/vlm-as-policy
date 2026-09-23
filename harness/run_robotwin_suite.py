"""Resumable full-task suite; every episode is a separate simulation process."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-robotwin310/bin/python'
MODELS=['qwen3-vl-plus','qwen3-vl-flash-2026-01-22','qwen-vl-plus','codex']
PROJECTS=['show_harness','robodawn']

def read(path):
 try:return json.loads(Path(path).read_text(encoding="utf-8"))
 except (FileNotFoundError,json.JSONDecodeError):return {}

def free_gpu_memory_mb():
 try:
  value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  return int(value.splitlines()[0])
 except Exception:return 0

def live_worker_counts(out):
 total=0;models={}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=(proc/'cmdline').read_bytes().decode().split('\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(ROOT/'run_robotwin_episode.py') not in args or '--output' not in args:continue
  if not args[args.index('--output')+1].startswith(str(out)+'/'):continue
  if '--model' not in args:continue
  model=args[args.index('--model')+1]
  if model not in MODELS:continue
  total+=1;models[model]=models.get(model,0)+1
 return total,models

def worker_is_alive(directory):
 try:
  pid=int((directory/'pid').read_text())
  args=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode()
  return 'run_robotwin_episode.py' in args and str(directory) in args
 except (FileNotFoundError,ValueError,ProcessLookupError):return False

def run_job(task,project,model,out,seed=100000,max_steps=100,timeout=7200):
 result=out/'result.json'
 existing=read(result)
 if existing.get('status') in ('completed','expert_validated','expert_unvalidated'):return existing
 # Adopt a still-running worker after a supervisor restart; never reset its trial.
 try:
  owned_pid=int((out/'pid').read_text())
  command=Path(f'/proc/{owned_pid}/cmdline').read_bytes().decode()
 except (FileNotFoundError,ValueError):owned_pid=None;command=''
 if owned_pid and str(out) in command and 'run_robotwin_episode.py' in command:
  deadline=time.monotonic()+timeout
  while time.monotonic()<deadline:
   current=read(result)
   if current.get('status') not in ('starting','running'):return current
   try: alive=Path(f'/proc/{owned_pid}/cmdline').read_bytes()
   except FileNotFoundError:alive=b''
   if not alive:break
   time.sleep(2)
  # A timed-out adopted process must stop before its directory can be archived.
  try:
   alive=Path(f'/proc/{owned_pid}/cmdline').read_bytes().decode()
   if str(out) in alive and 'run_robotwin_episode.py' in alive:
    os.killpg(owned_pid,signal.SIGTERM)
    time.sleep(2)
    try:os.killpg(owned_pid,signal.SIGKILL)
    except ProcessLookupError:pass
  except (FileNotFoundError,ProcessLookupError):pass
  existing=read(result)
 if project=='expert':
  # Include archived attempts from earlier restarts as well as the latest run.
  # A genuine failed expert scene should never be recomputed.
  attempt_files=[out/'seed_attempts.json',*(out.parent/'_attempts'/task).glob('*/seed_attempts.json')]
  for attempt_file in attempt_files:
   try:attempts=json.loads(attempt_file.read_text(encoding='utf-8'))
   except (FileNotFoundError,json.JSONDecodeError):continue
   if isinstance(attempts,list):
    rejected=[x['seed'] for x in attempts if x.get('valid') is False]
    if rejected:seed=max(seed,max(rejected)+1)
 if out.exists() and existing:
  archive=out.parent/'_attempts'/task/time.strftime('%Y%m%dT%H%M%S')
  archive.parent.mkdir(parents=True,exist_ok=True)
  shutil.move(str(out),str(archive))
 out.mkdir(parents=True,exist_ok=True)
 if project=='expert':
  while free_gpu_memory_mb()<14000:time.sleep(5)
 cmd=[str(PYTHON),str(ROOT/'run_robotwin_episode.py'),'--task',task,'--project',project,
      '--model',model,'--seed',str(seed),'--max-steps',str(max_steps),'--output',str(out)]
 env=os.environ.copy()
 env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
 with (out/'process.log').open('w', encoding="utf-8") as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  (out/'pid').write_text(str(process.pid), encoding="utf-8")
  try:code=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   code=124
  finally:
   # Curobo child planners must not outlive the episode worker.
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 current=read(result)
 if code or current.get('status') in ('starting','running') or not current:
  current.update(task=task,project=project,model=model,seed=seed,status='process_error',returncode=code,success=None)
  result.write_text(json.dumps(current,indent=2), encoding="utf-8")
 print('FINISHED',task,project,model,current.get('status'),current.get('success'),flush=True)
 return current

def summarize(out,tasks):
 rows=[]
 for project in PROJECTS:
  for model in MODELS:
   for task in tasks:
    d=read(out/project/model/task/'result.json')
    rows.append({'task':task,'project':project,'model':model,**d} if d else {'task':task,'project':project,'model':model,'status':'pending','success':None})
 data={'protocol':'50 tasks x 2 projects x 4 models x 1 episode; 100 policy decisions per episode; official success predicates; expert-validated seeds unless explicitly marked expert_unvalidated',
       'requested_episodes':len(rows),'completed_episodes':sum(r['status']=='completed' for r in rows),
       'successful_episodes':sum(r['status']=='completed' and r['success'] is True for r in rows),'episodes':rows}
 (out/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2), encoding="utf-8")
 return data

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/full_50_v1'))
 p.add_argument('--phase',choices=['seeds','episodes','all','summary'],default='all')
 p.add_argument('--workers',type=int,default=6);p.add_argument('--tasks',nargs='*')
 p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
 p.add_argument('--max-steps',type=int,default=100)
 p.add_argument('--wait-seeds',action='store_true')
 a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 selected_models=[model for model in a.models if model!='codex' or not (out/'CODEX_PAUSED.json').exists()]
 import yaml
 tasks=a.tasks or list(yaml.safe_load((ROOT/'RoboTwin/env_cfg/task_config/_eval_step_limit.yml').read_text(encoding="utf-8")))
 (out/'manifest.json').write_text(json.dumps({'tasks':tasks,'models':MODELS,'projects':PROJECTS,
   'episodes_per_task':1,'max_policy_decisions':a.max_steps,'robotwin_commit':'6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755',
   'show_harness_commit':'137d5718c3b7af0150764d8f9beeb252c9f2794a','task_config':'demo_clean'},indent=2), encoding="utf-8")
 if a.phase in ('seeds','all'):
  with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
   jobs=[ex.submit(run_job,t,'expert','none',out/'seeds'/t,timeout=1800) for t in tasks]
   for f in concurrent.futures.as_completed(jobs):f.result()
 if a.phase in ('episodes','all'):
  with concurrent.futures.ThreadPoolExecutor(max(a.workers,32)) as ex:
   jobs={}; submitted=set()
   while True:
    limit=int(read(out/'concurrency.json').get('workers',a.workers))
    model_limits=read(out/'model_limits.json')
    live_total,live_models=live_worker_counts(out)
    new_launches=0
    free_mb=free_gpu_memory_mb()
    waiting=False
    for task in tasks:
     seed=read(out/'seeds'/task/'result.json')
     if seed.get('status') not in ('expert_validated','expert_unvalidated'):
      if not seed or seed.get('status')=='starting':waiting=True
      continue
     for project in PROJECTS:
      for model in selected_models:
       key=(task,project,model)
       if key in submitted:continue
       current=read(out/project/model/task/'result.json')
       if current.get('status')=='completed':submitted.add(key);continue
       already_running=current.get('status') in ('starting','running') and worker_is_alive(out/project/model/task)
       if not already_running and (len(jobs)>=limit or live_total>=limit or
           live_models.get(model,0)>=int(model_limits.get(model,limit)) or
           new_launches>=1 or free_mb<10000):continue
       submitted.add(key)
       f=ex.submit(run_job,task,project,model,out/project/model/task,seed['seed'],a.max_steps)
       jobs[f]=key
       if not already_running:
        new_launches+=1;live_total+=1;live_models[model]=live_models.get(model,0)+1
    if jobs:
     done,_=concurrent.futures.wait(jobs,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
     for f in done:f.result();del jobs[f]
     if done:summarize(out,tasks)
    elif len(submitted)>=len(tasks)*len(PROJECTS)*len(selected_models):break
    elif not (a.wait_seeds and waiting) and not any(
      read(out/'seeds'/t/'result.json').get('status') in ('expert_validated','expert_unvalidated') and
      any((t,p,m) not in submitted for p in PROJECTS for m in selected_models) for t in tasks):break
    else:time.sleep(5)
 d=summarize(out,tasks)
 print('SUMMARY',d['completed_episodes'],d['requested_episodes'],flush=True)
if __name__=='__main__':main()
