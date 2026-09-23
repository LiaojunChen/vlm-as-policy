"""Bounded, resumable v3 cohort. Never overwrites a completed attempt."""
import argparse,concurrent.futures,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from source_layout import evaluation_sources
ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'robodawn_robotwin/.venv-robotwin310/bin/python'
def read(p):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
def gpu_free(gpu=0):
 try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[gpu])
 except Exception:return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+');p.add_argument('--projects',nargs='+',default=['robodawn','show_harness']);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-steps',type=int,default=20);p.add_argument('--timeout',type=int,default=1200)
 a=p.parse_args();out=a.output.resolve()
 if not out.is_relative_to(ROOT):raise ValueError('Outputs must stay inside v3 copy')
 out.mkdir(parents=True,exist_ok=True);tasks=a.tasks or sorted(p.name for p in (ROOT/'seeds').iterdir());jobs=[(t,p) for t in tasks for p in a.projects]
 manifest={'tasks':tasks,'projects':a.projects,'model':'qwen3-vl-plus','max_steps':a.max_steps,'timeout_s':a.timeout,'seed_source':'seeds','cohort':'development_seed_comparison','episodes':len(jobs),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 if (out/'manifest.json').exists():
  old=read(out/'manifest.json')
  for key in ('tasks','projects','model','max_steps'):assert old[key]==manifest[key],f'Cannot change frozen {key}'
 else:dump(out/'manifest.json',manifest)
 paths=evaluation_sources(ROOT)
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 if (out/'code_hashes.json').exists():assert read(out/'code_hashes.json')==hashes,'Source changed during frozen cohort'
 else:dump(out/'code_hashes.json',hashes)
 def job(task,project):
  directory=out/project/task;directory.mkdir(parents=True,exist_ok=True)
  if (directory/'result.json').exists():return read(directory/'result.json')
  cmd=[str(PYTHON),'-u',str(ROOT/'run_harness_v3.py'),'--task',task,'--project',project,'--max-steps',str(a.max_steps),'--output',str(directory)]
  with (directory/'process.log').open('w',encoding='utf-8') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   (directory/'pid').write_text(str(proc.pid))
   try:proc.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
    r=read(directory/'result.json');r.update(task=task,project=project,status='completed',success=False,end_reason='wall_time_budget',timeout_s=a.timeout);dump(directory/'result.json',r)
   r=read(directory/'result.json')
   if r.get('status') in (None,'starting','running'):
    r.update(task=task,project=project,status='error',success=None,error='Worker exited before final result',returncode=proc.returncode);dump(directory/'result.json',r)
   print('EPISODE',task,project,r.get('status'),r.get('success'),flush=True);return r
 def summary():
  rows=[{'task':t,'project':p,'status':'pending',**read(out/p/t/'result.json')} for t,p in jobs]
  data={'episodes':rows,'requested':len(jobs),'completed':sum(r['status']=='completed' for r in rows),'successes':sum(r.get('success') is True for r in rows),'errors':sum(r['status']=='error' for r in rows)}
  dump(out/'summary.json',data);return data
 pending=[j for j in jobs if not (out/j[1]/j[0]/'result.json').exists()]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  running={}
  while pending or running:
   while pending and len(running)<a.workers and gpu_free()>12000:
    task,project=pending.pop(0);running[pool.submit(job,task,project)]=(task,project);time.sleep(2)
   done,_=concurrent.futures.wait(running,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:future.result();del running[future]
   if not running and pending:time.sleep(10)
   d=summary()
   if done:print('PROGRESS',d['completed'],d['requested'],d['successes'],d['errors'],flush=True)
 summary()
if __name__=='__main__':main()
