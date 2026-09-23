"""Persistent CUDA worker, disjoint atomic job claims and per-episode logs."""
import argparse,contextlib,gc,json,os,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def live(pid):
 try:os.kill(pid,0);return True
 except (ProcessLookupError,ValueError):return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);a=p.parse_args();out=a.cohort.resolve()
 manifest=json.loads((out/'manifest.json').read_text());os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
 os.environ['VLM_JSON_SCHEMA']='1' if manifest.get('json_schema') else '0'
 from run_harness_v3 import main as episode,EpisodeBudgetExceeded
 for job in manifest['jobs']:
  directory=out/job['project']/job['task'];result=directory/'result.json';claim=out/'claims'/f"{job['project']}__{job['task']}.json"
  claim.parent.mkdir(exist_ok=True)
  if result.exists():continue
  try:
   with claim.open('x') as f:json.dump({'pid':os.getpid(),'started':time.time(),'job':job},f)
  except FileExistsError:continue
  directory.mkdir(parents=True,exist_ok=True)
  args=['--task',job['task'],'--project',job['project'],'--max-steps',str(manifest['max_steps']),'--output',str(directory)]
  if manifest.get('wide_head'):args+=['--wide-head']
  elif manifest.get('head_resolution'):args+=['--head-resolution',str(manifest['head_resolution'])]
  if job.get('seed_manifest'):args+=['--seed-manifest',job['seed_manifest']]
  def timeout(signum,frame):raise EpisodeBudgetExceeded('Episode wall-time budget exhausted')
  signal.signal(signal.SIGALRM,timeout);signal.alarm(int(manifest.get('timeout_s',1200)))
  with (directory/'process.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   episode(args)
  signal.alarm(0)
  print('WORKER_FINISHED',os.getpid(),job,flush=True);gc.collect()
if __name__=='__main__':main()
