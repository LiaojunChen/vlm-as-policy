"""Add two frozen-policy workers once older cohorts release the GPU; no extra episodes."""
import json,os,subprocess,time
from pathlib import Path
from run_suite_v3 import gpu_free
ROOT=Path(__file__).resolve().parent
EVAL=ROOT.parent/'robodawn_robotwin_harness_v3p_eval'
OUT=EVAL/'results/full_50'
PYTHON=ROOT.parent/'robodawn_robotwin/.venv-robotwin310/bin/python'
def older_gpu_jobs():
 count=0
 for entry in Path('/proc').glob('[0-9]*'):
  try:
   args=[p.decode(errors='replace') for p in (entry/'cmdline').read_bytes().split(b'\0') if p]
   cwd=Path(os.readlink(entry/'cwd'))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if not args or 'python' not in args[0]:continue
  if cwd.is_relative_to(ROOT.parent/'robodawn_robotwin') and any('episode' in p or 'replay' in p for p in args if p.endswith('.py')):count+=1
  if cwd.is_relative_to(ROOT.parent/'robodawn_robotwin_harness_v3t_eval') and any(p.endswith('cohort_worker_v3.py') for p in args):count+=1
 return count
def main():
 while not (OUT/'FINALIZED.json').exists():
  data=json.loads((OUT/'summary.json').read_text())
  if data['completed']+data['errors']>=data['requested']-4:return
  if older_gpu_jobs()==0 and gpu_free()>20000:break
  time.sleep(30)
 else:return
 workers=[]
 for index in range(2):
  log=(OUT/f'additional_worker_{index}.log').open('w')
  process=subprocess.Popen([str(PYTHON),'-u',str(EVAL/'cohort_worker_v3.py'),'--cohort',str(OUT)],cwd=EVAL,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  workers.append((process,log));time.sleep(2)
 (OUT/'THROUGHPUT_ADJUSTMENT.json').write_text(json.dumps(dict(added_workers=2,reason='Prior baseline and smoke cohorts finished; same frozen source, task list, seeds and per-episode budget',pids=[p.pid for p,l in workers]),indent=2))
 for process,log in workers:process.wait();log.close()
if __name__=='__main__':main()
