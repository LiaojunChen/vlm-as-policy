"""Run frozen v3w smoke and full cohort after older cohorts release capacity."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EVAL=ROOT.parent/'robodawn_robotwin_harness_v3w_eval'
PYTHON=ROOT.parent/'robodawn_robotwin/.venv-robotwin310/bin/python'
STATE=EVAL/'EVALUATION_QUEUE.json'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def status(stage,**more):
 payload=dict(stage=stage,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model='qwen3-vl-plus',protocol='enhanced_head_960x720_fov60',full_episodes=100,**more)
 temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(STATE);print(stage,flush=True)
def terminal(path):
 data=read(path)
 return data.get('requested',0)>0 and data.get('completed',0)+data.get('errors',0)==data['requested']
def verify():
 for name,digest in read(EVAL/'FROZEN_SOURCE_SHA256.json').items():
  if hashlib.sha256((EVAL/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+name)
def run(output,tasks,workers):
 verify()
 command=[str(PYTHON),'-u',str(EVAL/'run_cohort_v3.py'),'--output',str(EVAL/output),'--workers',str(workers),'--max-steps','30','--timeout','1200','--wide-head']
 if tasks:command+=['--tasks',*tasks]
 with (EVAL/(Path(output).name+'.supervisor.log')).open('a') as log:
  proc=subprocess.Popen(command,cwd=EVAL,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status('running_'+Path(output).name,pid=proc.pid,output=output)
  code=proc.wait()
 if code:raise RuntimeError(f'Cohort supervisor exited {code}')
def main():
 validation=read(ROOT/'results/robot_self_mask_probe/validation.json')
 if not validation.get('stationary_hold_passed'):raise RuntimeError('Real physics actuator validation must pass before launch')
 status('waiting_for_previous_regression_smoke')
 while not terminal(ROOT.parent/'robodawn_robotwin_harness_v3v_eval/results/regression_smoke/summary.json'):time.sleep(30)
 run('results/regression_smoke',['stack_blocks_two','click_bell'],2)
 smoke=read(EVAL/'results/regression_smoke/summary.json')
 if smoke.get('errors') or smoke.get('successes',0)<2:
  status('needs_review_before_full',smoke_summary={k:smoke.get(k) for k in ['requested','completed','successes','errors']});return
 status('waiting_for_standard_full_cohort')
 while not (ROOT.parent/'robodawn_robotwin_harness_v3p_eval/results/full_50/FINALIZED.json').exists():time.sleep(30)
 run('results/full_50',None,4)
 status('finished',summary={k:read(EVAL/'results/full_50/summary.json').get(k) for k in ['requested','completed','successes','errors']})
if __name__=='__main__':
 try:main()
 except Exception as exc:status('error',error=str(exc));raise
