"""Finalize the frozen all-task cohort and preserve infrastructure retry provenance."""
import csv,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
EVAL=ROOT.parent/'robodawn_robotwin_harness_v3p_eval'
OUT=EVAL/'results/full_50'
BASE=ROOT.parent/'robodawn_robotwin/results/full_50_v2_qwen3vlplus'
PYTHON=ROOT.parent/'robodawn_robotwin/.venv-robotwin310/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def normalize(r):
 # Native Show plan validation raises ValueError; this is a policy failure, not retryable infrastructure.
 error=r.get('error','')
 if r.get('status')=='error' and (('Subgoal ' in error and 'missing required' in error) or 'Dual planner returned two empty tracks' in error or 'Dual planner did not return JSON object' in error):
  r={**r,'source_status':'error','status':'completed','success':False,'end_reason':'invalid_model_plan','classification_note':'Native model-plan validation failure; original result retained; not retried'}
 if r.get('status')=='error' and r.get('end_reason')=='worker_interrupted' and r.get('observed_wall_s',0)>=r.get('wall_budget_s',float('inf')):
  r={**r,'source_status':'error','status':'completed','success':False,'end_reason':'wall_time_budget','classification_note':'Hard watchdog reached declared episode budget; original result retained; not retried'}
 return r
def collect(job):
 path=OUT/job['project']/job['task']/'result.json'
 r={**job,'status':'pending',**read(path)}
 if r.get('end_reason')=='worker_interrupted':
  claim=read(OUT/'claims'/f"{job['project']}__{job['task']}.json")
  if claim.get('started') and path.exists():r.update(observed_wall_s=path.stat().st_mtime-claim['started'],wall_budget_s=read(OUT/'manifest.json')['timeout_s'])
 return normalize(r)
def report(rows,final=False):
 baseline={}
 for path in BASE.glob('**/result.json'):
  r=read(path)
  if r.get('project') in ('robodawn','show_harness'):baseline[(r['project'],r['task'])]=r
 matched=[(r,baseline.get((r['project'],r['task']),{})) for r in rows]
 valid=[(r,b) for r,b in matched if r.get('status')=='completed' and b.get('status')=='completed']
 lines=['# RoboTwin v3p 与 v2 对照','',f"状态：{'已完成' if final else '运行中'}；更新时间 {time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())}",'',
  '固定 50 任务 × 2 项目 × 每任务 1 episode，仅 Qwen3-VL-Plus。相同开发种子、官方成功判定；不是留出种子泛化结果。',
  f"v3p：{sum(r.get('success') is True for r in rows)} 项成功，{sum(r.get('status')=='completed' for r in rows)} 项完成，{sum(r.get('status')=='error' for r in rows)} 项基础设施错误，共 100 项。",
  f"双方均完成的 {len(valid)} 个同种子 episode：v2 {sum(b.get('success') is True for r,b in valid)} 成功；v3p {sum(r.get('success') is True for r,b in valid)} 成功。",'',
  '仅对基础设施错误重跑；策略执行失败、提前结束、预算耗尽不做择优重试。初次结果和重试证据均保留。','',
  '| Task | Project | v2 | v3p | Status | Evidence |','|---|---|---|---|---|---|']
 for r,b in matched:
  evidence=r.get('evidence',f"{r['project']}/{r['task']}")
  lines.append(f"| {r['task']} | {r['project']} | {b.get('success')} | {r.get('success')} | {r.get('status')} | [Result]({evidence}/result.json) · [Video]({evidence}/continuous.mp4) |")
 (OUT/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
 (OUT/'delivery_summary.json').write_text(json.dumps(dict(finalized=final,episodes=rows),ensure_ascii=False,indent=2))
 with (OUT/'delivery_results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['task','project','model','seed','status','success','end_reason','policy_decisions','model_calls','elapsed_s','evidence','retry_of'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
def main():
 jobs=read(OUT/'manifest.json')['jobs']
 while True:
  rows=[collect(j) for j in jobs]
  report(rows)
  if all(r['status'] in ('completed','error') for r in rows):break
  time.sleep(30)
 for index,r in enumerate(rows):
  if r['status']!='error':continue
  directory=OUT/'infrastructure_retries'/r['project']/r['task'];directory.mkdir(parents=True,exist_ok=True)
  if not (directory/'result.json').exists():
   with (directory/'process.log').open('w') as f:
    proc=subprocess.Popen([str(PYTHON),'-u',str(EVAL/'run_harness_v3.py'),'--task',r['task'],'--project',r['project'],'--max-steps','30','--output',str(directory)],cwd=EVAL,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:proc.wait(timeout=1200)
    except subprocess.TimeoutExpired:
     import signal
     os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
     failed=read(directory/'result.json');failed.update(task=r['task'],project=r['project'],status='error',success=None,end_reason='infrastructure_retry_timeout');(directory/'result.json').write_text(json.dumps(failed,indent=2))
  retried=normalize(read(directory/'result.json'))
  if retried.get('status') in ('completed','error'):
   retried.update(retry_of=f"{r['project']}/{r['task']}",evidence=str(directory.relative_to(OUT)));rows[index]=retried
  report(rows)
 report(rows,True)
 (OUT/'FINALIZED.json').write_text(json.dumps(dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),successes=sum(r.get('success') is True for r in rows),episodes=len(rows)),indent=2))
 print('HARNESS_DELIVERY_FINALIZED',OUT,flush=True)
if __name__=='__main__':main()
