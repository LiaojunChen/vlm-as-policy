import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).parent; py=ROOT/'.venv-robotwin310/bin/python'
rows=[]
providers=[('mock',None),('codex',None)]+[('openai',m) for m in ('qwen3-vl-plus','qwen3-vl-flash-2026-01-22','qwen-vl-plus')]
for provider,model in providers:
 ok=0; steps=[]; errors=[]
 for i in range(5):
  t=time.time(); env=__import__('os').environ.copy();
  if model: env['VLM_MODEL']=model
  p=subprocess.run([str(py),'-m','robodawn_robotwin','--provider',provider],cwd=ROOT.parent,text=True,capture_output=True,timeout=120,env=env)
  try: d=json.loads(p.stdout); s=bool(d['success']); n=len(d['trace'])
  except Exception as e: s=False; n=None; errors.append((p.stderr or str(e))[-300:])
  ok+=s; steps += [n] if n is not None else []
 rows.append({'provider':provider,'model':model,'episodes':5,'successes':ok,'success_rate':ok/5,'mean_steps':sum(steps)/len(steps) if steps else None,'errors':errors,'elapsed_s':round(time.time()-t,2)})
Path(ROOT/'evaluation_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2))
