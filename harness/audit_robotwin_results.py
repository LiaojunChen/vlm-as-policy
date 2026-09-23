"""Audit completed episodes against raw executed actions and model call artifacts."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/full_50_v1')
issues=[];checked=0;actions=0;calls=0
for p in root.glob('*/*/*/result.json'):
 d=json.loads(p.read_text(encoding='utf-8'))
 if d.get('status')!='completed':continue
 checked+=1;directory=p.parent
 trace=directory/'trace.jsonl'
 rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()] if trace.exists() else []
 executed=sum('target_ee' in row.get('result',{}) for row in rows)
 actions+=executed
 if executed!=d.get('executed_actions'):issues.append([str(directory),'executed_action_count',executed,d.get('executed_actions')])
 if len(rows)>d['max_policy_decisions']:issues.append([str(directory),'decision_budget_exceeded'])
 if d['success'] and not d.get('final_observation',{}).get('success'):issues.append([str(directory),'success_without_environment_evidence'])
 for row in rows:
  for image in row['observation']['image_paths']:
   if not Path(image).is_file():issues.append([str(directory),'missing_observation_image',image])
  ee=row.get('result',{}).get('target_ee')
  if ee is not None and len(ee)!=16:issues.append([str(directory),'bad_ee_shape'])
 for request in (directory/'calls').glob('*.request.json'):
  calls+=1
  body=json.loads(request.read_text(encoding='utf-8'))
  if body.get('model')!=d['model']:issues.append([str(directory),'wrong_model'])
  if 'Authorization' in body or 'api_key' in body:issues.append([str(directory),'credential_in_request_artifact'])
  for message in body['messages']:
   if not isinstance(message['content'],list):continue
   for part in message['content']:
    if part.get('type')=='image_url' and not (request.parent/part['image_url']['url']).is_file():issues.append([str(directory),'missing_model_image'])
 if not list((directory/'calls').glob('*.response.json')):issues.append([str(directory),'no_real_model_response'])
 if d['model']=='codex':
  for response in (directory/'calls').glob('*.response.json'):
   body=json.loads(response.read_text(encoding='utf-8'))
   if body.get('response',{}).get('model')!='gpt-5.6-luna':
    issues.append([str(directory),'wrong_codex_underlying_model',response.name])
result={'completed_episodes_checked':checked,'executed_actions_checked':actions,'model_calls_checked':calls,'issues':issues}
(root/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(bool(issues))
