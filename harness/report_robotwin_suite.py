"""Produce transparent, progressively updated JSON/CSV/Markdown reports."""
import csv,json,sys
from pathlib import Path
from collections import Counter
from run_robotwin_suite import summarize
root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/full_50_v1').resolve()
manifest=json.loads((root/'manifest.json').read_text(encoding="utf-8"));tasks=manifest['tasks']
summary=summarize(root,tasks)
rows=summary['episodes']
seed_status=Counter(json.loads((root/'seeds'/task/'result.json').read_text(encoding='utf-8')).get('status')
                    if (root/'seeds'/task/'result.json').exists() else 'pending' for task in tasks)
fields=['task','project','model','status','success','seed','executed_actions','model_calls','elapsed_s','end_reason','error_type','error']
with (root/'episodes.csv').open('w', encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
text=['# RoboTwin 全任务覆盖结果','',f'目标 400 episodes；当前有效完成 {summary["completed_episodes"]}，成功 {summary["successful_episodes"]}。','',
 f'官方专家验证 seed：{seed_status["expert_validated"]}/50；未验证 seed：{seed_status["expert_unvalidated"]}/50。','',
 '统一配置：50 tasks × 2 projects × 4 models × 1 episode，demo_clean，官方专家筛选的共同 seed，每 episode 最多 100 次策略决策。成功判定使用官方环境；这不是 100 episodes/task 的 leaderboard 分数。','',
 '适配范围：Show-Harness 使用原生双臂子目标规划、控制器与文本记忆；Robodawn 使用原生 core 循环。两者均通过新增的 RoboTwin 适配器执行固定 4 cm 世界坐标平移和夹爪开合，未提供旋转动作。因此本报告衡量此适配配置下的表现；每任务仅 1 次，不能据此作稳定的模型优劣排名。','',
 'Codex CLI 组显式指定 `gpt-5.6-luna`，推理强度 `low`；之前未显式指定模型的轨迹归档在 `_attempts/`，不计入本轮结果。','',
 'Seed 说明：优先使用通过官方专家轨迹的共同 seed。若某任务连续 30 个实际场景均未通过专家筛选，则使用第一个可初始化场景的共同 seed 完成覆盖，并在 seed 结果及 episode 中标记 `expert_unvalidated`。此类任务不能与专家验证任务混称。','',
 '| 项目 | 模型 | 有效完成 | 成功 | 成功率（有效分母） | 错误 | 未完成 |','|---|---|---:|---:|---:|---:|---:|']
for project in manifest['projects']:
 for model in manifest['models']:
  subset=[r for r in rows if r['project']==project and r['model']==model]
  complete=[r for r in subset if r['status']=='completed']
  successes=sum(r['success'] for r in complete)
  errors=sum(r['status'] in ('error','process_error') for r in subset)
  rate=f'{successes/len(complete):.1%}' if complete else 'N/A'
 text.append(f'| {project} | {model} | {len(complete)}/50 | {successes} | {rate} | {errors} | {50-len(complete)-errors} |')
text+=['','## 结束原因','', '| 项目 | 模型 | 环境成功 | 决策预算耗尽 | 策略主动结束 | 非法模型动作 |', '|---|---|---:|---:|---:|---:|']
for project in manifest['projects']:
 for model in manifest['models']:
  counts=Counter(r.get('end_reason') for r in rows if r['project']==project and r['model']==model and r['status']=='completed')
  text.append('| '+project+' | '+model+' | '+' | '.join(str(counts[k]) for k in ['environment_success','decision_budget','policy_terminated','invalid_model_action'])+' |')
text+=['','## 逐任务','', '| Task | SH plus | SH flash | SH vl-plus | SH codex | RD plus | RD flash | RD vl-plus | RD codex |', '|---|---|---|---|---|---|---|---|---|']
for task in tasks:
 vals=[]
 for p in manifest['projects']:
  for m in manifest['models']:
   r=next(r for r in rows if r['task']==task and r['project']==p and r['model']==m)
   status=('SUCCESS' if r['success'] else 'FAIL') if r['status']=='completed' else r['status']
   link=f'{p}/{m}/{task}/result.json'
   vals.append(f'[{status}]({link})' if (root/link).exists() else status)
 text.append('| '+task+' | '+' | '.join(vals)+' |')
text+=['','所有失败与错误均保留，错误不算作策略失败。原始记录在各 episode 的 `trace.jsonl`、`frames/`、`calls/`、`process.log`；专家筛选证据在 `seeds/`。']
(root/'REPORT.md').write_text('\n'.join(text)+'\n', encoding="utf-8")
print(json.dumps({'completed':summary['completed_episodes'],'successes':summary['successful_episodes'],'statuses':dict(Counter(r['status'] for r in rows))}))
