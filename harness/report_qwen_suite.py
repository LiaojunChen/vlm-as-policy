"""Qwen-only report for the first 300 RoboTwin policy episodes."""
from __future__ import annotations
import csv,json,sys
from collections import Counter
from pathlib import Path
from run_robotwin_suite import MODELS,PROJECTS,read

ROOT=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent/'results/full_50_v1').resolve()
manifest=read(ROOT/'manifest.json')
tasks=manifest['tasks']
models=[model for model in MODELS if model!='codex']
rows=[]
for task in tasks:
 for project in PROJECTS:
  for model in models:
   result=read(ROOT/project/model/task/'result.json')
   rows.append({'task':task,'project':project,'model':model,**result}
               if result else {'task':task,'project':project,'model':model,'status':'pending','success':None})
completed=[row for row in rows if row['status']=='completed']
successes=sum(row['success'] is True for row in completed)
seed_status=Counter(read(ROOT/'seeds'/task/'result.json').get('status','pending') for task in tasks)
summary={'requested_episodes':300,'completed_episodes':len(completed),
         'successful_episodes':successes,'expert_validated_tasks':seed_status['expert_validated'],
         'models':models,'projects':PROJECTS,'episodes':rows}
summary_path=ROOT/'qwen_summary.json'
summary_tmp=summary_path.with_suffix('.json.tmp')
summary_tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
summary_tmp.replace(summary_path)
fields=['task','project','model','status','success','seed','executed_actions','model_calls',
        'elapsed_s','end_reason','error_type','error']
csv_path=ROOT/'qwen_episodes.csv'
csv_tmp=csv_path.with_suffix('.csv.tmp')
with csv_tmp.open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
csv_tmp.replace(csv_path)
text=['# RoboTwin Qwen 三模型全 50 任务结果','',
      f'有效完成 {len(completed)}/300，官方环境成功 {successes}；专家验证任务 {seed_status["expert_validated"]}/50。','',
      '协议：Show-Harness 与 Robodawn core 两个项目；三个 Qwen 视觉语言模型；50 个官方任务各 1 个 episode；相同的专家验证 seed；demo_clean；每个 episode 最多 100 次策略决策。成功仅依据 RoboTwin 官方环境。每任务 1 次不能视为稳定排名，也不是官方 100 次/任务排行榜复现。','',
      '动作适配只提供固定 4 cm 世界坐标平移与夹爪开合，没有旋转动作。头部相机位于工作区的 -Y 侧，工作区深处为 +Y；本适配层的 `MV_FWD` 却定义为 -Y，`MV_BACK` 为 +Y，因此方向词与画面直觉相反。这会影响策略表现，不能把本轮成功率直接解释为模型本身的能力排名。模型请求、响应、三路 RGB、逐步动作和仿真日志均在各 episode 目录。Codex CLI 结果不在本报告中。','',
      '调用遇到阿里云 429 限流后，总运行并发先调整为 6，`qwen3-vl-flash-2026-01-22` 的单模型并发先降为 1，稳定后试升到 2；后续在 Flash 上限 2 的约束下试升总并发到 8。传输层以指数退避重试 429。中断和网络错误保留归档并重新执行该 episode，成功率只统计完整有效的 episode。','',
      '模型输出不在动作白名单内时，保留原始响应并记为有效策略失败，不为挑选合法答案重新采样。`put_bottles_dustbin` 的 Show-Harness + `qwen-vl-plus` 原始轨迹曾因 `Grasp` 大小写被标作运行错误；核对原始调用和动作轨迹后原位更正为 `invalid_model_action`，结果文件保留了更正说明。','',
      '| 项目 | 模型 | 完成 | 成功 | 成功率 | 环境/接口错误 |',
      '|---|---|---:|---:|---:|---:|']
for project in PROJECTS:
 for model in models:
  subset=[row for row in rows if row['project']==project and row['model']==model]
  done=[row for row in subset if row['status']=='completed']
  hits=sum(row['success'] is True for row in done)
  errors=sum(row['status'] in ('error','process_error') for row in subset)
  rate=f'{hits/len(done):.1%}' if done else 'N/A'
  text.append(f'| {project} | {model} | {len(done)}/50 | {hits} | {rate} | {errors} |')
text+=['','## 结束原因','',
       '| 项目 | 模型 | 环境成功 | 决策预算耗尽 | 策略主动结束 | 非法模型动作 |',
       '|---|---|---:|---:|---:|---:|']
for project in PROJECTS:
 for model in models:
  reasons=Counter(row.get('end_reason') for row in completed
                  if row['project']==project and row['model']==model)
  text.append('| '+project+' | '+model+' | '+' | '.join(str(reasons[key]) for key in
     ('environment_success','decision_budget','policy_terminated','invalid_model_action'))+' |')
text+=['','## 逐任务','',
       '| Task | SH plus | SH flash | SH vl-plus | RD plus | RD flash | RD vl-plus |',
       '|---|---|---|---|---|---|---|']
index={(row['task'],row['project'],row['model']):row for row in rows}
for task in tasks:
 cells=[]
 for project in PROJECTS:
  for model in models:
   row=index[task,project,model]
   state=('SUCCESS' if row['success'] else 'FAIL') if row['status']=='completed' else row['status']
   link=f'{project}/{model}/{task}/result.json'
   cells.append(f'[{state}]({link})' if (ROOT/link).exists() else state)
 text.append('| '+task+' | '+' | '.join(cells)+' |')
text+=['','原始结果：`qwen_episodes.csv`、`qwen_summary.json`，以及每个 episode 的 `result.json`、`trace.jsonl`、`frames/`、`calls/`、`process.log`。专家筛选过程与共同 seed 在 `seeds/`。']
report_path=ROOT/'QWEN_REPORT.md'
report_tmp=report_path.with_suffix('.md.tmp')
report_tmp.write_text('\n'.join(text)+'\n',encoding='utf-8')
report_tmp.replace(report_path)
print(json.dumps({'completed':len(completed),'successes':successes,
                  'statuses':dict(Counter(row['status'] for row in rows))}))
