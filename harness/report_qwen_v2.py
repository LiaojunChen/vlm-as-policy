"""Audit and summarize the qwen3-vl-plus-only v2 suite without changing episodes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def audit_episode(directory: Path) -> dict:
    result = load(directory / 'result.json')
    trace_path = directory / 'trace.jsonl'
    trace = []
    if trace_path.exists():
        trace = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()
                 if line.strip()]
    calls = list((directory / 'calls').glob('call_*.request.json'))
    responses = list((directory / 'calls').glob('call_*.response.json'))
    errors = list((directory / 'calls').glob('call_*.error.json'))
    planner_fail = sum(
        status == 'Fail'
        for row in trace for sub in row.get('result', {}).get('subactions', [])
        for status in sub.get('planner_status', {}).values())
    sim_actions = sum(len(row.get('result', {}).get('subactions', [])) for row in trace)
    return {
        'trace_rows': len(trace), 'model_request_files': len(calls),
        'model_response_files': len(responses), 'model_error_files': len(errors),
        'planner_fail_count': planner_fail, 'robotwin_actions': sim_actions,
        'missing_trace': result.get('policy_decisions', 0) > 0 and not trace_path.exists(),
        'missing_calls': result.get('model_calls', 0) != len(calls),
        'missing_responses': len(calls) != len(responses) + len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    manifest = load(out / 'manifest.json')
    tasks = manifest.get('selected_tasks', [])
    projects = manifest.get('projects', [])
    models = manifest.get('models', [])
    if models != ['qwen3-vl-plus']:
        raise ValueError('This report only accepts the qwen3-vl-plus cohort')
    rows = []
    for task in tasks:
        for project in projects:
            directory = out / project / 'qwen3-vl-plus' / task
            result = load(directory / 'result.json')
            audit = audit_episode(directory)
            rows.append({'task': task, 'project': project,
                         'status': result.get('status', 'pending'),
                         'success': result.get('success'),
                         'end_reason': result.get('end_reason', ''),
                         'seed': result.get('seed', ''),
                         'decisions': result.get('policy_decisions', 0),
                         'model_calls': result.get('model_calls', 0),
                         'elapsed_s': result.get('elapsed_s', ''),
                         'error': result.get('error', ''), **audit})
    fields = list(rows[0]) if rows else []
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    totals = Counter(row['status'] for row in rows)
    reasons = Counter(row['end_reason'] for row in rows if row['status'] == 'completed')
    successes = {project: sum(row['success'] is True for row in rows
                              if row['project'] == project) for project in projects}
    audit_issues = [row for row in rows if any(row[key] for key in
                    ('missing_trace', 'missing_calls', 'missing_responses'))]
    lines = [
        '# qwen3-vl-plus × RoboTwin v2 全 50 任务结果', '',
        f'范围：{len(tasks)} 个官方任务 × {len(projects)} 个项目 × '
        'qwen3-vl-plus × 每组合 1 个 episode。Codex 和另外两个 Qwen 模型未进入本轮全量结果。',
        '',
        f'已完成 {totals["completed"]}/{len(rows)}；环境成功 '
        f'{sum(successes.values())}/{len(rows)}；运行错误 '
        f'{totals["error"] + totals["process_error"]}。',
        '',
        '| 项目 | 完成 | 成功 | 成功率 |',
        '|---|---:|---:|---:|',
    ]
    for project in projects:
        completed = sum(row['status'] == 'completed' for row in rows
                        if row['project'] == project)
        lines.append(f'| {project} | {completed}/{len(tasks)} | '
                     f'{successes[project]} | {successes[project] / len(tasks):.1%} |')
    lines += ['', '按官方 `check_success()` 判定；专家验证 seed 与 v1 相同。'
              '每 episode 上限为 30 次高层决策，技能可能执行多次底层动作。'
              'v2 采用新的 RGB 选点、深度换算及技能执行器，不能与 v1 作为同动作空间的直接对照。',
              '', '失败/结束原因：' + ', '.join(f'{k or "未定"}={v}' for k, v in reasons.items()),
              '', f'模型请求 {sum(row["model_request_files"] for row in rows)} 次；'
              f'已审计底层动作 {sum(row["robotwin_actions"] for row in rows)} 次；'
              f'规划器 Fail {sum(row["planner_fail_count"] for row in rows)} 次；'
              f'证据文件不完整的 episode {len(audit_issues)} 个。',
              '', '| 任务 | Show-Harness | Robodawn |',
              '|---|---|---|']
    lookup = {(row['task'], row['project']): row for row in rows}
    for task in tasks:
        cells = []
        for project in projects:
            row = lookup[task, project]
            state = ('成功' if row['success'] is True else
                     '失败' if row['status'] == 'completed' else row['status'])
            cells.append(f'[{state}]({project}/qwen3-vl-plus/{task}/result.json)')
        lines.append(f'| {task} | ' + ' | '.join(cells) + ' |')
    lines += ['', '完整逐次数据见 `summary.json`、`results.csv`，'
              '每个 episode 的 `trace.jsonl`、`calls/`、`grounding/`、`frames/`。'
              '协议和限制见 [FULL_ROBOTWIN_PROTOCOL_V2.md](../../FULL_ROBOTWIN_PROTOCOL_V2.md)。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'completed': totals['completed'], 'total': len(rows),
                      'success': sum(successes.values()), 'audit_issues': len(audit_issues)},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
