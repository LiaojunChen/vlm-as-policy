"""Summarize retained episodes without using evaluator state as policy input."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(root):
    totals = Counter()
    tasks = []
    for result_path in sorted(root.glob('robodawn/*/result.json')):
        result = json.loads(result_path.read_text())
        trace = result_path.with_name('trace.jsonl')
        rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        failures = Counter(row['result'].get('failure') or 'motion_succeeded_goal_unmet'
                           for row in rows if not row['result'].get('success'))
        if not result.get('success'):
            totals.update(failures)
        tasks.append(dict(task=result['task'], success=result.get('success'),
                          end_reason=result.get('end_reason'), decisions=len(rows),
                          failures=dict(failures), error=result.get('error')))
    return dict(source=str(root.resolve()), requested=len(tasks),
                successes=sum(t['success'] is True for t in tasks),
                failures=dict(totals.most_common()), tasks=tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.root), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + '\n')
    else:
        print(report)
