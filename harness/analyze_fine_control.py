"""Summarize audited RoboTwin movement accuracy, policy behavior, and wall-clock rate."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def summarize(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    trace=directory/'trace.jsonl'
    rows=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    actions={side:dict(Counter(row['action'][side] for row in rows)) for side in ('left','right')}
    accuracy=defaultdict(list)
    grippers=[]
    planner_failures=[]
    for row in rows:
        execution=row.get('result',{})
        for side in ('left','right'):
            token=row['action'][side]
            if token.startswith('MV_'):
                motion=execution.get('motion',{}).get(side)
                if motion:
                    requested=motion['requested_delta_m']
                    actual=motion['actual_delta_m']
                    axis=max(range(3),key=lambda i:abs(requested[i]))
                    accuracy[execution['step_m']].append({
                        'actual_axis_mm':abs(actual[axis])*1000,
                        'target_error_mm':motion['position_error_m']*1000})
            if token in ('GRASP','RELEASE'):
                grippers.append({'step':row['step'],'side':side,'token':token,
                    'gripper_after':execution.get('endpose_after',{}).get(side+'_gripper')})
            status=execution.get('planner',{}).get(side,{}).get('status')
            if status and status!='Success': planner_failures.append({'step':row['step'],'side':side,'status':status})
    # First decision includes one-time planning; report steady cycles separately.
    cycles=[row['cycle_elapsed_s'] for row in rows[1:] if 'cycle_elapsed_s' in row]
    decisions=[row['decision_elapsed_s'] for row in rows[1:] if 'decision_elapsed_s' in row]
    execution_times=[row['result']['action_elapsed_s'] for row in rows if 'action_elapsed_s' in row.get('result',{})]
    calls=[json.loads(file.read_text()) for file in sorted((directory/'calls').glob('*.response.json'))]
    report={
        'directory':str(directory),'status':result.get('status'),'success':result.get('success'),
        'end_reason':result.get('end_reason'),'step_m':result.get('step_m',.04),
        'show_prompt':result.get('show_prompt','legacy'),'executed_actions':result.get('executed_actions',len(rows)),
        'model_calls':result.get('model_calls',len(calls)),'actions':actions,
        'controller_fallbacks':sum(bool(row.get('model_fallback')) for row in rows),
        'max_stage_index':{side:max((row.get('stages',{}).get(side,0) for row in rows),default=0) for side in ('left','right')},
        'accuracy_by_requested_mm':{str(step*1000):{
            'n_arm_moves':len(values),
            'mean_actual_axis_mm':statistics.mean(v['actual_axis_mm'] for v in values),
            'mean_target_error_mm':statistics.mean(v['target_error_mm'] for v in values),
            'max_target_error_mm':max(v['target_error_mm'] for v in values)} for step,values in accuracy.items()},
        'gripper_commands':grippers,'planner_failures':planner_failures,
        'steady_cycle_mean_s':statistics.mean(cycles) if cycles else None,
        'steady_policy_hz':len(cycles)/sum(cycles) if cycles else None,
        'steady_decision_mean_s':statistics.mean(decisions) if decisions else None,
        'execution_mean_s':statistics.mean(execution_times) if execution_times else None,
        'call_mean_s':statistics.mean(c['elapsed_s'] for c in calls) if calls else None,
    }
    if rows and rows[0]['observation'].get('robot_geometry'):
        report['initial_robot_geometry']=rows[0]['observation']['robot_geometry']
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    payload=[summarize(directory) for directory in args.directories]
    rendered=json.dumps(payload,ensure_ascii=False,indent=2)
    if args.output: args.output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
