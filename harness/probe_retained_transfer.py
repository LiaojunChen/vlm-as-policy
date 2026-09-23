"""Bounded physical diagnosis of a retained model goal, NOT from-head scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
ROOT=Path(__file__).resolve().parent

class DiagnosticBudgetExceeded(RuntimeError):
    """Distinct from socket TimeoutError so HTTP retry cannot swallow the alarm."""


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--retained',type=Path)
    source.add_argument('--plan-response',type=Path,help='Diagnostic retained model response containing a complete mission')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-actions',type=int,default=3);parser.add_argument('--continue-on-error',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.plan_response:
        from robotwin_harness_v3 import parse_json
        response=json.loads(args.plan_response.read_text())
        retained=parse_json(response['response']['choices'][0]['message']['content'])
    else:retained=json.loads(args.retained.read_text())
    if not 1<=args.max_actions<=12:raise ValueError('Bounded diagnostic requires 1..12 actions')
    out.mkdir(parents=True,exist_ok=False)
    from source_layout import evaluation_sources
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                                  for f in evaluation_sources()},indent=2))
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1';os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    from robodawn.mission_planner import RobodawnPlanner,validate_mission
    from benchmark_clients import AuditedClient
    from task_contracts_v3 import contract
    install();seed=json.loads((ROOT/'seeds'/args.task/'result.json').read_text())
    config,_=load_task_args(dict(task_name=args.task,policy_name='robodawn',task_config='demo_clean'))
    config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
    env=class_decorator(args.task);bridge=None;history=[]
    report=dict(diagnostic_only=True,retained_model_input=retained)
    def timeout(*args):raise DiagnosticBudgetExceeded('Bounded diagnostic exhausted 600 seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
        instruction=seed['instruction']+'\nCompletion requirements: '+contract(args.task)
        bridge=Bridge(env,instruction,out,video=False)
        planner=RobodawnPlanner(AuditedClient('zdtaichu',out/'calls'),instruction)
        planner.steps=validate_mission(retained if args.plan_response else {'steps':[retained['retained_model_intent']]})
        from robodawn.mission_compiler import schedule_simultaneous_supports
        from robodawn.mission_safety import validate_resources
        planner.steps=schedule_simultaneous_supports(planner.steps,instruction)
        validate_resources(planner.steps,bridge.sensors())
        planner.memory.intentions.initialize(planner.steps);planner.revision=1
        report['compiled_steps']=planner.steps
        for index in range(args.max_actions):
            obs=bridge.observe();action=planner.plan(obs,history);result=bridge.execute(action)
            row=dict(observation=obs,action=action,result=result);history.append(row);planner.feedback(action,result)
            with (out/'trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('TRANSFER_DIAGNOSTIC',index,action['skill'],result['skill_success'],result['success'],result['failure'],flush=True)
            if result['success'] or (not result['skill_success'] and not args.continue_on_error):break
            if planner.index>=len(planner.steps):break
        report['final_observation']=bridge.observe()
    except Exception as exc:
        report.update(error=str(exc),error_type=type(exc).__name__)
    finally:
        if bridge:
            report['official_success']=bool(env.eval_success or env.check_success())
            if 'final_observation' not in report:report['final_observation']=bridge.observe()
        report['history']=history;(out/'report.json').write_text(json.dumps(report,indent=2))
        if bridge:bridge.close()
        env.close_env();signal.alarm(0)


if __name__=='__main__':main()
