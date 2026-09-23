"""Bounded physical diagnostic of a retained model-generated directed mission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-actions',type=int,default=6)
    args=p.parse_args();out=args.output.resolve()
    if not 1<=args.max_actions<=30:raise ValueError('max-actions must be 1..30')
    if out.exists():raise ValueError('Preserve prior diagnostic attempts')
    retained=json.loads(args.preflight.read_text());out.mkdir(parents=True)
    from source_layout import evaluation_sources
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources(ROOT)},indent=2))
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1';os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    from benchmark_clients import AuditedClient
    from robodawn.mission_planner import RobodawnPlanner
    install();seed=json.loads((ROOT/'seeds'/args.task/'result.json').read_text())
    config,_=load_task_args(dict(task_name=args.task,policy_name='robodawn',task_config='demo_clean'))
    config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
    env=class_decorator(args.task);bridge=None;history=[]
    try:
        env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
        bridge=Bridge(env,seed['instruction'],out,video=False);obs=bridge.observe()
        planner=RobodawnPlanner(AuditedClient('zdtaichu',out/'calls'),seed['instruction'])
        planner.steps=retained['steps'];planner.arm=retained['action']['arm']
        action=dict(retained['action'],observation_id=obs['observation_id'])
        for index in range(args.max_actions):
            result=bridge.execute(action);history.append(dict(action=action,result=result))
            (out/'trace.json').write_text(json.dumps(history,indent=2))
            obs=bridge.observe();planner.feedback(action,result)
            print('DIRECTED_PROBE',index,result['skill_success'],result['success'],result['failure'],flush=True)
            if not result['skill_success'] or result['success'] or index==args.max_actions-1:break
            action=planner.plan(obs,history)
        (out/'report.json').write_text(json.dumps(dict(diagnostic_only=True,history=history,final_observation=obs),indent=2))
    finally:
        if bridge:bridge.close()
        env.close_env()


if __name__=='__main__':main()
