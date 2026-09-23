"""Development-only first-action replay; never included in scored cohorts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',required=True)
    args=p.parse_args();cohort=args.cohort.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Choose a new directory; preserve prior attempts')
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1'
    os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    install();report=[]
    for task in args.tasks:
        seed=json.loads((ROOT/'seeds'/task/'result.json').read_text())
        row=json.loads((cohort/'robodawn'/task/'trace.jsonl').read_text().splitlines()[0])
        config,_=load_task_args(dict(task_name=task,policy_name='robodawn',task_config='demo_clean'))
        config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
        env=class_decorator(task);bridge=None
        try:
            env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
            bridge=Bridge(env,seed['instruction'],out/task,video=False)
            obs=bridge.observe()
            action=dict(row['action'],observation_id=obs['observation_id'])
            result=bridge.execute(action)
            final=bridge.observe()
            record=dict(task=task,diagnostic_only=True,source_action=row['action'],
                        executed_action=action,previous_result=row['result'],result=result,
                        final_observation=final)
            (out/task/'probe.json').write_text(json.dumps(record,indent=2))
            report.append(record)
            print('GRASP_PROBE',task,result.get('skill_success'),result.get('failure'),flush=True)
        finally:
            if bridge:bridge.close()
            env.close_env()
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
