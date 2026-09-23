"""Standard-camera startup check, without expert, model or policy actions."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    install();seed=json.loads((ROOT/'seeds'/args.task/'result.json').read_text())
    config,_=load_task_args(dict(task_name=args.task,policy_name='robodawn',task_config='demo_clean'))
    config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
    env=class_decorator(args.task);bridge=None
    try:
        env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
        bridge=Bridge(env,seed['instruction'],out,video=False)
        obs=bridge.observe()
        assert bridge.cloud.shape==(240,320,3)
        report=dict(diagnostic_only=True,startup_ok=True,policy_actions=0,model_calls=0,
                    task=args.task,seed=seed['seed'],observation=obs)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print('STANDARD_ENV_READY',args.task,seed['seed'],bridge.cloud.shape,flush=True)
    finally:
        if bridge:bridge.close()
        env.close_env()


if __name__=='__main__':main()
