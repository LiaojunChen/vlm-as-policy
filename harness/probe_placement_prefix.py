"""Replay retained pickup/place intents for geometry diagnosis, never scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True)
    args=p.parse_args();cohort=args.cohort.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Preserve all previous attempts')
    out.mkdir(parents=True)
    from source_layout import evaluation_sources
    (out/'code_hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in evaluation_sources(ROOT)},indent=2))
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1';os.environ['ROBOTWIN_REUSE_PLANNERS']='1'
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator
    from planner_cache_v3 import install
    from robotwin_harness_v3 import Bridge
    install();report=[]
    for task in args.tasks:
        seed=json.loads((ROOT/'seeds'/task/'result.json').read_text())
        rows=[json.loads(x) for x in (cohort/'robodawn'/task/'trace.jsonl').read_text().splitlines()[:2]]
        assert [r['action']['skill'] for r in rows]==['pick','place']
        config,_=load_task_args(dict(task_name=task,policy_name='robodawn',task_config='demo_clean'))
        config.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
        env=class_decorator(task);bridge=None;history=[]
        try:
            env.setup_demo(now_ep_num=0,seed=seed['seed'],is_test=True,**config)
            bridge=Bridge(env,seed['instruction'],out/task,video=False);obs=bridge.observe()
            for index,row in enumerate(rows):
                action=dict(row['action'],observation_id=obs['observation_id'])
                if index:action['arm']=history[0]['action']['arm']
                result=bridge.execute(action);history.append(dict(action=action,result=result,previous=row))
                (out/task/'trace.json').write_text(json.dumps(history,indent=2))
                obs=bridge.observe()
                print('PLACEMENT_PROBE',task,index,result['skill_success'],result['success'],result['failure'],flush=True)
                if not result['skill_success'] or result['success']:break
            record=dict(task=task,diagnostic_only=True,history=history,final_observation=obs)
            (out/task/'report.json').write_text(json.dumps(record,indent=2));report.append(record)
        finally:
            if bridge:bridge.close()
            env.close_env()
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
