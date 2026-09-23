"""Audited qwen3-vl-plus retest; all outputs belong to this independent copy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
from benchmark_clients import AuditedClient
from reviewed_interface import ACTION_NAMES, ContractError
from robodawn.reviewed import ReviewedRobodawnPlanner
from showharness.reviewed import ReviewedShowPolicy
from reviewed_mujoco import ReviewedGrasp, INSTRUCTIONS
from robotwin_grounding_v2 import GridGrounder
from robodawn.core import run_loop
from robotwin_bridge import PolicyOutputError

ROOT=Path(__file__).resolve().parent


def dump(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


class Stalled(RuntimeError):pass


def run(project,task,seed,directory,max_steps):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():raise RuntimeError('Refusing to overwrite an existing episode')
    result={'project':project,'task':task,'seed':seed,'model':'qwen3-vl-plus',
            'status':'starting','success':None,'protocol':'review-1',
            'max_policy_decisions':max_steps,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(directory/'result.json',result)
    env=client=None;started=time.monotonic();rows=[];consecutive_failures=0
    try:
        env=ReviewedGrasp(seed,directory,task)
        result['capabilities']=env.caps.to_dict()
        client=AuditedClient('qwen3-vl-plus',directory/'calls')
        grounder=GridGrounder(client,directory/'grounding')
        result['status']='running';dump(directory/'result.json',result)
        def record(row):
            nonlocal consecutive_failures
            rows.append(row)
            with (directory/'trace.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            consecutive_failures=0 if row['result'].get('ok') else consecutive_failures+1
            if consecutive_failures>=3:raise Stalled('Three consecutive execution failures')
        if project=='robodawn':
            planner=ReviewedRobodawnPlanner(client,grounder,env.caps,INSTRUCTIONS[task])
            run_loop(env,env,planner,max_steps=max_steps,stop_on_error=False,
                     allowed_actions=ACTION_NAMES,stop_on_noop=False,on_record=record)
        else:
            policy=ReviewedShowPolicy(client,grounder,env.caps,INSTRUCTIONS[task],directory)
            for step in range(max_steps):
                obs=env.observe()
                if obs['success']:break
                action,meta=policy.decide(obs)
                outcome=env.execute(action)
                policy.feedback(action,outcome,meta)
                record({'step':step,'observation':obs,'action':action,'result':outcome,'model_decision':meta})
                if outcome['success'] or outcome.get('terminal'):break
        result.update(status='completed',success=env.success,
            end_reason='environment_success' if env.success else
                       'policy_terminated' if rows and rows[-1]['action']['name']=='done' else 'decision_budget')
    except (ContractError,PolicyOutputError,Stalled) as exc:
        result.update(status='completed',success=False,
                      end_reason='stalled' if isinstance(exc,Stalled) else 'invalid_model_action',
                      error=str(exc),error_type=type(exc).__name__,
                      failure_code=getattr(exc,'code',None))
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    except Exception as exc:
        result.update(status='error',success=None,error=str(exc),error_type=type(exc).__name__)
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                final=env.observe()
                result.update(final_observation=final,max_lift_m=env.max_lift_m,
                              stable_goal_s=env.hold_s,skills_executed=env.index,
                              video=str(directory/'continuous.mp4'))
            except Exception as exc:result['final_observation_error']=str(exc)
            try:env.close()
            except Exception as exc:result['video_error']=str(exc)
        result.update(elapsed_s=time.monotonic()-started,model_calls=client.calls if client else 0,
                      policy_decisions=len(rows))
        dump(directory/'result.json',result)
        print(json.dumps({k:result.get(k) for k in ('project','task','seed','status','success',
              'end_reason','policy_decisions','model_calls','error','elapsed_s')},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',choices=['show_harness','robodawn'],required=True)
    p.add_argument('--task',choices=list(INSTRUCTIONS),required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--max-steps',type=int,default=12)
    p.add_argument('--output',type=Path,default=ROOT/'results/reviewed_v1')
    a=p.parse_args();output=a.output.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Review outputs must stay inside the copied workspace')
    for seed in a.seeds:run(a.project,a.task,seed,output/a.task/a.project/f'seed_{seed}',a.max_steps)
