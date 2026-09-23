"""Run real RoboTwin expert validation or one image-only policy episode."""
from __future__ import annotations
import argparse, faulthandler, json, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2), encoding="utf-8");temp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--project',choices=['expert','show_harness','robodawn','probe'],required=True)
    p.add_argument('--model',default='qwen3-vl-plus')
    p.add_argument('--seed',type=int,default=100000)
    p.add_argument('--seed-attempts',type=int,default=30)
    p.add_argument('--max-steps',type=int,default=100)
    p.add_argument('--step-m',type=float,default=.04,help='World-frame translation per policy action, in metres')
    p.add_argument('--show-prompt',choices=['legacy','fine'],default='legacy')
    p.add_argument('--controller-max-tokens',type=int,default=128)
    p.add_argument('--model-timeout-s',type=float,default=90)
    p.add_argument('--model-max-retries',type=int,default=6)
    p.add_argument('--structured-output',action='store_true',help='Send full JSON Schema to a supporting model server')
    p.add_argument('--probe-fine-control',action='store_true',help='Probe small moves and gripper actuation; no model policy')
    p.add_argument('--defer-render',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    from robotwin_bridge import make_frame_context
    frame_context=make_frame_context(a.step_m)
    if a.max_steps<=0 or a.controller_max_tokens<=0: p.error('step/token budgets must be positive')
    if a.model_timeout_s<=0 or a.model_max_retries<0: p.error('invalid transport timeout/retry budget')
    if a.probe_fine_control and a.project!='probe': p.error('--probe-fine-control requires --project probe')
    if (out/'trace.jsonl').exists(): p.error('output already contains a trace; choose a new directory')
    if a.project != 'expert':
        os.environ['ROBOTWIN_LAZY_BATCH_PLANNER']='1'
    faulthandler.dump_traceback_later(300,repeat=True)
    started=time.monotonic()
    result={'task':a.task,'project':a.project,'model':a.model,'seed':a.seed,'status':'starting',
            'success':None,'max_policy_decisions':a.max_steps,'task_config':'demo_clean',
            'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'deferred_render':bool(a.defer_render and a.project!='expert'),'lazy_batch_planner':a.project!='expert',
            'step_m':a.step_m,'show_prompt':a.show_prompt,'controller_max_tokens':a.controller_max_tokens,
            'structured_output':a.structured_output}
    dump(out/'result.json',result)
    os.chdir(ROOT/'RoboTwin');sys.path.insert(0,str(ROOT/'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args,class_decorator,build_instruction
    from robotwin_bridge import RobotWinBridge, PolicyOutputError
    env=None
    try:
        args,_=load_task_args(dict(task_name=a.task,policy_name=a.project,task_config='demo_clean'))
        args.update(eval_mode=True,save_data=False,eval_video_log=False,render_freq=0)
        attempts=[]
        if a.project=='expert':
            # Official expert filtering; skipped unstable seeds do not count as failures.
            for seed in range(a.seed,a.seed+a.seed_attempts):
                env=class_decorator(a.task)
                print('EXPERT_SETUP',seed,flush=True)
                try:
                    env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
                    info=env.play_once()
                    valid=bool(env.plan_success and env.check_success())
                    attempts.append({'seed':seed,'valid':valid,
                        'plan_success':bool(env.plan_success),
                        'environment_success':bool(env.check_success()),
                        'left_plan_statuses':[x.get('status') if isinstance(x,dict) else None for x in env.left_joint_path],
                        'right_plan_statuses':[x.get('status') if isinstance(x,dict) else None for x in env.right_joint_path]})
                    if valid:
                        result.update(seed=seed,success=True,status='expert_validated',
                            instruction=build_instruction(args,info,'seen',1),expert_info=info,
                            official_step_limit=env.step_lim)
                        break
                except Exception as exc:
                    attempts.append({'seed':seed,'error':str(exc),'type':type(exc).__name__})
                    if any(tag in str(exc).lower() for tag in ('cuda error','out of memory','graph capture')):
                        result['seed_attempts']=attempts
                        raise  # A poisoned CUDA context needs a fresh process, not a new seed.
                finally:
                    if env is not None:
                        try: env.close_env()
                        except Exception: pass
                        env=None
                        import gc, torch
                        gc.collect()
                        torch.cuda.empty_cache()
                dump(out/'seed_attempts.json',attempts)
            else: result.update(status='no_valid_seed',success=None)
            result['seed_attempts']=attempts
        else:
            env=class_decorator(a.task)
            print('POLICY_SETUP',a.project,a.model,a.task,a.seed,flush=True)
            env.setup_demo(now_ep_num=0,seed=a.seed,is_test=True,**args)
            import gc, torch
            result['cuda_before_cache_release']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
            gc.collect()
            torch.cuda.empty_cache()
            result['cuda_after_cache_release']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
            seed_path=out.parents[2]/'seeds'/a.task/'result.json'
            # Explicit sibling seed manifest supplied by the batch directory layout.
            if seed_path.exists(): instruction=json.loads(seed_path.read_text(encoding="utf-8"))['instruction']
            else: instruction=json.loads((ROOT/'RoboTwin/description/task_instruction'/f'{a.task}.json').read_text(encoding="utf-8"))['full_description']
            result['instruction']=instruction
            if seed_path.exists():result['seed_validation']=json.loads(seed_path.read_text(encoding="utf-8")).get('status')
            result['official_step_limit']=env.step_lim
            result['setup_elapsed_s']=time.monotonic()-started
            result['status']='running';dump(out/'result.json',result)
            bridge=RobotWinBridge(env,instruction,out/'frames',step_m=a.step_m,defer_render=a.defer_render,
                                 robot_geometry=a.show_prompt=='fine' or a.probe_fine_control)
            def record(row):
                faulthandler.dump_traceback_later(300,repeat=True)
                with (out/'trace.jsonl').open('a', encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print('DECISION',row['step'],row['action'],row.get('result',{}).get('success'),flush=True)
            if a.project=='probe':
                sequence=[(a.step_m,{'left':'MV_BACK','right':'STILL'})]
                if a.probe_fine_control:
                    sequence=[(distance,{'left':direction,'right':direction})
                        for distance in (.04,.01,.005,.002)
                        for direction in ('MV_BACK','MV_FWD')]
                    sequence += [(a.step_m,{'left':token,'right':token}) for token in ('GRASP','RELEASE')]
                for distance,tokens in sequence:
                    obs=bridge.observe();bridge.step_m=distance
                    r=bridge.execute_tokens(tokens)
                    before=obs['endpose']['left_endpose'][:3]
                    after=r['endpose_after']['left_endpose'][:3]
                    result['displacement']=[b-a for a,b in zip(before,after)]
                    record({'step':obs['step'],'observation':obs,'action':tokens,'result':r,'diagnostic_only':True})
                result['diagnostic_only']=True
            else:
                from benchmark_clients import AuditedClient,CODEX_MODEL
                client=AuditedClient(a.model,out/'calls')
                client.max_tokens=a.controller_max_tokens
                client.timeout_s=a.model_timeout_s
                client.max_retries=a.model_max_retries
                client.structured_output=a.structured_output
                if a.model=='codex':result['codex_model']=CODEX_MODEL
                if a.project=='robodawn':
                    from robodawn.core import run_loop
                    from robodawn.planner_v1 import RobodawnPlanner
                    run_loop(bridge,bridge,RobodawnPlanner(client,frame_context),max_steps=a.max_steps,
                             stop_on_noop=False,on_record=record)
                else:
                    from showharness.policy_v1 import ShowPolicy
                    policy=ShowPolicy(client,instruction,out,frame_context=frame_context,prompt_mode=a.show_prompt)
                    for step in range(a.max_steps):
                        cycle_started=time.monotonic()
                        obs=bridge.observe()
                        if obs['success']: break
                        if env.take_action_cnt>=env.step_lim: break
                        decision_started=time.monotonic()
                        tokens,done=policy.decide(obs)
                        decision_s=time.monotonic()-decision_started
                        if done:
                            record({'step':step,'observation':obs,'action':tokens,'result':{'ok':True,'terminal':True}})
                            break
                        r=bridge.execute_tokens(tokens)
                        policy.observe_execution(r)
                        record({'step':step,'observation':obs,'action':tokens,'result':r,
                                'model_decision':policy.last_decision.raw_text,
                                'model_fallback':bool(policy.last_decision.payload.get('fallback')),
                                'stages':policy.indices.copy(),'decision_elapsed_s':decision_s,
                                'cycle_elapsed_s':time.monotonic()-cycle_started})
                        if r['success']: break
                result['model_calls']=client.calls
            final_obs=bridge.observe()
            result.update(success=bool(final_obs['success']),status='completed',
                          executed_actions=bridge.index,final_observation=final_obs)
            result['end_reason']='environment_success' if result['success'] else ('decision_budget' if bridge.index>=a.max_steps else 'policy_terminated')
            if env.take_action_cnt>=env.step_lim and not result['success']: result['end_reason']='environment_budget'
            if a.project=='probe': result['end_reason']='diagnostic_complete'
    except KeyboardInterrupt:
        result.update(status='interrupted',success=None,end_reason='interrupted')
        if 'bridge' in locals(): result['executed_actions']=bridge.index
        if 'client' in locals(): result['model_calls']=client.calls
    except Exception as exc:
        result.update(status='error',error_type=type(exc).__name__,error=str(exc),success=None)
        if isinstance(exc,PolicyOutputError):
            result.update(status='completed',success=False,end_reason='invalid_model_action',
                          executed_actions=bridge.index,model_calls=client.calls,
                          final_observation=bridge.observe())
        (out/'error.txt').write_text(traceback.format_exc(), encoding="utf-8");traceback.print_exc()
    finally:
        if env is not None:
            try: env.close_env()
            except Exception: pass
        result['elapsed_s']=time.monotonic()-started
        faulthandler.cancel_dump_traceback_later()
        dump(out/'result.json',result)
        print('RESULT',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
