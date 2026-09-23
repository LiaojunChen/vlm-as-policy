"""Run one corrected RGB/depth-grounded Qwen episode on a validated RoboTwin seed."""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ('qwen3-vl-plus', 'qwen3-vl-flash-2026-01-22', 'qwen-vl-plus')


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--project', choices=['show_harness', 'robodawn'], required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed-manifest', required=True)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed = json.loads(Path(args.seed_manifest).read_text(encoding='utf-8'))
    if seed.get('status') not in ('expert_validated', 'expert_unvalidated'):
        raise ValueError('Seed has not been expert validated')
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER'] = '1'
    faulthandler.dump_traceback_later(300, repeat=True)
    started = time.monotonic()
    result = {'task': args.task, 'project': args.project,
              'model': args.model, 'seed': seed['seed'], 'status': 'starting',
              'success': None, 'protocol': 'v2_rgb_depth_grounded_skills',
              'task_config': 'demo_clean', 'max_policy_decisions': args.max_steps,
              'seed_validation': seed['status'], 'instruction': seed['instruction'],
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump(out / 'result.json', result)
    os.chdir(ROOT / 'RoboTwin')
    sys.path.insert(0, str(ROOT / 'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args, class_decorator
    from benchmark_clients import AuditedClient
    from robotwin_grounding_v2 import GroundedBridge, GridGrounder
    from robodawn.planner_v2 import RobodawnGroundedPlanner, SKILLS
    from showharness.policy_v2 import ShowGroundedPolicy
    from robodawn.core import run_loop
    env = bridge = client = None
    policy_terminated = False
    try:
        task_args, _ = load_task_args(dict(task_name=args.task,
                                           policy_name=args.project,
                                           task_config='demo_clean'))
        task_args.update(eval_mode=True, save_data=False, eval_video_log=False,
                         render_freq=0)
        env = class_decorator(args.task)
        print('V2_SETUP', args.project, args.model, args.task, seed['seed'], flush=True)
        env.setup_demo(now_ep_num=0, seed=seed['seed'], is_test=True, **task_args)
        result['official_step_limit'] = env.step_lim
        bridge = GroundedBridge(env, seed['instruction'], out / 'frames')
        client = AuditedClient(args.model, out / 'calls')
        grounder = GridGrounder(client, out / 'grounding')
        result['status'] = 'running'
        result['setup_elapsed_s'] = time.monotonic() - started
        dump(out / 'result.json', result)

        def record(row: dict) -> None:
            with (out / 'trace.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
            result['policy_decisions'] = max(result.get('policy_decisions', 0),
                                             row['step'] + 1)
            result['model_calls'] = client.calls
            result['sim_actions'] = bridge.index
            result['last_success'] = bool(row.get('result', {}).get('success'))
            dump(out / 'result.json', result)
            print('V2_DECISION', row['step'], row['action'].get('name'),
                  row.get('result', {}).get('ok'),
                  row.get('result', {}).get('success'), flush=True)

        if args.project == 'robodawn':
            planner = RobodawnGroundedPlanner(client, grounder, seed['instruction'])
            trace = run_loop(bridge, bridge, planner, max_steps=args.max_steps,
                             on_record=record, stop_on_error=False,
                             allowed_actions=(*SKILLS, 'done'))
            policy_terminated = bool(trace and trace[-1]['action']['name'] == 'done')
        else:
            policy = ShowGroundedPolicy(client, seed['instruction'], out, grounder)
            for step in range(args.max_steps):
                observation = bridge.observe()
                if observation['success']:
                    break
                actions, decision = policy.decide(observation)
                if not actions:
                    row = {'step': step, 'observation': observation,
                           'action': {'name': 'wait', 'controller_tokens':
                                      decision['controller_tokens']},
                           'result': {'ok': True, 'success': False,
                                      'native_done': decision['native_done']},
                           'model_decision': decision}
                    record(row)
                    if decision['native_done']:
                        policy_terminated = True
                        break
                    continue
                for action in actions:
                    outcome = bridge.execute(action)
                    policy.feedback(action, outcome)
                    record({'step': step, 'observation': observation,
                            'action': action, 'result': outcome,
                            'model_decision': decision})
                    if outcome['success']:
                        break
                if env.eval_success or env.check_success():
                    break
        final = bridge.observe()
        result.update(status='completed', success=bool(final['success']),
                      final_observation=final, model_calls=client.calls,
                      sim_actions=bridge.index)
        result['end_reason'] = ('environment_success' if result['success'] else
                                'policy_terminated' if policy_terminated else
                                'decision_budget')
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__,
                      error=str(exc), success=None,
                      model_calls=client.calls if client else 0,
                      sim_actions=bridge.index if bridge else 0)
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close_env()
            except Exception:
                pass
        result['elapsed_s'] = time.monotonic() - started
        dump(out / 'result.json', result)
        print('V2_RESULT', json.dumps({k: result.get(k) for k in
              ('task', 'project', 'model', 'status', 'success', 'end_reason',
               'policy_decisions', 'sim_actions', 'model_calls', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
