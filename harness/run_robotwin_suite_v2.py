"""Resume the isolated Qwen-only v2 RoboTwin 50-task evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv-robotwin310/bin/python'
MODELS = ('qwen3-vl-plus', 'qwen3-vl-flash-2026-01-22', 'qwen-vl-plus')
PROJECTS = ('show_harness', 'robodawn')


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def live_episode(directory: Path) -> bool:
    try:
        pid = int((directory / 'pid').read_text())
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        return 'run_robotwin_episode_v2.py' in args and str(directory) in args
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False


def gpu_free_mb() -> int:
    try:
        stdout = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'], text=True)
        return int(stdout.splitlines()[0])
    except Exception:
        return 0


def summarize(out: Path, tasks: list[str], models: list[str],
              projects: list[str], max_steps: int) -> dict:
    rows = []
    for project in projects:
        for model in models:
            for task in tasks:
                data = read(out / project / model / task / 'result.json')
                rows.append({'task': task, 'project': project, 'model': model,
                             'status': 'pending', 'success': None, **data})
    data = {
        'protocol': 'v2 RGB-selected, depth-grounded EE skills; official success; '
                    f'{len(tasks)} tasks x {len(projects)} project loops x '
                    f'{len(models)} models x 1 episode',
        'max_policy_decisions': max_steps,
        'requested_episodes': len(rows),
        'completed_episodes': sum(r['status'] == 'completed' for r in rows),
        'successful_episodes': sum(r['status'] == 'completed' and
                                   r['success'] is True for r in rows),
        'errors': sum(r['status'] in ('error', 'process_error') for r in rows),
        'episodes': rows,
    }
    path = out / 'summary.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding='utf-8')
    temp.replace(path)
    return data


def run_job(out: Path, task: str, project: str, model: str,
            max_steps: int, timeout: int) -> dict:
    directory = out / project / model / task
    result_path = directory / 'result.json'
    existing = read(result_path)
    if existing.get('status') == 'completed':
        return existing
    if live_episode(directory):
        pid = int((directory / 'pid').read_text())
        deadline = time.monotonic() + timeout
        while live_episode(directory) and time.monotonic() < deadline:
            time.sleep(3)
        if live_episode(directory):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        existing = read(result_path)
        if existing.get('status') == 'completed':
            return existing
    if directory.exists() and existing:
        archive = out / '_attempts' / project / model / task / time.strftime('%Y%m%dT%H%M%S')
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(archive))
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), str(ROOT / 'run_robotwin_episode_v2.py'),
           '--task', task, '--project', project, '--model', model,
           '--seed-manifest', str(out / 'seeds' / task / 'result.json'),
           '--max-steps', str(max_steps), '--output', str(directory)]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    with (directory / 'process.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        (directory / 'pid').write_text(str(process.pid), encoding='utf-8')
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    current = read(result_path)
    if code or current.get('status') in ('starting', 'running') or not current:
        current.update(task=task, project=project, model=model,
                       status='process_error', returncode=code, success=None)
        result_path.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                               encoding='utf-8')
    print('FINISHED', project, model, task, current.get('status'),
          current.get('success'), flush=True)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'results/full_50_v2'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--tasks', nargs='*')
    parser.add_argument('--models', nargs='+', choices=MODELS,
                        default=['qwen3-vl-plus'])
    parser.add_argument('--projects', nargs='+', choices=PROJECTS, default=list(PROJECTS))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_tasks = list(yaml.safe_load((ROOT / 'RoboTwin/env_cfg/task_config/_eval_step_limit.yml')
                                    .read_text(encoding='utf-8')))
    tasks = args.tasks or all_tasks
    for task in tasks:
        source = ROOT / 'results/full_50_v1/seeds' / task / 'result.json'
        target = out / 'seeds' / task / 'result.json'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if read(target).get('status') != 'expert_validated':
            raise RuntimeError(f'Seed is not expert validated: {task}')
    (out / 'CODEX_PAUSED.json').write_text(json.dumps({
        'status': 'paused', 'reason': 'User requested Qwen results first and a later '
                                      'explicit Codex model selection.'}, indent=2),
        encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps({
        'tasks': all_tasks, 'selected_tasks': tasks, 'projects': args.projects,
        'models': args.models, 'episodes_per_task': 1,
        'max_policy_decisions': args.max_steps, 'task_config': 'demo_clean',
        'seed_source': 'results/full_50_v1/seeds',
        'codex': 'paused by user',
        'robotwin_commit': '6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755',
        'show_harness_commit': '137d5718c3b7af0150764d8f9beeb252c9f2794a',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    jobs = [(task, project, model) for task in tasks
            for project in args.projects for model in args.models]
    pending = [(task, project, model) for task, project, model in jobs
               if read(out / project / model / task / 'result.json').get('status') != 'completed']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        running = {}
        while pending or running:
            while pending and len(running) < args.workers and gpu_free_mb() >= 8000:
                task, project, model = pending.pop(0)
                future = pool.submit(run_job, out, task, project, model,
                                     args.max_steps, args.timeout)
                running[future] = (task, project, model)
            if not running:
                time.sleep(5)
                continue
            done, _ = concurrent.futures.wait(
                running, timeout=10,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                del running[future]
            if done:
                data = summarize(out, tasks, args.models, args.projects,
                                 args.max_steps)
                print('PROGRESS', data['completed_episodes'],
                      data['requested_episodes'], data['successful_episodes'],
                      data['errors'], flush=True)
    data = summarize(out, tasks, args.models, args.projects, args.max_steps)
    print('SUMMARY', data['completed_episodes'],
          data['requested_episodes'], data['successful_episodes'], flush=True)


if __name__ == '__main__':
    main()
