"""Replay a scored-success episode's exact EE targets with RoboTwin's MP4 recorder."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    args = parser.parse_args()
    directory = args.episode.resolve()
    original = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    if original.get('status') != 'completed' or original.get('success') is not True:
        raise ValueError('Only an officially successful completed episode can be replayed')
    trace = [json.loads(line) for line in (directory / 'trace.jsonl')
             .read_text(encoding='utf-8').splitlines() if line.strip()]
    targets = [step['target_ee'] for row in trace
               for step in row.get('result', {}).get('subactions', [])]
    if not targets:
        raise ValueError('No recorded EE targets to replay')
    video_dir = directory / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    output = video_dir / 'official_replay.mp4'
    record = {'source_result': str(directory / 'result.json'),
              'source_success': True, 'source_seed': original['seed'],
              'replayed_targets': len(targets), 'video': str(output),
              'status': 'starting'}
    os.environ['ROBOTWIN_LAZY_BATCH_PLANNER'] = '1'
    os.chdir(ROOT / 'RoboTwin')
    sys.path.insert(0, str(ROOT / 'RoboTwin/scripts'))
    from eval_policy_xpolicylab import load_task_args, class_decorator
    task_args, _ = load_task_args(dict(task_name=original['task'],
                                       policy_name=original['project'],
                                       task_config='demo_clean'))
    task_args.update(eval_mode=True, save_data=False, eval_video_log=True,
                     eval_video_save_dir=str(video_dir), render_freq=0)
    env = class_decorator(original['task'])
    ffmpeg = None
    started = time.monotonic()
    try:
        env.setup_demo(now_ep_num=0, seed=original['seed'], is_test=True, **task_args)
        initial = env.get_obs()
        height, width = initial['observation']['head_camera']['rgb'].shape[:2]
        ffmpeg = subprocess.Popen([
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', '10', '-i', '-', '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264', '-crf', '23', str(output),
        ], stdin=subprocess.PIPE)
        env._set_eval_video_ffmpeg(ffmpeg)
        for target in targets:
            env.take_action(target, action_type='ee')
            env.get_obs()
            if env.eval_success or env.check_success():
                break
        record['official_success'] = bool(env.eval_success or env.check_success())
        record['status'] = 'completed' if record['official_success'] else 'replay_mismatch'
        record['executed_targets'] = env.take_action_cnt
        record['official_step_limit'] = env.step_lim
    except Exception as exc:
        record.update(status='error', error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if ffmpeg is not None:
            env._del_eval_video_ffmpeg()
        try:
            env.close_env()
        except Exception:
            pass
        record['elapsed_s'] = time.monotonic() - started
        (video_dir / 'replay_result.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: record.get(k) for k in
              ('status', 'official_success', 'video', 'executed_targets',
               'elapsed_s')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
