"""Finish the already-running single-model suite, retry known infrastructure errors,
then audit results and produce verified replay videos. Never launches Codex.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-robotwin310/bin/python'
p=argparse.ArgumentParser()
p.add_argument('--wait-pid',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve()
state_path=out/'completion_state.json'

def save(**data):
    state_path.write_text(json.dumps(data,ensure_ascii=False,indent=2))

def suite_alive():
    try:
        return b'run_robotwin_suite_v2.py' in Path(f'/proc/{a.wait_pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False

save(stage='waiting_for_existing_suite',pid=a.wait_pid)
while suite_alive():time.sleep(15)
for attempt in range(2):
    rows=[json.loads(f.read_text()) for f in out.glob('*/*/*/result.json')]
    errors=[r for r in rows if r.get('status') in ('error','process_error')]
    if not errors:break
    known=all(r.get('error_type')=='TypeError' and 'NoneType' in r.get('error','') for r in errors)
    if not known:
        save(stage='needs_error_review',errors=errors)
        break
    save(stage='retrying_fixed_replan_error',attempt=attempt+1,episodes=len(errors))
    subprocess.run([str(PYTHON),str(ROOT/'run_robotwin_suite_v2.py'),
                    '--output',str(out),'--models','qwen3-vl-plus',
                    '--workers','4','--max-steps','30'],check=True,cwd=ROOT)

subprocess.run([str(PYTHON),str(ROOT/'report_qwen_v2.py'),
                '--output',str(out)],check=True,cwd=ROOT)
for path in sorted(out.glob('*/*/*/result.json')):
    result=json.loads(path.read_text())
    if result.get('success') is not True:continue
    replay=path.parent/'videos/replay_result.json'
    if replay.exists() and json.loads(replay.read_text()).get('official_success') is True:
        continue
    save(stage='replaying_success',episode=str(path.parent))
    subprocess.run([str(PYTHON),str(ROOT/'replay_success_v2.py'),
                    '--episode',str(path.parent)],check=True,cwd=ROOT)
    video=path.parent/'videos/official_replay.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(video),
                    '-vf','setpts=10*PTS','-r','10','-pix_fmt','yuv420p',
                    '-c:v','libx264','-crf','20',str(video.with_name('official_replay_slow.mp4'))],check=True)
rows=[json.loads(f.read_text()) for f in out.glob('*/*/*/result.json')]
complete=sum(r.get('status')=='completed' for r in rows)
save(stage='complete' if complete==100 else 'needs_error_review',
     completed=complete,requested=100,success=sum(r.get('success') is True for r in rows))
