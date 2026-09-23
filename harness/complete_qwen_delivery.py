"""Resume only Qwen episodes, then audit and package the 300-result delivery."""
from __future__ import annotations
import fcntl,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
from source_layout import policy_sources
from run_robotwin_suite import MODELS,PROJECTS,read

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/full_50_v1'
PYTHON=ROOT/'.venv-robotwin310/bin/python'
QWEN=[model for model in MODELS if model!='codex']

def suite_running():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().decode().split('\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(ROOT/'run_robotwin_suite.py') in args and '--phase' in args:
   if args[args.index('--phase')+1] in ('episodes','all'):return True
 return False

def start_suite():
 args=[str(PYTHON),'-u',str(ROOT/'run_robotwin_suite.py'),'--phase','episodes',
       '--workers','14','--wait-seeds','--models',*QWEN]
 with (OUT/'qwen_supervisor.log').open('a',encoding='utf-8') as log:
  process=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
 print('STARTED_QWEN_SUPERVISOR',process.pid,flush=True)

def audit_and_package():
 audit=subprocess.run([str(PYTHON),str(ROOT/'audit_robotwin_results.py')],
                      capture_output=True,text=True,encoding='utf-8')
 if audit.returncode:raise RuntimeError('RoboTwin audit failed: '+audit.stdout[-1500:]+audit.stderr[-1500:])
 (OUT/'qwen_audit.json').write_bytes((OUT/'audit.json').read_bytes())
 audit_data=read(OUT/'qwen_audit.json')
 if audit_data.get('completed_episodes_checked')!=300 or audit_data.get('issues'):
  raise RuntimeError('Qwen audit incomplete or failed: '+str(audit_data))
 sources=['benchmark_clients.py','robotwin_bridge.py','show_robotwin_policy.py',
          'robodawn_core.py','run_robotwin_episode.py','run_robotwin_suite.py',
          'report_qwen_suite.py','audit_robotwin_results.py','complete_qwen_delivery.py',
          'FULL_ROBOTWIN_PROTOCOL.md','tests/test_bridge.py']
 implementation=OUT/'qwen_implementation';implementation.mkdir(exist_ok=True)
 sources += ['source_layout.py'] + [str(p.relative_to(ROOT)) for p in policy_sources(ROOT)]
 hashes={}
 for name in sources:
  data=(ROOT/name).read_bytes();target=implementation/name
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  hashes[name]=hashlib.sha256(data).hexdigest()
 (OUT/'qwen_source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
 (OUT/'qwen_robotwin_worktree.patch').write_bytes(subprocess.check_output(
     ['git','diff','--binary'],cwd=ROOT/'RoboTwin'))
 (OUT/'qwen_curobo_worktree.patch').write_bytes(subprocess.check_output(
     ['git','diff','--binary'],cwd=ROOT/'RoboTwin/envs/curobo'))
 paths=[OUT/'QWEN_REPORT.md',OUT/'qwen_episodes.csv',OUT/'qwen_summary.json',
        OUT/'qwen_audit.json',OUT/'qwen_source_sha256.json',
        OUT/'qwen_robotwin_worktree.patch',OUT/'qwen_curobo_worktree.patch',
        OUT/'qwen_implementation',OUT/'manifest.json',OUT/'environment.json',
        OUT/'validation_checks.json',
        OUT/'concurrency.json',OUT/'model_limits.json',
        OUT/'qwen_rate_limit_adjustment.json',OUT/'qwen_flash_rate_adjustment.json',
        OUT/'qwen_flash_rate_recovery.json',OUT/'qwen_throughput_trial.json',
        OUT/'seeds',ROOT/'results/20260916']
 tasks=read(OUT/'manifest.json')['tasks']
 paths.extend(OUT/project/model/task for task in tasks for project in PROJECTS for model in QWEN)
 paths.extend(OUT/project/model/'_attempts' for project in PROJECTS for model in QWEN
              if (OUT/project/model/'_attempts').exists())
 key=os.environ.get('VLM_API_KEY','').encode()
 if key:
  for path in paths:
   files=path.rglob('*') if path.is_dir() else [path]
   for file in files:
    if file.is_file() and file.suffix in ('.json','.jsonl','.log','.txt','.md','.py','.patch'):
     if key in file.read_bytes():raise RuntimeError('Credential found in delivery file '+str(file))
 marker=OUT/'QWEN_DELIVERY_COMPLETE.json'
 marker.write_text(json.dumps({'completed_episodes':300,'audited_episodes':300,
   'models':QWEN,'codex_paused':True},indent=2),encoding='utf-8')
 paths.append(marker)
 archive=OUT.parent/'robotwin_qwen_300_delivery.tar.gz'
 partial=archive.with_suffix('.gz.partial')
 with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
  for path in paths:
   if path.exists():
    arcname=str(path.relative_to(ROOT/'results')) if path.is_relative_to(ROOT/'results') else path.name
    tar.add(path,arcname=arcname)
 partial.replace(archive)
 digest=hashlib.sha256()
 with archive.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
 archive.with_suffix('.gz.sha256').write_text(digest.hexdigest()+'  '+archive.name+'\n',encoding='utf-8')
 print('QWEN_DELIVERY_COMPLETE',archive,digest.hexdigest(),flush=True)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'qwen_coordinator.lock').open('w')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'qwen_coordinator.pid').write_text(str(os.getpid()),encoding='utf-8')
 while True:
  report=subprocess.run([str(PYTHON),str(ROOT/'report_qwen_suite.py')],
                        capture_output=True,text=True,encoding='utf-8')
  if report.returncode:raise RuntimeError(report.stderr[-1500:])
  print(report.stdout.strip(),flush=True)
  summary=read(OUT/'qwen_summary.json')
  if summary.get('completed_episodes')==300:
   audit_and_package();return
  if not suite_running():start_suite()
  time.sleep(30)

if __name__=='__main__':main()
