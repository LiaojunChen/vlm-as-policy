"""Keep resumable benchmark supervisors alive and package a verified 400/400 run."""
from __future__ import annotations
import fcntl, hashlib, json, os, signal, subprocess, tarfile, time
from pathlib import Path
from source_layout import policy_sources

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/full_50_v1'
PYTHON=ROOT/'.venv-robotwin310/bin/python'

def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return {}

def active_phase(phase):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(ROOT/'run_robotwin_suite.py') in args and '--phase' in args:
            if args[args.index('--phase')+1] in (phase,'all'):return True
    return False

def stop_phase(phase):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(ROOT/'run_robotwin_suite.py') in args and '--phase' in args:
            if args[args.index('--phase')+1]==phase:
                try:os.kill(int(p.name),signal.SIGTERM)
                except ProcessLookupError:pass

def handle_unvalidated_seed(task):
    """Keep full task coverage possible when the official expert rejects 30 scenes."""
    directory=OUT/'seeds'/task
    current=read(directory/'result.json')
    if current.get('status') in ('expert_validated','expert_unvalidated'):return
    paths=[directory/'seed_attempts.json',*(OUT/'seeds/_attempts'/task).glob('*/seed_attempts.json')]
    rejected={}
    for path in paths:
        try:attempts=json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):continue
        if isinstance(attempts,list):
            for row in attempts:
                if row.get('valid') is False:rejected[row['seed']]=row
    if len(rejected)<30:return
    try:
        pid=int((directory/'pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        if str(directory) in cmd and 'run_robotwin_episode.py' in cmd:
            os.killpg(pid,signal.SIGTERM)
            print('STOPPED_EXPERT_AFTER_30_REJECTIONS',task,pid,flush=True)
            return
    except (FileNotFoundError,ValueError,ProcessLookupError):pass
    if active_phase('seeds'):return
    description=read(ROOT/'RoboTwin/description/task_instruction'/f'{task}.json')['full_description']
    fallback={'task':task,'project':'expert','model':'none','seed':min(rejected),
              'status':'expert_unvalidated','success':None,'instruction':description,
              'task_config':'demo_clean','actual_rejected_expert_seeds':len(rejected),
              'selection_method':'first initializable scene after 30 actual expert rejections',
              'warning':'The official expert failed; this seed is used only for task coverage.'}
    if directory.exists():
        archive=OUT/'seeds/_attempts'/task/time.strftime('%Y%m%dT%H%M%S')
        archive.parent.mkdir(parents=True,exist_ok=True)
        # Never overwrite an existing archive if two events share a second.
        if archive.exists():archive=archive.with_name(archive.name+'_'+str(os.getpid()))
        directory.rename(archive)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'result.json').write_text(json.dumps(fallback,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXPERT_UNVALIDATED_FALLBACK',task,'seed',min(rejected),flush=True)
    # The existing episode supervisor imported an older seed-status contract.
    stop_phase('episodes')

def launch(phase):
    args=[str(PYTHON),'-u',str(ROOT/'run_robotwin_suite.py'),'--phase',phase,
          '--workers','1' if phase=='seeds' else '14']
    if phase=='episodes':args+=['--wait-seeds']
    with (OUT/f'{phase}_automatic_recovery.log').open('a',encoding='utf-8') as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    print('LAUNCHED',phase,p.pid,flush=True)

def finalize():
    subprocess.run([str(PYTHON),str(ROOT/'audit_robotwin_results.py')],check=True)
    evidence=OUT/'implementation';evidence.mkdir(exist_ok=True)
    sources=['benchmark_clients.py','robotwin_bridge.py','show_robotwin_policy.py',
             'robodawn_core.py','run_robotwin_episode.py','run_robotwin_suite.py',
             'report_robotwin_suite.py','audit_robotwin_results.py',
             'complete_robotwin_delivery.py','run_full_robotwin.sh',
             'FULL_ROBOTWIN_PROTOCOL.md','tests/test_bridge.py']
    sources += ['source_layout.py'] + [str(p.relative_to(ROOT)) for p in policy_sources(ROOT)]
    hashes={}
    for name in sources:
        data=(ROOT/name).read_bytes();target=evidence/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    (OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT/'RoboTwin')
    (OUT/'robotwin_worktree.patch').write_bytes(patch)
    (OUT/'curobo_worktree.patch').write_bytes(subprocess.check_output(
        ['git','diff','--binary'],cwd=ROOT/'RoboTwin/envs/curobo'))
    # Scan the actual configured credential without displaying or storing it.
    key=os.environ.get('VLM_API_KEY','').encode()
    if key:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.md','.py','.sh','.patch'):
                if key in p.read_bytes():raise RuntimeError('Credential found in delivery file: '+str(p))
    archive=OUT.parent/'full_50_v1_delivery.tar.gz'
    partial=archive.with_suffix('.gz.partial')
    with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
        tar.add(OUT,arcname=OUT.name)
        tar.add(ROOT/'results/20260916',arcname='20260916')
    partial.replace(archive)
    digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
    if digest is None:
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        digest=h.hexdigest()
    archive.with_suffix('.gz.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    (OUT/'DELIVERY_COMPLETE.json').write_text(json.dumps({'completed_episodes':400,'archive':str(archive),'sha256':digest},indent=2),encoding='utf-8')
    print('DELIVERY_COMPLETE',archive,digest,flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'coordinator.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (OUT/'coordinator.pid').write_text(str(os.getpid()),encoding='utf-8')
    while True:
        subprocess.run([str(PYTHON),str(ROOT/'report_robotwin_suite.py')],check=True)
        summary=read(OUT/'summary.json')
        if summary.get('completed_episodes')==400:
            finalize();return
        tasks=read(OUT/'manifest.json')['tasks']
        for task in tasks:handle_unvalidated_seed(task)
        if any(read(OUT/'seeds'/t/'result.json').get('status') not in ('expert_validated','expert_unvalidated') for t in tasks):
            if not active_phase('seeds'):launch('seeds')
        if not active_phase('episodes'):launch('episodes')
        time.sleep(30)

if __name__=='__main__':main()
