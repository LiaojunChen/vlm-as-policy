"""Source inventory for cohort freezing and implementation snapshots."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def policy_sources(root=ROOT):
    """Include both policy packages and upstream code/configuration, not caches."""
    suffixes = {'.py', '.txt', '.json', '.yaml', '.yml', '.j2'}
    return sorted(
        path
        for package in ('robodawn', 'showharness')
        for path in (root / package).rglob('*')
        if path.is_file() and path.suffix in suffixes
        and not any(part.startswith('.') or part == '__pycache__'
                    for part in path.relative_to(root).parts)
    )


def evaluation_sources(root=ROOT):
    """All harness modules plus simulator files customized for this evaluation."""
    return sorted(set(root.glob('*.py')) | set((root / 'runtime').glob('*.py'))
                  | set((root / 'runtime/grammar_vendor').rglob('*.py'))
                  | set((root / 'runtime').glob('grammar_dependencies.json'))
                  | set((root / 'RoboTwin/assets/embodiments').rglob('*.yml'))
                  | set((root / 'RoboTwin/envs').glob('*.py'))
                  | set((root / 'RoboTwin/data').glob('*.py'))
                  | set((root / 'RoboTwin/envs/utils').glob('*.py'))
                  | set((root / 'RoboTwin/env_cfg/task_config').glob('*.yml'))
                  | set((root / 'seeds').glob('*/result.json'))
                  | set(policy_sources(root)) | {
        root / 'runtime/env.sh',
        root / 'runtime/nvidia_icd.json',
        root / 'RoboTwin/envs/robot/planner.py',
        root / 'RoboTwin/env_cfg/task_config/_camera_config.yml',
        root / 'RoboTwin/scripts/eval_policy_xpolicylab.py',
    })
