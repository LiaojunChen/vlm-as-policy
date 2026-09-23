"""Initial-image planning diagnostic only; no environment or scoring access."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_clients import AuditedClient
from robodawn.mission_planner import RobodawnPlanner
from source_layout import evaluation_sources, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in evaluation_sources()}
    (args.output / 'code_hashes.json').write_text(json.dumps(hashes, indent=2))
    for task in args.tasks:
        source = args.cohort / 'robodawn' / task
        prior = json.loads((source / 'result.json').read_text())
        instruction = prior['instruction'] + '\nCompletion requirements: ' + prior['task_contract']
        client = AuditedClient('zdtaichu', args.output / task / 'calls')
        obs = dict(observation_id=1,sensors={side: dict(holding=False) for side in ('left', 'right')},
                   image_paths=[str(source / 'frames/0001_head_camera.png')])
        planner = RobodawnPlanner(client, instruction)
        record = dict(diagnostic_only=True, source_observation=obs['image_paths'][0])
        try:
            planner._replan(obs, [])
            record['steps'] = planner.steps
        except Exception as exc:
            record.update(error=str(exc), error_type=type(exc).__name__)
        record['model_calls'] = client.calls
        (args.output / task / 'preflight.json').write_text(json.dumps(record, indent=2))
        print(task, json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
