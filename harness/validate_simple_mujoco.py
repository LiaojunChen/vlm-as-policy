"""Negative controls: lifting open fingers or missing the cube must not score."""
import json
from pathlib import Path
from simple_mujoco_grasp import ROOT, SimpleGrasp

out = ROOT / 'results/simple_mujoco_v1/negative_controls'
rows = []
for name, offset, close in [('open_fingers', 0., False), ('miss_cube', .12, True)]:
    directory = out / name
    if (directory / 'result.json').exists():
        rows.append(json.loads((directory / 'result.json').read_text()))
        continue
    env = SimpleGrasp(0, directory)
    point = env.data.body('cube').xpos.copy()
    x, y = point[0] + offset, point[1]
    env.move([x, y, .86], 1, 1)
    env.move([x, y, .724], 1, 1)
    env.move(opening=0 if close else 1, duration=1)
    env.move([x, y, .90], duration=1)
    env.move(duration=1)
    row = {'control': name, 'success': env.success,
           'max_lift_m': env.max_lift_m, 'hold_s': env.hold_s}
    env.close()
    (directory / 'result.json').write_text(json.dumps(row, indent=2))
    rows.append(row)
print(json.dumps(rows, indent=2))
assert len(rows) == 2 and all(not row['success'] for row in rows)
