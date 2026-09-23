"""Regression coverage for package isolation and source freezing after the split."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def isolated_python(code, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


def test_robodawn_imports_do_not_load_showharness(tmp_path):
    isolated_python('''
import sys
from robodawn.core import RulePlanner, SimulatedRobot, run_loop
from robodawn.planner_v1 import RobodawnPlanner as V1
from robodawn.planner_v2 import RobodawnGroundedPlanner
from robodawn.planner_v3 import RobodawnPlanner as V3
from robodawn.reviewed import ReviewedRobodawnPlanner
robot = SimulatedRobot()
assert run_loop(robot, robot, RulePlanner('fold'))[-1]['action']['name'] == 'done'
assert not any(n.split('.')[0] in ('showharness', 'core', 'plugins') for n in sys.modules)
''', tmp_path)


@pytest.mark.parametrize('module, symbol', [
    ('planner_v3', 'ShowPlanner'),
    ('policy_v1', 'ShowPolicy'),
    ('policy_v2', 'ShowGroundedPolicy'),
    ('reviewed', 'ReviewedShowPolicy'),
])
def test_showharness_imports_without_client_initialization(module, symbol, tmp_path):
    isolated_python(f'''
import sys
from showharness.{module} import {symbol}
from showharness.runtime import UPSTREAM_ROOT, ensure_upstream
ensure_upstream()
ensure_upstream()
assert sys.path.count(str(UPSTREAM_ROOT)) == 1
assert (UPSTREAM_ROOT / 'prompts/controller_dual.txt').is_file()
assert not any(n == 'robodawn' or n.startswith('robodawn.') for n in sys.modules)
''', tmp_path)


def test_legacy_imports_preserve_class_identity():
    from robodawn.planner_v3 import RobodawnPlanner
    from robotwin_harness_v3 import RobodawnPlanner as LegacyPlanner
    from showharness.planner_v3 import ShowPlanner
    from show_harness_v3 import ShowPlanner as LegacyShowPlanner
    from robodawn.planner_v1 import RobodawnPlanner as V1
    from robotwin_bridge import RobodawnPlanner as LegacyV1
    from robodawn.planner_v2 import RobodawnGroundedPlanner
    from robotwin_policies_v2 import RobodawnGroundedPlanner as LegacyV2
    from showharness.reviewed import ReviewedShowPolicy
    from reviewed_interface import ReviewedShowPolicy as LegacyReviewed
    assert RobodawnPlanner is LegacyPlanner
    assert ShowPlanner is LegacyShowPlanner
    assert V1 is LegacyV1
    assert RobodawnGroundedPlanner is LegacyV2
    assert ReviewedShowPolicy is LegacyReviewed


def test_frozen_sources_include_moved_code_and_prompt_assets():
    from source_layout import evaluation_sources
    sources = evaluation_sources()
    assert len(sources) == len(set(sources))
    assert all(path.is_file() for path in sources)
    assert set((ROOT / 'robodawn').glob('*.py')) <= set(sources)
    assert set((ROOT / 'showharness').glob('*.py')) <= set(sources)
    assert ROOT / 'showharness/upstream/core/vlm/dual_roles.py' in sources
    assert ROOT / 'showharness/upstream/prompts/controller_dual.txt' in sources
