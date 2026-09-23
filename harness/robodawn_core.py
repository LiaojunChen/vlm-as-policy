"""Compatibility entry point; implementation lives in robodawn.core."""
from robodawn.core import (ACTION_NAMES, ObservationSource, ActionExecutor, Planner,
                           RulePlanner, SimulatedRobot, validate_action, run_loop, main)

if __name__ == "__main__":
    main()
