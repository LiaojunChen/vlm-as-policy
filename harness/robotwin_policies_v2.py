"""Compatibility exports for the separated v2 policies."""


def __getattr__(name):
    """Resolve legacy policy imports without coupling shared code to a policy."""
    from importlib import import_module
    exports = {'RobodawnGroundedPlanner': 'robodawn.planner_v2', 'ROBO_CODES': 'robodawn.planner_v2', 'SKILLS': 'robodawn.planner_v2', 'ShowGroundedPolicy': 'showharness.policy_v2'}
    if name in exports:
        return getattr(import_module(exports[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
