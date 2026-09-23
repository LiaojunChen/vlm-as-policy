"""Robodawn-style VLM embodiment harness.

This module turns a camera observation into a small, safe action vocabulary and
executes actions in a see-think-act-observe loop.  It is intentionally hardware
agnostic: applications provide an ObservationSource and ActionExecutor.  The
offline simulator makes the idea reproducible without a robot or API key.
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, field
from typing import Any, Protocol

ACTION_NAMES = ("pick", "place", "push", "pull", "fold", "noop", "done")

class ObservationSource(Protocol):
    def observe(self) -> dict[str, Any]: ...

class ActionExecutor(Protocol):
    def execute(self, action: dict[str, Any]) -> dict[str, Any]: ...

class Planner(Protocol):
    def plan(self, observation: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]: ...

@dataclass
class RulePlanner:
    """Deterministic baseline used for tests and prompt/harness debugging."""
    goal: str
    def plan(self, observation, history):
        if observation.get("success") or observation.get("folded"):
            return {"name": "done", "reason": "goal satisfied"}
        if observation.get("shirt") and observation.get("flat"):
            return {"name": "fold", "object": "shirt", "confidence": .8}
        return {"name": "noop", "reason": "insufficient visual evidence", "confidence": .2}

@dataclass
class SimulatedRobot:
    state: dict[str, Any] = field(default_factory=lambda: {"shirt": True, "flat": True, "folded": False})
    def observe(self):
        return dict(self.state)
    def execute(self, action):
        name = action.get("name")
        if name not in ACTION_NAMES:
            return {"ok": False, "error": "unknown action"}
        if name == "fold" and self.state.get("shirt"):
            self.state["folded"] = True; self.state["flat"] = False
        return {"ok": True, "action": name, "state": self.observe()}

def validate_action(action: dict[str, Any], allowed_actions=ACTION_NAMES) -> dict[str, Any]:
    if not isinstance(action, dict) or action.get("name") not in allowed_actions:
        raise ValueError("planner returned an unsupported action")
    return action

def run_loop(source: ObservationSource, executor: ActionExecutor, planner: Planner,
             max_steps: int = 8, delay_s: float = 0.0, *,
             stop_on_noop: bool = True, on_record=None, stop_on_error: bool = True,
             allowed_actions=ACTION_NAMES) -> list[dict[str, Any]]:
    history = []; trace = []
    for step in range(max_steps):
        obs = source.observe()
        action = validate_action(planner.plan(obs, history), allowed_actions)
        record = {"step": step, "observation": obs, "action": action}
        if action["name"] == "done" or (action["name"] == "noop" and stop_on_noop):
            record["result"] = {"ok": True, "terminal": action["name"] == "done"}
            trace.append(record)
            if on_record is not None: on_record(record)
            break
        result = executor.execute(action); record["result"] = result
        trace.append(record); history.append(record)
        if on_record is not None: on_record(record)
        if (stop_on_error and not result.get("ok")) or result.get("success"): break
        if delay_s: time.sleep(delay_s)
    return trace

def main():
    p = argparse.ArgumentParser(); p.add_argument("--goal", default="fold the shirt")
    p.add_argument("--max-steps", type=int, default=8); p.add_argument("--json", action="store_true")
    a = p.parse_args(); robot = SimulatedRobot()
    trace = run_loop(robot, robot, RulePlanner(a.goal), a.max_steps)
    print(json.dumps({"goal": a.goal, "success": bool(trace and trace[-1]["action"]["name"] == "done"), "trace": trace}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
