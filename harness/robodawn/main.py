from __future__ import annotations
import argparse, json, os, subprocess, urllib.request
from dataclasses import dataclass, field
from typing import Any

ACTIONS = {"pick", "place", "push", "pull", "fold", "noop", "done"}
DEMO = {"observation": {"shirt": True, "flat": True}, "action": {"name": "fold", "object": "shirt"}, "result": {"folded": True}}

@dataclass
class RobotWinLikeEnv:
    state: dict[str, Any] = field(default_factory=lambda: {"shirt": True, "flat": True, "folded": False})
    def reset(self):
        self.state = {"shirt": True, "flat": True, "folded": False}; return self.observe()
    def observe(self): return dict(self.state)
    def step(self, action):
        name = action.get("name")
        if name not in ACTIONS: return self.observe(), {"ok": False, "error": "invalid action"}, True
        if name == "fold" and self.state["shirt"] and self.state["flat"]: self.state.update(folded=True, flat=False)
        done = name == "done" or self.state["folded"]
        return self.observe(), {"ok": True, "action": name, "folded": self.state["folded"]}, done

class MockPlanner:
    def plan(self, obs, history): return {"name": "done" if obs.get("folded") else "fold", "object": "shirt"}

class OpenAIPlanner:
    def __init__(self, endpoint, model, key, style="chat", timeout=45): self.endpoint, self.model, self.key, self.style, self.timeout = endpoint.rstrip("/"), model, key, style, timeout
    def _prompt(self, obs, history):
        return ("You control a simple RobotWin environment. Return JSON only. Allowed action names: " + ",".join(sorted(ACTIONS)) +
                "\nGoal: fold the shirt.\nSuccessful demonstration: " + json.dumps(DEMO) +
                "\nCurrent observation: " + json.dumps(obs) + "\nHistory: " + json.dumps(history[-3:]) +
                "\nReturn {\"name\": string, \"object\": string, \"reason\": string}.")
    def plan(self, obs, history):
        prompt = self._prompt(obs, history)
        if self.style == "responses":
            body = {"model": self.model, "input": prompt, "temperature": 0}
            url = self.endpoint + "/responses"
        else:
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
            url = self.endpoint + "/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json", "Authorization":"Bearer "+self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r: data = json.loads(r.read())
        text = data.get("output_text") or data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return parse_action(text)

class CodexPlanner:
    def __init__(self, binary="codex"): self.binary = binary
    def plan(self, obs, history):
        prompt = "Return JSON only with name in fold,done,noop. Goal fold shirt. Observation=" + json.dumps(obs)
        p = subprocess.run([self.binary, "exec", prompt], text=True, capture_output=True, check=True)
        return parse_action(p.stdout)

def parse_action(text):
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    try: value = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}"); value = json.loads(text[a:b+1])
    if value.get("name") not in ACTIONS: raise ValueError("VLM returned unsupported action")
    return value

def run(provider="mock", max_steps=8):
    env = RobotWinLikeEnv(); planner = MockPlanner()
    if provider == "openai": planner = OpenAIPlanner(os.environ.get("VLM_ENDPOINT", "https://api.openai.com/v1"), os.environ.get("VLM_MODEL", "gpt-4o-mini"), os.environ["VLM_API_KEY"], os.environ.get("VLM_API_STYLE", "chat"))
    elif provider == "codex": planner = CodexPlanner(os.environ.get("CODEX_BIN", "codex"))
    elif provider == "auto" and os.environ.get("VLM_API_KEY"): planner = OpenAIPlanner(os.environ.get("VLM_ENDPOINT", "https://api.openai.com/v1"), os.environ.get("VLM_MODEL", "gpt-4o-mini"), os.environ["VLM_API_KEY"], os.environ.get("VLM_API_STYLE", "chat"))
    trace=[]; history=[]
    for step in range(max_steps):
        obs=env.observe(); action=planner.plan(obs, history); new_obs,result,done=env.step(action)
        item={"step":step,"observation":obs,"action":action,"result":result}; trace.append(item); history.append(item)
        if done: break
    return {"provider":provider,"success":bool(trace and trace[-1]["result"].get("folded")),"trace":trace}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--provider", choices=["mock","openai","codex","auto"], default="auto"); p.add_argument("--max-steps", type=int, default=8); a=p.parse_args(); print(json.dumps(run(a.provider,a.max_steps), ensure_ascii=False, indent=2))
