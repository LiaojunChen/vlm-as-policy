"""Finish the Codex CLI / Astra Show-Harness cohort and audit its delivery."""
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL = BASE / "robodawn_robotwin_harness_astra_show_eval"
OUT = EVAL / "results/full_50"
PILOT = EVAL / "results/codex_integration_click_bell"
PYTHON = BASE / "robodawn_robotwin/.venv-robotwin310/bin/python"
STATE = EVAL / "ASTRA_DELIVERY_STATUS.json"
MODEL = "gpt-6-astra"
EFFORT = "high"


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def status(stage, **kwargs):
    data = dict(stage=stage, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model=MODEL, reasoning_effort=EFFORT, transport="codex_cli", project="show_harness", **kwargs)
    write(STATE, data)
    print("ASTRA_DELIVERY", stage, flush=True)


def code_integrity():
    frozen = read(OUT / "code_hashes.json")
    changed = [name for name, digest in frozen.items()
               if hashlib.sha256((EVAL / name).read_bytes()).hexdigest() != digest]
    return len(frozen), changed


def audited_calls(directory):
    issues = []
    for response_path in (directory / "calls").glob("call_*.response.json"):
        data = read(response_path).get("response", {})
        if (data.get("model"), data.get("reasoning_effort"), data.get("transport")) != (MODEL, EFFORT, "codex_cli"):
            issues.append(str(response_path.relative_to(OUT)) + ": model/config mismatch")
        event_path = response_path.with_name(response_path.name.replace(".response.json", ".codex.jsonl"))
        if not event_path.exists():
            issues.append(str(response_path.relative_to(OUT)) + ": missing CLI event log")
    return issues


def deliver():
    manifest = read(OUT / "manifest.json")
    if len(manifest.get("jobs", [])) != 50 or manifest.get("projects") != ["show_harness"]:
        raise RuntimeError("Expected exactly 50 Show-Harness jobs")
    rows, issues = [], []
    for job in manifest["jobs"]:
        task = job["task"]
        initial = OUT / "show_harness" / task
        initial_result = read(initial / "result.json")
        evidence, result = initial, initial_result
        if result.get("status") == "error":
            retry = OUT / "infrastructure_retries" / "show_harness" / task
            if not (retry / "result.json").exists():
                retry.mkdir(parents=True, exist_ok=True)
                command = [str(PYTHON), "-u", str(EVAL / "run_harness_v3.py"), "--task", task,
                           "--project", "show_harness", "--max-steps", "30", "--wide-head",
                           "--output", str(retry)]
                with (retry / "process.log").open("w") as log:
                    proc = subprocess.Popen(command, cwd=EVAL, env=os.environ.copy(), stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        proc.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=30)
                        write(retry / "result.json", dict(task=task, project="show_harness", model=MODEL,
                            reasoning_effort=EFFORT, status="completed", success=False,
                            end_reason="wall_time_budget", retry_of=str(initial.relative_to(OUT))))
            result = read(retry / "result.json")
            evidence = retry
        if result.get("model") != MODEL or result.get("reasoning_effort") != EFFORT:
            issues.append(task + ": wrong underlying model/effort")
        if result.get("status") not in ("completed", "error"):
            issues.append(task + ": nonterminal episode")
        issues.extend(audited_calls(initial))
        if evidence != initial:
            issues.extend(audited_calls(evidence))
        video = evidence / "continuous.mp4"
        if not video.exists() or video.stat().st_size == 0:
            issues.append(task + ": missing/empty video")
        rows.append(dict(task=task, project="show_harness", model=MODEL, reasoning_effort=EFFORT,
                         transport="codex_cli", seed=result.get("seed"), status=result.get("status"),
                         success=result.get("success"), end_reason=result.get("end_reason"),
                         policy_decisions=result.get("policy_decisions"), model_calls=result.get("model_calls"),
                         elapsed_s=result.get("elapsed_s"), evidence=str(evidence.relative_to(OUT)),
                         retried_infrastructure=evidence != initial))
        status("auditing", reviewed=len(rows), total=50)
    file_count, changed = code_integrity()
    issues.extend("changed frozen source: " + name for name in changed)
    summary = dict(model=MODEL, reasoning_effort=EFFORT, transport="codex_cli", project="show_harness",
                   requested=50, completed=sum(r["status"] == "completed" for r in rows),
                   successes=sum(r["success"] is True for r in rows),
                   errors=sum(r["status"] == "error" for r in rows),
                   success_rate_over_50=sum(r["success"] is True for r in rows) / 50,
                   head_resolution=[960, 720], head_fov_deg=60,
                   policy_decision_budget=30, wall_budget_s=3600,
                   frozen_source_files_checked=file_count, integrity_issues=issues, episodes=rows,
                   finalized_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write(OUT / "DELIVERY_SUMMARY.json", summary)
    with (OUT / "DELIVERY_RESULTS.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Codex CLI / GPT-6 Astra high / Show-Harness RoboTwin results", "",
             f"Official success: **{summary['successes']}/50 ({summary['success_rate_over_50']:.0%})**.",
             f"Completed: {summary['completed']}/50. Errors: {summary['errors']}. Integrity issues: {len(issues)}.",
             "", "Each task has one initial episode. Only infrastructure errors are retried; initial evidence is retained.",
             "Development seeds and a 960×720, 60° head camera; not a held-out or original-camera result.",
             "", "| Task | Status | Success | Decisions | Evidence |", "|---|---|---:|---:|---|"]
    for row in rows:
        path = row["evidence"]
        lines.append(f"| {row['task']} | {row['status']} | {row['success']} | {row['policy_decisions']} | "
                     f"[result]({path}/result.json) · [trace]({path}/trace.jsonl) · [video]({path}/continuous.mp4) |")
    if issues:
        lines += ["", "## Integrity issues", ""] + ["- " + issue for issue in issues]
    (OUT / "DELIVERY_REPORT.md").write_text("\n".join(lines) + "\n")
    status("finished", successes=summary["successes"], completed=summary["completed"],
           errors=summary["errors"], integrity_issues=len(issues))


def main():
    status("waiting_for_integration_episode")
    while True:
        result = read(PILOT / "show_harness/click_bell/result.json")
        if result.get("status") in ("completed", "error"):
            break
        time.sleep(30)
    if result["status"] == "error":
        raise RuntimeError("Integration episode failed; inspect pilot before full cohort")
    status("starting_full_50", pilot_success=result.get("success"))
    tasks = sorted(path.name for path in (EVAL / "seeds").iterdir() if path.is_dir())
    tasks = [task for task in ("click_bell", "press_stapler", "move_stapler_pad", "place_empty_cup") if task in tasks] + \
        [task for task in tasks if task not in ("click_bell", "press_stapler", "move_stapler_pad", "place_empty_cup")]
    if len(tasks) != 50:
        raise RuntimeError("Expected 50 official task seeds")
    command = [str(PYTHON), "-u", str(EVAL / "run_cohort_v3.py"), "--output", str(OUT),
               "--projects", "show_harness", "--workers", "4", "--max-steps", "30", "--timeout", "3600",
               "--wide-head", "--tasks", *tasks]
    with (EVAL / "full_50.supervisor.log").open("a") as log:
        proc = subprocess.Popen(command, cwd=EVAL, env=os.environ.copy(), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        status("running_full_50", supervisor_pid=proc.pid)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Full cohort supervisor exited with status {code}")
    deliver()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("error", error=str(exc))
        raise
