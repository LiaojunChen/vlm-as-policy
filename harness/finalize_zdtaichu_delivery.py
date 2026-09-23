"""Audit a completed RoboDawn-only ZDTaichu RoboTwin cohort and package its evidence."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from run_suite_v3 import read

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Delivery output must stay in the harness directory")
    manifest = read(out / "manifest.json")
    rows, issues = [], []
    for job in manifest["jobs"]:
        directory = out / job["project"] / job["task"]
        result = read(directory / "result.json")
        if result.get("status") not in ("completed", "error"):
            raise RuntimeError(f"Non-terminal episode: {job}")
        if result.get("model") != "zdtaichu":
            issues.append(f"{job['task']}: incorrect model label")
        trace_file = directory / "trace.jsonl"
        trace = [json.loads(line) for line in trace_file.read_text().splitlines()] if trace_file.exists() else []
        if len(trace) != result.get("policy_decisions", 0):
            issues.append(f"{job['task']}: trace/decision count mismatch")
        calls = sorted((directory / "calls").glob("*.response.json"))
        usage, latency = collections.Counter(), 0.0
        for call in calls:
            response = read(call)
            payload = response.get("response", {})
            if payload.get("model") != "zdtaichu":
                issues.append(f"{job['task']}: endpoint response model mismatch")
            usage.update(payload.get("usage", {}))
            latency += response.get("elapsed_s", 0)
        video = directory / "continuous.mp4"
        if not video.exists() or video.stat().st_size == 0:
            issues.append(f"{job['task']}: missing or empty video")
        rows.append({**result, "task": job["task"], "project": job["project"],
            "response_count": len(calls), "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"], "model_latency_s": round(latency, 3),
            "evidence": str(directory.relative_to(out))})
    for name, expected in read(out / "code_hashes.json").items():
        if sha256(ROOT / name) != expected:
            issues.append(f"frozen source changed: {name}")
    integrity = read(ROOT / "EVALUATION_INTEGRITY.json")
    for name, record in integrity.get("task_source_hashes", {}).items():
        if sha256(ROOT / name) != record["original"]:
            issues.append(f"official task source changed: {name}")
    totals = {"requested": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
        "successes": sum(r.get("success") is True for r in rows),
        "infrastructure_errors": sum(r["status"] == "error" for r in rows),
        "success_rate_all_requested": sum(r.get("success") is True for r in rows) / len(rows),
        "success_rate_completed": sum(r.get("success") is True for r in rows) / max(1, sum(r["status"] == "completed" for r in rows)),
        "end_reasons": dict(collections.Counter(r.get("end_reason", "unknown") for r in rows)),
        "model_calls": sum(r.get("model_calls", 0) for r in rows),
        "median_episode_s": statistics.median(r.get("elapsed_s", 0) for r in rows),
        "audit_issues": issues}
    (out / "delivery_summary.json").write_text(json.dumps({"totals": totals, "protocol": manifest, "episodes": rows}, ensure_ascii=False, indent=2))
    columns = ["task", "project", "model", "seed", "status", "success", "policy_decisions", "model_calls", "end_reason", "elapsed_s", "prompt_tokens", "completion_tokens", "model_latency_s", "evidence"]
    with (out / "delivery_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    report = ["# ZDTaichu5.0-9B + RoboDawn v3: RoboTwin delivery", "",
        f"Official successes: {totals['successes']}/{totals['requested']} ({totals['success_rate_all_requested']:.1%}).", "",
        "Protocol: 50 official tasks, one prevalidated development seed each, demo_clean, original 320px head camera, RGB-D/proprioception/contact, official success checker, 30 policy decisions and 1200 seconds per episode.",
        "Model: phyRSI/ZDTaichu5.0-9B, BF16, deterministic decoding (temperature 0), enable_thinking=False.",
        f"Completed: {totals['completed']}; infrastructure errors: {totals['infrastructure_errors']}; model calls: {totals['model_calls']}.",
        f"Audit issues: {len(issues)}.", "", "| Task | Success | Decisions | End reason | Evidence |", "|---|---:|---:|---|---|"]
    for row in rows:
        report.append(f"| {row['task']} | {row.get('success')} | {row.get('policy_decisions', 0)} | {row.get('end_reason', '')} | [{row['evidence']}]({row['evidence']}/result.json) |")
    (out / "DELIVERY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    archive = out.parent / f"{out.name}_delivery.tar.gz"
    partial = archive.with_suffix(".tar.gz.partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        tar.add(out, arcname=out.name)
    partial.replace(archive)
    digest = sha256(archive)
    archive.with_suffix(".tar.gz.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    (out / "DELIVERY_COMPLETE.json").write_text(json.dumps({"archive": str(archive), "sha256": digest, "audit_issues": issues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
