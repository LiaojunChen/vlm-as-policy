"""Audit and package the PhysBrain RoboDawn-only RoboTwin cohort."""
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
EXPECTED_MODEL = "physbrain1.5-8b-local"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Delivery output must stay in the harness directory")
    manifest = read(out / "manifest.json")
    if manifest.get("model") != EXPECTED_MODEL:
        raise ValueError(f"Unexpected cohort model: {manifest.get('model')}")

    rows, issues, videos = [], [], []
    for job in manifest["jobs"]:
        directory = out / job["project"] / job["task"]
        result = read(directory / "result.json")
        if result.get("status") not in ("completed", "error"):
            raise RuntimeError(f"Non-terminal episode: {job}")
        if result.get("model") != EXPECTED_MODEL:
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
            if payload.get("model") != EXPECTED_MODEL:
                issues.append(f"{job['task']}: endpoint response model mismatch")
            if not payload.get("constrained_json_schema"):
                issues.append(f"{job['task']}: unconstrained response in {call.name}")
            usage.update(payload.get("usage", {}))
            latency += response.get("elapsed_s", 0)
        video = directory / "continuous.mp4"
        if not video.exists() or video.stat().st_size == 0:
            issues.append(f"{job['task']}: missing or empty video")
        else:
            videos.append((job["task"], video))
        rows.append({**result, "task": job["task"], "project": job["project"],
            "response_count": len(calls), "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"], "model_latency_s": round(latency, 3),
            "video": str(video.relative_to(out)), "evidence": str(directory.relative_to(out))})

    for name, expected in read(out / "code_hashes.json").items():
        if sha256(ROOT / name) != expected:
            issues.append(f"frozen source changed: {name}")
    integrity = read(ROOT / "EVALUATION_INTEGRITY.json")
    for name, record in integrity.get("task_source_hashes", {}).items():
        if sha256(ROOT / name) != record["original"]:
            issues.append(f"official task source changed: {name}")

    completed = sum(row["status"] == "completed" for row in rows)
    successes = sum(row.get("success") is True for row in rows)
    totals = {"requested": len(rows), "completed": completed, "successes": successes,
        "infrastructure_errors": sum(row["status"] == "error" for row in rows),
        "success_rate_all_requested": successes / max(1, len(rows)),
        "success_rate_completed": successes / max(1, completed),
        "end_reasons": dict(collections.Counter(row.get("end_reason", "unknown") for row in rows)),
        "model_calls": sum(row.get("model_calls", 0) for row in rows),
        "median_episode_s": statistics.median(row.get("elapsed_s", 0) for row in rows),
        "videos": len(videos), "audit_issues": issues}
    (out / "delivery_summary.json").write_text(json.dumps(
        {"totals": totals, "protocol": manifest, "episodes": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["task", "project", "model", "seed", "status", "success", "policy_decisions",
        "model_calls", "end_reason", "elapsed_s", "prompt_tokens", "completion_tokens",
        "model_latency_s", "video", "evidence"]
    with (out / "delivery_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    report = ["# PhysBrain1.5-8B + RoboDawn v3: RoboTwin delivery", "",
        f"Official successes: {successes}/{len(rows)} ({totals['success_rate_all_requested']:.1%}).", "",
        "Protocol: 50 official tasks, one prevalidated development seed each, demo_clean, original 320px head camera, RGB-D/proprioception/contact, official success checker, 30 policy decisions and 1200 seconds per episode.",
        f"Model: {manifest['model_checkpoint']}, BF16, deterministic decoding (temperature 0), JSON-schema constrained actions.",
        f"Completed: {completed}; infrastructure errors: {totals['infrastructure_errors']}; model calls: {totals['model_calls']}; videos: {len(videos)}.",
        f"Audit issues: {len(issues)}.", "", "| Task | Success | Decisions | End reason | Result | Video |",
        "|---|---:|---:|---|---|---|"]
    for row in rows:
        report.append(f"| {row['task']} | {row.get('success')} | {row.get('policy_decisions', 0)} | {row.get('end_reason', '')} | [JSON]({row['evidence']}/result.json) | [MP4]({row['video']}) |")
    (out / "DELIVERY.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    video_archive = out.parent / f"{out.name}_videos.tar.gz"
    video_partial = video_archive.with_suffix(".tar.gz.partial")
    with tarfile.open(video_partial, "w:gz", compresslevel=1) as tar:
        for task, video in videos:
            tar.add(video, arcname=f"videos/{task}.mp4")
    video_partial.replace(video_archive)
    video_digest = sha256(video_archive)
    video_archive.with_suffix(".tar.gz.sha256").write_text(
        f"{video_digest}  {video_archive.name}\n", encoding="utf-8")

    archive = out.parent / f"{out.name}_delivery.tar.gz"
    partial = archive.with_suffix(".tar.gz.partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        tar.add(out, arcname=out.name)
    partial.replace(archive)
    digest = sha256(archive)
    archive.with_suffix(".tar.gz.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    (out / "DELIVERY_COMPLETE.json").write_text(json.dumps({
        "archive": str(archive), "sha256": digest,
        "video_archive": str(video_archive), "video_sha256": video_digest,
        "audit_issues": issues}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
