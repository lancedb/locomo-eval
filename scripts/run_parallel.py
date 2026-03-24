"""Run a benchmark across 4 parallel subprocesses, then merge results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

NUM_WORKERS = 4

BACKEND_SCRIPTS = {
    "memory-core": "scripts/run_memory_core.py",
    "memory-lancedb": "scripts/run_memory_lancedb.py",
    "memory-lancedb-pro": "scripts/run_memory_lancedb_pro.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark with 4 parallel workers")
    parser.add_argument("--backend", required=True, choices=BACKEND_SCRIPTS.keys())
    parser.add_argument("--input", required=True, help="Path to locomo10.json")
    parser.add_argument("--limit", type=int, default=None, help="Total QA rows to evaluate")
    parser.add_argument("--gateway", default="http://127.0.0.1:18789")
    parser.add_argument("--agent-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--gateway-model", default=None)
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-token", default=None)
    parser.add_argument("--gateway-token", default=None)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--agent-id", default="main")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--judge-concurrency", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    final_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{args.backend}-{timestamp}"
    final_dir.mkdir(parents=True, exist_ok=True)

    script = BACKEND_SCRIPTS[args.backend]
    shard_dirs: list[Path] = []
    procs: list[subprocess.Popen] = []

    for i in range(NUM_WORKERS):
        shard_dir = final_dir / f"_shard_{i}"
        shard_dirs.append(shard_dir)

        cmd = [
            sys.executable, script,
            "--input", args.input,
            "--gateway", args.gateway,
            "--agent-model", args.agent_model,
            "--judge-model", args.judge_model,
            "--settle-seconds", str(args.settle_seconds),
            "--timeout-seconds", str(args.timeout_seconds),
            "--agent-id", args.agent_id,
            "--judge-concurrency", str(args.judge_concurrency),
            "--output-dir", str(shard_dir),
            "--shard-index", str(i),
            "--shard-count", str(NUM_WORKERS),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.skip_ingest:
            cmd.append("--skip-ingest")
        if args.gateway_model:
            cmd += ["--gateway-model", args.gateway_model]
        if args.gateway_token:
            cmd += ["--gateway-token", args.gateway_token]
        if args.judge_base_url:
            cmd += ["--judge-base-url", args.judge_base_url]
        if args.judge_token:
            cmd += ["--judge-token", args.judge_token]

        print(f"[shard {i}] starting: {shard_dir}")
        procs.append(subprocess.Popen(cmd))

    failures = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failures.append(i)
            print(f"[shard {i}] FAILED (exit code {rc})")
        else:
            print(f"[shard {i}] done")

    if failures:
        print(f"\n{len(failures)} shard(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)

    # Merge outputs
    _merge_jsonl(shard_dirs, final_dir, "selected_rows.jsonl")
    _merge_jsonl(shard_dirs, final_dir, "qa_results.jsonl")
    _merge_jsonl(shard_dirs, final_dir, "qa_traces.jsonl")
    _merge_jsonl(shard_dirs, final_dir, "judged_results.jsonl")
    _merge_jsonl(shard_dirs, final_dir, "ingest_log.jsonl")

    # Recompute summary from merged data
    qa_results = _read_jsonl(final_dir / "qa_results.jsonl")
    judged_results = _read_jsonl(final_dir / "judged_results.jsonl")
    summary = _build_merged_summary(
        qa_results,
        judged_results,
        run_label=args.backend,
        input_path=args.input,
        limit=args.limit,
        skip_ingest=args.skip_ingest,
        shard_dirs=shard_dirs,
    )
    (final_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Copy memory status from first shard (all shards see the same store)
    for name in ("memory_status_before.json", "memory_status_after.json"):
        src = shard_dirs[0] / name
        if src.exists():
            (final_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Clean up shard dirs
    for shard_dir in shard_dirs:
        _rmtree(shard_dir)

    print(f"\nMerged results: {final_dir}")
    print(f"  {len(qa_results)} QA rows, {sum(1 for j in judged_results if j.get('result') == 'CORRECT')} correct")


def _merge_jsonl(shard_dirs: list[Path], dest_dir: Path, filename: str) -> None:
    with (dest_dir / filename).open("w", encoding="utf-8") as out:
        for shard_dir in shard_dirs:
            src = shard_dir / filename
            if src.exists():
                out.write(src.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_merged_summary(
    qa_results: list[dict],
    judged_results: list[dict],
    *,
    run_label: str,
    input_path: str,
    limit: int | None,
    skip_ingest: bool,
    shard_dirs: list[Path],
) -> dict:
    correct = sum(1 for j in judged_results if j.get("result") == "CORRECT")
    wrong = sum(1 for j in judged_results if j.get("result") == "WRONG")
    judged_count = len(judged_results)
    accuracy = correct / judged_count if judged_count else 0.0

    prompt_tokens = sum(
        (r.get("token_usage") or {}).get("prompt_tokens") or 0 for r in qa_results
    )
    completion_tokens = sum(
        (r.get("token_usage") or {}).get("completion_tokens") or 0 for r in qa_results
    )
    total_tokens = sum(
        (r.get("token_usage") or {}).get("total_tokens") or 0 for r in qa_results
    )

    latencies = [r["latency_seconds"] for r in qa_results if r.get("latency_seconds") is not None]
    average_latency = sum(latencies) / len(latencies) if latencies else None

    summary = {
        "run_label": run_label,
        "input_path": input_path,
        "limit": limit,
        "selected_rows": len(qa_results),
        "judged_rows": judged_count,
        "correct": correct,
        "wrong": wrong,
        "task_completion_rate": accuracy,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "average_latency_seconds": average_latency,
        "skip_ingest": skip_ingest,
        "workers": NUM_WORKERS,
    }

    # Include memory status from first shard
    status_path = shard_dirs[0] / "memory_status_before.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary["memory_status_before"] = status
        summary["memory_status_after"] = status

    return summary


def _rmtree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
