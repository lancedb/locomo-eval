from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.dataset import (
    build_memory_documents,
    build_sample_lookup,
    flatten_benchmark_rows,
    load_locomo_samples,
    qa_prompt,
    qa_session_key,
    select_rows,
    selected_samples,
    user_for_sample,
)
from src.gateway import GatewayClient, GatewayError
from src.judge import grade_results
from src.memory_core import (
    extract_indexed_memory_chunks,
    prepare_memory_root,
    reindex_memory,
    resolve_memory_index_paths,
    resolve_memory_status,
    write_memory_documents,
)
from src.memory_lancedb import (
    count_lancedb_rows,
    make_lancedb_config,
    migrate_legacy_lancedb_to_pro,
    prepare_lancedb_store,
    resolve_lancedb_config,
    resolve_lancedb_status,
    write_memory_chunks,
)
from src.openclaw_cli import OpenClawCliError, resolve_memory_slot, set_config_value
from src.schema import MemoryChunk, MemoryDocument, QaResult, TokenUsage
from src.summary import build_summary


def _benchmark_env_path() -> str | None:
    candidate = Path("locomo-bench/openclaw.env")
    if candidate.exists():
        return str(candidate)
    return None


def _benchmark_config_path() -> Path | None:
    from_env = os.getenv("OPENCLAW_CONFIG_PATH")
    if from_env:
        candidate = Path(from_env)
        if candidate.exists():
            return candidate
    candidate = Path("locomo-bench/openclaw.json")
    if candidate.exists():
        return candidate
    return None


load_dotenv()
load_dotenv(_benchmark_env_path(), override=False)

BENCHMARK_MEMORY_ROOT = "memory/locomo"


@dataclass(frozen=True)
class RunConfig:
    run_label: str
    input_path: str
    gateway_url: str
    gateway_token: str | None
    agent_model: str
    gateway_model: str | None
    output_dir: str | None
    limit: int | None
    judge_model: str
    judge_base_url: str | None
    judge_token: str | None
    settle_seconds: float
    timeout_seconds: float
    agent_id: str
    skip_ingest: bool
    concurrency: int = 1
    judge_concurrency: int = 10
    shard_index: int | None = None
    shard_count: int | None = None


def run_benchmark(config: RunConfig) -> Path:
    output_dir = _resolve_output_dir(config.run_label, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_locomo_samples(config.input_path)
    sample_lookup = build_sample_lookup(samples)
    all_rows = flatten_benchmark_rows(samples)
    selected = select_rows(all_rows, config.limit)
    if config.shard_index is not None and config.shard_count is not None:
        selected = selected[config.shard_index :: config.shard_count]
    if not selected:
        raise ValueError("No benchmark rows were selected")

    _write_jsonl(output_dir / "selected_rows.jsonl", (row.to_dict() for row in selected))

    memory_status_before, memory_status_after = _ingest_selected_memories(
        config,
        sample_lookup,
        selected,
        output_dir,
    )
    _write_json(output_dir / "memory_status_before.json", memory_status_before)
    _write_json(output_dir / "memory_status_after.json", memory_status_after)

    if config.settle_seconds > 0:
        time.sleep(config.settle_seconds)

    _configure_agent_model(config.agent_model)

    gateway = GatewayClient(
        base_url=config.gateway_url,
        token=config.gateway_token,
        model=config.gateway_model or config.agent_model,
        timeout_seconds=config.timeout_seconds,
    )

    qa_results, qa_traces = _run_qa(
        gateway, selected, memory_backend=config.run_label, concurrency=config.concurrency
    )
    _write_jsonl(output_dir / "qa_results.jsonl", (result.to_dict() for result in qa_results))
    _write_jsonl(output_dir / "qa_traces.jsonl", qa_traces)

    judged_results = grade_results(
        qa_results,
        model=config.judge_model,
        base_url=config.judge_base_url,
        token=config.judge_token,
        concurrency=config.judge_concurrency,
    )
    _write_jsonl(
        output_dir / "judged_results.jsonl",
        (result.to_dict() for result in judged_results),
    )

    summary = build_summary(
        qa_results,
        judged_results,
        run_label=config.run_label,
        input_path=config.input_path,
        limit=config.limit,
    )
    summary["skip_ingest"] = config.skip_ingest
    summary["memory_status_before"] = memory_status_before
    summary["memory_status_after"] = memory_status_after
    _write_json(output_dir / "summary.json", summary)
    return output_dir


def run_cli(run_label: str) -> Path:
    args = build_parser(run_label).parse_args()
    config = RunConfig(
        run_label=run_label,
        input_path=args.input,
        gateway_url=args.gateway,
        gateway_token=args.gateway_token or _default_gateway_token(),
        agent_model=args.agent_model,
        gateway_model=args.gateway_model,
        output_dir=args.output_dir,
        limit=args.limit,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_token=args.judge_token,
        settle_seconds=args.settle_seconds,
        timeout_seconds=args.timeout_seconds,
        agent_id=args.agent_id,
        skip_ingest=args.skip_ingest,
        concurrency=args.concurrency,
        judge_concurrency=args.judge_concurrency,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return run_benchmark(config)


def build_parser(run_label: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {run_label} LOCOMO benchmark")
    parser.add_argument("--input", required=True, help="Path to locomo10.json")
    parser.add_argument(
        "--gateway",
        default="http://127.0.0.1:18789",
        help="OpenClaw gateway base URL",
    )
    parser.add_argument(
        "--agent-model",
        default="openai/gpt-4.1-mini",
        help="OpenClaw agent model used for QA",
    )
    parser.add_argument(
        "--gateway-token",
        default=None,
        help="Optional bearer token for the gateway",
    )
    parser.add_argument(
        "--gateway-model",
        default=None,
        help="Optional model value sent to the OpenClaw responses API; defaults to --agent-model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of flattened QA rows to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write run artifacts into",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o-mini",
        help="Model used for LLM judging",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for the judge model",
    )
    parser.add_argument(
        "--judge-token",
        default=None,
        help="Optional API token for the judge model",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="Optional delay after memory reindex before QA starts",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Gateway request timeout in seconds",
    )
    parser.add_argument(
        "--agent-id",
        default="main",
        help="OpenClaw agent id used for memory status and reindex",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip corpus ingestion and query the existing store as-is",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent gateway QA requests (default: 1, serial; the gateway serializes via lane queuing)",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=10,
        help="Number of concurrent judge requests (default: 10)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index for parallel runs (0-based). Used by run_parallel.py.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total number of shards for parallel runs. Used by run_parallel.py.",
    )
    return parser


def _default_gateway_token() -> str | None:
    env_token = os.getenv("OPENCLAW_GATEWAY_TOKEN")
    if env_token:
        return env_token

    config_path = _benchmark_config_path()
    if not config_path:
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    gateway = payload.get("gateway")
    if not isinstance(gateway, dict):
        return None
    auth = gateway.get("auth")
    if not isinstance(auth, dict):
        return None
    token = auth.get("token")
    if isinstance(token, str) and token.strip():
        return token
    return None


def _collect_documents(
    sample_lookup: dict[str, dict[str, object]],
    rows,
) -> list[MemoryDocument]:
    documents: list[MemoryDocument] = []
    for sample in selected_samples(sample_lookup, rows):
        documents.extend(build_memory_documents(sample, memory_root=BENCHMARK_MEMORY_ROOT))
    return documents


def _ingest_selected_memories(
    config: RunConfig,
    sample_lookup: dict[str, dict[str, object]],
    rows,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if config.skip_ingest:
        return _reuse_existing_memories(config, output_dir)
    if config.run_label == "memory-core":
        return _ingest_memory_core(config, sample_lookup, rows, output_dir)
    if config.run_label == "memory-lancedb":
        return _ingest_memory_lancedb_backend(
            config, "memory-lancedb", sample_lookup, rows, output_dir
        )
    if config.run_label == "memory-lancedb-pro":
        return _ingest_memory_lancedb_pro(config, sample_lookup, rows, output_dir)
    raise ValueError(f"Unsupported run label: {config.run_label}")


def _reuse_existing_memories(
    config: RunConfig,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if config.run_label == "memory-core":
        _ensure_memory_slot("memory-core")
        status = resolve_memory_status(config.agent_id)
        payload = {
            "backend": status.backend,
            "workspace_dir": str(status.workspace_dir),
            "db_path": str(status.db_path) if status.db_path else None,
            "files": status.files,
            "chunks": status.chunks,
            "skip_ingest": True,
        }
    elif config.run_label in {"memory-lancedb", "memory-lancedb-pro"}:
        _ensure_memory_slot(config.run_label)
        lancedb_config = resolve_lancedb_config(config.run_label)
        status = resolve_lancedb_status(
            lancedb_config,
            row_count=count_lancedb_rows(lancedb_config),
        )
        payload = {
            **status.raw,
            "skip_ingest": True,
        }
    else:
        raise ValueError(f"Unsupported run label: {config.run_label}")

    _write_jsonl(output_dir / "ingest_log.jsonl", [])
    (output_dir / "reindex.log").write_text(
        "Ingest skipped; benchmark queried the prebuilt store already configured for this backend.\n",
        encoding="utf-8",
    )
    return payload, payload


def _ingest_memory_core(
    config: RunConfig,
    sample_lookup: dict[str, dict[str, object]],
    rows,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    _ensure_memory_slot("memory-core")
    memory_status_before = resolve_memory_status(config.agent_id)
    documents = _collect_documents(sample_lookup, rows)
    prepare_memory_root(memory_status_before.workspace_dir, BENCHMARK_MEMORY_ROOT)
    ingest_log = write_memory_documents(memory_status_before.workspace_dir, documents)
    _write_jsonl(output_dir / "ingest_log.jsonl", ingest_log)

    reindex_stdout = reindex_memory(config.agent_id)
    (output_dir / "reindex.log").write_text(reindex_stdout, encoding="utf-8")

    memory_status_after = resolve_memory_status(config.agent_id)
    before_payload = {
        "backend": memory_status_before.backend,
        "workspace_dir": str(memory_status_before.workspace_dir),
        "db_path": str(memory_status_before.db_path) if memory_status_before.db_path else None,
        "files": memory_status_before.files,
        "chunks": memory_status_before.chunks,
    }
    after_payload = {
        "backend": memory_status_after.backend,
        "workspace_dir": str(memory_status_after.workspace_dir),
        "db_path": str(memory_status_after.db_path) if memory_status_after.db_path else None,
        "files": memory_status_after.files,
        "chunks": memory_status_after.chunks,
    }
    return before_payload, after_payload


def _ingest_memory_lancedb_backend(
    config: RunConfig,
    backend: str,
    sample_lookup: dict[str, dict[str, object]],
    rows,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    _ensure_memory_slot(backend)
    lancedb_config = resolve_lancedb_config(backend)
    chunks = _build_core_equivalent_chunks(
        sample_lookup,
        rows,
        output_dir,
        agent_id=config.agent_id,
    )
    memory_status_before = resolve_lancedb_status(lancedb_config)
    prepare_lancedb_store(lancedb_config.db_path)
    ingest_log = write_memory_chunks(lancedb_config, chunks)
    _write_jsonl(output_dir / "ingest_log.jsonl", ingest_log)
    memory_status_after = resolve_lancedb_status(lancedb_config, row_count=len(chunks))
    return memory_status_before.raw, memory_status_after.raw


def _ingest_memory_lancedb_pro(
    config: RunConfig,
    sample_lookup: dict[str, dict[str, object]],
    rows,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    backend = "memory-lancedb-pro"
    _ensure_memory_slot(backend)
    lancedb_config = resolve_lancedb_config(backend)
    chunks = _build_core_equivalent_chunks(
        sample_lookup,
        rows,
        output_dir,
        agent_id=config.agent_id,
    )
    memory_status_before = resolve_lancedb_status(
        lancedb_config, row_count=count_lancedb_rows(lancedb_config)
    )
    temp_legacy_db = output_dir / "legacy-memory-lancedb"
    temp_legacy_config = make_lancedb_config(
        "memory-lancedb",
        temp_legacy_db,
        template=lancedb_config,
    )
    prepare_lancedb_store(temp_legacy_config.db_path)
    ingest_log = write_memory_chunks(temp_legacy_config, chunks)
    migration_stdout = migrate_legacy_lancedb_to_pro(temp_legacy_config.db_path)
    _write_jsonl(output_dir / "ingest_log.jsonl", ingest_log)
    with (output_dir / "reindex.log").open("a", encoding="utf-8") as handle:
        handle.write("\n[memory-lancedb-pro migration]\n")
        handle.write(migration_stdout)
    memory_status_after = resolve_lancedb_status(
        lancedb_config, row_count=count_lancedb_rows(lancedb_config)
    )
    return memory_status_before.raw, memory_status_after.raw


def _build_core_equivalent_chunks(
    sample_lookup: dict[str, dict[str, object]],
    rows,
    output_dir: Path,
    *,
    agent_id: str,
) -> list[MemoryChunk]:
    workspace_dir, db_path = resolve_memory_index_paths(agent_id)
    documents = _collect_documents(sample_lookup, rows)
    prepare_memory_root(workspace_dir, BENCHMARK_MEMORY_ROOT)
    document_log = write_memory_documents(workspace_dir, documents)
    _write_jsonl(output_dir / "document_log.jsonl", document_log)

    reindex_stdout = reindex_memory(agent_id)
    (output_dir / "reindex.log").write_text(reindex_stdout, encoding="utf-8")
    return extract_indexed_memory_chunks(db_path, documents)


def _ensure_memory_slot(expected: str) -> None:
    try:
        actual = resolve_memory_slot()
    except OpenClawCliError as exc:
        raise ValueError(str(exc)) from exc
    if actual != expected:
        raise ValueError(
            f"OpenClaw memory slot is '{actual}', but this benchmark run requires '{expected}'."
        )


def _configure_agent_model(model: str) -> None:
    try:
        set_config_value("agents.defaults.model.primary", model)
    except OpenClawCliError as exc:
        raise ValueError(f"Failed to set OpenClaw agent model to '{model}': {exc}") from exc


def _run_qa(
    gateway: GatewayClient,
    rows,
    *,
    memory_backend: str,
    concurrency: int = 1,
) -> tuple[list[QaResult], list[dict[str, object]]]:

    def _query_one(idx, row):
        user = user_for_sample(row.sample_id)
        session_key = qa_session_key(row.sample_id, row.benchmark_id)
        try:
            response = gateway.send_message(
                user=user,
                session_key=session_key,
                message=qa_prompt(row.question, memory_backend),
            )
            result = QaResult(
                benchmark_id=row.benchmark_id,
                sample_id=row.sample_id,
                qa_index=row.qa_index,
                question=row.question,
                answer=row.answer,
                category=row.category,
                evidence=row.evidence,
                response=response.text,
                latency_seconds=response.latency_seconds,
                token_usage=response.token_usage,
                error=None,
                user=user,
                session_key=session_key,
            )
            trace = {
                "benchmark_id": row.benchmark_id,
                "sample_id": row.sample_id,
                "qa_index": row.qa_index,
                "question": row.question,
                "user": user,
                "session_key": session_key,
                "response_text": response.text,
                "output_types": _output_types(response.raw_body),
                "function_call_count": len(_function_calls(response.raw_body)),
                "function_calls": _function_calls(response.raw_body),
                "usage": response.raw_body.get("usage"),
                "raw_body": response.raw_body,
            }
        except GatewayError as exc:
            result = QaResult(
                benchmark_id=row.benchmark_id,
                sample_id=row.sample_id,
                qa_index=row.qa_index,
                question=row.question,
                answer=row.answer,
                category=row.category,
                evidence=row.evidence,
                response=None,
                latency_seconds=None,
                token_usage=_empty_token_usage(),
                error=str(exc),
                user=user,
                session_key=session_key,
            )
            trace = {
                "benchmark_id": row.benchmark_id,
                "sample_id": row.sample_id,
                "qa_index": row.qa_index,
                "question": row.question,
                "user": user,
                "session_key": session_key,
                "response_text": None,
                "output_types": [],
                "function_call_count": 0,
                "function_calls": [],
                "usage": None,
                "raw_body": None,
                "error": str(exc),
            }
        return idx, result, trace

    if concurrency <= 1:
        pairs = [_query_one(i, row) for i, row in enumerate(rows)]
    else:
        pairs = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_query_one, i, row): i for i, row in enumerate(rows)}
            for future in as_completed(futures):
                idx, result, trace = future.result()
                pairs[idx] = (idx, result, trace)

    results = [p[1] for p in pairs]
    traces = [p[2] for p in pairs]
    return results, traces


def _empty_token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)


def _output_types(body: dict[str, object]) -> list[str]:
    output = body.get("output")
    if not isinstance(output, list):
        return []
    return [str(item.get("type")) for item in output if isinstance(item, dict) and item.get("type")]


def _function_calls(body: dict[str, object]) -> list[dict[str, object]]:
    output = body.get("output")
    if not isinstance(output, list):
        return []

    calls: list[dict[str, object]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        calls.append(
            {
                "id": item.get("id"),
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "status": item.get("status"),
                "arguments": item.get("arguments"),
            }
        )
    return calls


def _resolve_output_dir(run_label: str, override: str | None) -> Path:
    if override:
        return Path(override)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"{run_label}-{timestamp}"


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
