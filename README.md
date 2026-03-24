# LOCOMO Benchmark for OpenClaw Memory

Minimal harness for memory benchmarking using OpenClaw on the [LOCOMO](https://github.com/snap-research/locomo) dataset.

This repo benchmarks OpenClaw against LOCOMO with three backends:

- `memory-core`: writes lossless LOCOMO session markdown into the OpenClaw workspace, then reindexes
- `memory-lancedb`: writes the same LOCOMO session markdown, reuses the exact `memory-core` indexed chunks and embeddings, then stores those chunks in the built-in LanceDB plugin table
- `memory-lancedb-pro`: starts from that same chunk-aligned LanceDB corpus and uses the `memory-lancedb-pro` plugin for more advanced retrieval tuning

QA still runs through a local OpenClaw gateway, and an LLM judge scores the answers.

## Quick Start

### 1. Install OpenClaw

```bash
npm install -g openclaw@latest
openclaw onboard
```

### 2. Add your OpenAI key

Put this in `.env` at the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

### 3. Download the dataset

```bash
mkdir -p datasets
curl -fsSL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -o datasets/locomo10.json
```

### 4. Install Python dependencies

```bash
uv sync
```

### 5. Configure OpenClaw for the backend you want to benchmark

```bash
./setup_memory_core.sh
```

For the LanceDB leg, use:

```bash
./setup_memory_lancedb.sh
```

For the LanceDB Pro leg, use:

```bash
./setup_memory_lancedb_pro.sh
```

Before the first LanceDB Pro run, install the plugin once:

```bash
openclaw plugins install memory-lancedb-pro@beta
```

### 6. Start the gateway

```bash
./start_gateway.sh
```

The benchmark expects the gateway at `http://127.0.0.1:18789`.

`start_gateway.sh` hardcodes the OpenClaw agent model to `openai/gpt-4.1-mini` so the gateway does not fall back to another provider by default.

### 7. Build the backend corpus once

For `memory-core`, prebuild the workspace markdown and SQLite index:

```bash
./setup_memory_core.sh

uv run python scripts/build_memory_core_corpus.py \
  --input datasets/locomo10.json
```

That writes the full LOCOMO session markdown into the benchmark workspace and runs the built-in `openclaw memory index --force`. It records a manifest at `locomo-bench/prebuilt-memory-core.json`.

For `memory-lancedb`, prebuild the chunk-aligned LanceDB store:

```bash
./setup_memory_lancedb.sh

uv run python scripts/build_memory_lancedb_corpus.py \
  --input datasets/locomo10.json
```

That writes the full LOCOMO session markdown into the workspace, runs the built-in `memory-core` indexer, reads back the exact indexed chunk text and embeddings from SQLite, and writes those chunks into the configured `memory-lancedb` store. It records a manifest at `locomo-bench/prebuilt-memory-lancedb.json`.

For `memory-lancedb-pro`, build from the existing `memory-lancedb` store:

```bash
./setup_memory_lancedb_pro.sh

uv run python scripts/build_memory_lancedb_pro_corpus.py \
  --source-db locomo-bench/lancedb
```

**This script requires that the `memory-lancedb` store pre-exist (to migrate already-computed embeddings to the format expected by the `memory-lancedb-pro` plugin).

That uses the plugin's migration path to materialize a separate `memory-lancedb-pro` store from the already-built chunk-aligned `memory-lancedb` corpus without re-embedding the corpus again. It records a manifest at `locomo-bench/prebuilt-memory-lancedb-pro.json`.

### 8. Run a subset of the benchmark

Use the `--limit` parameter to specify the number of QA pairs to benchmark.

> [!NOTE]
> `--limit` applies to the flattened LOCOMO QA rows, not the number of dialogues. So, the loader flattens every sample's `qa` array into one ordered benchmark list. Here, `--limit 5` means "run the first 5 QA rows". If those rows all come from one dialogue, the harness still ingests the full source dialogue for that selected sample. It does not ingest the entire LOCOMO dataset unless your selected rows span the entire dataset.

The benchmark exposes three model controls:

- `--agent-model`: the actual OpenClaw agent model used for QA
- `--gateway-model`: optional model value sent to OpenClaw `/v1/responses`; if omitted it defaults to `--agent-model`
- `--judge-model`: model used by the LLM judge

By default, judge calls run with `--judge-concurrency 10` (10 parallel requests). Gateway QA calls run serially (`--concurrency 1`) because the OpenClaw gateway serializes requests through a single lane queue. For full parallelism across QA calls, use `scripts/run_parallel.py` which splits rows across 4 subprocesses. Output format is identical regardless of concurrency settings.

In a second terminal, enter the following for `memory-core`:

```bash
uv run python scripts/run_memory_core.py \
  --input datasets/locomo10.json \
  --limit 5 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

For the LanceDB leg:

```bash
uv run python scripts/run_memory_lancedb.py \
  --input datasets/locomo10.json \
  --limit 5 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

For the LanceDB Pro leg:

```bash
uv run python scripts/run_memory_lancedb_pro.py \
  --input datasets/locomo10.json \
  --limit 5 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

All runs write artifacts under `outputs/` by default.

### 8b. Parallel runs with `run_parallel.py`

The OpenClaw gateway serializes requests through a single lane queue, so in-process concurrency for QA calls doesn't help. For faster large-scale runs, use `run_parallel.py` which splits rows across 4 subprocesses:

```bash
uv run python scripts/run_parallel.py \
  --backend memory-lancedb-pro \
  --input datasets/locomo10.json \
  --limit 100 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

This spawns 4 worker processes, each handling a quarter of the rows. When all workers finish, the script merges the JSONL outputs and recomputes `summary.json` into a single output directory. The output format is identical to a single-process run.

All flags from the single-process scripts are supported (`--limit`, `--skip-ingest`, `--judge-concurrency`, etc.). The `--backend` flag selects which runner script to use (`memory-core`, `memory-lancedb`, or `memory-lancedb-pro`).

### Concurrency controls

The benchmark exposes two concurrency settings:

- `--concurrency`: number of concurrent gateway QA requests per process (default: 1, serial). Increasing this is not useful because the gateway serializes via lane queuing.
- `--judge-concurrency`: number of concurrent LLM judge requests per process (default: 10). This parallelizes calls to the OpenAI API for grading answers.

These flags apply to both the single-process scripts and the parallel wrapper.

## Sample Results

The latest summaries under [`outputs/`](/Users/prrao/code/locomo-eval/outputs) show the following picture for recent `--limit 10` runs:

| Backend | Rows | Correct | Wrong | Completion Rate | Avg latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory-core` | 50 | 27 | 23 | 0.54 | 6.7 |
| `memory-lancedb` | 50 | 32 | 18 | 0.64 | 4.1 |
| `memory-lancedb-pro` | 50 | 36 | 14 | 0.72 |  10.9 |

Source summaries are from the output files after running the benchmark using each memory plugin.

### 9. Large-scale runs: prebuild the stores once, then benchmark in query-only mode

For real benchmark runs, repeatedly ingesting the same corpus is wasteful. The recommended workflow is:

1. build the `memory-core` corpus once
2. build the `memory-lancedb` corpus once
3. build the `memory-lancedb-pro` corpus once from that prebuilt `memory-lancedb` store
4. run each benchmark leg with `--skip-ingest`

After the corpora exist, benchmark in query-only mode.

For `memory-core`:

```bash
./setup_memory_core.sh
./start_gateway.sh
```

In a second terminal:

```bash
uv run python scripts/run_memory_core.py \
  --input datasets/locomo10.json \
  --limit 100 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

For `memory-lancedb`:

```bash
./setup_memory_lancedb.sh
./start_gateway.sh
```

In a second terminal:

```bash
uv run python scripts/run_memory_lancedb.py \
  --input datasets/locomo10.json \
  --limit 100 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

For `memory-lancedb-pro`:

```bash
./setup_memory_lancedb_pro.sh
./start_gateway.sh
```

In a second terminal:

```bash
uv run python scripts/run_memory_lancedb_pro.py \
  --input datasets/locomo10.json \
  --limit 100 \
  --gateway http://127.0.0.1:18789 \
  --agent-model openai/gpt-4.1-mini \
  --judge-model openai/gpt-4.1-mini \
  --skip-ingest
```

`--skip-ingest` means:

- the benchmark does not touch the existing store
- the run is query-only
- `memory_status_before.json` and `memory_status_after.json` reflect the prebuilt store as-is
- for `memory-core`, this only works after `build_memory_core_corpus.py` has populated the workspace markdown and SQLite index

## What the Runner Does

For `memory-core`, the benchmark:

- writes raw LOCOMO session markdown into `workspace/memory/locomo/`
- clears only that benchmark-managed subtree before each run
- runs `openclaw memory index --force`
- asks QA through the gateway after reindexing
- does no summarization during ingest

For `memory-lancedb`, the benchmark:

- writes the same LOCOMO session markdown used by `memory-core`
- runs the built-in `openclaw memory index --force`
- reads the exact indexed `memory-core` chunks and stored embeddings from the SQLite `chunks` table
- writes those chunks directly into the plugin's `memories` table
- clears the benchmark-managed LanceDB directory before each run
- uses the bundled `memory-lancedb` plugin with its Node dependency installed once
- does no summarization during ingest

When `--skip-ingest` is set, it skips the write/reset step and queries the already-built store.

The standalone [build_memory_core_corpus.py](./scripts/build_memory_core_corpus.py) script performs that same write-and-index step once up front so later `memory-core` runs can safely use `--skip-ingest`.

For `memory-lancedb-pro`, the benchmark:

- writes the same LOCOMO session markdown used by `memory-core`
- runs the built-in `openclaw memory index --force`
- reads the exact indexed `memory-core` chunks and stored embeddings from the SQLite `chunks` table
- materializes a temporary legacy `memory-lancedb` store from those chunks
- migrates that temporary legacy store into `memory-lancedb-pro` so Pro sees the same chunk corpus and vectors
- uses the installed `memory-lancedb-pro` plugin and the retrieval settings from `setup_memory_lancedb_pro.sh`
- does no summarization during ingest

This is intentional. `memory-lancedb-pro` maintains additional search/index state, so the benchmark migrates through the plugin's supported path instead of treating it like the simpler built-in LanceDB store.

When `--skip-ingest` is set, it skips the import step and queries the already-built Pro store.

## Run Artifacts (What is Output)

Each benchmark run writes a directory under:

- [./outputs](./outputs)

Typical files in one run directory:

- `selected_rows.jsonl`
  - the flattened LOCOMO QA rows selected by `--limit`
  - useful for seeing exactly which benchmark questions were included
- `ingest_log.jsonl`
  - one row per stored memory unit written during ingest
  - for `memory-core`, this is one row per session markdown file
  - for `memory-lancedb`, this is one row per `memory-core` chunk stored in LanceDB
  - for `memory-lancedb-pro`, this is one row per `memory-core` chunk stored in LanceDB Pro
- `reindex.log`
  - stdout from `openclaw memory index --force` for `memory-core`
  - stdout from the chunk-source `openclaw memory index --force` step for both LanceDB legs
  - for `memory-lancedb-pro`, this file also includes the plugin migration output
- `document_log.jsonl`
  - the session markdown files written before chunk extraction
  - produced for both LanceDB legs because their chunk source is now the same session-document corpus used by `memory-core`
- `memory_status_before.json`
  - backend status before ingest
  - for `memory-core`, this records the workspace, SQLite path, and indexed file/chunk counts
  - for `memory-lancedb`, this records the LanceDB path and whether the store already existed
  - for `memory-lancedb-pro`, this records the LanceDB Pro path and whether the store already existed
- `memory_status_after.json`
  - backend status after ingest
  - for `memory-core`, this confirms how many files and chunks were indexed
  - for `memory-lancedb`, this confirms the LanceDB path and how many chunk rows were written
  - for `memory-lancedb-pro`, this confirms the LanceDB Pro path and how many chunk rows were written
- `qa_results.jsonl`
  - raw benchmark answers returned by the gateway for each selected QA row
  - includes latency, token usage if present, and any gateway errors
  - token usage for gateway-mediated runs should be treated as approximate until the gateway usage fields are fully normalized by the harness
- `judged_results.jsonl`
  - the QA rows after LLM judging
  - marks each answer as `CORRECT` or `WRONG` with short reasoning
- `summary.json`
  - small run-level summary
  - includes task completion rate, counts, token totals, latency, and memory status before/after

## Notes

- Start with `memory-core` as the baseline, then compare it against `memory-lancedb` and `memory-lancedb-pro`.
- The same `.env` key is used for both OpenClaw and judge calls.
- For the cleanest benchmark, keep unrelated files out of the active OpenClaw workspace memory corpus.
- `setup_memory_lancedb_pro.sh` is the comparable baseline script.
- `setup_memory_lancedb_pro_tune.sh` is the experimental tuned script for retrieval sweeps and A/B testing.
- For large runs, prefer the prebuilt-store workflow and `--skip-ingest`. Re-ingesting the same corpus on every run is mainly useful for debugging, not for full benchmarks.
- `memory-lancedb-pro` still depends on the prebuilt `memory-lancedb` store as its corpus source.
