#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

STATE_DIR_NAME="locomo-bench"
STATE_DIR_ROOT="$PWD"
STATE_DIR="${STATE_DIR_ROOT%/}/${STATE_DIR_NAME}"
AGENT_ID="main"
BENCH_CONFIG_PATH="${STATE_DIR}/openclaw.json"
ENV_FILE="${STATE_DIR}/openclaw.env"

WORKSPACE_DIR="${STATE_DIR}/workspace"
SQLITE_DIR="${STATE_DIR}/sqlite"
LANCEDB_DIR="${STATE_DIR}/lancedb"
LANCEDB_PRO_DIR="${STATE_DIR}/lancedb-pro"
PLUGIN_EXTENSIONS_DIR="${WORKSPACE_DIR}/.openclaw/extensions"
LOCAL_PLUGIN_DIR="${PLUGIN_EXTENSIONS_DIR}/memory-lancedb"
LOCAL_PLUGIN_PRO_DIR="${PLUGIN_EXTENSIONS_DIR}/memory-lancedb-pro"
REPO_PLUGIN_DIR="${PWD}/local-plugins/memory-lancedb-pro"

# Tune these values.
# The values below are a LOCOMO-oriented precision hybrid profile:
# vector-first retrieval, constrained BM25 influence, and no temporal/lifecycle
# bias during benchmark recall.
AUTO_RECALL="false"
SMART_EXTRACTION="false"
RETRIEVAL_MODE="hybrid"
RETRIEVAL_VECTOR_WEIGHT="1.0"
RETRIEVAL_BM25_WEIGHT="0.0"
RETRIEVAL_MIN_SCORE="0.40"
RETRIEVAL_HARD_MIN_SCORE="0.40"
RETRIEVAL_CANDIDATE_POOL_SIZE="20"
RETRIEVAL_RERANK="none"
RETRIEVAL_FILTER_NOISE="true"
RETRIEVAL_RECENCY_HALF_LIFE_DAYS="0"
RETRIEVAL_RECENCY_WEIGHT="0"
RETRIEVAL_LENGTH_NORM_ANCHOR="500"
RETRIEVAL_TIME_DECAY_HALF_LIFE_DAYS="0"
RETRIEVAL_REINFORCEMENT_FACTOR="0"
RETRIEVAL_MAX_HALF_LIFE_MULTIPLIER="1"

# Optional reranker overrides. Leave empty to use plugin defaults.
# For Jina, set JINA_API_KEY in .env and this script will pick it up automatically.
RERANK_API_KEY="${JINA_API_KEY:-}"
RERANK_MODEL=""
RERANK_ENDPOINT=""
RERANK_PROVIDER=""

mkdir -p "${STATE_DIR}" "${WORKSPACE_DIR}" "${SQLITE_DIR}" "${LANCEDB_DIR}" "${LANCEDB_PRO_DIR}" "${PLUGIN_EXTENSIONS_DIR}"

export OPENCLAW_STATE_DIR="${STATE_DIR}"
export OPENCLAW_CONFIG_PATH="${BENCH_CONFIG_PATH}"

if [[ ! -f "${REPO_PLUGIN_DIR}/index.ts" ]]; then
  echo "Could not find local memory-lancedb-pro plugin copy at ${REPO_PLUGIN_DIR}" >&2
  exit 1
fi

rm -rf "${LOCAL_PLUGIN_DIR}"
rm -rf "${LOCAL_PLUGIN_PRO_DIR}"

if [[ ! -d "${REPO_PLUGIN_DIR}/node_modules/@lancedb/lancedb" ]]; then
  npm install --omit=dev --prefix "${REPO_PLUGIN_DIR}"
fi

EXISTING_TOKEN=""
if [[ -f "${BENCH_CONFIG_PATH}" ]]; then
  EXISTING_TOKEN="$(sed -n 's/.*"token": "\(.*\)".*/\1/p' "${BENCH_CONFIG_PATH}" | head -n 1)"
fi

if [[ -n "${EXISTING_TOKEN}" ]]; then
  GATEWAY_TOKEN="${EXISTING_TOKEN}"
elif command -v openssl >/dev/null 2>&1; then
  GATEWAY_TOKEN="$(openssl rand -hex 24)"
else
  GATEWAY_TOKEN="$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')"
fi

RERANK_API_KEY_JSON=""
if [[ -n "${RERANK_API_KEY}" ]]; then
  RERANK_API_KEY_JSON=$(printf ',\n            "rerankApiKey": "%s"' "${RERANK_API_KEY}")
fi

RERANK_MODEL_JSON=""
if [[ -n "${RERANK_MODEL}" ]]; then
  RERANK_MODEL_JSON=$(printf ',\n            "rerankModel": "%s"' "${RERANK_MODEL}")
fi

RERANK_ENDPOINT_JSON=""
if [[ -n "${RERANK_ENDPOINT}" ]]; then
  RERANK_ENDPOINT_JSON=$(printf ',\n            "rerankEndpoint": "%s"' "${RERANK_ENDPOINT}")
fi

RERANK_PROVIDER_JSON=""
if [[ -n "${RERANK_PROVIDER}" ]]; then
  RERANK_PROVIDER_JSON=$(printf ',\n            "rerankProvider": "%s"' "${RERANK_PROVIDER}")
fi

cat > "${BENCH_CONFIG_PATH}" <<EOF
{
  "env": {
    "OPENAI_API_KEY": "\${OPENAI_API_KEY}"
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "openai/gpt-4.1-mini"
      },
      "workspace": "${WORKSPACE_DIR}",
      "memorySearch": {
        "store": {
          "path": "${SQLITE_DIR}/{agentId}.sqlite"
        }
      }
    }
  },
  "gateway": {
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "${GATEWAY_TOKEN}"
    },
    "http": {
      "endpoints": {
        "responses": {
          "enabled": true
        }
      }
    }
  },
  "plugins": {
    "allow": [
      "memory-core",
      "memory-lancedb-pro"
    ],
    "load": {
      "paths": [
        "${REPO_PLUGIN_DIR}"
      ]
    },
    "slots": {
      "memory": "memory-lancedb-pro"
    },
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "embedding": {
            "apiKey": "\${OPENAI_API_KEY}"
          },
          "dbPath": "${LANCEDB_PRO_DIR}",
          "autoCapture": false,
          "autoRecall": ${AUTO_RECALL},
          "smartExtraction": ${SMART_EXTRACTION},
          "retrieval": {
            "mode": "${RETRIEVAL_MODE}",
            "vectorWeight": ${RETRIEVAL_VECTOR_WEIGHT},
            "bm25Weight": ${RETRIEVAL_BM25_WEIGHT},
            "minScore": ${RETRIEVAL_MIN_SCORE},
            "hardMinScore": ${RETRIEVAL_HARD_MIN_SCORE},
            "candidatePoolSize": ${RETRIEVAL_CANDIDATE_POOL_SIZE},
            "rerank": "${RETRIEVAL_RERANK}",
            "filterNoise": ${RETRIEVAL_FILTER_NOISE},
            "recencyHalfLifeDays": ${RETRIEVAL_RECENCY_HALF_LIFE_DAYS},
            "recencyWeight": ${RETRIEVAL_RECENCY_WEIGHT},
            "lengthNormAnchor": ${RETRIEVAL_LENGTH_NORM_ANCHOR},
            "timeDecayHalfLifeDays": ${RETRIEVAL_TIME_DECAY_HALF_LIFE_DAYS},
            "reinforcementFactor": ${RETRIEVAL_REINFORCEMENT_FACTOR},
            "maxHalfLifeMultiplier": ${RETRIEVAL_MAX_HALF_LIFE_MULTIPLIER}${RERANK_API_KEY_JSON}${RERANK_MODEL_JSON}${RERANK_ENDPOINT_JSON}${RERANK_PROVIDER_JSON}
          }
        }
      }
    }
  }
}
EOF

cat > "${ENV_FILE}" <<EOF
OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR}"
OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH}"
OPENCLAW_GATEWAY_TOKEN="${GATEWAY_TOKEN}"
OPENAI_API_KEY="${OPENAI_API_KEY}"
EOF

cat <<EOF
OpenClaw benchmark state configured for memory-lancedb-pro (tuned).

OPENCLAW_STATE_DIR=${OPENCLAW_STATE_DIR}
OPENCLAW_CONFIG_PATH=${OPENCLAW_CONFIG_PATH}
OPENCLAW_GATEWAY_TOKEN=${GATEWAY_TOKEN}
AGENT_ID=${AGENT_ID}

memory-lancedb store:
  ${LANCEDB_DIR}

memory-lancedb-pro store:
  ${LANCEDB_PRO_DIR}

loaded memory-lancedb-pro plugin:
  ${REPO_PLUGIN_DIR}

Current retrieval settings:
  mode=${RETRIEVAL_MODE}
  vectorWeight=${RETRIEVAL_VECTOR_WEIGHT}
  bm25Weight=${RETRIEVAL_BM25_WEIGHT}
  minScore=${RETRIEVAL_MIN_SCORE}
  hardMinScore=${RETRIEVAL_HARD_MIN_SCORE}
  candidatePoolSize=${RETRIEVAL_CANDIDATE_POOL_SIZE}
  rerank=${RETRIEVAL_RERANK}
  rerankApiKey=$([[ -n "${RERANK_API_KEY}" ]] && printf 'configured' || printf 'unset')
  filterNoise=${RETRIEVAL_FILTER_NOISE}
  recencyHalfLifeDays=${RETRIEVAL_RECENCY_HALF_LIFE_DAYS}
  recencyWeight=${RETRIEVAL_RECENCY_WEIGHT}
  lengthNormAnchor=${RETRIEVAL_LENGTH_NORM_ANCHOR}
  timeDecayHalfLifeDays=${RETRIEVAL_TIME_DECAY_HALF_LIFE_DAYS}
  reinforcementFactor=${RETRIEVAL_REINFORCEMENT_FACTOR}
  maxHalfLifeMultiplier=${RETRIEVAL_MAX_HALF_LIFE_MULTIPLIER}
  autoRecall=${AUTO_RECALL}
  smartExtraction=${SMART_EXTRACTION}

Edit the variable block at the top of ./setup_memory_lancedb_pro_tune.sh to change these settings.

To switch back to the comparable baseline:
  ./setup_memory_lancedb_pro.sh

Next step:
  ./start_gateway.sh
EOF
