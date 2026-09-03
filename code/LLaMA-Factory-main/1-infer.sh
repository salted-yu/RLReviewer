#!/usr/bin/env bash

set -euo pipefail

dataset=cr_sft_test
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPLICATION_ROOT="${REPLICATION_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
model_name_or_path="${SFT_MODEL_PATH:-${REPLICATION_ROOT}/models/sft_model}"
path_to_merged_model_base="${RL_CHECKPOINT_DIR:-${REPLICATION_ROOT}/checkpoints/rl_model}"

adapter_name_or_path="${path_to_merged_model_base}/actor/lora_adapter"
save_name="${dataset}.jsonl"

python scripts/vllm_infer.py \
    --model_name_or_path "${model_name_or_path}" \
    --adapter_name_or_path "${adapter_name_or_path}" \
    --dataset "${dataset}" \
    --temperature 0 \
    --cutoff_len 8192 \
    --max_new_tokens 1024 \
    --save_name "${save_name}"
