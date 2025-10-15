#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH_1="${MODEL_PATH_1:-/data1/yxfeng/models/qwen/Qwen2.5-0.5B-Instruct}"
MODEL_PATH_2="${MODEL_PATH_2:-/data1/yxfeng/models/qwen/Qwen2.5-1.5B-Instruct}"
MODEL_PATH_3="${MODEL_PATH_3:-/data1/yxfeng/models/qwen/Qwen2.5-3B-Instruct}"

# Accept prompt from arg; if empty, use a safe default without tricky quoting
PROMPT="${1:-}"
if [ -z "$PROMPT" ]; then
  PROMPT="Please provide a detailed introduction to the basic ideas of reinforcement learning, derivation formulas, and important algorithms such as 'PPO', 'GRPO', 'GSPO', etc. Then, rate your generated content on a scale of 0 to 10, with 10 being the highest. Place the score at the end using the format: ####Score."
fi

python3 "$SCRIPT_DIR/quick_gen.py" \
  --model "$MODEL_PATH_1" \
  --prompt "$PROMPT" \
  --use-chat \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.8

python3 "$SCRIPT_DIR/quick_gen.py" \
  --model "$MODEL_PATH_2" \
  --prompt "$PROMPT" \
  --use-chat \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.8

python3 "$SCRIPT_DIR/quick_gen.py" \
  --model "$MODEL_PATH_3" \
  --prompt "$PROMPT" \
  --use-chat \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.8

