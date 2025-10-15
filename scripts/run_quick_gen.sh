#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/data1/yxfeng/models/qwen/Qwen2.5-3B-Instruct}"
PROMPT="${1:-用三句话介绍强化学习的基本思想}"

python3 "$SCRIPT_DIR/quick_gen.py" \
  --model "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --use-chat \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9

