#!/usr/bin/env bash

set -euo pipefail

# Restrict to the specified GPUs only
export CUDA_VISIBLE_DEVICES=6,7,8,9

PY="/home/yxfeng/miniconda3/envs/fyx_verl/bin/python"
REPO_ROOT="/home/yxfeng/verl"
EVAL_SCRIPT="$REPO_ROOT/scripts/eval_math.py"

# MATH validation parquet
VAL_PARQUET="/data1/yxfeng/data/math/test.parquet"

# Output directories
OUT_DIR="/home/yxfeng/verl/accuracy"
TMP_DIR="$OUT_DIR/tmp"
mkdir -p "$TMP_DIR"

# Six target models (absolute paths)
MODELS=(
  "/data1/yxfeng/models/qwen/Qwen2.5-0.5B-Instruct"
  "/data1/yxfeng/models/qwen/Qwen2.5-1.5B-Instruct"
  "/data1/yxfeng/models/qwen/Qwen2.5-3B-Instruct"
  "/data1/yxfeng/models/qwen/Qwen2.5-7B-Instruct"
  "/data1/yxfeng/models/qwen/Qwen2.5-Math-1.5B"
  "/data1/yxfeng/models/qwen/Qwen2.5-Math-7B"
)

# Reasonable defaults (adjust if needed)
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_GEN_LEN=${MAX_GEN_LEN:-512}

echo "[INFO] Evaluating ${#MODELS[@]} models on MATH (VAL: $VAL_PARQUET) using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2

RESULT_FILES=()
for MODEL_DIR in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_DIR")
  if [ ! -d "$MODEL_DIR" ]; then
    echo "[WARN] Skip $MODEL_NAME: directory not found at $MODEL_DIR" >&2
    continue
  fi

  OUT_FILE="$TMP_DIR/${MODEL_NAME}.json"
  echo "[INFO] -> $MODEL_NAME" >&2
  "$PY" -u "$EVAL_SCRIPT" \
    --model_dir "$MODEL_DIR" \
    --val_parquet "$VAL_PARQUET" \
    --batch_size "$BATCH_SIZE" \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_gen_len "$MAX_GEN_LEN" \
    --out "$OUT_FILE"

  RESULT_FILES+=("$OUT_FILE")
done

# Combine results into a single JSON
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUT_DIR/math_accuracy_summary_${TIMESTAMP}.json"

"$PY" - <<'PY'
import json, os, glob

out_dir = "/home/yxfeng/verl/accuracy"
tmp_dir = os.path.join(out_dir, "tmp")
summary = {"results": [], "meta": {"dataset": "MATH", "val_parquet": "/data1/yxfeng/data/math/test.parquet"}}

for fp in sorted(glob.glob(os.path.join(tmp_dir, "*.json"))):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        model_name = os.path.splitext(os.path.basename(fp))[0]
        # Prefer key "accuracy"; fallback to "accuracy_em"
        acc = obj.get("accuracy")
        if acc is None:
            acc = obj.get("accuracy_em")
        summary["results"].append({
            "model": model_name,
            "num_examples": obj.get("num_examples"),
            "num_correct": obj.get("num_correct"),
            "accuracy": acc,
        })
    except Exception as e:
        summary["results"].append({
            "model": os.path.splitext(os.path.basename(fp))[0],
            "error": str(e),
        })

summary_path = os.environ.get("SUMMARY_FILE_PATH")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Wrote:", summary_path)
PY

SUMMARY_FILE_PATH="$SUMMARY_FILE" SUMMARY_FILE_PATH="$SUMMARY_FILE" "$PY" - <<'PY'
import os
print(os.environ["SUMMARY_FILE_PATH"])  # placeholder to ensure env var exported for the previous block
PY >/dev/null 2>&1 || true

echo "[OK] Combined results saved to: $SUMMARY_FILE" >&2
echo "$SUMMARY_FILE"


