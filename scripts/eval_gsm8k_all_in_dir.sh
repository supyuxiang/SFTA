#!/usr/bin/env bash

set -euo pipefail

# User-configurable paths
ROOT_DIR="/data1/chzhang/verl/checkpoints/gsm8k_qwen25_1d5b_sft_baseline_20251009_010627"
VAL_PARQUET="/data1/chzhang/data/gsm8k/test.parquet"

# Working/output paths
SCRIPTS_DIR="/data1/chzhang/verl/scripts"
OUT_DIR="/data1/chzhang/verl/eval_results/gsm8k_sft_10627"
AGG_JSON="${OUT_DIR}/aggregate_results.json"

mkdir -p "${OUT_DIR}"

echo "[INFO] Scanning checkpoints under: ${ROOT_DIR}" >&2

shopt -s nullglob
for STEP_DIR in "${ROOT_DIR}"/global_step_*; do
  [ -d "${STEP_DIR}" ] || continue
  STEP_BN=$(basename "${STEP_DIR}")

  # Decide merged dir and local_dir for model_merger
  if [ -d "${STEP_DIR}/actor" ]; then
    LOCAL_DIR="${STEP_DIR}/actor"
    MERGED_DIR="${STEP_DIR}_merged"
  else
    LOCAL_DIR="${STEP_DIR}"
    MERGED_DIR="${STEP_DIR}_merged"
  fi

  # Merge if needed
  if [ ! -d "${MERGED_DIR}" ] || [ ! -f "${MERGED_DIR}/model.safetensors" ]; then
    echo "[INFO] Merging checkpoint: ${LOCAL_DIR} -> ${MERGED_DIR}" >&2
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${LOCAL_DIR}" \
      --target_dir "${MERGED_DIR}"
  else
    echo "[INFO] Reusing existing merged model: ${MERGED_DIR}" >&2
  fi

  # Evaluate
  OUT_JSON="${OUT_DIR}/${STEP_BN}.json"
  if [ -f "${OUT_JSON}" ]; then
    echo "[INFO] Skip eval (exists): ${OUT_JSON}" >&2
  else
    echo "[INFO] Evaluating ${MERGED_DIR} on GSM8K -> ${OUT_JSON}" >&2
    python "${SCRIPTS_DIR}/eval_gsm8k.py" \
      --model_dir "${MERGED_DIR}" \
      --val_parquet "${VAL_PARQUET}" \
      --prompt_key extra_info \
      --answer_key extra_info \
      --batch_size 4 \
      --max_prompt_len 256 \
      --max_gen_len 256 \
      --out "${OUT_JSON}"
  fi
done

echo "[INFO] Aggregating results into: ${AGG_JSON}" >&2
python - <<'PY'
import json, os, re
from pathlib import Path

out_dir = Path("/data1/chzhang/verl/eval_results/gsm8k_sft_10627")
agg_path = out_dir/"aggregate_results.json"

results = []
pat = re.compile(r"global_step_(\d+)\.json$")
for p in sorted(out_dir.glob("global_step_*.json")):
    m = pat.search(p.name)
    if not m:
        continue
    step = int(m.group(1))
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize keys across possible variants
        acc = data.get("accuracy") or data.get("accuracy_em") or data.get("score_mean")
        results.append({
            "step": step,
            "file": p.name,
            "num_examples": data.get("num_examples"),
            "num_correct": data.get("num_correct"),
            "accuracy": acc,
        })
    except Exception as e:
        results.append({"step": step, "file": p.name, "error": str(e)})

results.sort(key=lambda x: x.get("step", 0))
with open(agg_path, "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, ensure_ascii=False, indent=2)

print(json.dumps({"results_path": str(agg_path)}, ensure_ascii=False))
PY

echo "[DONE] All evaluations aggregated at: ${AGG_JSON}" >&2


