import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error


def extract_final_answer(text: str) -> str:
    r"""Extract final answer from GSM8K-like outputs.
    Order:
      1) Try Markdown heading lines like '#### xxx'
      2) Try LaTeX box family: \boxed{...}, \bbox{...}{...}, \fbox{...}, \colorbox{...}{...}, \framebox{...}
      3) Fallback to last number in text
      4) Fallback to stripped text
    """
    import re

    # 1) Markdown heading '#### xxx'
    m = re.findall(r"(?m)^\s*####\s*([^\n]+)", text)
    if m:
        candidate = m[-1].strip()
        if candidate:
            return candidate

    # 2) LaTeX box family
    box_patterns = [
        r"\\boxed\s*\{\s*([^}]+)\s*\}",                  # \\boxed{...}
        r"boxed\s*\{\s*([^}]+)\s*\}",                     # boxed{...}（无反斜杠）
        r"\\bbox\s*\{[^}]*\}\s*\{\s*([^}]+)\s*\}",    # \\bbox{..}{...} / \\bbox[..]{...}
        r"\\fbox\s*\{\s*([^}]+)\s*\}",                  # \\fbox{...}
        r"fbox\s*\{\s*([^}]+)\s*\}",                     # fbox{...}（无反斜杠）
        r"\\colorbox\s*\{[^}]*\}\s*\{\s*([^}]+)\s*\}",# \\colorbox{color}{...}
        r"\\framebox\s*\{\s*([^}]+)\s*\}",              # \\framebox{...}
    ]
    for pat in box_patterns:
        hits = re.findall(pat, text)
        if hits:
            candidate = hits[-1].strip()
            if candidate:
                return candidate

    # 3) Fallback: last number in the string
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if nums:
        return nums[-1]

    # 4) Final fallback: stripped text
    return text.strip()


def _to_text(value, prefer_keys):
    # If already string, try JSON parse fallback
    if isinstance(value, str):
        v = value
        try:
            obj = json.loads(value)
            value = obj
        except Exception:
            return v
    if isinstance(value, dict):
        for k in prefer_keys:
            if k in value and value[k] is not None:
                return str(value[k])
        # fallback: first non-empty field
        for k, v in value.items():
            if v is not None:
                return str(v)
        return ""
    return str(value)


def _build_judge_prompt(question: str, model_output: str, answer: str) -> str:
    return (
        "You are a math answer judge. Given the question, the model's solution, and the correct answer, decide whether the final answer is correct."
        " Output only a single score of 0 or 1, where 1 means correct and 0 means incorrect. Do not output anything else.\n"
        f"Question: {question}\n"
        f"Model solution: {model_output}\n"
        f"Correct answer: {answer}\n"
        "Score:"
    )


def _run_generation(tokenizer, model, prompts, max_len: int):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
    outputs = tokenizer.batch_decode(
        gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return outputs


def evaluate(model_dir: str, val_parquet: str, prompt_key: str, answer_key: str,
             max_prompt_len: int = 256, max_gen_len: int = 128, batch_size: int = 4, device: str = "cuda",
             out_path: str | None = None, judge_model_dir: str | None = None, judge_max_len: int = 64):
    # Silence HF warnings about unused generation flags
    set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Optional judge model
    judge_tok = None
    judge_model = None
    if judge_model_dir:
        judge_tok = AutoTokenizer.from_pretrained(judge_model_dir, trust_remote_code=False)
        judge_tok.padding_side = "left"
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        judge_model.eval()

    df = pd.read_parquet(val_parquet)
    raw_prompts = df[prompt_key].tolist()
    raw_labels = df[answer_key].tolist()

    prompts = [_to_text(x, ["question", "content", "prompt"]) for x in raw_prompts]
    labels = [_to_text(x, ["answer", "solution", "final_answer"]) for x in raw_labels]

    # Progress: dataset size
    print(f"Total samples: {len(prompts)}")

    total = 0
    num_correct = 0
    judge_sum = 0.0
    combined_sum = 0.0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        # Progress: per-batch progress
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_prompt_len).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_gen_len, do_sample=False)
        outputs = tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # EM reward
        batch_labels = labels[i:i+batch_size]
        for out, label, q in zip(outputs, batch_labels, batch_prompts):
            pred = extract_final_answer(out)
            gold = extract_final_answer(str(label))
            em = 1.0 if pred == gold else 0.0
            num_correct += int(em == 1.0)
            total += 1

            judge_score = None
            if judge_model is not None:
                jp = _build_judge_prompt(q, out, gold)
                judge_out = _run_generation(judge_tok, judge_model, [jp], max_len=judge_max_len)[0]
                # parse score
                import re
                m = re.search(r"[-+]?[0-9]*\.?[0-9]+", judge_out)
                if m:
                    try:
                        judge_score = float(m.group(0))
                    except Exception:
                        judge_score = None
                if judge_score is None:
                    judge_score = em  # fallback
                judge_score = max(0.0, min(1.0, judge_score))
                judge_sum += judge_score
                combined_sum += 0.5 * (em + judge_score)
            else:
                combined_sum += em

            # Periodic progress every 100 samples
            if total % 100 == 0:
                curr_acc = num_correct / max(1, total)
                print(f"Processed {total} samples, current EM accuracy: {curr_acc:.4f}")

    acc = num_correct / max(1, total)
    if judge_model is not None:
        mean_judge = judge_sum / max(1, total)
        mean_combined = combined_sum / max(1, total)
        result = {
            "num_examples": total,
            "num_correct": num_correct,
            "accuracy_em": acc,
            "judge_mean": mean_judge,
            "score_mean": mean_combined
        }
    else:
        result = {"num_examples": total, "num_correct": num_correct, "accuracy": acc}

    # Final human-readable summary
    print("\n=== GSM8K Evaluation Results ===")
    print(f"Total examples: {total}")
    print(f"Correct answers (EM): {num_correct}")
    if judge_model is not None:
        print(f"Accuracy (EM): {acc:.4f} ({acc*100:.2f}%)")
        print(f"Judge mean: {mean_judge:.4f}")
        print(f"Score mean: {mean_combined:.4f}")
    else:
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("================================")

    print(json.dumps(result, ensure_ascii=False))

    if out_path:
        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--val_parquet", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--judge_model_dir", type=str, default=None)
    parser.add_argument("--judge_max_len", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        val_parquet=args.val_parquet,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        max_prompt_len=args.max_prompt_len,
        max_gen_len=args.max_gen_len,
        batch_size=args.batch_size,
        out_path=args.out,
        judge_model_dir=args.judge_model_dir,
        judge_max_len=args.judge_max_len,
    )


if __name__ == "__main__":
    main()


'''
python /home/yxfeng/verl/scripts/eval_gsm8k.py \
  --model_dir /home/yxfeng/models/qwen/Qwen2.5-3B-Instruct \
  --val_parquet /home/yxfeng/data/gsm8k/test.parquet \
  --prompt_key extra_info --answer_key extra_info \
  --batch_size 4 --max_prompt_len 256 --max_gen_len 256 \
  --judge_model_dir /home/yxfeng/models/qwen/Qwen2.5-3B-Instruct \
  --out /home/yxfeng/output_verl_accuracy/gsm8k_eval_Qwen2.5-3B-Instruct_with_judge.json
  '''