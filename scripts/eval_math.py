#!/usr/bin/env python3
"""
MATH数据集评估脚本
用于评估模型在MATH验证集上的准确率
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error


def extract_final_answer_math(text: str) -> str:
    """
    从MATH模型输出中提取最终答案
    MATH数据集使用\\boxed{}格式
    """
    # 首先尝试匹配\\boxed{}格式
    boxed_patterns = [
        r"\\boxed\{([^}]+)\}",  # \boxed{answer}
        r"\\boxed\{([^}]*)\}",  # \boxed{} (可能为空)
        r"boxed\{([^}]+)\}",    # boxed{answer} (没有反斜杠)
        r"boxed\{([^}]*)\}",    # boxed{} (没有反斜杠，可能为空)
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 返回最后一个匹配的答案
            answer = matches[-1].strip()
            if answer:  # 如果答案不为空
                return answer
    
    # 如果没有找到\\boxed{}格式，尝试其他常见格式
    # 匹配各种答案格式
    answer_patterns = [
        # —— LaTeX 盒子/框线类 ——
        r"\\boxed\s*\{\s*([^}]+)\s*\}",                  # \\boxed{...}
        r"boxed\s*\{\s*([^}]+)\s*\}",                     # boxed{...}（无反斜杠）
        r"\\bbox\s*\{[^}]*\}\s*\{\s*([^}]+)\s*\}",    # \\bbox{..}{...} / \\bbox[..]{...}
        r"\\fbox\s*\{\s*([^}]+)\s*\}",                  # \\fbox{...}
        r"fbox\s*\{\s*([^}]+)\s*\}",                     # fbox{...}（无反斜杠）
        r"\\colorbox\s*\{[^}]*\}\s*\{\s*([^}]+)\s*\}",# \\colorbox{color}{...}
        r"\\framebox\s*\{\s*([^}]+)\s*\}",              # \\framebox{...}
        r"\\ovalbox\s*\{\s*([^}]+)\s*\}",               # \\ovalbox{...}
        r"\\doublebox\s*\{\s*([^}]+)\s*\}",             # \\doublebox{...}

        # —— Markdown/标题/显式标签 ——
        r"(?m)^\s*####\s*([^\n]+)",                         # ####xxx（四级标题）
        r"(?m)^\s*\*?\*?Answer\*?\*?\s*[:：]?\s*([^\n]+)", # Answer: / **Answer:**
        r"(?m)^\s*Ans(?:wer)?\.?\s*[:：]?\s*([^\n]+)",     # Ans: / Answer.
        r"(?m)^\s*A\s*[:：]\s*([^\n]+)",                   # A: xxx

        # —— 英文常见提示 ——
        r"Answer:\s*([^\n]+)",                               # Answer: xxx
        r"The answer is\s*([^\n]+)",                         # The answer is xxx
        r"Final answer:\s*([^\n]+)",                         # Final answer: xxx
        r"Therefore,?\s*([^\n]+)",                           # Therefore, xxx
        r"So,?\s*([^\n]+)",                                  # So, xxx
        r"Result:\s*([^\n]+)",                               # Result: xxx

        # —— 中文常见提示 ——
        r"(?:答案|最终答案|结论|结果|解)\s*[:：]\s*([^\n]+)",   # 答案: xxx 等

        # —— 行尾等号表达 ——
        r"=\s*([^\n]+)$",                                    # = xxx

        # —— LaTeX 行内/行间数学 ——
        r"\\\(([^)]+)\\\)",                               # \\( ... \\)
        r"\\\[([^\]]+)\\\]",                             # \\\[ ... \\\]
        r"\$\$([^$\n]+)\$\$",                              # $$ ... $$（单行）
        r"\$([^$\n]+)\$",                                   # $ ... $（单行）

        # —— 兼容原有兜底 ——
        r"\\$([^$]+)\\$",                                   # $xxx$（原有）
        r"\\\\([^\\\\]+)\\\\",                      # \\\\xxx\\\\（原有）
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            if answer:
                return answer
    
    # 最后尝试提取文本中的最后一个数字或表达式
    # 匹配数学表达式和数字
    math_patterns = [
        r"([-+]?[0-9]*\.?[0-9]+)",                    # 数字
        r"([-+]?[0-9]*\.?[0-9]+/[0-9]*\.?[0-9]+)",   # 分数
        r"([-+]?[0-9]*\.?[0-9]+\^[0-9]+)",           # 指数
        r"([a-zA-Z]+)",                               # 变量
    ]
    
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    # 如果什么都没找到，返回原始文本的最后一部分
    return text.strip().split()[-1] if text.strip() else ""


def _to_text(value, prefer_keys):
    """将值转换为文本格式"""
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
    """运行模型生成"""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
    outputs = tokenizer.batch_decode(
        gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return outputs


def evaluate_math(model_dir: str, val_parquet: str, 
                 max_prompt_len: int = 512, max_gen_len: int = 512, 
                 batch_size: int = 4, device: str = "cuda",
                 out_path: str | None = None,
                 judge_model_dir: str | None = None,
                 judge_max_len: int = 64):
    """
    在MATH数据集上评估模型
    
    Args:
        model_dir: 模型目录路径
        val_parquet: MATH验证集parquet文件路径
        max_prompt_len: 最大提示长度
        max_gen_len: 最大生成长度
        batch_size: 批处理大小
        device: 设备
        out_path: 输出结果文件路径
    """
    # 静默HF警告
    set_verbosity_error()

    print(f"Loading model from {model_dir}")
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

    print(f"Loading MATH dataset from {val_parquet}")
    df = pd.read_parquet(val_parquet)
    
    # 提取提示和标签
    prompts = []
    labels = []
    
    for _, row in df.iterrows():
        # 提取提示内容
        prompt_data = row['prompt']
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            prompt_text = prompt_data[0]['content']
        else:
            prompt_text = str(prompt_data)
        prompts.append(prompt_text)
        
        # 提取标准答案
        reward_model = row['reward_model']
        if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
            ground_truth = reward_model['ground_truth']
        else:
            ground_truth = str(reward_model)
        labels.append(ground_truth)

    print(f"Total samples: {len(prompts)}")
    
    total = 0
    num_correct = 0
    judge_sum = 0.0
    combined_sum = 0.0
    detailed_results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_prompt_len).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_gen_len, do_sample=False)
        outputs = tokenizer.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 评估每个样本（对预测与标签均进行统一的答案提取与规范化）
        for j, (out, label, prompt) in enumerate(zip(outputs, batch_labels, batch_prompts)):
            pred = extract_final_answer_math(out)
            gold = extract_final_answer_math(str(label))

            pred_clean = pred.strip().lower()
            gold_clean = gold.strip().lower()

            is_correct = pred_clean == gold_clean
            
            if is_correct:
                num_correct += 1
            total += 1
            
            # 保存详细结果
            detailed_results.append({
                "index": i + j,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "ground_truth": gold,
                "prediction": pred,
                "model_output": out[:500] + "..." if len(out) > 500 else out,
                "correct": is_correct
            })
            
            if total % 100 == 0:
                print(f"Processed {total} samples, current accuracy: {num_correct/total:.4f}")

            # Judge scoring (optional)
            if judge_model is not None:
                jp = _build_judge_prompt(prompt, out, gold)
                judge_out = _run_generation(judge_tok, judge_model, [jp], max_len=judge_max_len)[0]
                import re as _re
                m = _re.search(r"[-+]?[0-9]*\.?[0-9]+", judge_out)
                judge_score = None
                if m:
                    try:
                        judge_score = float(m.group(0))
                    except Exception:
                        judge_score = None
                if judge_score is None:
                    judge_score = 1.0 if is_correct else 0.0
                judge_score = max(0.0, min(1.0, judge_score))
                judge_sum += judge_score
                combined_sum += 0.5 * ((1.0 if is_correct else 0.0) + judge_score)

    accuracy = num_correct / max(1, total)

    if judge_model is not None:
        mean_judge = judge_sum / max(1, total)
        mean_combined = combined_sum / max(1, total)
        result = {
            "num_examples": total,
            "num_correct": num_correct,
            "accuracy_em": accuracy,
            "judge_mean": mean_judge,
            "score_mean": mean_combined,
            "detailed_results": detailed_results[:10],
        }
    else:
        result = {
            "num_examples": total,
            "num_correct": num_correct,
            "accuracy": accuracy,
            "detailed_results": detailed_results[:10],  # 只保存前10个详细结果作为示例
        }

    print(f"\n=== MATH Evaluation Results ===")
    print(f"Total examples: {total}")
    print(f"Correct answers: {num_correct}")
    if judge_model is not None:
        print(f"Accuracy (EM): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Judge mean: {mean_judge:.4f}")
        print(f"Score mean: {mean_combined:.4f}")
    else:
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"================================")

    # 保存结果
    if out_path:
        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MATH dataset")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory")
    parser.add_argument("--val_parquet", required=True, help="Path to MATH validation parquet file")
    parser.add_argument("--max_prompt_len", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--max_gen_len", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--out", type=str, default=None, help="Output file path for results")
    parser.add_argument("--judge_model_dir", type=str, default=None, help="Optional judge model directory")
    parser.add_argument("--judge_max_len", type=int, default=64, help="Max new tokens for judge model")
    
    args = parser.parse_args()

    evaluate_math(
        model_dir=args.model_dir,
        val_parquet=args.val_parquet,
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
cd /home/yxfeng/verl && python3 scripts/eval_math.py \
  --model_dir /home/yxfeng/verl/checkpoints/gsm8k_qwen25_1d5b_sft_baseline_20251009_010627/global_step_6990_merged \
  --val_parquet /home/yxfeng/data/math/test.parquet \
  --batch_size 2 \
  --max_prompt_len 512 \
  --max_gen_len 512 \
  --out /home/yxfeng/verl/qwen2.5-1.5b_sft_step6990_math_eval_results.json
'''

'''python3 /home/yxfeng/verl/scripts/eval_math.py \
  --model_dir /home/yxfeng/models/qwen/Qwen2.5-1.5B-Instruct \
  --val_parquet /home/yxfeng/data/math/test.parquet \
  --batch_size 12 \
  --max_prompt_len 1024 \
  --max_gen_len 1024 \
  --out /home/yxfeng/verl/qwen2.5-1.5b-instruct_math_eval_results.json'''