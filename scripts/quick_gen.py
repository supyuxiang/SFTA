#!/usr/bin/env python3
"""
Minimal demo: load a HF causal LM (e.g., Qwen2.5-3B-Instruct) and generate from a prompt.

Usage examples:
  python3 quick_gen.py --model /data1/yxfeng/models/qwen/Qwen2.5-3B-Instruct --prompt "Write a haiku about the moon."
  python3 quick_gen.py --model /data1/yxfeng/models/qwen/Qwen2.5-3B-Instruct --use-chat \
      --prompt "帮我用中文解释牛顿第二定律" --system "You are a helpful AI assistant."
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple HF text generation demo")
    parser.add_argument("--model", required=True, help="Path or name of the HF model directory")
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--system", default="", help="System message (chat mode only)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-chat", action="store_true", help="Use tokenizer chat template if available")
    parser.add_argument("--no-stream", action="store_true", help="Disable incremental printing")
    return parser.parse_args()


def build_inputs(tokenizer: AutoTokenizer, prompt: str, system: str = "", use_chat: bool = False) -> torch.Tensor:
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    inputs = build_inputs(tokenizer, args.prompt, args.system, args.use_chat)

    do_stream = not args.no_stream
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    if do_stream:
        generated = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        print("\nAssistant:", end=" ", flush=True)
        for _ in range(args.max_new_tokens):
            out = model.generate(
                input_ids=generated,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=gen_kwargs["do_sample"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                eos_token_id=gen_kwargs["eos_token_id"],
                pad_token_id=gen_kwargs["pad_token_id"],
            )
            new_token = out[0, -1:]
            generated = out
            if attention_mask is not None:
                # attention_mask has shape [batch, seq_len]; append a column of ones
                ones_col = torch.ones(
                    (attention_mask.size(0), 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, ones_col], dim=-1)
            token_text = tokenizer.decode(new_token, skip_special_tokens=True)
            print(token_text, end="", flush=True)
            if gen_kwargs["eos_token_id"] is not None and new_token.item() == gen_kwargs["eos_token_id"]:
                break
        print()
    else:
        output_ids = model.generate(**inputs, **gen_kwargs)
        gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\nAssistant:", gen_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
