#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import Any, Iterable, Optional


def _print_example(idx: int, example: dict[str, Any]) -> None:
    question = example.get("question") or example.get("query") or example.get("input")
    answer = example.get("answer") or example.get("output") or example.get("solution")
    print(f"===== Example {idx} =====")
    print("Question:")
    print(question if isinstance(question, str) else json.dumps(question, ensure_ascii=False))
    print("\nAnswer:")
    print(answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False))
    print()


def _load_from_hf(split: str, name: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import datasets. Please install it via `pip install datasets`."
        ) from e

    # gsm8k has configurations: "main" (train/test), "socratic" etc.
    ds = load_dataset("gsm8k", name, split=split)
    return (ds[i] for i in range(len(ds)))


def _load_from_local(path: str) -> Iterable[dict[str, Any]]:
    # Supports a directory containing JSONL files or a single JSONL file
    candidates: list[str] = []
    if os.path.isdir(path):
        for fn in os.listdir(path):
            if fn.endswith(".jsonl") or fn.endswith(".json"):
                candidates.append(os.path.join(path, fn))
    elif os.path.isfile(path):
        candidates.append(path)
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    def _iter_files(files: list[str]) -> Iterable[dict[str, Any]]:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # try whole-file JSON array fallback
                        f.seek(0)
                        arr = json.load(f)
                        if isinstance(arr, list):
                            for it in arr:
                                if isinstance(it, dict):
                                    yield it
                        break
                    else:
                        if isinstance(obj, dict):
                            yield obj

    return _iter_files(candidates)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show GSM8K examples from Hugging Face Datasets or a local JSONL file",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument(
        "--name",
        default="main",
        help="HF config name for gsm8k (default: main)",
    )
    parser.add_argument("--num", type=int, default=3, help="Number of examples to display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--local_path",
        type=str,
        default=os.environ.get("GSM8K_PATH", ""),
        help="Optional local path to JSONL/JSON files for GSM8K",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before selecting the first N examples",
    )
    args = parser.parse_args(argv)

    examples: list[dict[str, Any]] = []
    try:
        if args.local_path:
            it = _load_from_local(args.local_path)
        else:
            it = _load_from_hf(split=args.split, name=args.name)
        for i, ex in enumerate(it):
            examples.append(ex)
            # avoid loading entire dataset if not needed when using local JSONL
            if not args.local_path and i > 200000:  # hard guard
                break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    if not examples:
        print("No examples loaded.")
        return 1

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(examples)

    take = min(args.num, len(examples))
    for i in range(take):
        _print_example(i + 1, examples[i])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


