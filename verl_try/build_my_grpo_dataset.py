from datasets import Dataset
import pandas as pd
import os


def build_rows(split: str):
    rows = []
    n = 100 if split == "train" else 10
    for i in range(n):
        # 模型需自评：最后一行输出严格格式 "SCORE: d"，d 为 0~9 的整数
        question = (
            f"请计算 12 + {i} 的结果，并只在####后给出答案。"
            "然后对你的答案进行自我评价，给出 0~9 的分数，9 表示非常确定，"
            "0 表示完全不确定。最后一行严格以 'SCORE: d' 的格式输出分数（d 为 0~9）。"
        )
        ground_truth = str(12 + i)
        rows.append({
            "data_source": "my_dataset",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": split, "index": i, "answer": ground_truth, "question": question},
        })
    return rows


def main(out_dir: str = "/home/yxfeng/data/my_grpo"):
    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.DataFrame(build_rows("train"))
    test_df = pd.DataFrame(build_rows("test"))
    Dataset.from_pandas(train_df).to_parquet(os.path.join(out_dir, "train.parquet"))
    Dataset.from_pandas(test_df).to_parquet(os.path.join(out_dir, "test.parquet"))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()


