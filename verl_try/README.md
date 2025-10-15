VERL GRPO quick practice

Contents
- build_my_grpo_dataset.py: creates a tiny toy dataset at /home/yxfeng/data/my_grpo
- reward_function_my.py: simple rule-based reward comparing model answer with ground truth
- train_grpo_try.sh: example launch script wired to the above dataset and reward

Steps
1) Build dataset
   python3 /home/yxfeng/verl/verl_try/build_my_grpo_dataset.py

2) Train (adjust CUDA ids/model path as needed)
   bash /home/yxfeng/verl/verl_try/train_grpo_try.sh

Notes
- Uses absolute paths to avoid Ray working_dir issues
- Truncation set to right and overlong prompt filtering enabled
- vLLM chunked prefill disabled to avoid max_num_batched_tokens vs max_model_len mismatch

