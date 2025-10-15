#!/bin/bash
# SFT training script for Qwen2.5-1.5B on GSM8K

set -x

export CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH="${PYTHONPATH}:/home/yxfeng/verl"
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Model and data paths
model_path=/data1/yxfeng/verl/models/qwen/Qwen2.5-1.5B-Instruct
train_file=/data1/yxfeng/verl/data/gsm8k/train.parquet
val_file=/data1/yxfeng/verl/data/gsm8k/test.parquet
save_dir=/data1/yxfeng/verl/SERA/baseline/qwen/Qwen2.5-1.5B-Instruct/gsm8k/gsm8k_qwen25_1d5b_sft_$(date +%Y%m%d_%H%M%S)

# Training parameters
nproc_per_node=2
train_batch_size=32
micro_batch_size_per_gpu=4
learning_rate=1e-4
total_epochs=10
save_freq=100
test_freq=25

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.train_batch_size=$train_batch_size \
    data.max_length=512 \
    data.truncation=right \
    data.prompt_key=prompt \
    data.response_key=prompt \
    data.prompt_dict_keys='["content"]' \
    data.response_dict_keys='["content"]' \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    model.partial_pretrain="$model_path" \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=False \
    model.enable_gradient_checkpointing=True \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules="all-linear" \
    optim.lr=$learning_rate \
    trainer.default_local_dir="$save_dir" \
    trainer.project_name=verl_baseline_sft_qwen2.5_1.5b_gsm8k \
    trainer.experiment_name=qwen2.5_1.5b_sft_baseline_gsm8k \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.resume_mode=disable \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true $@