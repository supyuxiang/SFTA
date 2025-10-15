#!/bin/bash
# GRPO training script for Qwen2.5-0.5B on MATH with reference quality scoring

set -x

export NCCL_P2P_DISABLE=1
export WANDB_API_KEY=
#export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=5,6,7,8
n_gpus_per_node=4

export HF_ENDPOINT="https://hf-mirror.com"
export ACCELERATE_LOG_LEVEL=info

export MODEL_PATH=/data1/yxfeng/models/qwen/Qwen2.5-0.5B-Instruct
export DATA_PATH=/data1/yxfeng/data/math

model_path=/data1/yxfeng/models/qwen/Qwen2.5-0.5B-Instruct
save_root=/data1/yxfeng/grpo/baseline/qwen/Qwen2.5-0.5B-Instruct/math/math_qwen25_0d5b_grpo_baseline_$(date +%Y%m%d_%H%M%S)
train_file=/data1/yxfeng/data/math/train.parquet
val_file=/data1/yxfeng/data/math/test.parquet

batch_size=128
ppo_micro_batch_size_per_gpu=16
total_epochs=25
lr=1e-6
lora_rank=32
lora_alpha=16
save_freq=400
test_freq=10

kl_loss_coef=0.0005
entropy_coeff=0.001
use_kl_loss=True

run_name=qwen2.5_0.5b_grpo_math_baseline_lora_ep${total_epochs}_lr${lr}
save_root=${save_root}/${run_name}
mkdir -p $save_root

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=$lora_rank \
    actor_rollout_ref.model.lora_alpha=$lora_alpha \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.ref_quality_score.enable=False \
    algorithm.ref_quality_score.max_weight=0.3 \
    algorithm.ref_quality_score.warmup_ratio=0.7 \
    algorithm.ref_quality_score.schedule=linear \
    algorithm.ref_quality_score.steps=3 \
    algorithm.repr_align.enable=False \
    algorithm.repr_align.ref_alignment.enable=False \
    algorithm.repr_align.ref_alignment.max_weight=0.2 \
    algorithm.repr_align.ref_alignment.warmup_ratio=0.7 \
    algorithm.repr_align.ref_alignment.schedule=sigmoid \
    algorithm.repr_align.ref_alignment.entropy_high_threshold=0.3 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.resume_mode=disable \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_grpo_qwen2.5_0.5b_math' \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.default_local_dir=$save_root \
    actor_rollout_ref.nccl_timeout=1200 \
    ray_kwargs.ray_init.num_cpus=16 $@


