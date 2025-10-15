#!/bin/bash
# GRPO training script for Qwen2.5-0.5B on MATH with reference quality scoring

set -x

#export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=5,6,7,8
n_gpus_per_node=4
epochs=30
save_freq=400
test_freq=10
train_batch_size=128
lr=1e-6
ref_quality_score_enable=True
repr_align_enable=True


model_path=/data1/yxfeng/verl/models/qwen/Qwen2.5-1.5B-Instruct
train_file=/data1/yxfeng/verl/data/math/train.parquet
val_file=/data1/yxfeng/verl/data/math/test.parquet
SAVE_DIR=/data1/yxfeng/verl/SERA/ours/qwen/Qwen2.5-1.5B-Instruct/math/math_qwen25_1d5b_ours_$(date +%Y%m%d_%H%M%S)

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.ref_quality_score.enable=$ref_quality_score_enable \
    algorithm.ref_quality_score.max_weight=0.3 \
    algorithm.ref_quality_score.warmup_ratio=0.7 \
    algorithm.ref_quality_score.schedule=linear \
    algorithm.ref_quality_score.steps=3 \
    algorithm.repr_align.enable=$repr_align_enable \
    algorithm.repr_align.ref_alignment.enable=$ref_align_enable \
    algorithm.repr_align.ref_alignment.max_weight=0.2 \
    algorithm.repr_align.ref_alignment.warmup_ratio=0.7 \
    algorithm.repr_align.ref_alignment.schedule=sigmoid \
    algorithm.repr_align.ref_alignment.entropy_high_threshold=0.3 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.resume_mode=disable \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_ours_qwen2.5_1.5b_math' \
    trainer.experiment_name='qwen2.5_1.5b_grpo_lora_math_ours_grpo' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$epochs \
    trainer.default_local_dir=$SAVE_DIR \
    actor_rollout_ref.nccl_timeout=1200 \
    ray_kwargs.ray_init.num_cpus=16 $@


