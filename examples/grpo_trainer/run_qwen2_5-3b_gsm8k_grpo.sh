#!/bin/bash
# GRPO training script with reference model quality scoring
# This script enables the reference model to score actor-generated responses
# and blend the quality scores with original rewards during PPO training

set -x

#export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=1,2,3,4
n_gpus_per_node=4

# 简单的环境变量设置
#export NCCL_SOCKET_IFNAME=lo
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=29500

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/yxfeng/data/gsm8k/train.parquet \
    data.val_files=/home/yxfeng/data/gsm8k/test.parquet \
    data.train_batch_size=1536 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=/home/yxfeng/models/qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=384 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.ref_quality_score.enable=True \
    algorithm.ref_quality_score.max_weight=0.3 \
    algorithm.ref_quality_score.warmup_ratio=0.7 \
    algorithm.ref_quality_score.schedule=linear \
    algorithm.ref_quality_score.steps=3 \
    algorithm.repr_align.enable=True \
    algorithm.repr_align.ref_alignment.enable=True \
    algorithm.repr_align.ref_alignment.max_weight=0.2 \
    algorithm.repr_align.ref_alignment.warmup_ratio=0.7 \
    algorithm.repr_align.ref_alignment.schedule=sigmoid \
    algorithm.repr_align.ref_alignment.entropy_high_threshold=0.2 \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_grpo_qwen2.5_3b_gsm8k' \
    trainer.experiment_name='qwen2.5_3b_grpo_full_ref_quality' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    actor_rollout_ref.nccl_timeout=1200 \
    ray_kwargs.ray_init.num_cpus=16 $@
