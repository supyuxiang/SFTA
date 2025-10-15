set -x

# 使用2个GPU设备
nproc_per_node=2
save_path=/home/yxfeng/verl/models/sft_gpu2
save_name='char_count-sft-Qwen2.5-1.5B-Instruct-2GPU-20250922'
train_data_path=/home/yxfeng/data/char_count/sft/train.parquet
test_data_path=/home/yxfeng/data/char_count/sft/test.parquet

#export HF_ENDPOINT=https://hf-mirror.com
# 使用GPU 8,9
export CUDA_VISIBLE_DEVICES=8,9

# 添加调试环境变量
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export TORCH_USE_CUDA_DSA=1

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data_path \
    data.val_files=$test_data_path \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=1024 \
    data.truncation=right \
    data.train_batch_size=32 \
    use_remove_padding=True \
    model.partial_pretrain='Qwen/Qwen2.5-1.5B-Instruct' \
    trainer.default_local_dir=$save_path \
    trainer.project_name=char_count-sft \
    trainer.experiment_name=$save_name \
    trainer.total_epochs=2 \
    trainer.logger=['console','swanlab'] \
    trainer.test_freq=2 \
    trainer.save_freq=1000 \
    trainer.seed=1 \
    model.lora_rank=32 \
    model.lora_alpha=16
