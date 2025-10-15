set -x

#export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=1,2,3,8
n_gpus_per_node=4

model_path=/data1/yxfeng/verl/models/qwen/Qwen2.5-0.5B-Instruct
train_file=/data1/yxfeng/verl/data/gsm8k/train.parquet
val_file=/data1/yxfeng/verl/data/gsm8k/test.parquet
SAVE_DIR=/data1/yxfeng/verl/SERA/baseline/qwen/Qwen2.5-0.5B-Instruct/gsm8k/gsm8k_qwen25_0d5b_sft_baseline_$(date +%Y%m%d_%H%M%S)
train_batch_size=32
micro_batch_size_per_gpu=4
learning_rate=1e-4
total_epochs=10
save_freq=100
test_freq=25

python3 -m verl.trainer.fsdp_sft_trainer \
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
    model.gradient_checkpointing_kwargs.max_chunk_size=8192 \
    trainer.project_name=verl_baseline_sft_qwen2.5_0.5b_gsm8k \
    trainer.experiment_name=qwen2.5_0.5b_sft_baseline_gsm8k \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.save_dir=$SAVE_DIR









