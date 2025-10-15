set -x

#export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=6,7,8,9
n_gpus_per_node=4

model_path=/data1/yxfeng/models/qwen/Qwen2.5-0.5B-Instruct
train_file=/data1/yxfeng/data/math/train.parquet
val_file=/data1/yxfeng/data/math/test.parquet
save_root=/data1/yxfeng/grpo/baseline/qwen/Qwen2.5-0.5B-Instruct/math/math_qwen25_0d5b_sft_baseline_$(date +%Y%m%d_%H%M%S)
train_batch_size=32
micro_batch_size_per_gpu=4
learning_rate=1e-4
total_epochs=10
save_freq=100
test_freq=25
run_name=qwen2.5_0.5b_sft_baseline_math_ep${total_epochs}_lr${learning_rate}

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
    trainer.project_name=verl_baseline_sft_qwen2.5_0.5b_math \
    trainer.experiment_name=$run_name \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.total_epochs=$total_epochs \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.save_dir=$save_root









