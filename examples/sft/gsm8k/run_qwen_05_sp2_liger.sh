set -x


if [ "$#" -lt 0 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

CUDA_VISIBLE_DEVICES=1,2,3,4
nproc_per_node=4
model_dir=/home/yxfeng/models/qwen/Qwen2.5-1.5B-Instruct
save_path=/home/yxfeng/models/qwen/Qwen2.5-1.5B-Instruct-sft

# Shift the arguments so $@ refers to the rest
shift 0

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=$model_dir \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-1.5b-instruct-sp2-liger \
    trainer.logger=console $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
