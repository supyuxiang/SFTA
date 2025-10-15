# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer

这是一个基于 PyTorch FSDP (Fully Sharded Data Parallel) 的监督微调(SFT)训练器。
主要功能包括：
1. 支持 FSDP 和 FSDP2 两种分布式训练策略
2. 支持序列并行 (Sequence Parallelism) 和 Ulysses 优化
3. 支持 Liger Kernel 加速
4. 支持 LoRA 微调
5. 支持梯度检查点和混合精度训练
6. 支持模型检查点保存和恢复
7. 支持训练过程中的 entropy 计算和记录

TODO(zhangchi.usc1992)
- Add calculation of mfu (Model FLOPs Utilization)
- Add validation
"""

import os

# 设置环境变量
os.environ["NCCL_DEBUG"] = "INFO"  # 启用 NCCL 调试信息
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 启用 tokenizer 并行处理

import logging
import re
from contextlib import nullcontext

# 核心依赖
import hydra  # 配置管理
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf  # 配置处理
from peft import LoraConfig, TaskType, get_peft_model  # LoRA 微调
from tensordict import TensorDict  # 张量字典
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # 设备网格
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy  # FSDP 组件
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # FSDP 主类
from torch.utils.data import Dataset, DistributedSampler  # 数据加载
from torchdata.stateful_dataloader import StatefulDataLoader  # 状态化数据加载器
from tqdm import tqdm  # 进度条
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel  # HuggingFace 模型

# VERL 工具库导入
import verl.utils.hdfs_io as hdfs_io  # HDFS 文件系统操作
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename  # 检查点管理
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager  # FSDP 检查点管理器
from verl.utils.dataset import SFTDataset  # 单轮对话数据集
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset  # 多轮对话数据集
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available  # 设备管理
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group  # 分布式初始化
from verl.utils.fs import copy_to_local  # 文件系统操作
from verl.utils.fsdp_utils import (  # FSDP 工具函数
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank  # 分布式日志
from verl.utils.profiler import log_gpu_memory_usage  # GPU 内存监控
from verl.utils.py_functional import convert_to_regular_types  # 类型转换
from verl.utils.torch_dtypes import PrecisionType  # 精度类型
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup  # 学习率调度
from verl.utils.tracking import Tracking  # 实验跟踪
from verl.utils.ulysses import (  # Ulysses 序列并行工具
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager  # Ulysses 分片管理器

# 根据设备类型导入相应的 Flash Attention 实现
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # CUDA Flash Attention
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input  # NPU Flash Attention

# 设置日志
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    """
    从检查点路径中提取步数
    
    Args:
        path (str): 检查点路径，格式如 "global_step_1000"
        
    Returns:
        int: 提取的步数，如果未找到则返回 None
    """
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    """
    FSDP 监督微调训练器
    
    这是一个基于 PyTorch FSDP 的监督微调训练器，支持：
    - FSDP/FSDP2 分布式训练
    - 序列并行 (Ulysses)
    - LoRA 微调
    - Liger Kernel 加速
    - 梯度检查点
    - 混合精度训练
    - 检查点保存/恢复
    - Entropy 计算和记录
    """
    
    def __init__(
        self,
        config,  # 训练配置
        device_mesh: DeviceMesh,  # 全局设备网格 (用于 FSDP)
        ulysses_device_mesh: DeviceMesh,  # Ulysses 设备网格 (用于序列并行)
        tokenizer,  # 分词器
        train_dataset: Dataset,  # 训练数据集
        val_dataset: Dataset,  # 验证数据集
    ):
        # 保存配置和组件
        self.config = config
        self.device_mesh = device_mesh  # 全局设备网格
        self.ulysses_device_mesh = ulysses_device_mesh  # Ulysses 设备网格
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)  # 分片管理器
        self.tokenizer = tokenizer
        
        # 检查聊天模板配置
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # 标准化数据并行大小
        self._normalize_config_bsz()

        # 设置序列并行参数
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        # Initialize runtime device name before building dataloaders/models
        # Used by pin_memory_device and tensor transfers
        self.device_name = get_device_name()

        # 构建数据加载器
        self._build_dataloader(train_dataset, val_dataset)

        # 初始化恢复相关变量
        self.resume_global_step = 0

        # 构建模型和优化器
        self._build_model_optimizer()

        # 初始化检查点管理器
        self._init_checkpoint_manager()

        # 加载检查点
        self.load_checkpoint()

        # 打印配置信息
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        """
        标准化批处理大小配置
        
        根据数据并行大小调整全局批处理大小，确保：
        1. 全局批处理大小能被数据并行大小整除
        2. 全局批处理大小能被每个GPU的微批处理大小整除
        """
        # 获取数据并行大小
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        # 检查全局批处理大小是否能被数据并行大小整除
        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        # 调整全局批处理大小
        self.config.data.train_batch_size //= dp_size

        # 检查调整后的批处理大小是否能被微批处理大小整除
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        """
        构建数据加载器
        
        根据是否使用序列并行来选择不同的 rank 和 world_size：
        - 如果使用序列并行：使用 Ulysses 设备网格的本地 rank 和 size
        - 如果不使用序列并行：使用全局设备网格的 rank 和 size
        """
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # 根据序列并行配置选择 rank 和 world_size
        if self.config.ulysses_sequence_parallel_size > 1:
            # 使用序列并行时，使用 Ulysses 设备网格的本地信息
            rank = self.ulysses_device_mesh.get_local_rank("dp")  # 获取当前进程在数据并行组中的本地 rank
            world_size = self.ulysses_device_mesh.size(0)  # 获取数据并行组中的总进程数
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            # 不使用序列并行时，使用全局设备网格信息
            rank = self.device_mesh.get_rank()  # 获取当前进程在全局设备网格中的 rank
            world_size = self.device_mesh.size()  # 获取全局设备网格中的总进程数
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            # Avoid binding CUDA contexts to rank-0 GPU from DataLoader workers
            # Keep pinned memory on CPU to prevent stray contexts on device 0
            pin_memory_device=self.device_name,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=self.device_name,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """
        计算损失和熵，并可选地执行反向传播
        
        这个方法支持两种模式：
        1. 标准模式：不使用序列并行
        2. 序列并行模式：使用 Ulysses 序列并行和 remove padding
        
        Args:
            batch: 输入批次数据，包含 input_ids, attention_mask, position_ids, loss_mask
            do_backward: 是否执行反向传播
            
        Returns:
            tuple: (loss, entropy) 损失值和平均熵
        """
        # 判断是否使用序列并行
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # 将输入数据移动到 GPU 并准备损失掩码
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)  # 移除最后一个token的损失
        loss_fct = nn.CrossEntropyLoss(reduction="none")  # 不进行reduction，保持每个token的损失
        
        # 初始化熵跟踪变量
        total_entropy = 0.0
        total_valid_tokens = 0

        # 根据是否使用序列并行选择上下文管理器
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # ========== 标准模式：不使用序列并行 ==========
                # 准备标签（输入序列向右偏移一位）
                labels = input_ids[:, 1:].contiguous()
                
                # 前向传播
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                # 准备用于损失计算的 logits 和 labels
                shift_logits = logits[..., :-1, :].contiguous()  # 移除最后一个token的logits
                shift_labels = labels.contiguous()
                
                # 展平张量以便计算损失
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)  # (batch_size * seq_len, vocab_size)
                shift_labels = shift_labels.view(-1)  # (batch_size * seq_len,)
                
                # 确保标签在正确的设备上
                shift_labels = shift_labels.to(shift_logits.device)
                
                # 计算损失
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)  # 只计算有效token的损失
                
                # ========== 计算熵 ==========
                # 计算每个token位置的熵
                probs = torch.softmax(shift_logits, dim=-1)  # 概率分布
                log_probs = torch.log_softmax(shift_logits, dim=-1)  # 对数概率
                entropy = -torch.sum(probs * log_probs, dim=-1)  # 熵: H = -Σ(p * log(p))
                
                # 只计算有效token的熵
                valid_entropy = entropy * loss_mask.to(entropy.device)
                total_entropy = torch.sum(valid_entropy)
                total_valid_tokens = torch.sum(loss_mask)
            else:
                # ========== 序列并行模式：使用 Ulysses 序列并行 ==========
                # 重要假设：同一个序列可以在不同的SP rank之间分片
                # 即：每个GPU处理 <1 个序列，每个SP组处理1个序列
                # 1. 所有SP rank会接收到*相同*的批次
                # 2. 不同的SP组会接收到*不同*的批次
                # 这通过 DistributedSampler 实现

                batch_size, seqlen = input_ids.shape
                
                # ========== 移除填充 ==========
                # 移除padding tokens以提高效率
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # 移除position_ids的填充以对齐旋转位置编码
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # ========== 序列并行处理 ==========
                # 为序列并行填充和切片输入
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                
                # 为计算损失准备标签（向右滚动一位）
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # ========== 前向传播 ==========
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Flash attention varlen 不需要 attention_mask
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # ========== 计算损失和熵 ==========
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                
                # 计算序列并行情况下的熵
                probs_rmpad = torch.softmax(logits_rmpad, dim=-1)
                log_probs_rmpad = torch.log_softmax(logits_rmpad, dim=-1)
                entropy_rmpad = -torch.sum(probs_rmpad * log_probs_rmpad, dim=-1)
                
                # ========== 序列并行聚合 ==========
                # 从所有 Ulysses rank 收集并移除填充
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                entropy_rmpad = gather_outputs_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # ========== 恢复原始形状 ==========
                # 这是从所有 Ulysses rank 收集的损失
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                
                # 移除最后一个token的损失和熵
                full_loss = full_loss.squeeze(-1)[:, :-1]
                full_entropy = full_entropy.squeeze(-1)[:, :-1]
                full_loss = full_loss.reshape(-1)
                full_entropy = full_entropy.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask
                
                # 计算有效token的熵
                valid_entropy = full_entropy * loss_mask.to(full_entropy.device)
                total_entropy = torch.sum(valid_entropy)
                total_valid_tokens = torch.sum(loss_mask)

            # ========== 损失和熵的最终计算 ==========
            valid_token_this_rank = torch.sum(loss_mask)

            # 如果启用了数据并行token平衡，则进行all_reduce
            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            # 计算平均损失
            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
            
            # 计算有效token的平均熵
            if total_valid_tokens > 0:
                avg_entropy = total_entropy / (total_valid_tokens + 1e-8)
            else:
                avg_entropy = torch.tensor(0.0, device=loss.device)

            # 可选的反向传播
            if do_backward:
                loss.backward()
            return loss, avg_entropy

    def training_step(self, batch: TensorDict):
        """
        执行一个训练步骤
        
        包括：
        1. 梯度清零
        2. 微批次处理
        3. 损失和熵计算
        4. 梯度裁剪
        5. 优化器步骤
        6. 学习率调度
        
        Args:
            batch: 输入批次数据
            
        Returns:
            dict: 包含损失、熵和学习率的指标字典
        """
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        # 清零梯度
        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        # 将批次分割为微批次以节省内存
        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        step_entropy = torch.tensor(0.0, device=self.device_name)
        
        # 处理每个微批次
        for micro_batch in micro_batches:
            loss, entropy = self._compute_loss_and_backward(batch=micro_batch)
            step_loss += loss.item() / n_micro_batches
            step_entropy += entropy / n_micro_batches

        # ========== 梯度裁剪 ==========
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # ========== 优化器步骤 ==========
        # 如果梯度范数不是有限的，跳过更新
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        # ========== 学习率调度 ==========
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        # ========== 分布式聚合 ==========
        # 在所有数据并行rank之间平均损失和熵
        step_loss = torch.tensor(step_loss).to(self.device_name)
        # step_entropy 已经是tensor，不需要再转换
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(step_entropy, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            torch.distributed.all_reduce(step_entropy)
            step_loss /= self.device_mesh.size(0)
            step_entropy /= self.device_mesh.size(0)
            
        return {
            "train/loss": step_loss.detach().item(), 
            "train/entropy": step_entropy.detach().item(),
            "train/lr(1e-3)": lr * 1e3
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, entropy = self._compute_loss_and_backward(batch, do_backward=False)
            # loss 和 entropy 已经是 tensor；直接 all_reduce 后转为 float 返回
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(entropy, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                torch.distributed.all_reduce(entropy)
                loss /= self.device_mesh.size(0)
                entropy /= self.device_mesh.size(0)
            return {"val/loss": loss.item(), "val/entropy": entropy.item()}

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        """
        主训练循环
        
        执行完整的训练过程，包括：
        1. 初始化实验跟踪
        2. 计算总训练步数
        3. 训练循环（支持断点续训）
        4. 定期验证和保存检查点
        5. 早停机制
        """
        rank = self.device_mesh.get_rank()

        # ========== 初始化实验跟踪 ==========
        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        # ========== 计算总训练步数 ==========
        global_step = self.resume_global_step  # 从恢复的步数开始
        last_valid_metric = None
        
        # 计算总训练步数（主要用于早停）
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # 如果配置中指定了总训练步数，则使用配置的值
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # ========== 断点续训处理 ==========
        # 使用 StatefulDataLoader，不需要手动计算epoch和步数
        # 数据加载器会自动从上次停止的地方恢复
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # 计算从哪个epoch开始（用于sampler.set_epoch()）
        start_epoch = global_step // self.steps_per_epoch

        # ========== 主训练循环 ==========
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # 设置epoch以确保数据打乱的一致性
            self.train_sampler.set_epoch(epoch=epoch)

            # 遍历训练数据
            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,  # 只在rank 0显示进度条
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                
                # 执行训练步骤
                metric = self.training_step(data)
                
                # 记录训练指标
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # ========== 检查是否需要验证、保存或停止 ==========
                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # ========== 验证步骤 ==========
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # 执行验证
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_metric = self.validation_step(val_data)
                        val_metrics.append(val_metric)
                    
                    if rank == 0:
                        # 平均验证指标（val_metrics 中元素是 float）
                        avg_val_loss = sum(m["val/loss"] for m in val_metrics) / len(val_metrics)
                        avg_val_entropy = sum(m["val/entropy"] for m in val_metrics) / len(val_metrics)
                        metric = {
                            "val/loss": float(avg_val_loss),
                            "val/entropy": float(avg_val_entropy)
                        }
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                # ========== 保存检查点 ==========
                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                # ========== 早停检查 ==========
                if is_last_step:
                    # 在退出前各 rank 同步，避免销毁时竞争
                    try:
                        torch.distributed.barrier()
                    except Exception:
                        pass
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    """
    运行监督微调训练
    
    这是训练的主入口函数，负责：
    1. 初始化分布式环境
    2. 创建设备网格
    3. 加载模型和数据集
    4. 创建训练器
    5. 执行训练
    6. 清理资源
    
    Args:
        config: 训练配置对象
    """
    # ========== 初始化分布式环境 ==========
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    # ========== 创建设备网格 ==========
    # 全局设备网格（用于FSDP）
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    
    # Ulysses设备网格（用于序列并行）
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    
    # ========== 构建分词器和数据集 ==========
    from verl.utils import hf_tokenizer

    # 下载模型到本地
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    
    # 创建分词器
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    
    # 创建训练和验证数据集
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    # ========== 创建训练器并开始训练 ==========
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # 执行训练
    trainer.fit()

    # ========== 清理资源 ==========
    # 同步并释放 GPU 缓存，降低销毁通信组时的显存压力
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    try:
        torch.distributed.barrier()
    except Exception:
        pass
    
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    """
    主函数入口
    
    使用 Hydra 进行配置管理，然后调用 run_sft 开始训练
    """
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """
    创建SFT数据集
    
    根据配置选择合适的数据集类型：
    1. 自定义数据集类
    2. 多轮对话数据集
    3. 单轮对话数据集（默认）
    
    Args:
        data_paths: 数据文件路径
        data_config: 数据配置
        tokenizer: 分词器
        
    Returns:
        Dataset: 创建的数据集对象
    """
    # 首先检查是否指定了自定义数据集类
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # 然后检查是否应该使用多轮对话数据集
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # 默认使用单轮对话数据集
    else:
        dataset_cls = SFTDataset

    # 根据选择的类创建数据集
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
