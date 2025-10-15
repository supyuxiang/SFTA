# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
基于Ray的PPO训练器，使用单一控制器架构。
该训练器支持与HuggingFace兼容的模型无关模型初始化。

主要功能：
1. 分布式PPO训练，支持多节点多GPU
2. 管理Actor rollout、Critic训练和奖励计算
3. 支持多种模型架构：FSDP、Megatron、vLLM、SGLang
4. 使用Ray作为分布式计算后端
"""

import json
import os
import uuid
from collections import defaultdict
import numpy as _np
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from transformers import AutoModelForCausalLM, AutoTokenizer


def _linear_schedule(current_step: int, total_steps: int, max_weight: float, warmup_ratio: float) -> float:
    """Linearly increase weight from 0 to max_weight over warmup_ratio of total_steps.

    Args:
        current_step: current global step starting from 1
        total_steps: total planned training steps (>0)
        max_weight: upper bound for the weight
        warmup_ratio: fraction in (0,1], the portion of total steps to reach max_weight
    """
    warmup_steps = max(1, int(total_steps * max(1e-6, min(1.0, warmup_ratio))))
    progress = min(1.0, max(0.0, current_step) / warmup_steps)
    return float(max_weight * progress)

def _sigmoid_schedule(current_step: int, total_steps: int, max_weight: float, warmup_ratio: float) -> float:
    """Non-linearly increase weight from 0 to max_weight over warmup_ratio of total_steps.
    """
    warmup_steps = max(1, int(total_steps * max(1e-6, min(1.0, warmup_ratio))))
    progress = min(1.0, max(0.0, current_step) / warmup_steps)
    # 修复sigmoid函数，使其从0开始
    # 使用 sigmoid(6 * progress - 3) 来确保在progress=0时接近0，在progress=1时接近1
    sigmoid_val = 1 / (1 + np.exp(-(6 * progress - 3)))
    return float(max_weight * sigmoid_val)


def _cosine_schedule(current_step: int, total_steps: int, max_weight: float, warmup_ratio: float) -> float:
    """Cosine ramp up: slow start and end, faster mid-course."""
    warmup_steps = max(1, int(total_steps * max(1e-6, min(1.0, warmup_ratio))))
    x = min(1.0, max(0.0, current_step / warmup_steps))
    v = 0.5 * (1 - np.cos(np.pi * x))
    return float(max_weight * v)


def _exp_schedule(current_step: int, total_steps: int, max_weight: float, warmup_ratio: float, k: float = 5.0) -> float:
    """Exponential ramp up: very small early, rapid later. Larger k is steeper."""
    warmup_steps = max(1, int(total_steps * max(1e-6, min(1.0, warmup_ratio))))
    x = min(1.0, max(0.0, current_step / warmup_steps))
    v = (np.exp(k * x) - 1) / (np.exp(k) - 1)
    return float(max_weight * v)


def _step_schedule(current_step: int, total_steps: int, max_weight: float, warmup_ratio: float, steps: int = 3) -> float:
    """Piecewise constant staircase schedule."""
    warmup_steps = max(1, int(total_steps * max(1e-6, min(1.0, warmup_ratio))))
    x = min(1.0, max(0.0, current_step / warmup_steps))
    idx = int(x * steps)
    idx = min(idx, steps)
    return float(max_weight * idx / steps)


def compute_avg_response_entropy(entropy_matrix: torch.Tensor, response_mask: torch.Tensor) -> float:
    """Compute batch-mean entropy over response tokens only.

    Args:
        entropy_matrix: Tensor [B, T], token-level entropies
        response_mask: Tensor [B, T], 1 for response tokens, 0 otherwise

    Returns:
        float: batch mean entropy over response region
    """
    return masked_mean(entropy_matrix, mask=response_mask, axis=-1).mean().item()


def entropy_to_weight_linear(avg_entropy: float, max_weight: float, entropy_high_threshold: float) -> float:
    """Linear decay: weight = max_w * (1 - e/e_high), clipped to [0, max_w]."""
    if entropy_high_threshold <= 1e-6:
        return 0.0
    ratio = max(0.0, min(1.0, avg_entropy / entropy_high_threshold))
    return max_weight * (1.0 - ratio)


def entropy_to_weight_exp(avg_entropy: float, max_weight: float, entropy_high_threshold: float, k: float = 3.0) -> float:
    """Exponential decay on normalized entropy in [0,1]: weight = max_w * exp(-k * e_norm)."""
    if entropy_high_threshold <= 1e-6:
        return 0.0
    e_norm = max(0.0, min(1.0, avg_entropy / entropy_high_threshold))
    return float(max_weight * float(torch.exp(torch.tensor(-k * e_norm))))


def entropy_to_weight_cosine(avg_entropy: float, max_weight: float, entropy_high_threshold: float) -> float:
    """Cosine decay on normalized entropy: weight = max_w * 0.5 * (1 + cos(pi * e_norm))."""
    if entropy_high_threshold <= 1e-6:
        return 0.0
    e_norm = max(0.0, min(1.0, avg_entropy / entropy_high_threshold))
    return float(max_weight * 0.5 * (1.0 + float(torch.cos(torch.tensor(3.141592653589793 * e_norm)))))


def entropy_to_weight_piecewise(avg_entropy: float, max_weight: float, entropy_low_threshold: float, entropy_high_threshold: float) -> float:
    """Piecewise: full weight below low, zero above high, linear in-between."""
    if entropy_low_threshold >= entropy_high_threshold:
        return 0.0
    if avg_entropy <= entropy_low_threshold:
        return max_weight
    if avg_entropy >= entropy_high_threshold:
        return 0.0
    t = (avg_entropy - entropy_low_threshold) / (entropy_high_threshold - entropy_low_threshold)
    return max_weight * (1.0 - float(t))


def _schedule_weight(cfg: dict, current_step: int, total_steps: int) -> float:
    """Pick a scheduler by name and compute current weight."""
    sched = str(cfg.get("schedule", "linear")).lower()
    max_w = float(cfg.get("max_weight", 0.0))
    warm = float(cfg.get("warmup_ratio", 0.5))
    if sched == "sigmoid":
        return _sigmoid_schedule(current_step, total_steps, max_w, warm)
    if sched == "cosine":
        return _cosine_schedule(current_step, total_steps, max_w, warm)
    if sched == "exp":
        return _exp_schedule(current_step, total_steps, max_w, warm)
    if sched == "step":
        return _step_schedule(current_step, total_steps, max_w, warm, steps=int(cfg.get("steps", 3)))
    return _linear_schedule(current_step, total_steps, max_w, warm)


def _compute_ref_quality_score(
    batch: DataProto,
    ref_model_wg,
    response_mask: torch.Tensor,
    device: str = "cuda",
    tokenizer=None,
) -> torch.Tensor:
    """让参考模型对actor生成的回答进行质量评分。
    
    直接调用参考模型生成质量分数，而不是使用对数似然。
    
    Args:
        batch: 包含输入和输出的数据批次
        ref_model_wg: 参考模型工作器组
        response_mask: 响应部分的掩码
        device: 计算设备
        
    Returns:
        torch.Tensor: 参考模型给出的质量评分，形状为 [batch_size, response_length]
    """
    try:
        # 优先从token解码构造原始prompt/response文本
        raw_prompt = None
        if tokenizer is not None and ("prompts" in batch.batch and "responses" in batch.batch):
            try:
                prompt_texts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                output_texts = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                raw_prompt = [{"input": q, "output": a} for q, a in zip(prompt_texts, output_texts, strict=False)]
            except Exception:
                raw_prompt = None

        # 若未能从token构造，则回退到已有的non-tensor文本
        if raw_prompt is None:
            input_texts = batch.non_tensor_batch.get("input_texts", [])
            output_texts = batch.non_tensor_batch.get("output_texts", [])
            if input_texts and output_texts:
                raw_prompt = [{"input": q, "output": a} for q, a in zip(input_texts, output_texts, strict=False)]

        # 若仍无可用文本，则返回零分数
        if not raw_prompt:
            batch_size = batch.batch["response_mask"].shape[0]
            response_length = batch.batch["response_mask"].shape[1]
            return torch.zeros(batch_size, response_length, device=device)

        # 构建评分用的批次数据（使用raw_prompt，兼容worker侧实现）
        score_batch = DataProto()
        score_batch.non_tensor_batch = {
            "raw_prompt": np.array(raw_prompt, dtype=object),
        }

        # 调用参考模型生成质量分数（worker返回样本级分数）
        quality_scores_dp = ref_model_wg.generate_quality_score(score_batch)
        quality_scores = quality_scores_dp.batch["quality_scores"]  # [B]

        # 规范分数范围并扩展到token级别，仅在响应区间生效
        # 注意：ref worker 返回的分数通常在 CPU 上，这里需要对齐设备与 dtype
        # 要求：若分数不在[0,1]（或为NaN/Inf），则直接置为0.5
        invalid_mask = ~(torch.isfinite(quality_scores)) | (quality_scores < 0.0) | (quality_scores > 1.0)
        if invalid_mask.any():
            quality_scores = torch.where(invalid_mask, torch.full_like(quality_scores, 0.5), quality_scores)
        quality_scores = quality_scores.to(response_mask.device, dtype=torch.float32)
        quality_scores = quality_scores.unsqueeze(-1) * response_mask  # [B, T]

        print(f"INFO: quality_scores: {quality_scores}")

        return quality_scores

    except Exception as e:
        print(f"Error in _compute_ref_quality_score: {e}")
        # 出错时返回零分数
        batch_size = batch.batch["response_mask"].shape[0]
        response_length = batch.batch["response_mask"].shape[1]
        return torch.zeros(batch_size, response_length, device=device)


def _blend_ref_quality_score(
    token_level_scores: torch.Tensor,
    ref_quality_scores: torch.Tensor,
    response_mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """将参考模型的质量评分与原始奖励进行混合。
    
    Args:
        token_level_scores: 原始token级奖励分数
        ref_quality_scores: 参考模型的质量评分
        response_mask: 响应部分的掩码
        weight: 混合权重 (0-1)
        
    Returns:
        torch.Tensor: 混合后的奖励分数
    """
    if weight <= 0:
        return token_level_scores
    
    # 确保两个tensor形状一致
    if token_level_scores.shape != ref_quality_scores.shape:
        print(f"Warning: Shape mismatch - token_level_scores: {token_level_scores.shape}, ref_quality_scores: {ref_quality_scores.shape}")
        return token_level_scores
    
    # 只在响应部分进行混合
    blended = (1.0 - weight) * token_level_scores + weight * ref_quality_scores
    blended = blended * response_mask  # 确保只在响应部分有效
    
    return blended


@dataclass
class ResourcePoolManager:
    """
    资源池管理器，定义和管理Ray集群中的GPU资源分配。
    
    资源池将首先被初始化，用于管理分布式训练中的GPU资源分配。
    支持FSDP和Megatron等不同的并行策略。
    """

    resource_pool_spec: dict[str, list[int]]  # 资源池规格：{pool_name: [每个节点的GPU数量列表]}
    mapping: dict[Role, str]  # 角色到资源池的映射
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)  # 实际的资源池字典

    def create_resource_pool(self):
        """创建Ray资源池用于分布式训练。

        基于资源池规格初始化资源池，每个池管理跨多个节点的GPU资源。
        - FSDP后端：使用max_colocate_count=1来合并WorkerGroups
        - Megatron后端：使用max_colocate_count>1来支持不同模型
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count表示每个RayResourcePool中WorkerGroups（即进程）的数量
            # FSDP后端：推荐使用max_colocate_count=1，将所有WorkerGroups合并为一个
            # Megatron后端：推荐使用max_colocate_count>1，可以为不同模型使用不同的WorkerGroup
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """获取指定角色对应的资源池"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """获取集群中GPU的总数量"""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """检查Ray集群中是否有足够的资源满足资源池需求"""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # 检查总可用GPU数量是否满足需求
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"总可用GPU数量 {total_available_gpus} 少于所需GPU数量 {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """对token级别的奖励应用KL散度惩罚。

    该函数计算参考策略和当前策略之间的KL散度，
    然后基于这个散度对token级别的奖励应用惩罚。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据
        kl_ctrl (core_algos.AdaptiveKLController): 自适应KL惩罚控制器
        kl_penalty (str, optional): 要应用的KL惩罚类型。默认为"kl"

    Returns:
        tuple: 包含以下内容的元组：
            - 经过KL惩罚调整的token级别奖励的更新数据
            - 与KL惩罚相关的指标字典
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 计算参考策略和当前策略之间的KL散度
    # 当应用KL惩罚时，algorithm.use_kl_in_reward=True，所以参考模型已被启用
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask  # 只对响应部分计算KL散度
    beta = kl_ctrl.value  # 获取当前KL惩罚系数

    # 应用KL惩罚：token级别奖励 = 原始分数 - beta * KL散度
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前KL散度的平均值（用于更新控制器）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # 在序列维度上平均
    current_kl = torch.mean(current_kl, dim=0).item()

    # 根据HuggingFace TRL的实现更新KL控制器
    # https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    # 记录KL惩罚相关的指标
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """计算序列中响应部分的注意力掩码。

    该函数提取与模型响应对应的注意力掩码部分，
    用于掩码只应应用于响应token的计算。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据

    Returns:
        torch.Tensor: 响应token的注意力掩码
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """计算策略优化的优势估计。

    该函数使用各种估计器（如GAE、GRPO、REINFORCE++等）计算优势估计。
    优势估计用于指导强化学习算法中的策略优化。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据
        adv_estimator (AdvantageEstimator): 要使用的优势估计器（如GAE、GRPO、REINFORCE++）
        gamma (float, optional): 未来奖励的折扣因子。默认为1.0
        lam (float, optional): GAE的lambda参数。默认为1.0
        num_repeat (int, optional): 重复计算的次数。默认为1
        norm_adv_by_std_in_grpo (bool, optional): 在GRPO中是否通过标准差标准化优势。默认为True
        config (dict, optional): 算法设置的配置字典。默认为None

    Returns:
        DataProto: 包含计算出的优势和回报的更新数据
    """
    # 向后兼容：如果数据中没有响应掩码，则计算一个
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # 根据不同的优势估计器类型进行计算
    if adv_estimator == AdvantageEstimator.GAE:
        # 使用广义优势估计（GAE）计算优势和回报
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        
        # 如果启用了PF-PPO，则重新加权数据
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # 初始化GRPO计算的掩码
        grpo_calculation_mask = data.batch["response_mask"]
        # 调用GRPO结果优势计算函数
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # 处理除GAE和GRPO之外的所有其他优势估计器类型
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # 可选的uid参数
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # 可选的奖励基线参数
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # 计算优势估计器
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """使用Ray进行可扩展强化学习的分布式PPO训练器。

    该训练器协调跨多个节点和GPU的分布式PPO训练，
    使用Ray后端管理actor rollout、critic训练和奖励计算。
    支持多种模型架构，包括FSDP、Megatron、vLLM和SGLang集成。
    """

    # TODO: 支持每个角色都有独立的ray_worker_group_cls，
    # 即支持不同角色的不同后端
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        使用Ray后端初始化分布式PPO训练器。
        注意：该训练器在单个CPU/GPU节点上的驱动进程中运行。

        Args:
            config: 包含训练参数的配置对象
            tokenizer: 用于编码和解码文本的分词器
            role_worker_mapping (dict[Role, WorkerType]): 从角色到工作器类的映射
            resource_pool_manager (ResourcePoolManager): Ray资源池管理器
            ray_worker_group_cls (RayWorkerGroup, optional): Ray工作器组类。默认为RayWorkerGroup
            processor: 可选的数据处理器，用于多模态数据
            reward_fn: 训练期间计算奖励的函数
            val_reward_fn: 验证期间计算奖励的函数
            train_dataset (Optional[Dataset], optional): 训练数据集。默认为None
            val_dataset (Optional[Dataset], optional): 验证数据集。默认为None
            collate_fn: 将数据样本整理成批次的函数
            train_sampler (Optional[Sampler], optional): 训练数据集的采样器。默认为None
            device_name (str, optional): 训练设备名称（如"cuda"、"cpu"）。默认为None
        """

        # 存储分词器用于文本处理
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # 检查是否使用混合引擎（目前只支持混合引擎）
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        # 设置角色和工作器映射
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        
        # 确定需要哪些组件
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        
        # 初始化验证生成日志记录器
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # 懒加载一个仅用于表征提取的模型与分词器（当需要表征对齐时才会加载）
        self._embed_model = None
        self._embed_tokenizer = None

        # 如果ref_in_actor为True，参考策略将是未应用LoRA的actor
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # 定义奖励中的KL控制
        # 目前不支持KL损失控制
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # ---------- small helper for safe text truncation in logs ----------
    def _trunc_text(self, txt, max_len: int = 256) -> str:
        try:
            s = str(txt)
        except Exception:
            return "<unprintable>"
        return (s[:max_len] + "...<truncated>") if len(s) > max_len else s

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        创建训练和验证数据加载器。
        """
        # TODO: 必须确保批次大小能被数据并行大小整除
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # 创建训练数据集（如果未提供）
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        # 创建验证数据集（如果未提供）
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # 创建训练采样器（如果未提供）
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        # 创建整理函数（如果未提供）
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # 创建训练数据加载器
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # 创建验证数据加载器
        val_batch_size = self.config.data.val_batch_size  # 优先使用配置值
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        # 确保数据加载器不为空
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"训练数据加载器大小: {len(self.train_dataloader)}, 验证数据加载器大小: "
            f"{len(self.val_dataloader)}"
        )

        # 计算总训练步数
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"总训练步数: {self.total_training_steps}")

        # 设置优化器的总训练步数
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"警告: 无法在配置中设置total_training_steps。结构缺失？错误: {e}")

    def _ensure_embedder(self):
        """确保用于表征提取的HF模型与分词器已加载。放在driver侧，避免改worker接口。

        注意：这会额外占用显存/内存，建议仅在repr_align启用且触发时调用。####
        """
        if self._embed_model is not None and self._embed_tokenizer is not None:
            return
        model_path = self.config.actor_rollout_ref.model.path
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="eager",  # 使用eager attention避免sdpa问题
        )
        device = torch.device(self.device_name if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._embed_tokenizer = tok
        self._embed_model = model

    def _get_response_embedding(self, prompt_text: str, response_text: str, max_length: int = 4096, is_prefix: bool = True) -> torch.Tensor:
        """计算单条样本的response表征：将prompt+response拼接，取最后一层hidden在response段的均值。

        返回：tensor [hidden_dim]，在_embed_model所在设备。####
        """
        try:
            tok = self._embed_tokenizer
            model = self._embed_model
            
            # 限制文本长度以避免内存问题
            prefix_text = 'First list the key steps you used to solve the problem in 2–4 short sentences. Then state the final result on a separate line as: Answer: <number>.'
            
            # 构建完整的prompt（包含prefix）
            if is_prefix:
                full_prompt = prefix_text + prompt_text
            else:
                full_prompt = prompt_text
            
            full_text = full_prompt + response_text

            if len(full_text) > max_length:
                # 如果文本太长，截断prompt部分
                prompt_len = len(full_prompt)
                if prompt_len > max_length // 2:
                    # 保持prefix完整，只截断原始prompt
                    if is_prefix:
                        remaining_len = max_length // 2 - len(prefix_text)
                        if remaining_len > 0:
                            prompt_text = prompt_text[-(remaining_len):]
                            full_prompt = prefix_text + prompt_text
                        else:
                            full_prompt = prefix_text[:max_length // 2]
                    else:
                        full_prompt = prompt_text[-(max_length // 2):]
                
                full_text = full_prompt + response_text
                if len(full_text) > max_length:
                    full_text = full_text[:max_length]
            
            inputs = tok(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states[-1]  # [1, L, H]
            
            # 计算response部分的长度（使用完整的prompt）
            prompt_inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=max_length)
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            # 确保prompt_len不超过实际序列长度
            seq_len = hs.shape[1]
            prompt_len = min(prompt_len, seq_len - 1)  # 至少保留1个token给response
            
            resp_h = hs[0, prompt_len:, :]
            if resp_h.shape[0] == 0:
                # 如果response部分为空，返回全零向量
                return torch.zeros(hs.shape[-1], device=model.device)
            
            return resp_h.mean(dim=0)
            
        except Exception as e:
            print(f"Error in _get_response_embedding: {e}")
            raise e
            '''# 返回零向量作为fallback
            if hasattr(self, '_embed_model') and self._embed_model is not None:
                hidden_dim = self._embed_model.config.hidden_size
                return torch.zeros(hidden_dim, device=self._embed_model.device)
            else:
                return torch.zeros(4096)  # 默认维度'''

    def _handle_embedding_dimension_mismatch(self, emb1: torch.Tensor, emb2: torch.Tensor, method: str = "truncate") -> tuple:
        """
        处理表征维度不匹配的多种方法
        
        Args:
            emb1: 第一个表征向量
            emb2: 第二个表征向量
            method: 处理方法 ("truncate", "pad", "interpolate", "project", "average")
        
        Returns:
            tuple: 处理后的两个表征向量
        """
        if emb1.shape == emb2.shape:
            return emb1, emb2
        
        dim1, dim2 = emb1.shape[0], emb2.shape[0]
        device = emb1.device
        
        print(f"Applying {method} method: {dim1} -> {dim2}")
        
        if method == "truncate":
            # 方法1: 截断到较小维度
            min_dim = min(dim1, dim2)
            return emb1[:min_dim], emb2[:min_dim]
        
        elif method == "pad":
            # 方法2: 填充到较大维度
            max_dim = max(dim1, dim2)
            
            if dim1 < max_dim:
                # 用零填充emb1
                padding = torch.zeros(max_dim - dim1, device=device, dtype=emb1.dtype)
                emb1 = torch.cat([emb1, padding], dim=0)
            elif dim2 < max_dim:
                # 用零填充emb2
                padding = torch.zeros(max_dim - dim2, device=device, dtype=emb2.dtype)
                emb2 = torch.cat([emb2, padding], dim=0)
            
            return emb1, emb2
        
        elif method == "interpolate":
            # 方法3: 插值到目标维度
            target_dim = (dim1 + dim2) // 2  # 使用平均维度作为目标
            
            if dim1 != target_dim:
                emb1 = self._interpolate_embedding(emb1, target_dim)
            if dim2 != target_dim:
                emb2 = self._interpolate_embedding(emb2, target_dim)
            
            return emb1, emb2
        
        elif method == "project":
            # 方法4: 投影到公共子空间
            min_dim = min(dim1, dim2)
            return emb1[:min_dim], emb2[:min_dim]
        
        elif method == "average":
            # 方法5: 平均池化到固定维度
            target_dim = 512  # 固定目标维度
            
            if dim1 != target_dim:
                emb1 = self._average_pool_embedding(emb1, target_dim)
            if dim2 != target_dim:
                emb2 = self._average_pool_embedding(emb2, target_dim)
            
            return emb1, emb2
        
        elif method == "weighted_truncate":
            # 方法6: 加权截断（保留重要部分）
            min_dim = min(dim1, dim2)
            
            # 计算每个维度的重要性（使用绝对值）
            if dim1 > min_dim:
                importance = torch.abs(emb1)
                _, top_indices = torch.topk(importance, min_dim)
                emb1 = emb1[top_indices]
            
            if dim2 > min_dim:
                importance = torch.abs(emb2)
                _, top_indices = torch.topk(importance, min_dim)
                emb2 = emb2[top_indices]
            
            return emb1, emb2
        
        else:
            # 默认使用截断方法
            print(f"Unknown method {method}, using truncate")
            min_dim = min(dim1, dim2)
            return emb1[:min_dim], emb2[:min_dim]
    
    def _interpolate_embedding(self, emb: torch.Tensor, target_dim: int) -> torch.Tensor:
        """使用线性插值调整表征维度"""
        current_dim = emb.shape[0]
        
        if current_dim == target_dim:
            return emb
        
        # 创建插值索引
        indices = torch.linspace(0, current_dim - 1, target_dim, device=emb.device)
        
        # 线性插值
        interpolated = torch.zeros(target_dim, device=emb.device, dtype=emb.dtype)
        for i, idx in enumerate(indices):
            if idx.is_integer():
                # 整数索引，直接取值
                interpolated[i] = emb[int(idx)]
            else:
                # 非整数索引，线性插值
                lower = int(torch.floor(idx))
                upper = int(torch.ceil(idx))
                weight = idx - lower
                
                if upper < current_dim:
                    interpolated[i] = (1 - weight) * emb[lower] + weight * emb[upper]
                else:
                    interpolated[i] = emb[lower]
        
        return interpolated
    
    def _average_pool_embedding(self, emb: torch.Tensor, target_dim: int) -> torch.Tensor:
        """使用平均池化调整表征维度"""
        current_dim = emb.shape[0]
        
        if current_dim == target_dim:
            return emb
        
        if current_dim < target_dim:
            # 上采样：重复并平均
            repeat_factor = target_dim // current_dim
            remainder = target_dim % current_dim
            
            pooled = []
            for i in range(current_dim):
                # 每个元素重复repeat_factor次
                pooled.append(emb[i].repeat(repeat_factor))
                
                # 处理余数
                if i < remainder:
                    pooled.append(emb[i])
            
            return torch.cat(pooled, dim=0)
        
        else:
            # 下采样：平均池化
            pool_size = current_dim // target_dim
            remainder = current_dim % target_dim
            
            pooled = []
            start_idx = 0
            
            for i in range(target_dim):
                # 计算当前池的大小
                current_pool_size = pool_size + (1 if i < remainder else 0)
                
                # 提取并平均
                pool_emb = emb[start_idx:start_idx + current_pool_size]
                pooled.append(pool_emb.mean(dim=0))
                
                start_idx += current_pool_size
            
            return torch.stack(pooled, dim=0)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """将rollout/验证样本转储为JSONL格式。"""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """将验证样本表记录到配置的日志记录器（wandb或swanlab）"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # 创建(input, output, score)元组并按输入文本排序
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # 按输入文本排序

        # 使用固定随机种子进行确定性洗牌
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # 洗牌后取前N个样本
        samples = samples[:generations_to_log]

        # 记录到每个配置的日志记录器
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """从批次中提取用于生成的批次数据。"""
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # 为生成弹出这些键
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # 对于代理循环，我们需要奖励模型键来计算分数
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        '''# 强制在生成提示末尾加入规范答案格式要求（仅对 GSM8K）
        try:
            ds = batch.non_tensor_batch.get("data_source", None)
            if ds is not None:
                # 若全是 openai/gsm8k 或其中包含该数据源，则追加提示后缀
                if isinstance(ds, (list, tuple)):
                    has_gsm = any(str(x) == "openai/gsm8k" for x in ds)
                else:
                    has_gsm = str(ds) == "openai/gsm8k"
                if has_gsm:
                    suffix = '\nFinally, output the answer on a new last line as "#### <number>" (no extra text after the number).'
                    suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                    if len(suffix_ids) > 0:
                        input_ids = gen_batch.batch.get("input_ids")
                        attn = gen_batch.batch.get("attention_mask")
                        pos = gen_batch.batch.get("position_ids")
                        if input_ids is not None and attn is not None and pos is not None:
                            import torch as _torch
                            bsz = input_ids.shape[0]
                            add = _torch.tensor(suffix_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0).repeat(bsz, 1)
                            gen_batch.batch["input_ids"] = _torch.cat([input_ids, add], dim=1)
                            add_attn = _torch.ones((bsz, add.shape[1]), dtype=attn.dtype, device=attn.device)
                            gen_batch.batch["attention_mask"] = _torch.cat([attn, add_attn], dim=1)
                            last_pos = pos[:, -1:]
                            incr = _torch.arange(1, add.shape[1] + 1, device=pos.device, dtype=pos.dtype).unsqueeze(0).repeat(bsz, 1)
                            gen_batch.batch["position_ids"] = _torch.cat([pos, last_pos + incr], dim=1)
        except Exception:
            pass'''

        return gen_batch

    def _validate(self):
        """执行验证过程，生成样本并计算指标。"""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # 用于收集表格样本的列表
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # 如果没有uid，则生成一个
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # 重复测试批次
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # 我们只对基于规则的奖励模型进行验证
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # 存储原始输入
            input_ids = test_batch.batch["input_ids"]
            # TODO: 除了填充token外，我们能否保留特殊token？
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # 获取真实标签
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # 准备生成批次
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"测试生成批次元信息: {test_gen_batch.meta_info}")

            # 填充到能被数据并行大小整除
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            
            # 生成序列
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # 取消填充
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("验证生成结束")

            # 存储生成的输出
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # 在验证期间打印模型输出和参考模型评价（轻量打印，带截断）
            print(f"\n{'='*80}")
            print(f"VALIDATION SAMPLE (Step {self.global_steps})")
            print(f"{'='*80}")
            for i in range(min(3, len(input_texts))):  # 只显示前3个样本
                print(f"\n--- Sample {i+1} ---")
                print(f"Question: {self._trunc_text(input_texts[i])}")
                print(f"Model Answer: {self._trunc_text(output_texts[i])}")
                gt_i = ground_truths[i] if i < len(ground_truths) else 'N/A'
                print(f"Ground Truth: {self._trunc_text(gt_i)}")
                print("-" * 50)

            # 合并批次数据
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # 若缺少 response_mask，则根据 attention_mask 和 prompts 构造一个
            if "response_mask" not in test_batch.batch:
                try:
                    attn = test_batch.batch.get("attention_mask", None)
                    prompts = test_batch.batch.get("prompts", None)
                    responses = test_batch.batch.get("responses", None)
                    if responses is not None:
                        device = responses.device
                        seq_resp = responses.shape[1]
                        pad_id = getattr(self.tokenizer, "pad_token_id", None)

                        # 基于 responses 的非 padding 掩码（优先）
                        if pad_id is not None:
                            nonpad_mask = responses.ne(pad_id)
                        else:
                            nonpad_mask = torch.ones_like(responses, dtype=torch.bool)

                        # 若能从 attention_mask 中切出响应区间，则与 nonpad 取交集
                        if attn is not None and prompts is not None:
                            prompt_len = int(prompts.shape[1])
                            start = max(0, min(prompt_len, attn.shape[1]))
                            end = max(start, min(start + seq_resp, attn.shape[1]))
                            slice_len = end - start
                            if slice_len < seq_resp:
                                # 不足部分用全1补齐，避免形状不一致
                                tail = torch.ones((attn.shape[0], seq_resp - slice_len), device=attn.device, dtype=attn.dtype)
                                resp_attn_slice = torch.cat([attn[:, start:end], tail], dim=1)
                            else:
                                resp_attn_slice = attn[:, start:end]
                            resp_attn_bool = resp_attn_slice.to(torch.bool).to(device)
                            resp_mask = (nonpad_mask.to(device) & resp_attn_bool)
                        else:
                            resp_mask = nonpad_mask.to(device)

                        # 统一为 bool，形状与 responses 一致
                        if resp_mask.dtype is not torch.bool:
                            resp_mask = resp_mask.to(torch.bool)
                        assert resp_mask.shape == responses.shape, f"response_mask shape {resp_mask.shape} != responses {responses.shape}"
                        test_batch.batch["response_mask"] = resp_mask
                        # 如果某些样本响应掩码全为False，打印其uid便于排查
                        try:
                            zero_mask = (resp_mask.sum(dim=-1) == 0)
                            if bool(zero_mask.any()):
                                uids = test_batch.non_tensor_batch.get("uid", None)
                                bad_indices = [int(i) for i in torch.nonzero(zero_mask, as_tuple=False).flatten().tolist()]
                                if uids is not None:
                                    bad_uids = [str(uids[i]) for i in bad_indices]
                                    print(f"Warning: response_mask all-zero for indices {bad_indices}, uids={bad_uids}")
                                else:
                                    print(f"Warning: response_mask all-zero for indices {bad_indices}")
                        except Exception as _ee:
                            print(f"Warning: response_mask zero-check failed: {_ee}")
                except Exception as _e:
                    print(f"Warning: build response_mask failed in validation: {_e}")

            # 使用奖励函数评估
            if self.val_reward_fn is None:
                raise ValueError("验证必须提供val_reward_fn。")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            # 打印奖励分数
            print(f"\nReward Scores: {scores[:min(3, len(scores))]}")
            
            # 如果启用了参考模型质量评分，也显示质量评分
            ref_quality_cfg = self.config.algorithm.get("ref_quality_score", None)
            if (
                ref_quality_cfg is not None
                and ref_quality_cfg.get("enable", False)
                and self.use_reference_policy
            ):
                try:
                    # 为 ref 质量评分也做等分填充，避免“only support equal chunk”错误
                    ref_batch_padded, ref_pad = pad_dataproto_to_divisor(test_batch, size_divisor)

                    # 计算参考模型的质量评分（在填充后的 batch 上）
                    if not self.ref_in_actor:
                        ref_quality_scores = _compute_ref_quality_score(
                            batch=ref_batch_padded,
                            ref_model_wg=self.ref_policy_wg,
                            response_mask=ref_batch_padded.batch["response_mask"],
                            device=ref_batch_padded.batch["responses"].device,
                            tokenizer=self.tokenizer,
                        )
                    else:
                        ref_quality_scores = _compute_ref_quality_score(
                            batch=ref_batch_padded,
                            ref_model_wg=self.actor_rollout_wg,
                            response_mask=ref_batch_padded.batch["response_mask"],
                            device=ref_batch_padded.batch["responses"].device,
                            tokenizer=self.tokenizer,
                        )

                    # 取消填充到原始 batch 大小
                    if ref_pad > 0:
                        orig_bsz = len(test_batch)
                        ref_quality_scores = ref_quality_scores[:orig_bsz]
                    
                    # 计算质量分数（使用mean以更直观地展示平均token评分）
                    quality_scores = ref_quality_scores.mean(dim=-1).cpu().tolist()
                    print(f"Reference Model Quality Scores: {quality_scores[:min(3, len(quality_scores))]}")
                    
                    # 显示质量评分的详细信息
                    for i in range(min(3, len(quality_scores))):
                        print(f"Sample {i+1} Quality Score: {quality_scores[i]:.3f}")
                        
                except Exception as e:
                    print(f"Error computing reference quality scores during validation: {e}")
                    print("Reference Model Quality Scores: [0.0]")
            
            print(f"{'='*80}\n")

            # 收集奖励额外信息
            reward_extra_infos_dict["reward"].extend(scores)
            print(f"奖励额外信息字典长度: {len(reward_extra_infos_dict['reward'])}")
            
            # 计算正确率
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"奖励额外信息字典['{key}']长度: {len(reward_extra_infos_dict[key])}")
                    
                    # 如果包含正确率信息，添加到指标中
                    if key == "accuracy":
                        accuracies = [item["accuracy"] if isinstance(item, dict) else item for item in lst]
                        reward_extra_infos_dict["acc"].extend(accuracies)
                        print(f"正确率信息长度: {len(accuracies)}")
                        print(f"正确率样本: {accuracies[:min(3, len(accuracies))]}")
                        print(f"当前批次正确率: {sum(accuracies)/len(accuracies):.3f}")

            # 收集每个提示的轮数
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        # 记录验证生成样本
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # 转储生成样本
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        # 验证数据长度一致性
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # 处理验证指标
        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # 添加轮数统计
        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """使用Ray后端初始化分布式训练工作器。

        创建：
        1. 从配置创建Ray资源池
        2. 为每个角色创建工作器组（actor、critic等）
        """
        # 创建资源池
        self.resource_pool_manager.create_resource_pool()

        # 初始化资源池到类的映射
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 创建actor和rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # 创建critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 如果需要，创建参考策略
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # 如果reward_fn为None，创建奖励模型
        if self.use_rm:
            # 在这里创建奖励模型
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 初始化工作器组
        # 注意：如果你想为每个角色使用不同的资源池，支持不同的并行大小，
        # 你不应该使用`create_colocated_worker_cls`。
        # 相反，直接将不同的资源池传递给不同的工作器组。
        # 更多信息请参见：https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb
        all_wg = {}
        wg_kwargs = {}  # 为RayWorkerGroup设置kwargs
        
        # 设置Ray等待注册中心超时
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        
        # 设置性能分析步骤
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # 只有当工具是nsys时才需要nsight工作器选项
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "使用nsys和profile_steps时必须设置worker_nsight_options"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        # 为每个资源池创建工作器组
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # 初始化各个工作器组
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # 我们应该在最后创建rollout，这样vllm可以更好地估计kv缓存内存
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # 创建异步rollout管理器和请求调度器
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        """保存训练检查点。"""
        from verl.utils.fs import local_mkdir_safe

        # 路径：给定路径 + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"本地全局步骤文件夹: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        # 处理检查点保留策略
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "警告: remove_previous_ckpt_in_save已弃用，"
                + "请设置max_actor_ckpt_to_keep=1和max_critic_ckpt_to_keep=1"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # 保存actor检查点
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # 保存critic检查点
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # 保存数据加载器状态
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # 最新检查点迭代跟踪器（用于原子操作）
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """加载训练检查点。"""
        if self.config.trainer.resume_mode == "disable":
            return 0

        # 从hdfs加载
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("从hdfs加载尚未实现")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: 检查路径
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # 如果没有最新的则为None

        # 查找global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("从头开始训练")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "恢复检查点必须是字符串类型"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "恢复检查点必须指定global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"从检查点文件夹加载: {global_step_folder}")
        # 设置全局步骤
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"设置全局步骤为 {self.global_steps}")
        print(f"从 {global_step_folder} 恢复")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # 加载actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # 加载critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # 加载数据加载器
        # TODO: 从远程加载尚未实现
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"警告: 在 {dataloader_local_path} 未找到数据加载器状态，将从头开始")

    def _start_profiling(self, do_profile: bool) -> None:
        """如果启用了性能分析，为所有工作器组启动性能分析。"""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """如果启用了性能分析，停止所有工作器组的性能分析。"""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """在单一控制器上重新排序数据，使每个数据并行rank获得相似的总token数"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # 基于索引重新排序。数据将通过调度函数自动等分
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        PPO的训练循环。
        驱动进程只需要通过RPC调用工作器组的计算函数来构建PPO数据流。
        轻量级的优势计算在驱动进程上完成。
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # 初始化日志记录器
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 在做任何事情之前加载检查点
        self._load_checkpoint()

        # 在训练前执行验证
        # 目前，我们只支持使用奖励函数进行验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"初始验证指标: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 如果启用了跳过rollout，则包装生成序列函数
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # 添加进度条
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="训练进度")
        except ImportError:
            # 如果没有tqdm，使用一个简单的计数器
            class SimpleProgressBar:
                def __init__(self, total, initial=0, desc=""):
                    self.total = total
                    self.current = initial
                    self.desc = desc
                def update(self, n=1):
                    self.current += n
                    if self.current % 10 == 0:  # 每10步打印一次
                        print(f"{self.desc}: {self.current}/{self.total}")
                def close(self):
                    pass
            progress_bar = SimpleProgressBar(total=self.total_training_steps, initial=self.global_steps, desc="训练进度")

        # 我们从步骤1开始
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # 性能分析相关变量
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # 主训练循环
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # 启动性能分析
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # 创建数据批次
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 为批次添加uid
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # 获取生成批次
                gen_batch = self._get_gen_batch(batch)

                # 传递global_steps用于跟踪
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # 生成一个批次
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # 处理REMAX优势估计器
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("REMAX优势估计需要reward_fn。")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # 重复以与rollout中的重复响应对齐
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # 计算响应掩码
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # 平衡数据并行rank之间的有效token数量
                    # 注意：这通常会改变`batch`中数据的顺序，
                    # 这不会影响优势计算（因为它基于uid），
                    # 但可能会影响损失计算（由于mini-batching的变化）。
                    # TODO: 解耦数据并行平衡和mini-batching
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效token数量
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # 计算奖励模型分数
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 计算奖励
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # 重新计算old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # 将token级entropy保存到batch，供后续repr对齐等逻辑使用
                        batch.batch["entropys"] = entropys
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        # 计算调试指标
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: 我们可能还想添加概率差异
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    # 计算参考策略log概率
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # 计算价值函数
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # 我们与基于规则的奖励模型结合
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 计算奖励。如果可用则应用KL惩罚
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 可选：将reference模型的质量评分与原始奖励做加权融合
                        # 配置位置：algorithm.ref_quality_score.{enable,max_weight,warmup_ratio}
                        ref_quality_cfg = self.config.algorithm.get("ref_quality_score", None)
                        if (
                            ref_quality_cfg is not None
                            and ref_quality_cfg.get("enable", False)
                            and self.use_reference_policy
                        ):
                            with marked_timer("ref_quality_score", timing_raw, color="purple"):
                                # 计算参考模型的质量评分
                                if not self.ref_in_actor:
                                    ref_quality_scores = _compute_ref_quality_score(
                                        batch=batch,
                                        ref_model_wg=self.ref_policy_wg,
                                        response_mask=batch.batch["response_mask"],
                                        device=batch.batch["token_level_rewards"].device,
                                        tokenizer=self.tokenizer,
                                    )
                                else:
                                    ref_quality_scores = _compute_ref_quality_score(
                                        batch=batch,
                                        ref_model_wg=self.actor_rollout_wg,
                                        response_mask=batch.batch["response_mask"],
                                        device=batch.batch["token_level_rewards"].device,
                                        tokenizer=self.tokenizer,
                                    )
                                
                               

                                # 基于actor熵的线性权重：熵越高权重越低，最高不超过阈值
                                # 计算批次平均响应区间熵
                                ent = batch.batch.get("entropys")  # [B, T]
                                resp_mask = batch.batch.get("response_mask")
                                assert ent is not None and resp_mask is not None, "entropys/response_mask missing"
                                avg_ent = masked_mean(ent, mask=resp_mask, axis=-1).mean().item()

                                max_weight_cfg = float(ref_quality_cfg.get("max_weight", 0.3))
                                # 参考阈值：当平均熵达到该阈值时，权重降为 ~0
                                ent_high = float(ref_quality_cfg.get("entropy_high_threshold", 1.0))
                                # 线性映射到 [0, max_weight]，并裁剪
                                norm = 0.0 if ent_high <= 1e-6 else min(1.0, max(0.0, avg_ent / ent_high))
                                weight = max_weight_cfg * (1.0 - norm)
                                
                                # 混合参考模型质量评分与原始奖励
                                blended = _blend_ref_quality_score(
                                    token_level_scores=batch.batch["token_level_rewards"],
                                    ref_quality_scores=ref_quality_scores,
                                    response_mask=batch.batch["response_mask"],
                                    weight=weight,
                                )
                                batch.batch["token_level_rewards"] = blended
                                
                                # 记录指标 - 确保这些指标被正确记录
                                ref_quality_metrics = {
                                    "actor/ref_quality/weight": weight,
                                    "actor/ref_quality/enabled": 1.0,
                                    "actor/ref_quality/avg_entropy": avg_ent,
                                    "actor/ref_quality/mean_score": ref_quality_scores.mean().item(),
                                    "actor/ref_quality/max_score": ref_quality_scores.max().item(),
                                    "actor/ref_quality/min_score": ref_quality_scores.min().item(),
                                    "actor/ref_quality/std_score": ref_quality_scores.std().item(),
                                }
                                metrics.update(ref_quality_metrics)
                        else:
                            # 即使未启用，也记录一些调试信息
                            metrics.update({
                                "actor/ref_quality/enabled": 0.0,
                                "actor/ref_quality/weight": 0.0,
                            })

                        # 奖励归一化（防止自博弈时奖励尺度失控）
                        # 配置：algorithm.reward_norm in {"tanh","clip","none"}, algorithm.reward_clip_value (默认1.0)
                        reward_norm_method = str(self.config.algorithm.get("reward_norm", "tanh")).lower()
                        raw_mean = batch.batch["token_level_rewards"].mean().item()
                        raw_std = batch.batch["token_level_rewards"].std().item()
                        if reward_norm_method == "tanh":
                            batch.batch["token_level_rewards"] = torch.tanh(batch.batch["token_level_rewards"])  # [-1,1]
                        elif reward_norm_method == "clip":
                            clip_v = float(self.config.algorithm.get("reward_clip_value", 1.0))
                            batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"].clamp(-clip_v, clip_v)
                        else:
                            pass  # none
                        metrics.update({
                            "actor/reward_norm/raw_mean": raw_mean,
                            "actor/reward_norm/raw_std": raw_std,
                            "actor/reward_norm/mean": batch.batch["token_level_rewards"].mean().item(),
                            "actor/reward_norm/std": batch.batch["token_level_rewards"].std().item(),
                        })


                        # 高entropy时执行gold表征对齐（driver侧HF前向，简化实现）##########
                        repr_cfg = self.config.algorithm.get("repr_align", None)
                        print(f"[DEBUG] repr_cfg: {repr_cfg}")
                        if repr_cfg is not None and repr_cfg.get("enable", False):
                            ref_align_cfg = repr_cfg.get("ref_alignment", {})
                            print(f"[DEBUG] ref_align_cfg: {ref_align_cfg}")
                            if ref_align_cfg.get("enable", False):
                                # 计算batch平均entropy，用于触发
                                ent = batch.batch.get("entropys")  # [batch_sz, seq_len]
                                assert ent is not None, f"entropys is None"
                                if ent is not None:
                                    resp_mask = batch.batch.get("response_mask")  # [batch_sz, seq_len]
                                    assert resp_mask is not None, f"response_mask is None"
                                    if resp_mask is not None:
                                        avg_ent_scalar = masked_mean(ent, mask=resp_mask, axis=-1).mean().item()
                                        thresh = float(ref_align_cfg.get("entropy_high_threshold", 0.32))
                                        print(f"[DEBUG] avg_ent_scalar: {avg_ent_scalar}, thresh: {thresh}")
                                        if avg_ent_scalar >= thresh:
                                            print(f"[DEBUG] 熵阈值达到，开始表征对齐计算")
                                            # 懒加载表征模型
                                            self._ensure_embedder()
                                            
                                            # 检查必要的tensor是否存在
                                            prompts = batch.batch.get("prompts")
                                            responses = batch.batch.get("responses")
                                            assert prompts is not None, f"prompts is None"
                                            assert responses is not None, f"responses is None"
                                            if prompts is not None and responses is not None:
                                                # 准备文本（注意：使用prompts/responses token反解到文本）
                                                prompt_texts = self.tokenizer.batch_decode(
                                                    prompts, skip_special_tokens=True
                                                )
                                                resp_texts = self.tokenizer.batch_decode(
                                                    responses, skip_special_tokens=True
                                                )
                                                
                                                # gold答案从reward_model中获取（若不存在则跳过对齐）
                                                gold_answers = None
                                                try:
                                                    # 遍历batch中的每个样本，优先在GSM8K上使用推理过程文本（solution/rationale），否则回退到ground_truth
                                                    gold_answers = []
                                                    for item in batch:
                                                        ds = item.non_tensor_batch.get("data_source", None)
                                                        reward_data = item.non_tensor_batch.get("reward_model", {})
                                                        extra_info = item.non_tensor_batch.get("extra_info", {})
                                                        gold_text = None

                                                        if ds == "openai/gsm8k":
                                                            # GSM8K优先使用含推理的CoT文本：extra_info.answer
                                                            if isinstance(extra_info, dict):
                                                                gold_text = extra_info.get("answer")
                                                            if not gold_text and isinstance(reward_data, dict):
                                                                # 回退到reward_model里的可能字段
                                                                gold_text = (
                                                                    reward_data.get("solution")
                                                                    or reward_data.get("rationale")
                                                                    or reward_data.get("reasoning")
                                                                    or reward_data.get("gold_solution")
                                                                    or reward_data.get("ground_truth")
                                                                )
                                                        else:
                                                            # 其他数据集保持原逻辑
                                                            if isinstance(reward_data, dict):
                                                                gold_text = reward_data.get("ground_truth", None)

                                                        # 最后再做一层宽松回退（极端情况下）
                                                        if not gold_text:
                                                            gold_text = (
                                                                (extra_info.get("answer") if isinstance(extra_info, dict) else None)
                                                                or item.non_tensor_batch.get("solution")
                                                                or item.non_tensor_batch.get("rationale")
                                                                or item.non_tensor_batch.get("reasoning")
                                                                or item.non_tensor_batch.get("ground_truth")
                                                            )

                                                        gold_answers.append(gold_text)

                                                    # 检查是否有有效文本
                                                    if not any((g is not None and str(g).strip() != "") for g in gold_answers):
                                                        gold_answers = None
                                                        print("[DEBUG] No valid gold reasoning/ground truth found in batch metadata")

                                                except Exception as e:
                                                    print(f"Error extracting gold reasoning/answer from batch: {e}")
                                                    gold_answers = None
                                                
                                                # 逐条计算对齐加成（为控算力，仅在gold可用时）
                                                if gold_answers is not None and any(g is not None for g in gold_answers):
                                                    print(f"[DEBUG] Found {sum(1 for g in gold_answers if g is not None)} valid gold answers out of {len(gold_answers)} total samples")
                                                    print(f"[DEBUG] Sample gold answers: {[g[:100] + '...' if g and len(str(g)) > 100 else g for g in gold_answers[:3]]}")
                                                    print(f"[DEBUG] 开始计算表征对齐相似度")
                                                    align_scores = []
                                                    
                                                    # 导入tqdm用于显示进度条
                                                    try:
                                                        from tqdm import tqdm
                                                        progress_bar = tqdm(
                                                            zip(prompt_texts, resp_texts, gold_answers),
                                                            total=len(prompt_texts),
                                                            desc="计算表征对齐相似度",
                                                            unit="样本",
                                                            ncols=80
                                                        )
                                                    except ImportError:
                                                        # 如果没有tqdm，使用普通的zip
                                                        progress_bar = zip(prompt_texts, resp_texts, gold_answers)
                                                    
                                                    for i, (q, gen, gold) in enumerate(progress_bar):
                                                        if gold is None:
                                                            align_scores.append(0.0)
                                                            if hasattr(progress_bar, 'set_postfix'):
                                                                progress_bar.set_postfix({"状态": "gold为None", "相似度": "0.000"})
                                                            continue
                                                        try:
                                                            gen_emb = self._get_response_embedding(q, gen)
                                                            gold_emb = self._get_response_embedding(q, gold)
                                                            
                                                            # 检查表征维度是否匹配
                                                            if gen_emb.shape != gold_emb.shape:
                                                                print(f"Warning: Embedding dimension mismatch - gen: {gen_emb.shape}, gold: {gold_emb.shape}")
                                                                
                                                                # 多种处理策略
                                                                gen_emb, gold_emb = self._handle_embedding_dimension_mismatch(
                                                                    gen_emb, gold_emb, method="truncate"
                                                                )
                                                            
                                                            # 确保表征不为零向量
                                                            gen_norm = torch.norm(gen_emb)
                                                            gold_norm = torch.norm(gold_emb)
                                                            
                                                            if gen_norm == 0 or gold_norm == 0:
                                                                print(f"Warning: Zero embedding detected - gen_norm: {gen_norm}, gold_norm: {gold_norm}")
                                                                align_scores.append(0.0)
                                                                if hasattr(progress_bar, 'set_postfix'):
                                                                    progress_bar.set_postfix({"状态": "零向量", "相似度": "0.000"})
                                                                continue
                                                            
                                                            # 计算余弦相似度
                                                            sim = torch.nn.functional.cosine_similarity(
                                                                gen_emb.unsqueeze(0), gold_emb.unsqueeze(0)
                                                            )[0]
                                                            sim_value = float(sim.detach().cpu())
                                                            align_scores.append(sim_value)
                                                            if hasattr(progress_bar, 'set_postfix'):
                                                                progress_bar.set_postfix({"相似度": f"{sim_value:.3f}"})
                                                        except Exception as e:
                                                            print(f"Error computing embedding similarity: {e}")
                                                            align_scores.append(0.0)
                                                            if hasattr(progress_bar, 'set_postfix'):
                                                                progress_bar.set_postfix({"状态": "错误", "相似度": "0.000"})
                                                    
                                                    # 关闭进度条
                                                    if hasattr(progress_bar, 'close'):
                                                        progress_bar.close()
                                                    
                                                    # 打印对齐样本与统计并记录到指标
                                                    try:
                                                        if len(align_scores) > 0:
                                                            import numpy as _np
                                                            sims_np = _np.array(align_scores, dtype=float)
                                                            sim_mean = float(sims_np.mean())
                                                            sim_std = float(sims_np.std())
                                                            sim_min = float(sims_np.min())
                                                            sim_max = float(sims_np.max())
                                                            metrics["actor/repr/ref_gold_align/sim_mean"] = sim_mean
                                                            metrics["actor/repr/ref_gold_align/sim_std"] = sim_std
                                                            metrics["actor/repr/ref_gold_align/sim_min"] = sim_min
                                                            metrics["actor/repr/ref_gold_align/sim_max"] = sim_max
                                                            print(f"[DEBUG] 表征对齐metrics已设置: sim_mean={sim_mean:.4f}")

                                                            print("\n[Representation Alignment] Cosine similarity stats:")
                                                            print(f"mean={sim_mean:.4f}, std={sim_std:.4f}, min={sim_min:.4f}, max={sim_max:.4f}")

                                                            # 打印前3条样本（带截断）以便在终端观察
                                                            top_k = min(3, len(align_scores))
                                                            for i in range(top_k):
                                                                print(f"\n--- Align Sample {i+1} ---")
                                                                pt = prompt_texts[i] if i < len(prompt_texts) else ''
                                                                rt = resp_texts[i] if i < len(resp_texts) else ''
                                                                gt = gold_answers[i] if i < len(gold_answers) else ''
                                                                print(f"Prompt: {self._trunc_text(pt, 200)}")
                                                                print(f"Generated: {self._trunc_text(rt, 200)}")
                                                                print(f"Gold: {self._trunc_text(gt, 200)}")
                                                                print(f"CosineSim: {align_scores[i]:.4f}")
                                                    except Exception as _e:
                                                        print(f"Error logging representation alignment stats: {_e}")

                                                    # 将标量sim变为token级加成，并按调度权重缩放
                                                    w_align = _schedule_weight(
                                                        ref_align_cfg, self.global_steps, self.total_training_steps
                                                    )
                                                    print(f"[DEBUG] w_align: {w_align}")
                                                    if w_align > 0:
                                                        print(f"[DEBUG] 权重>0，开始融合表征对齐奖励")
                                                        resp_mask_f = resp_mask.float() # [batch_sz, seq_len]
                                                        token_counts = resp_mask_f.sum(-1, keepdim=True).clamp_min(1.0) # [batch_sz, 1],其中设置下限为1.0以保证分母不为0
                                                        # 每样本均匀分配到响应token
                                                        add_mat = (
                                                            torch.tensor(align_scores, device=resp_mask.device).unsqueeze(-1) # [batch_sz, 1]
                                                            * resp_mask_f # -> broadcast to [batch_sz, seq_len]
                                                            / token_counts # [batch_sz, 1] (broadcasted)
                                                        ) # [batch_sz, seq_len]
                                                        # 融合加成（归一化前先记录均值/方差）
                                                        rewards_pre = batch.batch["token_level_rewards"] + w_align * add_mat
                                                        # 记录融合前统计
                                                        try:
                                                            m_pre = float(rewards_pre.mean().item())
                                                            v_pre = float(rewards_pre.var(unbiased=False).item())
                                                            metrics["actor/reward/mean_pre"] = m_pre
                                                            metrics["actor/reward/var_pre"] = v_pre
                                                        except Exception:
                                                            pass
                                                        batch.batch["token_level_rewards"] = rewards_pre
                                                        # 对响应区间内的token级奖励进行按样本[min,max]归一化到[0,1]
                                                        rewards = batch.batch["token_level_rewards"] # [batch_sz, seq_len]
                                                        mask_f = resp_mask_f
                                                        very_large = torch.tensor(float("inf"), device=rewards.device, dtype=rewards.dtype)
                                                        very_small = torch.tensor(float("-inf"), device=rewards.device, dtype=rewards.dtype)
                                                        masked_min = rewards.masked_fill(mask_f == 0, very_large).amin(dim=-1, keepdim=True)
                                                        masked_max = rewards.masked_fill(mask_f == 0, very_small).amax(dim=-1, keepdim=True)
                                                        denom = (masked_max - masked_min).clamp_min(1e-6)
                                                        rewards_norm = (rewards - masked_min) / denom
                                                        rewards_norm = rewards_norm * mask_f
                                                        batch.batch["token_level_rewards"] = rewards_norm
                                                        print(f"INFO: final_rewards: {rewards_norm}")
                                                        # 记录归一化后统计
                                                        try:
                                                            m_post = float(rewards_norm.mean().item())
                                                            v_post = float(rewards_norm.var(unbiased=False).item())
                                                            metrics["actor/reward/mean_post"] = m_post
                                                            metrics["actor/reward/var_post"] = v_post
                                                        except Exception:
                                                            pass

                                                        metrics["actor/repr/ref_gold_align/weight"] = float(w_align)
                                                       
                                                        print(f"actor/repr/ref_gold_align/weight: {w_align}")
                                                else:
                                                    print("Warning: No gold answers available for representation alignment")
                                                    # 设置默认的metrics值，确保仪表板上曲线连续
                                                    metrics.setdefault("actor/repr/ref_gold_align/weight", 0.0)
                                                    metrics.setdefault("actor/repr/ref_gold_align/sim_mean", 0.0)
                                                    metrics.setdefault("actor/repr/ref_gold_align/sim_std", 0.0)
                                                    metrics.setdefault("actor/repr/ref_gold_align/sim_min", 0.0)
                                                    metrics.setdefault("actor/repr/ref_gold_align/sim_max", 0.0)
                                            else:
                                                print("Warning: Missing prompts or responses for representation alignment")
                                                # 保证仪表板上有连续时间序列（未触发时写0）
                                                metrics.setdefault("actor/repr/ref_gold_align/weight", 0.0)
                                                metrics.setdefault("actor/repr/ref_gold_align/sim_mean", 0.0)
                                                metrics.setdefault("actor/repr/ref_gold_align/sim_std", 0.0)
                                                metrics.setdefault("actor/repr/ref_gold_align/sim_min", 0.0)
                                                metrics.setdefault("actor/repr/ref_gold_align/sim_max", 0.0)
                                    else:
                                            print("Warning: Missing response_mask for representation alignment")
                                            metrics.setdefault("actor/repr/ref_gold_align/weight", 0.0)
                                            metrics.setdefault("actor/repr/ref_gold_align/sim_mean", 0.0)
                                            metrics.setdefault("actor/repr/ref_gold_align/sim_std", 0.0)
                                            metrics.setdefault("actor/repr/ref_gold_align/sim_min", 0.0)
                                            metrics.setdefault("actor/repr/ref_gold_align/sim_max", 0.0)
                                else:
                                    print("Warning: Missing entropys for representation alignment")
                                    metrics.setdefault("actor/repr/ref_gold_align/weight", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_mean", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_std", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_min", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_max", 0.0)
                                # 未达到高熵阈值时也记录0，确保仪表板上曲线连续
                                if avg_ent_scalar < thresh:
                                    print(f"[DEBUG] 熵值未达到阈值，设置表征对齐metrics为0")
                                    metrics.setdefault("actor/repr/ref_gold_align/weight", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_mean", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_std", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_min", 0.0)
                                    metrics.setdefault("actor/repr/ref_gold_align/sim_max", 0.0)

                        # 计算优势，在驱动进程上执行
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO优势标准化因子

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # 更新critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # 实现critic预热
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # 如果启用，记录rollout生成
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # 验证
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 检查ESI（弹性服务器实例）/训练计划是否接近过期
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # 检查保存检查点的条件是否满足
                # 条件包括一个强制条件(1)和以下可选条件之一(2/3/4)：
                # 1. 保存频率设置为正值
                # 2. 这是最后一个训练步骤
                # 3. 当前步骤数是保存频率的倍数
                # 4. ESI（弹性服务器实例）/训练计划接近过期
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("强制保存检查点：ESI实例即将过期。")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # 停止性能分析
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # 更新步骤持续时间
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # 训练指标
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # 收集指标
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: 实现实际的tflpo和理论tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # 这是实验性的，将来可能会更改/删除，以支持通用目的
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: 创建一个支持各种后端的规范日志记录器
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                # 内存快照
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                # 检查是否为最后一步
                if is_last_step:
                    pprint(f"最终验证指标: {last_val_metrics}")
                    progress_bar.close()
                    return

                # 这是实验性的，将来可能会更改/删除
                # 以支持通用数据缓冲池
                if hasattr(self.train_dataset, "on_batch_end"):
                    # 数据集可能会在每个训练批次后更改
                    self.train_dataset.on_batch_end(batch=batch)
