## VERL GRPO 训练与“参考打分融合 + 表征对齐”改动说明

本文面向本仓库的 GRPO 训练实现，系统性讲解：
1) 我为你新增的功能与关键代码改动；
2) GRPO 在 VERL 中的完整数据流与主要代码位置，便于二次开发与学习。

---

### 一、此次新增与修改概览

目标：在 GRPO 训练中，基于参考模型（reference policy）和“gold 表征”进行奖励塑形，帮助模型在不同阶段兼顾稳健性与探索性。

新增功能包含两部分：
- 参考打分融合（ref_reward_blend）：将reference model的 `ref_log_prob` 归一化后，与原始 token 级奖励按调度权重加权混合。
- 表征对齐（repr_align.ref_alignment）：当 batch 平均 entropy 偏高时，对齐“gold 表征”（prompt+标准答案的表征）与“生成表征”（prompt+生成的表征），用相似度作为奖励加成。

主要修改文件：
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/trainer/config/ppo_trainer.yaml`
- `verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`

#### 1.1 配置项

在 `ppo_trainer.yaml` 新增并注释了：

```yaml
algorithm:
  ref_reward_blend:
    enable: False
    max_weight: 0.6
    warmup_ratio: 0.7
    schedule: sigmoid   # linear|sigmoid|cosine|exp|step
    steps: 3            # step 调度时的阶梯数

  repr_align:
    enable: False
    ref_alignment:
      enable: True
      max_weight: 0.2
      warmup_ratio: 0.7
      schedule: sigmoid
      entropy_high_threshold: 1.0
      # Gold 定义：将 (prompt+gold_answer) 喂入同一 HF 模型，取最后层 hidden 在回答段的平均池化向量
```

对应示例脚本（`run_qwen2_5-3b_gsm8k_grpo_lora.sh`）已加入：

```bash
algorithm.ref_reward_blend.enable=True \
algorithm.ref_reward_blend.max_weight=0.6 \
algorithm.ref_reward_blend.warmup_ratio=0.7 \
algorithm.ref_reward_blend.schedule=sigmoid \
algorithm.ref_reward_blend.steps=3 \

algorithm.repr_align.enable=True \
algorithm.repr_align.ref_alignment.enable=True \
algorithm.repr_align.ref_alignment.max_weight=0.2 \
algorithm.repr_align.ref_alignment.warmup_ratio=0.7 \
algorithm.repr_align.ref_alignment.schedule=sigmoid \
algorithm.repr_align.ref_alignment.entropy_high_threshold=1.0 \
```

#### 1.2 参考打分融合（ref_reward_blend）

关键位置：`ray_trainer.py` 训练循环中，生成 `token_level_scores` 后、优势估计前：

- 若 `algorithm.use_kl_in_reward=True` 则先执行 KL 惩罚 `apply_kl_penalty`；否则 `token_level_rewards = token_level_scores`；
- 若 `ref_reward_blend.enable=True` 且 batch 中有 `ref_log_prob`：
  1) 使用 `_linear/_sigmoid/_cosine/_exp/_step` 调度器计算当前步权重 `weight`；
  2) 将 `ref_log_prob` 在响应段做 min-max 归一化到 [0,1]；
  3) 令 `rewards = (1-weight)*rewards + weight*ref_norm`。

相关函数：

```python
_linear_schedule/_sigmoid_schedule/_cosine_schedule/_exp_schedule/_step_schedule
_schedule_weight(cfg, current_step, total_steps)
_blend_ref_reward(token_level_scores, ref_log_prob, response_mask, weight)
```

#### 1.3 表征对齐（repr_align.ref_alignment）

关键位置：同样位于优势估计前，在 `token_level_rewards` 已就绪的阶段：

- 读取 batch 的 token 级 entropy（见 1.4 修复）并计算 batch 平均 entropy；
- 当 `avg_entropy >= entropy_high_threshold`：
  - 懒加载一个 HF 模型与分词器（`_ensure_embedder`），用于表征抽取；
  - 将 `batch.prompts` 与 `batch.responses` 反解为文本；
  - 从 `batch.non_tensor_batch` 读取 gold（`reward_model.ground_truth`）；
  - 逐样本：
    - `gen_emb = embed(prompt, generated_text)`
    - `gold_emb = embed(prompt, gold_text)`
    - `sim = cosine(gen_emb, gold_emb)`
  - 用调度权重 `w_align` 将标量相似度投影到 token 级：
    - `add_mat = sim * response_mask / num_tokens`
    - `token_level_rewards += w_align * add_mat`

表征抽取函数：

```python
_ensure_embedder()          # 首次触发时加载 HF 模型与 tokenizer
_get_response_embedding()   # 将 (prompt+response) 输入模型，取最后层 hidden 的响应段均值
```

记录指标：`actor/repr/ref_gold_align/weight`

#### 1.4 Bug 修复（entropy 的存储）

原因：`old_log_prob.batch["entropys"]` 在 union 回 batch 前被 pop 掉，后续读取失败。

修复：在 pop 前，显式复制到 `batch.batch["entropys"]`，后续从 `batch.batch` 读取。

---

### 二、VERL 中 GRPO 训练数据流（代码级导读）

入口：`verl/verl/trainer/main_ppo.py`

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)
```

Hydra 将 `verl/verl/trainer/config/ppo_trainer.yaml` 作为基底，并用 CLI 键值覆盖。核心训练逻辑在 `ray_trainer.py` 的 `RayPPOTrainer.fit()`。

#### 2.1 初始化与 Worker 组
- `init_workers()` 创建 Ray 资源池、actor/critic/ref/rm 等 WorkerGroup，并调用各自 `init_model()`。
- 当 `actor_rollout_ref.rollout.mode == "async"` 时，使用 AgentLoop 做异步 roll-out；否则同步。

#### 2.2 DataLoader
- `_create_dataloader()`：构造 `StatefulDataLoader`，并计算 `total_training_steps`。同时把总步数写入优化器/调度器配置中。

#### 2.3 训练循环（每个 batch）

1) 采样/生成
- 将原始 batch `DataProto` 精简为生成需要的字段 `_get_gen_batch()`；
- 重复 n 次（rollout.n），调用 `actor_rollout_wg.generate_sequences` 生成 responses；
- 可选：REMAX 需要 baseline（greedy）进行对比，会额外生成一份 baseline。

2) 对齐、mask、平衡
- 若缺响应 mask，调用 `compute_response_mask`；
- 可选：`_balance_batch` 对全局 token 数做 DP 平衡（改变样本顺序，不影响基于 UID 的优势估计）。

3) 奖励计算
- 若使用 RM：`rm_wg.compute_rm_score(batch)` 将 RM 分数并入 batch；
- 自定义 reward：`compute_reward()`（或异步 `compute_reward_async`）；
- `token_level_scores` 存入 `batch.batch`。

4) old_log_prob 与 entropy
- `actor_rollout_wg.compute_log_prob(batch)`，得到 `old_log_probs` 与 token 级 `entropys`；
- 统计 `actor/entropy` 指标；
- 将 `entropys` 复制到 `batch.batch["entropys"]`，再从 `old_log_prob` 中删除并 union 回 batch。

5) 参考策略 log prob（可选）
- 若启用参考策略：调用 `ref_policy_wg.compute_ref_log_prob`（或 actor 内部）并存入 batch。

6) 价值函数（可选）
- `critic_wg.compute_values(batch)`，将值函数并入 batch。

7) 奖励塑形与优势估计
- KL 惩罚：`apply_kl_penalty`（若 `algorithm.use_kl_in_reward=True`）；
- 否则 `token_level_rewards = token_level_scores`；
- 参考打分融合：`ref_reward_blend`（按调度权重混合 ref 归一化分数）；
- 表征对齐：当 entropy 高于阈值时，以 gold/resp 表征相似度做加成；
- 最终调用 `compute_advantage()`：
  - GRPO：`compute_grpo_outcome_advantage`（按 UID 进行分组聚合）
  - 其他：GAE/REINFORCE++ 等对应实现

8) 更新与日志
- critic 更新（若启用）；
- actor 更新（在 `critic_warmup` 之后生效）；
- 统计并记录 throughput、timing、数据统计、内存快照等；
- 到达保存/测试频率时，`_save_checkpoint()` 与 `_validate()`。

#### 2.4 验证流程 `_validate()`
- 对验证集重复生成，调用 `val_reward_fn`（通常为规则/模型奖励），
- 产出多种 val metrics，并可按配置将样本落盘或在 logger 中展示。

---

### 三、调度器与权重设计

五种调度器（均位于 `ray_trainer.py` 顶部）：

- `linear`：线性从 0 递增到 `max_weight`；
- `sigmoid`：S 形（前慢后快）；
- `cosine`：余弦上升（首尾慢，中间快）；
- `exp`：指数上升（前极慢、后期迅速）；
- `step`：阶梯式上升（段数由 `steps` 控制）。

统一接口：

```python
_schedule_weight(cfg, current_step, total_steps)
```

在 `ref_reward_blend` 与 `repr_align.ref_alignment` 均可通过配置选择不同调度策略。

---

### 四、注意事项与后续优化建议

1) 表征提取的开销
- 当前在 driver 侧用 HF 模型二次前向，方便快速落地。若需规模化训练，建议：
  - 在 actor worker 暴露“返回响应段 hidden states”的 RPC，避免二次前向；
  - 或使用轻量级编码器做表征近似。

2) gold 来源
- 文档示例默认从 `reward_model.ground_truth` 读取；若你的数据结构不同，请在表征对齐处调整读取逻辑。

3) token 级加成的分配
- 目前用“均匀分配到响应 token”的方式（便于实现且稳定）。可改为长度归一化、权重按位分配等更细致策略。

4) KL 与参考融合的叠加
- 先执行 KL 惩罚，再进行参考融合与表征对齐；若你想调整顺序或做互斥条件，也可在对应代码区块切换位置或增加条件。

---

### 五、最小可用配置与运行

脚本中已启用：

```bash
algorithm.ref_reward_blend.enable=True \
algorithm.ref_reward_blend.max_weight=0.6 \
algorithm.ref_reward_blend.warmup_ratio=0.7 \
algorithm.ref_reward_blend.schedule=sigmoid \
algorithm.ref_reward_blend.steps=3 \

algorithm.repr_align.enable=True \
algorithm.repr_align.ref_alignment.enable=True \
algorithm.repr_align.ref_alignment.max_weight=0.2 \
algorithm.repr_align.ref_alignment.warmup_ratio=0.7 \
algorithm.repr_align.ref_alignment.schedule=sigmoid \
algorithm.repr_align.ref_alignment.entropy_high_threshold=1.0 \
```

运行脚本后，可在 logger（console/swanlab）查看：
- `actor/ref_blend/weight`：参考融合权重
- `actor/repr/ref_gold_align/weight`：表征对齐权重
- 以及标准的 entropy、throughput、val-core/val-aux 等指标。

---

如需把表征提取下沉到 worker 或替换为更轻的编码器，我可以继续为你改造。也欢迎提出你希望观察的更多指标或可视化（例如对齐相似度的分布、按题型的分层统计）。

