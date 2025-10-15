### tmux 使用速查（持久化训练会话）

本指南帮助你使用 tmux 在服务器上后台稳定运行训练任务，即使断线/关闭 SSH 也能继续跑。

---

### 1) 创建会话并进入

```bash
tmux new -s verl_train
```

提示：`verl_train` 是自定义会话名，按需替换。

---

### 2) 在 tmux 中激活环境并运行脚本（建议记录日志）

方式 A（推荐）：不依赖 `conda init` 的兼容激活
```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)" || source ~/miniconda3/etc/profile.d/conda.sh
conda activate fyx_verl

cd /home/yxfeng/verl
mkdir -p logs
bash examples/grpo_trainer/run_qwen2_5-1.5b_gsm8k_grpo_lora.sh \
  2>&1 | tee -a logs/gsm8k_$(date +%Y%m%d_%H%M%S).log
```

方式 B：不进入环境，直接用 `conda run` 一次性执行
```bash
tmux new -s verl_train
conda run -n fyx_verl bash -lc 'cd /home/yxfeng/verl && mkdir -p logs && \
  bash examples/grpo_trainer/run_qwen2_5-1.5b_gsm8k_grpo_lora.sh \
  2>&1 | tee -a logs/gsm8k_$(date +%Y%m%d_%H%M%S).log'
```

---

### 3) 脱离/重连会话

- 脱离（训练不中断）：
  - 按键：`Ctrl-b` 然后 `d`

- 重新连接：
```bash
tmux attach -t verl_train     # 或简写 tmux a -t verl_train
```

- 如果忘记会话名：
```bash
tmux ls
tmux a -t <会话名或编号>
```

- 若提示“已在别处连接”，强制接管：
```bash
tmux attach -d -t verl_train1008
```

---

### 4) 查看与管理会话

- 列出会话：
```bash
tmux ls
```

- 杀掉指定会话（推荐按会话名或编号）：
```bash
tmux kill-session -t verl_train
# 或
tmux kill-session -t 0
```

- 在会话内关闭当前 pane/窗口：
```bash
exit           # 退出当前 shell 即可关闭 pane
tmux kill-pane
tmux kill-window
```

- 杀掉所有 tmux（慎用）：
```bash
tmux kill-server
```

---

### 5) 常见问题与排查

- Conda 激活报错 `Run 'conda init' before 'conda activate'`：
  - 用“方式 A”命令行：`eval "$(~/miniconda3/bin/conda shell.bash hook)" || source ~/miniconda3/etc/profile.d/conda.sh`

- 误输路径（如多空格、被终端折行）：
  - 建议先 `cd /home/yxfeng/verl` 再执行脚本；可用 `pwd` 校验。

- 日志体积过大：
  - 用 `tee -a` 持续追加到按时间命名的日志文件，定期清理 `logs/` 目录。

---

### 6) 快捷键常用（前缀均为 `Ctrl-b`）

- `d` 脱离会话  ·  `c` 新建窗口  ·  `,` 重命名窗口
- `p/n` 上/下一个窗口  ·  `x` 关闭当前 pane  ·  `"`/`%` 分屏
- `[` 进入复制模式（用方向键翻页，回车复制，`q` 退出）

---

### 7) 最小工作流（复习）

```bash
tmux new -s verl_train
eval "$(~/miniconda3/bin/conda shell.bash hook)" || source ~/miniconda3/etc/profile.d/conda.sh
conda activate fyx_verl
cd /home/yxfeng/verl && mkdir -p logs
bash examples/grpo_trainer/run_qwen2_5-1.5b_gsm8k_grpo_lora.sh \
  2>&1 | tee -a logs/gsm8k_$(date +%Y%m%d_%H%M%S).log
# 脱离：Ctrl-b, d
# 重连：tmux a -t verl_train
```


