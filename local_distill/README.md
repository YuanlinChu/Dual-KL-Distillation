# 本地 On-Policy 蒸馏（无需 Tinker）

该原型用 Hugging Face 生态在本地复现实验：学生在 prompt 上按自身策略采样，教师仅提供逐 token 的 logprob，用反向 KL 构造优势并更新学生。

- 学生采样续写；教师对同一序列给出 log q。
- 反向 KL：Δ=log p_student − log p_teacher，优势 A=−coef·Δ（可选折扣）；只对续写部分求损失。
- 损失：policy gradient 形式的加权 NLL，loss=−∑ A_t·log p_student。

## 快速开始

1) 安装依赖（Python ≥3.10）：

   pip install torch transformers peft datasets accelerate wandb deepspeed

2) CPU 小模型演示（玩具数据）：

   python -m local_distill.train_on_policy_local \
     --student_model sshleifer/tiny-gpt2 \
     --teacher_model sshleifer/tiny-gpt2 \
     --steps 3 --batch_size 2 --max_new_tokens 16 --group_size 2

3) 单机多卡（2×A800 或 4×A800）分布式训练（建议）：

   # 2 卡（bf16），使用默认的 accelerate 配置
   accelerate launch --config_file local_distill/accelerate_config_multi_gpu.yaml \
     -m local_distill.train_on_policy_local \
     --student_model Qwen/Qwen3-8B \
     --teacher_model Qwen/Qwen3-32B \
     --dataset tulu3 \
     --batch_size 256 --group_size 4 --grad_accum 8 \
     --max_new_tokens 512 --use_lora --lora_r 64 \
     --dtype bf16 --save_every 200 \
     --wandb_project onpolicy-distill --wandb_name qwen8b_tulu3

   # 4 卡：将 accelerate_config_multi_gpu.yaml 中 num_processes 改为 4 即可

   # 可选：DeepSpeed ZeRO-2（显存友好），示例：
   accelerate launch --config_file local_distill/accelerate_config_multi_gpu.yaml \
     --deepspeed_config_file local_distill/ds_zero2_bf16.json \
     -m local_distill.train_on_policy_local ...（其余参数同上）

   # 重要：当教师为 32B 且显存紧张时，建议仅对教师启用 ZeRO-3 推理分片，学生继续用 DDP：
   accelerate launch --config_file local_distill/accelerate_config_multi_gpu.yaml \
     -m local_distill.train_on_policy_local \
     --student_model Qwen/Qwen3-8B --teacher_model Qwen/Qwen3-32B \
     --dataset tulu3 --batch_size 256 --group_size 4 --grad_accum 8 \
     --max_new_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
     --teacher_ds_zero3  # 可选：--teacher_ds_config path/to/ds_zero3_infer.json

说明
- 可用 `--dataset deepmath|tulu3`（需 datasets）或通过 `--prompts_file` 指定自定义 prompt（每行一个）。
- 默认自动将 pad token 设为 EOS。
- 检查点保存到 `--output_dir`。
- `--group_size` 对齐 Tinker 的“每个 prompt 多次采样”；`--grad_accum` 对齐 `num_substeps`；`--dtype bf16` 适合 A800；`--use_lora` 适合 4B–8B 学生。

## 配置文件
- `local_distill/accelerate_config_multi_gpu.yaml`：单机多卡（默认 2 进程，改 `num_processes` 为 4 即四卡）。
- `local_distill/ds_zero2_bf16.json`：DeepSpeed ZeRO-2（bf16）示例，可与 accelerate 联用。

## 日志与可视化
- 传入 `--wandb_project your_project` 即可启用 W&B；`--wandb_name` 指定 run 名称；`--wandb_mode offline|disabled` 控制模式。

## 超参数建议（A800 环境）

目标：策略与 Tinker 实现一致，优先稳定性与吞吐。

- 通用（建议起点）
  - 学习率：`--learning_rate 1e-4`（LoRA 情况下），权重衰减 `--weight_decay 0.0`。
  - 反向 KL：`--kl_coef 1.0`，`--kl_discount 0.0`（先不开启折扣，保证对齐）。
  - 采样：`--temperature 0.8`，`--top_p 0.95`；`--max_new_tokens 512`（数学可 1024）。
  - AMP：`--dtype bf16`（A800 推荐）。
  - 评估与保存：`--eval_every 50`，`--save_every 200`（按训练时长调整）。

- 学生 1.7B Base（LoRA，单机 2×A800）
  - `--use_lora --lora_r 64 --lora_alpha 32 --lora_dropout 0.05`
  - 吞吐配置：`--batch_size 512 --group_size 4 --grad_accum 8 --max_new_tokens 512 --dtype bf16`
  - 如显存吃紧：先降 `--group_size`（到 2），再降 `--batch_size`，最后降 `--max_new_tokens`。

- 学生 4B Base（LoRA，单机 2×A800）
  - `--use_lora --lora_r 64~96 --lora_alpha 32~64`
  - 吞吐配置：`--batch_size 384 --group_size 4 --grad_accum 8 --max_new_tokens 512`
  - 数学/长输出场景：`--max_new_tokens 1024`，同时适当降低 `--batch_size` 或提升 `--grad_accum`。

- 学生 8B Base（LoRA，单机 2×A800；更稳妥为 4×A800）
  - `--use_lora --lora_r 64 --lora_alpha 32~64`
  - 2 卡起点：`--batch_size 256 --group_size 4 --grad_accum 8 --max_new_tokens 512`
  - 4 卡起点：`--batch_size 512 --group_size 4 --grad_accum 8 --max_new_tokens 512`
  - 若报 OOM：优先把 `--group_size` 调到 2；或改用 DeepSpeed ZeRO-2（见下）。

- 教师 32B Instruct（推理）
  - 建议 bf16；KV Cache 对显存敏感，`--max_new_tokens` 直接影响显存与速度。
  - 确保教师只做前向且不参与梯度；本脚本已强制 teacher.no_grad()。

- DeepSpeed（可选）
  - 学生：建议保持 Accelerate DDP 以保证速度。
  - 教师：使用 `--teacher_ds_zero3` 将教师以 ZeRO-3 推理分片方式加载，避免在每张卡上重复常驻 32B 权重导致 OOM。
  - 若需自定义配置可用 `--teacher_ds_config your_ds_config.json`（不提供则用内置零三推理配置）。
  - 与 accelerate 的 `--deepspeed_config_file`（面向训练/优化器的 ZeRO）可独立使用，两者互不冲突。

- Token 统计与吞吐
  - 脚本会记录 reverse KL、loss、token 数（续写部分）。
  - 吞吐主要由：`batch_size × group_size × max_new_tokens × 卡数` 决定；调参时优先保持 `batch_size×group_size` 大、`max_new_tokens` 适中以稳定训练。

- 训练时长与步数
  - 以 Tinker 示例为参照，可从 `--steps 20k~50k` 起步，视目标任务与资源继续延长。
  - 建议开启 W&B 观察 reverse KL 与 loss 是否收敛、是否出现发散或模式坍塌（可适度调低 `--learning_rate` 或增大 `--grad_accum`）。

- 其他细节
  - 只对续写 token 计算 KL 与梯度（prompt 掩码在脚本中已处理）。
  - 使用内置数据集：`--dataset tulu3|deepmath`；或 `--prompts_file your_prompts.txt`（每行一个 prompt）。
  - 想完全贴齐原策略，可将 `--kl_discount` 保持 0；如需对长序列更稳，可尝试 `0.95` 并对比实验。
