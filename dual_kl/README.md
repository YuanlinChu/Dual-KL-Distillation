# 双向 KL 蒸馏（Dual-KL）本地原型

该原型在不改动模型与训练框架（Accelerate/LoRA/数据管线）的前提下，将损失改为**正向 KL 与反向 KL 的组合**，并提供**软/硬门控**：当学生与老师分布差异较小时偏向 on-policy（反向 KL），差异较大时增加 forward KL 约束，避免学生在坏模态上收窄。

核心要点
- 反向 KL（RKL）：按学生分布对齐老师，on-policy 稳定，易 mode-seeking。
- 正向 KL（FKL）：用 teacher 分布覆盖学生，mode-covering，补盲区。
- 门控（默认软门控）：`λ_t = sigmoid(α(τ − D_rkl(t)))`，逐 token 平滑切换 RKL 与 FKL 权重。

安装
- pip install torch transformers peft datasets accelerate wandb deepspeed

快速启动（2 卡 A800，bf16）
- accelerate launch --config_file local_distill/accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_local \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset tulu3 --batch_size 256 --group_size 4 --grad_accum 8 \
  --max_new_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
  --tau 1.0 --alpha 5.0 --gating soft --rkl exact --fkl full \
  --wandb_project dualkl-distill --wandb_name qwen8b_dualkl \
  --teacher_ds_zero3  # 教师 32B 建议启用 ZeRO-3 推理分片；可选 --teacher_ds_config path/to/ds_zero3_infer.json
  --gen_micro_batch 4 --lp_micro_batch 8  # 学生生成/前向的微批，避免 OOM

参数说明（与 local_distill 对齐的基础上新增）
- 门控与 KL：
  - `--tau`：门控阈值，越小越容易走 forward KL。
  - `--alpha`：门控平滑度（sigmoid 斜率），越大越接近硬门控。
  - `--gating`：`soft|hard`，软/硬门控。
  - `--rkl`：`exact|mc`，反向 KL 的计算方式；`exact` 为全词表，`mc` 为采样 token 近似（与 on-policy 版本一致）。
  - `--fkl`：`full|argmax`，正向 KL 的计算方式；`full` 用全词表期望（cross-entropy），`argmax` 用老师 argmax token 的 CE 近似。
- 其余与 local_distill 相同：LoRA、bf16、group_size、grad_accum、wandb、dataset/prompts。

建议超参（与 local_distill 相同基础上）：
- 1.7B：`--tau 1.0 --alpha 5.0 --gating soft --rkl exact --fkl full`；batch/group/acc 参考 local_distill/README。
- 4B/8B：同上；OOM 时优先降 `--group_size` 与 `--max_new_tokens`，或启用 DeepSpeed ZeRO-2。

说明
- exact RKL/FKL 需要同时拿到学生与老师全词表分布，显存/算力开销较 `mc/argmax` 更大；可先用 `rkl=mc` + `fkl=full` 折中。
- 只对续写 token 计算损失；prompt 段掩码不参与。
- 未定细节（如 top-k 截断 KL、分布平滑、阈值自适应）可在后续迭代中加入。
- 强烈建议：学生保持 Accelerate DDP；教师仅做推理并用 DeepSpeed ZeRO-3 分片，避免每张卡常驻 32B 权重导致 OOM。
 - 对 decoder-only 模型（如 Qwen3），已自动设置 `tokenizer.padding_side = 'left'` 以减少生成阶段的无效计算。
 - 若仍 OOM，先降低 `--gen_micro_batch`，再降低 `--lp_micro_batch`，然后再考虑调小 `--group_size` / `--batch_size` / `--max_new_tokens`。
