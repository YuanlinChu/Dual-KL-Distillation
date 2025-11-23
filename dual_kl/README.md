# 双向 KL 蒸馏（Dual-KL）本地原型

该原型在不改动模型与训练框架（Accelerate/LoRA/数据管线）的前提下，将损失改为“对称的 MC 双向 KL 蒸馏”：
- rKL-MC（学生采样）：在学生采样 token 上最小化 (log p_s − log p_t) 的 PG 估计
- fKL-MC（教师采样）：在教师分布逐位采样 token 上最小化 (log p_t − log p_s) 的 PG 估计
- 对称 gating（标量）：g_R = relu((s_g − t_g)·s_g)，g_F = relu((t_g − s_g)·t_g)，λ_R = g_R/(g_R+g_F)，λ_F = g_F/(g_R+g_F)
- 最终 loss：loss = λ_R·rKL-MC + λ_F·fKL-MC，仅在续写且非 pad 的位置聚合

核心要点
- 纯 MC 实现，无需全词表 exact KL；工程开销接近于“argmax FKL”，仅多一次逐位 multinomial 采样
- 教师前向全程 no_grad；学生前向参与反传；teacher use_cache=True，student use_cache=False 以降低显存

安装
- pip install torch transformers peft datasets accelerate wandb deepspeed

快速启动（4 卡 A800，bf16）
- accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset tulu3 --batch_size 256 --group_size 4 --grad_accum 8 \
  --max_new_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name qwen4b_dualkl \
  --wandb_project dualkl-distill --wandb_name qwen4b_dualkl \
  --teacher_ds_zero3  # 教师 32B 建议启用 ZeRO-3 推理分片；可选 --teacher_ds_config path/to/ds_zero3_infer.json
  --gen_micro_batch 4 --lp_micro_batch 4  # 学生生成/前向的微批，避免 OOM
  --no_progress  # 如果需要禁用 tqdm 进度条则加上这个参数

参数说明
- 已不再提供 `--tau/--alpha/--gating/--rkl/--fkl` 等旧参数；当前实现采用对称 MC 双向 KL 与标量 gating，无需额外开关。
- 其余与 on_policy_distill 相同：LoRA、bf16、group_size、grad_accum、wandb、dataset/prompts。

建议超参（与 on_policy_distill 相同基础上）：
- 1.7B/4B/8B：batch/group/acc 参考 on_policy_distill/README；OOM 时优先降 `--group_size` 与 `--max_new_tokens`，或启用 DeepSpeed ZeRO-2。

说明
- 只对续写且非 pad token 计算损失；prompt 段掩码不参与。
- 强烈建议：学生保持 Accelerate DDP；教师仅做推理并用 DeepSpeed ZeRO-3 分片，避免每张卡常驻 32B 权重导致 OOM。
- 对 decoder-only 模型（如 Qwen3），已自动设置 `tokenizer.padding_side = 'left'`；若仍 OOM，先降 `--gen_micro_batch`、再降 `--lp_micro_batch`、再调小 `--group_size`/`--batch_size`/`--max_new_tokens`。
