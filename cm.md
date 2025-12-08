# baseline: on-policy distillation   / tulu3
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset tulu3 --batch_size 8 --group_size 4 --grad_accum 8 \
  --gen_micro_batch 4 --lp_micro_batch 4 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 512 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project opd-distill --wandb_name opd-4b-32b \
  --teacher_ds_zero3 --output_dir ./out/opd-4b-32b \
  --no_progress

# baseline: on-policy distillation   / deepmath
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 4 \
  --gen_micro_batch 2 --lp_micro_batch 2 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 512 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project opd-distill --wandb_name opd-4b-32b-deepmath \
  --teacher_ds_zero3 --output_dir ./out/opd-4b-32b-deepmath \
  --no_progress

# 新版 (mc-1 修改了fkl的方向)
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset tulu3 --batch_size 8 --group_size 4 --grad_accum 8 \
  --max_new_tokens 512 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name qwen4b_dualkl_mc-1 \
  --teacher_ds_zero3 --gen_micro_batch 4 --lp_micro_batch 4 \
  --output_dir ./out/dual-kl-mc-1 \
  --no_progress


# 新版（mc-2 修改了fkl的loss计算方式）
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new_2 \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset tulu3 --batch_size 8 --group_size 4 --grad_accum 8 \
  --max_new_tokens 512 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name qwen4b_dualkl_mc-2 \
  --teacher_ds_zero3 --gen_micro_batch 4 --lp_micro_batch 4 \
  --output_dir ./out/dual-kl-mc-2 \
  --no_progress


# eval
python -m eval.run_eval \
  --task gsm8k \
  --model ./out/dual-kl-mc-1/step-1000 \
  --base_model Qwen/Qwen3-4B \
  --dtype bf16 \
  --save_outputs dualkl-mc-1-out.jsonl

evalscope eval --eval-type llm_ckpt \
 --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
 --model-args "hub='huggingface', precision='auto'" \
 --datasets gsm8k --limit 5

python -m evalscope.run --eval-type llm_ckpt \
        --model "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
        --model-args "hub='huggingface', precision='auto'" \
        --datasets gsm8k --limit 5

evalscope eval --eval-type llm_ckpt \
        --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
        --model-args "hub='huggingface', lora_path='/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-4b-32b-deepmath-long/step-500',
  precision='auto'" \
        --datasets gsm8k --limit 5

python -m evalscope.run --eval-type llm_ckpt \
        --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
        --model-args "hub='huggingface', lora_path='~/Documents/Dual-KL-Distillation/out/opd-4b-32b-deepmath-long/step-500',
  precision='auto'" \
        --datasets gsm8k --limit 5