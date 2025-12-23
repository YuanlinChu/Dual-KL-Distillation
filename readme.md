# setup 一共三个conda环境

1. 配置训练环境
- 安装conda环境
- 安装依赖
  ```bash
  conda env create -f environment.yml
  conda activate dualkl
  ```

2. 采用vllm搭配evalscope评估模型

- vllm:0.12.0
- evalscope: https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html

# command：

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
  --max_new_tokens 1024 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
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

## deepmath
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new_2 \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 4 \
  --max_new_tokens 1024 --max_prompt_tokens 128 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name qwen4b_dualkl_mc-2-deepmath \
  --teacher_ds_zero3 --gen_micro_batch 4 --lp_micro_batch 4 \
  --output_dir ./out/dual-kl-mc-2-deepmath \
  --no_progress

# eval

## base model
torchrun --nproc_per_node 4 -m eval.run_eval \
 --task gsm8k --model Qwen/Qwen3-4B --dtype bf16 \
 --batch_size 8 --save_outputs ./eval_out

## opd-4b-32b
torchrun --nproc_per_node 4 -m eval.run_eval \
 --task gsm8k --model ./out/opd-4b-32b-deepmath/step-500 \
 --base_model Qwen/Qwen3-4B --dtype bf16 \
 --batch_size 8 --save_outputs ./eval_out

## dual-kl-mc-2
torchrun --nproc_per_node 4 -m eval.run_eval \
 --task gsm8k --model ./out/dual-kl-mc-2/step-1000 \
 --base_model Qwen/Qwen3-4B --dtype bf16 \
 --batch_size 8 --save_outputs ./eval_out

torchrun --nproc_per_node 4 -m eval.run_eval \
 --task gsm8k --model ./out/dual-kl-mc-2-deepmath/step-500 \
 --base_model Qwen/Qwen3-4B --dtype bf16 \
 --batch_size 8 --save_outputs ./eval_out







## opd和dkl的比较：选择deepmath作为训练集，最大token数为4096，最大prompt为256

# baseline: on-policy distillation   / deepmath
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --gen_micro_batch 1 --lp_micro_batch 1 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name opd-4b-32b-deepmath \
  --teacher_ds_zero3 --output_dir ./out/opd-4b-32b-deepmath-long \
  --no_progress
# dual-kl / deepmath
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new_2 \
  --student_model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-4b-32b-deepmath \
  --teacher_ds_zero3 --gen_micro_batch 1 --lp_micro_batch 1 \
  --output_dir ./out/dkl-4b-32b-deepmath-long \
  --no_progress



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




# dual-kl / deepmath / qwen3 1.7B
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new_2 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath-long \
  --no_progress

accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --gen_micro_batch 2 --lp_micro_batch 2 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name opd-1.7b-32b-deepmath \
  --teacher_ds_zero3 --output_dir ./out/opd-1.7b-32b-deepmath-long \
  --no_progress


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model Qwen/Qwen3-1.7B --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd500-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-1.7b-32b-deepmath-long/step-500 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-opd500-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl500-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-long/step-500 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl500-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"



# train new-3  该版本去除归一化权重
accelerate launch --config_file accelerate_config_multi_gpu.yaml \
  -m dual_kl.train_dualkl_new_3 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath-lamr1f0.5 \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath-lamr1f0.5 \
  --lam_r 1 --lam_f 0.5 \
  --no_progress

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_3 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 8 --group_size 4 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath-lamr0f1 \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath-lamr0f1 \
  --lam_r 0 --lam_f 1

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl500-8192-lamr1f0.5=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f0.5/step-500 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl500-8192-lamr1f0.5 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"


evalscope eval --model 1.7b-dkl500-8192-lamr1f0.5 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets truthful_qa

evalscope eval --model 1.7b-opd500-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets truthful_qa

evalscope eval --model Qwen/Qwen3-1.7B --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets truthful_qa


bash run_compute_kl.sh 2>&1 | tee run_log.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl500-8192-lamr1f0.2=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f0.2/step-500 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl500-8192-lamr1f0.2 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

torchrun --nproc_per_node=8 compute_dual_kl_qwen.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train --max_samples 30 --max_new_tokens 2048 --dtype bf16 --ddp \
  --output_json output-computekl-dkl/dual_kl_metrics.json --plot_dir output-computekl-dkl/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95 \
  --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-long/step-500

  torchrun --nproc_per_node=8 compute_dual_kl_qwen.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train --max_samples 30 --max_new_tokens 2048 --dtype bf16 --ddp \
  --output_json output-computekl-dklr0f1/dual_kl_metrics.json --plot_dir output-computekl-dklr0f1/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95 \
  --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr0f1/step-500

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl500-8192-lamr0f1=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr0f1/step-500 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl500-8192-lamr0f1 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl1000-8192-lamr1f0.2=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f0.2/step-1000 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl1000-8192-lamr1f0.2 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

# new-4   修改fkl优势裁剪，增加位置衰减参数[--fkl_pos_decay]

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 8 --group_size 4 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath-lamr1f1-noposdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath-lamr1f1-noposdecay \
  --lam_r 1 --lam_f 1 \
  --no_progress

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 8 --group_size 4 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath-lamr1f1-posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay \
  --fkl_pos_decay \
  --lam_r 1 --lam_f 1 \
  --no_progress