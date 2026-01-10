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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl200_8-8192-lamr1f1-posdecay=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-200 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl200_8-8192-lamr1f1-posdecay --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

torchrun --nproc_per_node=8 compute_dual_kl_qwen2.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train --max_samples 30 --max_new_tokens 2048 --dtype bf16 --ddp \
  --output_json output-computekl-dklr1f1-posdecay/dual_kl_metrics.json --plot_dir output-computekl-dklr1f1-posdecay/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95 \
  --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-200

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd400-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-1.7b-32b-deepmath-long/step-400 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-opd400-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

# baseline opd 0.6B 32B

accelerate launch --config_file accelerate_config_multi_4gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-0.6B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 4 --group_size 4 --grad_accum 2 \
  --gen_micro_batch 2 --lp_micro_batch 2 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name opd-0.6b-32b-deepmath \
  --teacher_ds_zero3 --output_dir ./out/opd-0.6b-32b-deepmath \
  --no_progress

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-0.6B --teacher_model Qwen/Qwen3-32B \
  --dataset deepmath --batch_size 8 --group_size 4 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-0.6b-32b-deepmath-lamr1f1-posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-0.6b-32b-deepmath-lamr1f1-posdecay \
  --fkl_pos_decay \
  --lam_r 1 --lam_f 1 \
  --no_progress

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules 0.6b-dkl300_8-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-0.6b-32b-deepmath-lamr1f1-posdecay/step-300 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 0.6b-dkl300_8-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-0.6B --enable-lora --lora-modules 0.6b-opd500_4-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-0.6b-32b-deepmath/step-500 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 0.6b-opd500_4-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model Qwen/Qwen3-0.6B --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

torchrun --nproc_per_node=8 compute_dual_kl_qwen2.py --teacher_model Qwen/Qwen3-32B --student_model Qwen/Qwen3-1.7B \
  --dataset aime24 --aime_split train --max_samples 30 --max_new_tokens 2048 --dtype bf16 --ddp \
  --output_json output-computekl-dklr1f1-posdecay-t0.1/dual_kl_metrics.json --plot_dir output-computekl-dklr1f1-posdecay-t0.1/entropy_plots \
  --do_sample --temperature 0.1 --top_p 0.95 \
  --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-200

torchrun --nproc_per_node=8 compute_dual_kl_qwen2.py --teacher_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --student_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset aime24 --aime_split train --max_samples 30 --max_new_tokens 2048 --dtype bf16 --ddp \
  --output_json computekl-ds7B-1.5B/dual_kl_metrics.json --plot_dir computekl-ds7B-1.5B/entropy_plots \
  --do_sample --temperature 0.7 --top_p 0.95 \
  --student_lora /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-long/step-500

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl300_8-8192-lamr1f1-posdecay=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-300 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl300_8-8192-lamr1f1-posdecay --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl100_8-8192-lamr1f1-posdecay=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-100 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl100_8-8192-lamr1f1-posdecay --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl500_8-8192-lamr1f1-posdecay=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-posdecay/step-400 --max-lora-rank 64 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl500_8-8192-lamr1f1-posdecay --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd700_4-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-1.7b-32b-deepmath-long/step-700 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-opd700_4-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

## 1.7b-dkl200_8-8192-lamr1f1-noposdecay
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl200_8-8192-lamr1f1-noposdecay=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath-lamr1f1-noposdecay/step-200 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl300_8-8192-lamr1f1-noposdecay --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"




















# sample数据量32k，训练1000step

python sample_deepmath.py --source zwhe99/DeepMath-103K --split train --num-samples 32000 --output-dir DeepMath-32k

## baseline 方案 -- opd
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --gen_micro_batch 2 --lp_micro_batch 2 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name opd-1.7b-32b-deepmath_sample32k \
  --teacher_ds_zero3 --output_dir ./out/opd-1.7b-32b-deepmath_sample32k \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve out/opd-1.7b-32b-deepmath_sample32k-step1000-merged --served-model-name 1.7b-opd1000_8-sample32k-8192 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-opd1000_8-sample32k-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

## new-4 方案 -- posdecay 和noposdecay
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath_sample32k-noposdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath_sample32k-noposdecay \
  --lam_r 1 --lam_f 1

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath_sample32k-noposdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 --learning_rate 1e-6 \
  --output_dir ./out/dkl-1.7b-32b-deepmath_sample32k-noposdecay \
  --lam_r 1 --lam_f 1

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-32B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 256 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-32b-deepmath_sample32k-posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-32b-deepmath_sample32k-posdecay \
  --lam_r 1 --lam_f 1 --fkl_pos_decay

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve out/dkl-1.7b-32b-deepmath_sample32k-posdecay-step800-merged --served-model-name 1.7b-dkl800_8_posdecay-sample32k-8192 --tensor-parallel-size 8 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl800_8_posdecay-sample32k-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 5 --dataset-args "$(cat dataset_args.json)"

## base 模型  Qwen3-32B
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model Qwen/Qwen3-32B --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

## base 模型  Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-1.7B --tensor-parallel-size 2 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model Qwen/Qwen3-1.7B --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-1.7B --tensor-parallel-size 2 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8802

evalscope eval --model Qwen/Qwen3-1.7B --api-url http://127.0.0.1:8802/v1 --api-key EMPTY --eval-type openai_api --datasets math_500 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

## eval 多个step的lora结果

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd800_8-sample32k-8192=out/opd-1.7b-32b-deepmath_sample32k/step-800 --max-lora-rank 64 --tensor-parallel-size 1 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-opd800_8-sample32k-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd600_8-sample32k-8192=out/opd-1.7b-32b-deepmath_sample32k/step-600 --max-lora-rank 64 --tensor-parallel-size 1 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8802

evalscope eval --model 1.7b-opd600_8-sample32k-8192 --api-url http://127.0.0.1:8802/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd400_8-sample32k-8192=out/opd-1.7b-32b-deepmath_sample32k/step-400 --max-lora-rank 64 --tensor-parallel-size 1 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8803

evalscope eval --model 1.7b-opd400_8-sample32k-8192 --api-url http://127.0.0.1:8803/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-opd200_8-sample32k-8192=out/opd-1.7b-32b-deepmath_sample32k/step-200 --max-lora-rank 64 --tensor-parallel-size 1 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8804

evalscope eval --model 1.7b-opd200_8-sample32k-8192 --api-url http://127.0.0.1:8804/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules 1.7b-dkl1000_8_posdecay-sample32k-8192=/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/dkl-1.7b-32b-deepmath_sample32k-posdecay/step-1000 --max-lora-rank 64 --tensor-parallel-size 4 --trust-remote-code --max-model-len 10000 --gpu-memory-utilization 0.8 --port 8801

evalscope eval --model 1.7b-dkl1000_8_posdecay-sample32k-8192 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets aime24 --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":8192}' --repeats 3 --dataset-args "$(cat dataset_args.json)"

# 测输出长度 max 2048 4096
Qwen3-32B  avg_len  1857.44  3297
dkl-1.7b-32b-deepmath_sample32k-step400-merged  avg_len 2029  3390
opd-1.7b-32b-deepmath_sample32k-step400-merged  avg_len 2091  4139



# teacher用8b跑实验

### opd
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m on_policy_distill.train_on_policy_local \
  --student_model Qwen/Qwen3-1.7B-Base --teacher_model Qwen/Qwen3-8B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --gen_micro_batch 2 --lp_micro_batch 2 --kl_coef 1.0 --kl_discount 0.0 \
  --max_new_tokens 2048 --max_prompt_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name opd-1.7b-8b-deepmath_32k \
  --teacher_ds_zero3 --output_dir ./out/opd-1.7b-8b-deepmath_32k \
  --use_chat_template

### dkl: relu-fkl + no posdecay
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B-Base --teacher_model Qwen/Qwen3-8B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-8b-deepmath_32k-no_posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-8b-deepmath_32k-no_posdecay \
  --lam_r 1 --lam_f 1 --use_chat_template

### dkl: relu-fkl + posdecay
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4 \
  --student_model Qwen/Qwen3-1.7B-Base --teacher_model Qwen/Qwen3-8B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-8b-deepmath_32k-posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-8b-deepmath_32k-posdecay \
  --lam_r 1 --lam_f 1 --fkl_pos_decay \
  --use_chat_template

### dkl: no-relu-fkl + posdecay
accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.train_dualkl_new_4-noclipfkl \
  --student_model Qwen/Qwen3-1.7B-Base --teacher_model Qwen/Qwen3-8B \
  --dataset data/DeepMath-32k --batch_size 32 --group_size 1 --grad_accum 1 \
  --max_new_tokens 2048 --max_prompt_tokens 512 --use_lora --lora_r 64 --dtype bf16 \
  --wandb_project dualkl-distill --wandb_name dkl-1.7b-8b-deepmath_32k-noclipfkl-posdecay \
  --teacher_ds_zero3 --gen_micro_batch 2 --lp_micro_batch 2 \
  --output_dir ./out/dkl-1.7b-8b-deepmath_32k-noclipfkl-posdecay \
  --lam_r 1 --lam_f 1 --fkl_pos_decay \
  --use_chat_template