accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.dualkl_opd \
  --student_model /home/wangyadao.wyd/project/ScaleAligner/model/Qwen/Qwen3-4B-Base --teacher_model /data/oss_bucket_0/zhulin/models/Qwen3-8B \
  --dataset /home/wangyadao.wyd/project/Dual-KL-Distillation/data/DeepMath-32k --batch_size 256 --group_size 1 --grad_accum 1 \
  --max_tokens 2048 --steps 125 --gen_micro_batch 2 --lp_micro_batch 2 \
  --swanlab_project dualkl-distill --swanlab_name dkl-4b_base_sft_500-4b_instruct-r1f0 \
  --teacher_ds_zero3 --output_dir /data/oss_bucket_0/zhulin/output/opd-out/dkl-4b_base_sft_500-8b-r1f0 \
  --lam_r 1 --lam_f 0 --learning_rate 1e-5


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.dualkl_opd \
  --student_model /data/oss_bucket_0/zhulin/output/Qwen3-4B-Base-sft-checkpoint-500 --teacher_model /data/oss_bucket_0/zhulin/models/Qwen3-8B \
  --dataset /home/wangyadao.wyd/project/Dual-KL-Distillation/data/DeepMath-32k --batch_size 256 --group_size 1 --grad_accum 1 \
  --max_tokens 2048 --steps 125 --gen_micro_batch 2 --lp_micro_batch 2 \
  --swanlab_project dualkl-distill --swanlab_name dkl-4b_base_sft_500-8b-r1f0 \
  --teacher_ds_zero3 --output_dir /data/oss_bucket_0/zhulin/output/opd-out/dkl-4b_base_sft_500-8b-r1f0 \
  --lam_r 1 --lam_f 0 --learning_rate 1e-5
  --resume_from_step 25    

accelerate launch --config_file accelerate_config_multi_8gpu.yaml \
  -m dual_kl.dualkl_opd \
  --student_model /data/oss_bucket_0/zhulin/output/Qwen3-4B-Base-sft-checkpoint-500 --teacher_model /data/oss_bucket_0/zhulin/models/Qwen3-8B \
  --dataset /home/wangyadao.wyd/project/Dual-KL-Distillation/data/DeepMath-32k --batch_size 256 --group_size 1 --grad_accum 1 \
  --max_tokens 2048 --steps 125 --gen_micro_batch 2 --lp_micro_batch 2 \
  --swanlab_project dualkl-distill --swanlab_name dkl-4b_base_sft_500-8b-r1f1 \
  --teacher_ds_zero3 --output_dir /data/oss_bucket_0/zhulin/output/opd-out/dkl-4b_base_sft_500-8b-r1f1 \
  --lam_r 1 --lam_f 1 --learning_rate 5e-6