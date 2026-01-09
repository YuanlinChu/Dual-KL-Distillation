#!/usr/bin/env bash
set -euo pipefail

SESSION="vllm-dkl-4inst"

MODEL="Qwen/Qwen3-1.7B"
MAX_LEN=10000
MEM_UTIL=0.8
RANK=64
TP=2

# 端口 -> GPU 组
declare -A GPUS=(
  [8801]="0,1"
  [8802]="2,3"
  [8803]="4,5"
  [8804]="6,7"
)

# 端口 -> LoRA 名称
declare -A LORA_NAME=(
  [8801]="1.7b-dkl200_8_posdecay-sample32k-8192"
  [8802]="1.7b-dkl400_8_posdecay-sample32k-8192"
  [8803]="1.7b-dkl600_8_posdecay-sample32k-8192"
  [8804]="1.7b-dkl800_8_posdecay-sample32k-8192"
)

# 端口 -> LoRA 路径
declare -A LORA_PATH=(
  [8801]="../out/dkl-1.7b-32b-deepmath_sample32k-posdecay/step-200"
  [8802]="../out/dkl-1.7b-32b-deepmath_sample32k-posdecay/step-400"
  [8803]="../out/dkl-1.7b-32b-deepmath_sample32k-posdecay/step-600"
  [8804]="../out/dkl-1.7b-32b-deepmath_sample32k-posdecay/step-800"
)

# 如果 session 已存在，先 kill，避免重复起
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# 新建 session（第一个 pane）
tmux new-session -d -s "$SESSION"

# 顺序启动 4 个 pane
for port in 8801 8802 8803 8804; do
  # 第一个 pane 已存在，其余 split
  if [ "$port" != "8801" ]; then
    tmux split-window -t "$SESSION" -h
    tmux select-layout -t "$SESSION" tiled
  fi

  # 给 pane 标题（tmux 3.x 支持，方便你看）
  tmux select-pane -t "$SESSION" -T "port:${port} gpu:${GPUS[$port]}"

  CMD="source ~/anaconda3/etc/profile.d/conda.sh && \
  conda activate vllm && \
  CUDA_VISIBLE_DEVICES=${GPUS[$port]} \
  vllm serve ${MODEL} \
  --enable-lora \
  --lora-modules ${LORA_NAME[$port]}=${LORA_PATH[$port]} \
  --max-lora-rank ${RANK} \
  --tensor-parallel-size ${TP} \
  --trust-remote-code \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization ${MEM_UTIL} \
  --port ${port}"

  tmux send-keys -t "$SESSION" "$CMD" C-m
done

# 固定成 2x2
tmux select-layout -t "$SESSION" tiled

# 自动 attach
tmux attach -t "$SESSION"
