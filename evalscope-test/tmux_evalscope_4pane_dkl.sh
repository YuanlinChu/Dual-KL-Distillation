#!/usr/bin/env bash
set -euo pipefail

SESSION="evalscope-math_500-4"

# 你要评测的共同参数
DATASET="math_500"
REPEATS=3
API_KEY="EMPTY"
DATASET_ARGS_FILE="dataset_args.json"
GEN_CFG='{"do_sample":true,"temperature":0.7,"max_tokens":8192}'

# 4 个实例的 (model, port)
MODELS=(
  "1.7b-dkl200_8_posdecay-sample32k-8192"
  "1.7b-dkl400_8_posdecay-sample32k-8192"
  "1.7b-dkl600_8_posdecay-sample32k-8192"
  "1.7b-dkl800_8_posdecay-sample32k-8192"
)
PORTS=(8801 8802 8803 8804)

# 如果 session 已存在，先干掉（避免重复启动）
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# 新建 session（第一个 pane）
tmux new-session -d -s "$SESSION" -n eval

# 先在第一个 pane 里跑第一个任务
for i in 0 1 2 3; do
  if [ "$i" -gt 0 ]; then
    tmux split-window -t "$SESSION" -h
    tmux select-layout -t "$SESSION" tiled
  fi

  MODEL="${MODELS[$i]}"
  PORT="${PORTS[$i]}"
  API_URL="http://127.0.0.1:${PORT}/v1"

  # 给每个 pane 一个标题，方便你看（状态栏里会显示）
  tmux select-pane -t "$SESSION".0 -T "${MODEL}@${PORT}"
  
  CMD="source ~/anaconda3/etc/profile.d/conda.sh && \
  conda activate evalscope && \
  evalscope eval \
--model ${MODEL} \
--api-url ${API_URL} \
--api-key ${API_KEY} \
--eval-type openai_api \
--datasets ${DATASET} \
--generation-config '${GEN_CFG}' \
--repeats ${REPEATS} \
--dataset-args \"\$(cat ${DATASET_ARGS_FILE})\""

  tmux send-keys -t "$SESSION" "$CMD" C-m
done

# 排成 2x2
tmux select-layout -t "$SESSION" tiled

# attach 进去同时看四个
tmux attach -t "$SESSION"
