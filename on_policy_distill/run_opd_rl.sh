#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash on_policy_distill/run_opd_rl.sh 0.6b standard_opd
#   bash on_policy_distill/run_opd_rl.sh 1.7b standard_opd
#   bash on_policy_distill/run_opd_rl.sh 1.7b dual_kl_equal
#   bash on_policy_distill/run_opd_rl.sh 1.7b dual_kl_always_on
#   bash on_policy_distill/run_opd_rl.sh 4b dual_kl_mid_f
#
# Key experiments:
#   standard_opd:
#     纯 rKL 的 RL-style OPD 基线
#   dual_kl_equal:
#     fKL 初始权重为 1，并在训练前 30% 线性衰减到 0
#   dual_kl_always_on:
#     fKL 始终保持开启，用于验证“持续 fKL 是否不利于数学蒸馏”
#   dual_kl_mild_f / dual_kl_mid_f:
#     不同初始 fKL 强度的衰减版实验
#
# Optional environment overrides:
#   STUDENT_ROOT=/data/oss_bucket_0/zhulin/models
#   TEACHER_MODEL=/data/oss_bucket_0/zhulin/models/Qwen3-8B
#   OUTPUT_ROOT=/data/oss_bucket_0/zhulin/output/opd-rl
#   DATASET=/data/oss_bucket_0/zhulin/datasets/DeepMath-103K
#   MAX_NEW_TOKENS=4096
#   GEN_MICRO_BATCH=...
#   LP_MICRO_BATCH=...
#   SWANLAB_PROJECT=opd-rl
#   SWANLAB_NAME=qwen3-1.7b-dual-kl
#   SWANLAB_MODE=offline

MODEL_SIZE="${1:-}"
EXPERIMENT="${2:-standard_opd}"

if [[ -z "${MODEL_SIZE}" ]]; then
  echo "Usage: bash on_policy_distill/run_opd_rl.sh <0.6b|1.7b|4b> [experiment]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config_multi_8gpu.yaml}"
YAML_CONFIG="${YAML_CONFIG:-on_policy_distill/opd_rl_experiments.yaml}"

STUDENT_ROOT="${STUDENT_ROOT:-/data/oss_bucket_0/zhulin/output}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/oss_bucket_0/zhulin/models/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/oss_bucket_0/zhulin/output/test}"

DATASET="${DATASET:-/data/oss_bucket_0/zhulin/datasets/DeepMath-103K}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-opd}"
SWANLAB_MODE="${SWANLAB_MODE:-offline}"

case "${MODEL_SIZE}" in
  0.6b)
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-0.6B-Base-sft-checkpoint-79"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-0.6b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=32
    LP_MICRO_BATCH_DEFAULT=2
    ;;
  1.7b)
    # STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-1.7B-Base"
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-1.7B-Base-sft-checkpoint-79"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-1.7b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=32
    LP_MICRO_BATCH_DEFAULT=2
    ;;
  4b)
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-4B-Base-sft-checkpoint-79"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-4b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=32
    LP_MICRO_BATCH_DEFAULT=1
    ;;
  *)
    echo "Unknown model size: ${MODEL_SIZE}"
    echo "Supported values: 0.6b | 1.7b | 4b"
    exit 1
    ;;
esac

STUDENT_MODEL="${STUDENT_MODEL:-${STUDENT_MODEL_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${OUTPUT_NAME_DEFAULT}}"
GEN_MICRO_BATCH="${GEN_MICRO_BATCH:-${GEN_MICRO_BATCH_DEFAULT}}"
LP_MICRO_BATCH="${LP_MICRO_BATCH:-${LP_MICRO_BATCH_DEFAULT}}"
SWANLAB_NAME="${SWANLAB_NAME:-${OUTPUT_NAME_DEFAULT}}"

echo "Running OPD-RL"
echo "  model_size:        ${MODEL_SIZE}"
echo "  experiment:        ${EXPERIMENT}"
echo "  student_model:     ${STUDENT_MODEL}"
echo "  teacher_model:     ${TEACHER_MODEL}"
echo "  output_dir:        ${OUTPUT_DIR}"
echo "  dataset:           ${DATASET}"
echo "  max_new_tokens:    ${MAX_NEW_TOKENS}"
echo "  gen_micro_batch:   ${GEN_MICRO_BATCH}"
echo "  lp_micro_batch:    ${LP_MICRO_BATCH}"
echo "  swanlab_project:   ${SWANLAB_PROJECT}"
echo "  swanlab_name:      ${SWANLAB_NAME}"
echo "  swanlab_mode:      ${SWANLAB_MODE}"
echo "  accelerate_config: ${ACCELERATE_CONFIG}"
echo "  yaml_config:       ${YAML_CONFIG}"

CMD=(
  accelerate launch --config_file "${ACCELERATE_CONFIG}"
  -m on_policy_distill.train_on_policy_rl_local
  --config "${YAML_CONFIG}"
  --experiment "${EXPERIMENT}"
  --student_model "${STUDENT_MODEL}"
  --teacher_model "${TEACHER_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --dataset "${DATASET}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --gen_micro_batch "${GEN_MICRO_BATCH}"
  --lp_micro_batch "${LP_MICRO_BATCH}"
  --teacher_ds_zero3
  --swanlab_mode "${SWANLAB_MODE}"
)

if [[ -n "${SWANLAB_PROJECT}" ]]; then
  CMD+=(--swanlab_project "${SWANLAB_PROJECT}")
fi

if [[ -n "${SWANLAB_NAME}" ]]; then
  CMD+=(--swanlab_name "${SWANLAB_NAME}")
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
