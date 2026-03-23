#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash on_policy_distill/run_opd_rl.sh 0.6b
#   bash on_policy_distill/run_opd_rl.sh 1.7b standard_opd
#   bash on_policy_distill/run_opd_rl.sh 4b dual_kl_mid_f
#
# Optional environment overrides:
#   STUDENT_ROOT=/models/Qwen
#   TEACHER_MODEL=/models/Qwen/Qwen3-8B
#   OUTPUT_ROOT=/outputs/opd-rl
#   DATASET=deepmath
#   MAX_NEW_TOKENS=4096
#   GEN_MICRO_BATCH=...
#   LP_MICRO_BATCH=...

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

STUDENT_ROOT="${STUDENT_ROOT:-/path/to}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/out}"

DATASET="${DATASET:-deepmath}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

case "${MODEL_SIZE}" in
  0.6b)
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-0.6B-Base"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-0.6b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=16
    LP_MICRO_BATCH_DEFAULT=8
    ;;
  1.7b)
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-1.7B-Base"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-1.7b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=8
    LP_MICRO_BATCH_DEFAULT=4
    ;;
  4b)
    STUDENT_MODEL_DEFAULT="${STUDENT_ROOT}/Qwen3-4B-Base"
    OUTPUT_NAME_DEFAULT="opd-rl-qwen3-4b-${EXPERIMENT}"
    GEN_MICRO_BATCH_DEFAULT=4
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
echo "  accelerate_config: ${ACCELERATE_CONFIG}"
echo "  yaml_config:       ${YAML_CONFIG}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
  -m on_policy_distill.train_on_policy_rl_local \
  --config "${YAML_CONFIG}" \
  --experiment "${EXPERIMENT}" \
  --student_model "${STUDENT_MODEL}" \
  --teacher_model "${TEACHER_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --gen_micro_batch "${GEN_MICRO_BATCH}" \
  --lp_micro_batch "${LP_MICRO_BATCH}" \
  --teacher_ds_zero3
