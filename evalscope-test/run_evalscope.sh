#!/usr/bin/env bash
set -euo pipefail

# GSM8K（限制3条）
evalscope eval --model qwen3.4b.dkl500 \
  --api-url http://127.0.0.1:8801/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":4096}' \
  --repeats 3 \
  --limit 3

# AIME24（全部数据）
evalscope eval --model qwen3.4b.dkl500 \
  --api-url http://127.0.0.1:8801/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets aime24 \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":4096}' \
  --repeats 3
