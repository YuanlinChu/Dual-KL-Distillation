# Eval Suite

This folder provides a modular evaluation suite for math reasoning and, later, instruction following. It supports evaluating:

- Teacher model (HF model id or local path)
- Student model before distillation (HF id/path)
- Student model after distillation (weights produced by this repo; supports both full and LoRA adapters)

Currently implemented tasks
- GSM8K: exact numeric accuracy using gold “#### answer” extraction
- MATH (hendrycks/competition_math): accuracy via LaTeX \boxed{...} or numeric extraction
- MATH-500 (HuggingFaceH4/MATH-500): accuracy with boxed/numeric extraction
- AIME (HuggingFaceH4/aime_2024): numeric accuracy with split selection

Instruction following
- 本仓库不再内置指令跟随评测，请使用官方/开源的 AlpacaEval 或 MT-Bench 仓库进行评测。

## Quickstart

Evaluate GSM8K with a base model:

```
python -m eval.run_eval \
  --task gsm8k \
  --model Qwen/Qwen3-7B \
  --dtype bf16 \
  --max_new_tokens 256 \
  --batch_size 8 \
  --save_outputs outputs_gsm8k.jsonl
```

Evaluate a LoRA adapter produced by dual_kl training:

```
python -m eval.run_eval \
  --task gsm8k \
  --model /path/to/dual-kl-out/step-2000 \
  --base_model Qwen/Qwen3-7B \
  --dtype bf16 \
  --save_outputs out.jsonl
```

Evaluate MATH (hendrycks/competition_math):

```
python -m eval.run_eval \
  --task math \
  --model /path/to/final-out \
  --base_model Qwen/Qwen3-7B \
  --dtype bf16 \
  --max_new_tokens 512
```

Evaluate MATH-500 (HuggingFaceH4/MATH-500):

```
python -m eval.run_eval \
  --task math500 \
  --math500_split test \
  --model Qwen/Qwen3-7B \
  --dtype bf16 \
  --max_new_tokens 512
```

Evaluate on a local AIME-style JSONL (temporary until dataset hookup):

```
python -m eval.run_eval \
  --task aime_jsonl \
  --jsonl_file /path/to/aime.jsonl \
  --model Qwen/Qwen3-7B \
  --dtype bf16
```

Where each JSONL line is:

```
{"question": "...", "answer": "..."}
```

## Notes
- Uses HF `datasets` when available. If offline, provide `--jsonl_file` for custom eval.
- For decoder-only models, padding side is set to left for efficient generation.
- If `--model` directory contains a LoRA adapter (adapter_config.json), pass `--base_model` to load base weights + adapter.
Evaluate AIME (HuggingFaceH4/aime_2024):

```
python -m eval.run_eval \
  --task aime \
  --aime_split test \
  --model Qwen/Qwen3-7B \
  --dtype bf16 \
  --max_new_tokens 256
```

如需与本仓库模型评测配合，可先用本套件导出模型预测，再按 AlpacaEval/MT-Bench 官方工具的输入格式进行评测。
