#!/usr/bin/env python3
"""
ModelScope 适配版启动器：
- 复用 dual_kl/train_dualkl_new_4.py 的训练主逻辑与参数
- 仅在进入训练前，把 `--student_model` 与 `--teacher_model` 解析为本地目录：
  - 若传入的是本地路径：直接使用
  - 若传入的是 ModelScope 的模型 ID：通过 modelscope.snapshot_download 下载到本地后再使用

优点：不改动原训练脚本，不影响 wandb 等现有逻辑；只负责把模型从 ModelScope 拉到本地并把路径传给原脚本。

用法示例（与原脚本参数一致）：
  accelerate launch -m dual_kl.train_dualkl_new_4_modelscope \
    --student_model ZhipuAI/xxx-1.7B \
    --teacher_model ZhipuAI/xxx-32B \
    --dataset data/DeepMath-32k --dataset_field question \
    --batch_size 32 --group_size 1 --grad_accum 1 \
    --gen_micro_batch 2 --lp_micro_batch 2 \
    --max_new_tokens 2048 --max_prompt_tokens 256 \
    --use_lora --lora_r 64 --dtype bf16 \
    --output_dir ./out/opd-ms

注意：
- 需要安装 modelscope：pip install modelscope
- 内网可访问 ModelScope 时，传模型 ID 会自动下载；若不能访问，请直接传本地模型目录路径。
"""

from __future__ import annotations

import os
from typing import Optional

try:
    # 仅用于解析/下载模型到本地
    from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    _HAVE_MS = True
except Exception:
    _HAVE_MS = False

import dual_kl.train_dualkl_new_4 as base


def _is_local_dir(p: str) -> bool:
    try:
        return os.path.isdir(p)
    except Exception:
        return False


def _resolve_model_path(model: str, cache_dir: Optional[str] = None) -> str:
    """将传入的模型 ID/路径解析为本地目录。

    - 若已是本地目录，直接返回
    - 否则尝试通过 ModelScope 下载到本地并返回下载目录
    """
    if _is_local_dir(model):
        return model
    if not _HAVE_MS:
        raise RuntimeError("未安装 modelscope，请先 pip install modelscope，或直接传入本地模型目录路径")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    print(f"[info] Using ModelScope snapshot_download for: {model}")
    local_dir = snapshot_download(model, **kwargs)
    print(f"[ok] ModelScope downloaded to: {local_dir}")
    return local_dir


def main() -> None:
    # 直接复用原脚本的参数解析与主逻辑
    cfg = base.parse_args()

    # 可选：通过环境变量控制 ModelScope 缓存目录（不新增命令行参数，保持与原脚本一致）
    ms_cache = os.environ.get("MODELSCOPE_CACHE", None)

    # 将模型 ID 解析成本地目录
    cfg.student_model = _resolve_model_path(cfg.student_model, cache_dir=ms_cache)
    cfg.teacher_model = _resolve_model_path(cfg.teacher_model, cache_dir=ms_cache)

    # 交给原脚本继续执行（包括 wandb 逻辑等）
    base.main(cfg)


if __name__ == "__main__":
    main()

