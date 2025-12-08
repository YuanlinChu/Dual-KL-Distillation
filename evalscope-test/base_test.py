import os
from evalscope import TaskConfig, run_task

# 请替换为你的实际路径（注意：必须指向 snapshots/<REV_SHA> 快照目录）
HF_SNAPSHOT_DIR = os.path.expanduser(
    "/hpc2hdd/home/ychu763/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)

# 如需加载 LoRA，请把下面路径替换为你的 LoRA 目录；不需要 LoRA 时设为 None
LORA_DIR = os.path.expanduser(
    "/hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/out/opd-4b-32b-deepmath-long/step-500"
)
USE_LORA = True  # 没有 LoRA 时改为 False 或置 LORA_DIR=None

def build_task(hf_model_dir: str, lora_dir: str | None = None) -> TaskConfig:
    """
    构造 TaskConfig：
    - 使用 HuggingFace 本地快照作为基座
    - 可选加载 LoRA 适配器
    - 生成参数为适合推理评测的“稳中带少量探索”的设置
    """
    model_args = {
        # 指定走 HF 加载路径（不改变项目默认行为）
        "hub": "huggingface",
        # tokenizer 默认与模型同目录；如需替换自定义模板或分词器，可改为其他路径
        "tokenizer_path": hf_model_dir,
        # 自动选择精度（如果模型权重和硬件支持会优先半精度）
        "precision": "auto",
        # 也可显式给 device_map（例如 'auto' 或 {'': 0}），不传时默认 'auto'
        # "device_map": "auto",
    }

    if lora_dir:
        model_args["lora_path"] = lora_dir

    # 生成配置：偏向推理评测，适度采样，稳定性较好
    generation_cfg = {
        "max_tokens": 2048,   # 允许输出较长的推理过程与答案
        "do_sample": True,    # 打开采样以激活一定的推理探索能力
        "temperature": 0.7,   # 少量随机性（0.1~0.3 区间常用于推理）
        "top_p": 0.95,        # nucleus sampling，覆盖面较广
        "top_k": 50,          # 限定 top-k，提升稳定性
        "n": 1,               # 单样本输出
    }

    task = TaskConfig(
        # 指定本地 HF 模型快照目录
        model=hf_model_dir,
        # 本地权重评测
        eval_type="llm_ckpt",
        # 评测数据集：GSM8K（已内置 few-shot，测数学推理）
        datasets=["gsm8k"],
        # 可按需限制条数（None 为全量）
        limit=None,
        repeats=1,
        # 生成参数
        generation_config=generation_cfg,
        # 传递给加载器的参数（含 hub、lora_path 等）
        model_args=model_args,
        # 批大小：推理评测时建议从 1 开始，便于控制显存与稳定性
        eval_batch_size=32,
        # 固定随机种子，结果更可复现
        seed=42,
        # 如需更详细日志：debug=True
        debug=False,
    )
    return task


if __name__ == "__main__":
    if USE_LORA and (not LORA_DIR or not os.path.exists(LORA_DIR)):
        raise FileNotFoundError(f"LoRA 目录不存在：{LORA_DIR}")
    if not os.path.exists(HF_SNAPSHOT_DIR):
        raise FileNotFoundError(f"HF 模型快照目录不存在：{HF_SNAPSHOT_DIR}")

    task_cfg = build_task(HF_SNAPSHOT_DIR, LORA_DIR if USE_LORA else None)
    result = run_task(task_cfg)
    print("Eval finished. Result summary keys:", list(result.keys()))