import os
from typing import List, Optional, Dict

from evalscope import TaskConfig, run_task
from evalscope.report.combinator import gen_table


# ===================== 用户需根据本机实际情况修改的路径 =====================
# 基座模型：必须指向 HF 的 snapshots/<REV_SHA> 目录（包含 config.json、tokenizer.*、pytorch_model*.bin 等）
HF_SNAPSHOT_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)

# LoRA 列表（可多个）。name 仅用于命名输出目录、汇总展示；path 为 LoRA 适配器本地目录或 HF 适配器仓库名
LORA_CANDIDATES: List[Dict[str, str]] = [
    {
        "name": "lora-deepmath",
        "path": os.path.expanduser(
            "~/Documents/Dual-KL-Distillation/out/opd-4b-32b-deepmath-long/step-500"
        ),
    },
    # 可继续添加：{"name": "lora-xxx", "path": "/path/to/another/lora"},
]

# 评测数据集（可替换或叠加，如 ["gsm8k", "mmlu"]）
DATASETS = ["gsm8k"]

# 每个模型运行输出根目录（每个 variant 会有子目录）
OUTPUT_ROOT = "./outputs/hf_eval_compare"


# ===================== 统一的生成参数（偏推理） =====================
GENERATION_CFG = {
    "max_tokens": 1024,  # 给足推理空间
    "do_sample": True,   # 开启采样利于推理探索
    "temperature": 0.2,  # 低温度，稳定又有少许随机性
    "top_p": 0.95,
    "top_k": 50,
    "n": 1,
}


def build_task(
    variant_name: str,
    hf_model_dir: str,
    lora_path: Optional[str] = None,
) -> TaskConfig:
    """构造 TaskConfig：
    - 使用 HF 本地快照作为基座
    - 可选加载 LoRA 适配器
    - 生成参数偏推理
    - 固定每个 variant 的输出目录，方便对比与日志查找
    """
    model_args = {
        "hub": "huggingface",
        "tokenizer_path": hf_model_dir,
        "precision": "auto",
        # 如需显式设备映射，可开启：
        # "device_map": "auto",
    }
    if lora_path:
        model_args["lora_path"] = lora_path

    work_dir = os.path.join(OUTPUT_ROOT, variant_name)

    task = TaskConfig(
        model=hf_model_dir,
        eval_type="llm_ckpt",
        datasets=DATASETS,
        limit=50,  # 先小样本验证；满意后可去掉或增大
        generation_config=GENERATION_CFG,
        model_args=model_args,
        eval_batch_size=1,
        seed=42,
        debug=False,
        work_dir=work_dir,
        no_timestamp=True,  # 固定目录，避免追加时间戳
    )
    return task


def main():
    # 基础检查
    if not os.path.exists(HF_SNAPSHOT_DIR):
        raise FileNotFoundError(f"HF 模型快照目录不存在：{HF_SNAPSHOT_DIR}")
    for lora in LORA_CANDIDATES:
        if not os.path.exists(lora["path"]):
            raise FileNotFoundError(f"LoRA 目录不存在：{lora['path']}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 评测序列：先跑基座，再跑每个 LoRA
    variants = [("base", None)] + [(item["name"], item["path"]) for item in LORA_CANDIDATES]

    report_dirs: List[str] = []
    results_by_variant: Dict[str, dict] = {}

    for name, lora_path in variants:
        task_cfg = build_task(variant_name=name, hf_model_dir=HF_SNAPSHOT_DIR, lora_path=lora_path)
        log_path = os.path.join(task_cfg.work_dir, "logs", "eval_log.log")
        print(f"[EvalScope] 正在评测：{name} | 日志：{log_path}")
        print("[EvalScope] 首次加载模型可能较慢，请耐心等待...")

        result = run_task(task_cfg)
        results_by_variant[name] = result
        report_dirs.append(os.path.join(task_cfg.work_dir, "reports"))

    # 生成跨模型汇总表（将所有 runs 的 report 汇总到一张表）
    print("\n[EvalScope] 生成跨模型汇总表...")
    summary_txt = os.path.join(OUTPUT_ROOT, "summary.txt")
    try:
        table = gen_table(reports_path_list=report_dirs, add_overall_metric=True)
        print(table)
        # 顺便保存一份到文件
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"\n[EvalScope] 汇总表已保存：{summary_txt}")
    except Exception as e:
        print(f"[EvalScope] 生成汇总表失败：{e}")
        print("[EvalScope] 你仍可手动查看每个模型的报告目录：")
        for name, _ in variants:
            print(f"  - {os.path.join(OUTPUT_ROOT, name, 'reports')}")

    # 可选：打印返回结果的高层结构键，帮助定位具体分数
    print("\n[EvalScope] 各模型结果字典的顶层键：")
    for name in results_by_variant:
        print(f"  - {name}: {list(results_by_variant[name].keys())}")

    print("\n[EvalScope] 完成。所有输出目录：")
    for name, _ in variants:
        print(f"  - {os.path.join(OUTPUT_ROOT, name)}")


if __name__ == "__main__":
    main()

