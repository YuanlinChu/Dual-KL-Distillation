import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

TARGET_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def find_token_triplet(obj: Any) -> Optional[Dict[str, int]]:
    """
    在任意嵌套 JSON 对象里递归查找同时包含
    input_tokens / output_tokens / total_tokens 的 dict。
    找到就返回 {"input_tokens": int, "output_tokens": int, "total_tokens": int}
    否则返回 None。
    """
    if isinstance(obj, dict):
        # 当前层就命中
        if all(k in obj for k in TARGET_KEYS):
            it, ot, tt = obj.get("input_tokens"), obj.get("output_tokens"), obj.get("total_tokens")
            # 允许是 int 或者可转 int 的字符串
            try:
                it_i = int(it)
                ot_i = int(ot)
                tt_i = int(tt)
                return {"input_tokens": it_i, "output_tokens": ot_i, "total_tokens": tt_i}
            except (TypeError, ValueError):
                pass

        # 递归子节点
        for v in obj.values():
            hit = find_token_triplet(v)
            if hit is not None:
                return hit

    elif isinstance(obj, list):
        for v in obj:
            hit = find_token_triplet(v)
            if hit is not None:
                return hit

    return None


def token_means_from_jsonl(path: str, debug_lines: int = 5) -> Tuple[float, float, float, int]:
    input_sum = 0
    output_sum = 0
    total_sum = 0
    count = 0

    # 诊断：前几行的顶层 keys
    seen_preview = 0
    matched_preview = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 打印前几行的顶层结构，帮助你确认 schema
            if seen_preview < debug_lines:
                if isinstance(item, dict):
                    print(f"[preview line {line_no}] top-level keys: {list(item.keys())[:30]}")
                else:
                    print(f"[preview line {line_no}] top-level type: {type(item).__name__}")
                seen_preview += 1

            hit = find_token_triplet(item)
            if hit is None:
                continue

            input_sum += hit["input_tokens"]
            output_sum += hit["output_tokens"]
            total_sum += hit["total_tokens"]
            count += 1

            if matched_preview < debug_lines:
                print(f"[matched line {line_no}] usage found: {hit}")
                matched_preview += 1

    if count == 0:
        raise ValueError(
            "仍然未找到任何同时包含 input_tokens/output_tokens/total_tokens 的记录。\n"
            "请检查：字段名是否不同（如 prompt_tokens/completion_tokens/total_tokens），"
            "或 tokens 是否在别的键下。可把任意一行 JSON（脱敏）贴出来我帮你对齐。"
        )

    return input_sum / count, output_sum / count, total_sum / count, count


def main():
    if len(sys.argv) < 2:
        print("用法: python count_tokens.py /hpc2hdd/home/ychu763/Documents/Dual-KL-Distillation/evalscope-test/outputs/predictions/Qwen3-32B/aime24_default.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    input_mean, output_mean, total_mean, n = token_means_from_jsonl(path)

    print("\n==== Summary ====")
    print(f"有效记录数: {n}")
    print(f"input_tokens 均值:  {input_mean:.4f}")
    print(f"output_tokens 均值: {output_mean:.4f}")
    print(f"total_tokens 均值:  {total_mean:.4f}")


if __name__ == "__main__":
    main()
