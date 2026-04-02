# python test/instruct_test.py

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def load_prompts_from_dataset(dataset_path: str, max_samples: int = 5) -> List[str]:
    """从数据集加载指定数量的问题"""
    try:
        from datasets import load_from_disk
        
        obj = load_from_disk(dataset_path)
        # 可能是 DatasetDict 或 Dataset
        if hasattr(obj, "keys"):
            split_name = "train" if "train" in obj.keys() else list(obj.keys())[0]
            ds = obj[split_name]
        else:
            ds = obj
        
        # 尝试找到问题字段
        for field in ["question", "prompt", "input", "text"]:
            if field in ds.column_names:
                return [str(v) for v in ds[field][:max_samples]]
        
        raise ValueError("未找到问题字段")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        # 返回一些默认问题
        return [
            "What is 2+2?",
            "Explain quantum computing.",
            "How does photosynthesis work?",
            "What is the capital of France?",
            "Solve: x^2 + 5x + 6 = 0",
            "What is machine learning?",
            "Describe the water cycle.",
            "What causes seasons?",
            "Explain DNA replication.",
            "What is the speed of light?"
        ]

def apply_chat_format_with_thinking(
    tok: AutoTokenizer,
    question: str,
    system_prompt: str,
    enable_thinking: bool
) -> str:
    """应用 chat template，可选是否启用 thinking"""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

def main():
    # 配置
    # teacher_model_path = "/data/oss_bucket_0/zhiyi.lx/output/ScaleAligner/output/output_sft-pipeline-ori-Qwen3-4B-Base/sft_train_ori-0/checkpoint-2999/hf"  # 修改为你的教师模型路径
    # teacher_model_path = "/data/oss_bucket_0/zhulin/output/sft-pipeline-ori-Qwen3-8B-Base-3000_hf"  # 修改为你的教师模型路径
    # teacher_model_path = "/data/oss_bucket_0/zhulin/models/Qwen3-8B"
    teacher_model_path = "/data/oss_bucket_0/zhulin/output/Qwen3-1.7B-Base-sft-checkpoint-79"
    dataset_path = "/home/chuyuanlin.cyl/notebook/Dual-KL-Distillation/data/DeepMath-32k"  # 修改为你的数据集路径
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
    
    print("=" * 100)
    print("加载教师模型和 tokenizer...")
    print("=" * 100)
    
    # 加载模型和 tokenizer
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(teacher_model_path)
    
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    
    print(f"模型加载完成: {teacher_model_path}")
    print()
    
    # 加载问题
    print("=" * 100)
    print("加载数据集...")
    print("=" * 100)
    questions = load_prompts_from_dataset(dataset_path, max_samples=5)
    print(f"加载了 {len(questions)} 个问题")
    print()
    
    # 对每个问题进行测试
    for idx, question in enumerate(questions):
        print("=" * 100)
        print(f"问题 {idx + 1}/{len(questions)}: {question[:100]}...")
        print("=" * 100)
        print()
        
        # 测试 1: enable_thinking=False
        print("-" * 100)
        print("【测试 1】enable_thinking=False")
        print("-" * 100)
        
        prompt_no_thinking = apply_chat_format_with_thinking(
            tok, question, system_prompt, enable_thinking=False
        )
        
        print("[formatted_prompt]")
        print(repr(prompt_no_thinking))  # 只显示最后200个字符
        print()
        
        # 生成
        inputs = tok(prompt_no_thinking, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        # 解码完整输出
        full_text = tok.decode(outputs[0], skip_special_tokens=False)
        # 只显示生成的部分（去掉 prompt）
        generated_text = tok.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=False)
        
        # print("[sample_full]")
        # print(full_text[-2000:])  # 显示最后2000个字符
        # print()
        
        print("[generated_part]")
        print(generated_text[:2000])  # 显示前300个字符
        print()
        
        # 显示前5个生成的 token
        print("[first_5_tokens]")
        first_5_tokens = outputs[0][inputs['input_ids'].size(1):inputs['input_ids'].size(1)+5]
        for i, token_id in enumerate(first_5_tokens):
            token_text = tok.convert_ids_to_tokens([token_id.item()])[0]
            print(f"  Token {i+1}: {token_id.item()} -> {repr(token_text)}")
        print()
        
        # 测试 2: enable_thinking=True (默认)
        print("-" * 100)
        print("【测试 2】enable_thinking=True (或不设置)")
        print("-" * 100)
        
        prompt_with_thinking = apply_chat_format_with_thinking(
            tok, question, system_prompt, enable_thinking=True
        )
        
        print("[formatted_prompt]")
        print(repr(prompt_with_thinking))  # 只显示最后200个字符
        print()
        
        # 生成
        inputs = tok(prompt_with_thinking, return_tensors="pt").to(teacher.device)
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        # 解码完整输出
        full_text = tok.decode(outputs[0], skip_special_tokens=False)
        # 只显示生成的部分（去掉 prompt）
        generated_text = tok.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=False)
        
        # print("[sample_full]")
        # print(full_text[-2000:])  # 显示最后2000个字符
        # print()
        
        print("[generated_part]")
        print(generated_text[:2000])  # 显示前2000个字符
        print()
        
        # 显示前5个生成的 token
        print("[first_5_tokens]")
        first_5_tokens = outputs[0][inputs['input_ids'].size(1):inputs['input_ids'].size(1)+5]
        for i, token_id in enumerate(first_5_tokens):
            token_text = tok.convert_ids_to_tokens([token_id.item()])[0]
            print(f"  Token {i+1}: {token_id.item()} -> {repr(token_text)}")
        print()
        
        print()

if __name__ == "__main__":
    main()