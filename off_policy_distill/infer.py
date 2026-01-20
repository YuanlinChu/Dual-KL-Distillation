from transformers import AutoModelForCausalLM
from peft import PeftModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tinker-cookbook"))

from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

model_name = "/home/chuyuanlin.cyl/notebook/models/Qwen/Qwen3-4B-Base"
adapter_path = "/home/chuyuanlin.cyl/tinker-examples/distillation/sft-openthoughts3-local--home-chuyuanlin.cyl-notebook-models-Qwen-Qwen3-4B-Base-128rank-0.001lr-128batch-2026-01-18-22-15/step-3000"  # 训练输出目录

base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)

tokenizer = get_tokenizer(model_name)
renderer = renderers.get_renderer("qwen3", tokenizer=tokenizer)

question = "Let $A$, $B$, $C$, and $D$ be point on the hyperbola $\\frac{x^2}{20}- \\frac{y^2}{24} = 1$ such that $ABCD$ is a rhombus whose diagonals intersect at the origin. Find the greatest real number that is less than $BD^2$ for all such rhombi.\nPlease reason step by step, and put your final answer within \\boxed{}."
messages = [{"role": "user", "content": question}]
model_input = renderer.build_generation_prompt(messages)
input_ids = model_input.to_ints()

import torch
inputs = torch.tensor([input_ids], device=model.device)
out = model.generate(
    input_ids=inputs,
    do_sample=False,  # 先用确定性生成排查
    max_new_tokens=200,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))