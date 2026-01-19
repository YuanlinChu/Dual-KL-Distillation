from transformers import AutoModelForCausalLM
from peft import PeftModel
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

model_name = "Qwen/Qwen3-8B-Base"
adapter_path = "/path/to/adapter"  # 训练输出目录

base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)

tokenizer = get_tokenizer(model_name)
renderer = renderers.get_renderer("qwen3", tokenizer=tokenizer)

question = "你的问题..."
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