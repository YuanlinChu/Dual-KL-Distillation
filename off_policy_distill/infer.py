from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base", torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

model = PeftModel.from_pretrained(base, "/path/to/adapter")  # 训练输出目录
prompt = "...你的问题..."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))