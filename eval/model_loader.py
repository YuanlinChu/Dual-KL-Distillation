from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def detect_lora_adapter(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        or os.path.exists(os.path.join(path, "adapter_model.bin"))
        or os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


def load_model_and_tokenizer(
    model_or_adapter: str,
    base_model: Optional[str] = None,
    dtype: str = "bf16",
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a CausalLM and tokenizer.

    - If `model_or_adapter` is a HF model id or full checkpoint dir, loads directly.
    - If it looks like a LoRA adapter, requires `base_model` and merges at runtime.
    """
    torch_dtype = None
    if dtype.lower() == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype.lower() == "fp16":
        torch_dtype = torch.float16
    elif dtype.lower() == "fp32":
        torch_dtype = None

    is_adapter = detect_lora_adapter(model_or_adapter)
    if is_adapter:
        if not PEFT_AVAILABLE:
            raise RuntimeError("LoRA adapter detected but peft is not installed. Please `pip install peft`." )
        if base_model is None:
            raise ValueError("--base_model is required when --model points to a LoRA adapter directory.")
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype if torch_dtype else None)
        tok = AutoTokenizer.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_or_adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_or_adapter, dtype=torch_dtype if torch_dtype else None)
        tok = AutoTokenizer.from_pretrained(model_or_adapter)

    # Prefer left padding for decoder-only generation efficiency
    try:
        if getattr(tok, "pad_token_id", None) is not None:
            tok.padding_side = "left"
    except Exception:
        pass
    ensure_pad_token(tok)
    # Move to device if provided
    if device is not None:
        try:
            model.to(device)
        except Exception:
            pass
    return model, tok
