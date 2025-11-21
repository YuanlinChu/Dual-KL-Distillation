from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_batched(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 8,
) -> List[str]:
    """Simple micro-batched generation returning decoded strings."""
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    outs: List[str] = []
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    for i in range(0, len(prompts), max(1, batch_size)):
        chunk = prompts[i : i + max(1, batch_size)]
        batch = tok(chunk, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device_of(model_for_gen)) for k, v in batch.items()}
        out_ids = model_for_gen.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
        # Decode only the generated part for readability
        # Here we decode full sequence; answer extraction downstream will handle parsing
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        outs.extend(texts)
    return outs

