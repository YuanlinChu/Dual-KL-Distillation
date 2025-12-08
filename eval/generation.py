from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


try:
    from tqdm.auto import tqdm  # type: ignore
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


@torch.no_grad()
def generate_batched(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 8,
    show_progress: bool = True,
    desc: str = "gen",
) -> List[str]:
    """Simple micro-batched generation returning decoded strings."""
    model_for_gen = getattr(model, "module", model)
    model_for_gen.eval()
    outs: List[str] = []
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    bar = None
    if show_progress and _HAVE_TQDM:
        bar = tqdm(total=len(prompts), desc=desc, leave=False, dynamic_ncols=True)
    it = range(0, len(prompts), max(1, batch_size))
    for i in it:
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
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)
        outs.extend(texts)
        if bar is not None:
            bar.update(len(chunk))
    if bar is not None:
        bar.close()
    return outs
