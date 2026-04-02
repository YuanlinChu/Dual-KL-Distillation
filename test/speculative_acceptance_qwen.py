"""
Classic speculative decoding evaluation focused on draft-token acceptance rate.

Standard algorithm used here:
1. Draft model proposes gamma tokens autoregressively.
2. Target model evaluates those draft tokens under the same sampling distribution.
3. Each draft token is accepted with probability min(1, p(x) / q(x)).
4. On first rejection, sample one correction token from the residual distribution
   proportional to max(0, p - q), then continue from that corrected prefix.
5. If all draft tokens are accepted, sample one extra token from the target model.

Recommended usage for full-finetuned draft models:

python evalscope-test/speculative_acceptance_qwen.py \
  --target_model /path/to/target_model \
  --draft_model /path/to/opd_full_finetune_model \
  --dataset aime24 \
  --aime_split train \
  --max_samples 32 \
  --gamma 4 \
  --max_new_tokens 512 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.95 \
  --output_json out/spec_accept_opd.json

python evalscope-test/speculative_acceptance_qwen.py \
  --target_model /path/to/target_model \
  --draft_model /path/to/sft_full_finetune_model \
  --dataset aime24 \
  --aime_split train \
  --max_samples 32 \
  --gamma 4 \
  --max_new_tokens 512 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.95 \
  --output_json out/spec_accept_sft.json

LoRA loading is kept only as an optional compatibility path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

try:
    from datasets import load_dataset  # type: ignore

    _HAVE_DATASETS = True
except Exception:
    _HAVE_DATASETS = False

try:
    from peft import PeftModel  # type: ignore

    _HAVE_PEFT = True
except Exception:
    _HAVE_PEFT = False

try:
    from tqdm.auto import tqdm  # type: ignore

    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


@dataclass
class Args:
    target_model: str
    target_lora: Optional[str]
    draft_model: str
    draft_lora: Optional[str]
    tokenizer_path: Optional[str]
    dataset: str
    aime_split: str
    jsonl_file: Optional[str]
    input_key: Optional[str]
    max_samples: Optional[int]
    max_prompt_tokens: Optional[int]
    max_new_tokens: int
    gamma: int
    temperature: float
    top_p: float
    do_sample: bool
    seed: int
    dtype: str
    device_map: str
    system_prompt: Optional[str]
    enable_thinking: bool
    output_json: Optional[str]
    print_samples: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Classic speculative decoding evaluation for draft-token acceptance rate."
    )
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--target_lora", type=str, default=None)
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--draft_lora", type=str, default=None, help="Optional compatibility path for LoRA draft models.")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="aime24")
    parser.add_argument("--aime_split", type=str, default="train")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gamma", type=int, default=4, help="Draft tokens proposed per speculative block.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--print_samples", type=int, default=2)
    ns = parser.parse_args()
    return Args(**vars(ns))


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tok: PreTrainedTokenizerBase) -> None:
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass


def device_of(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def load_model(
    model_name: str,
    lora_path: Optional[str],
    dtype: str,
    device_map: str,
) -> PreTrainedModel:
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None if device_map == "none" else device_map,
    )
    if lora_path:
        if not _HAVE_PEFT:
            raise RuntimeError("peft not installed; cannot load LoRA adapter. pip install peft")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model


def load_models_and_tokenizer(args: Args) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
    target = load_model(args.target_model, args.target_lora, args.dtype, args.device_map)
    draft = load_model(args.draft_model, args.draft_lora, args.dtype, args.device_map)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.draft_model)
    ensure_pad_token(tok)

    target_vocab = getattr(target.config, "vocab_size", None)
    draft_vocab = getattr(draft.config, "vocab_size", None)
    if target_vocab is not None and draft_vocab is not None and target_vocab != draft_vocab:
        raise ValueError(
            f"Target vocab_size ({target_vocab}) != draft vocab_size ({draft_vocab}). "
            "Classic speculative decoding requires token-space alignment."
        )
    return target, draft, tok


def apply_chat_template(tok: PreTrainedTokenizerBase, question: str, args: Args) -> str:
    if hasattr(tok, "apply_chat_template") and callable(tok.apply_chat_template):
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": question})
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if args.enable_thinking:
            kwargs["enable_thinking"] = True
        try:
            return tok.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tok.apply_chat_template(messages, **kwargs)
    if args.system_prompt:
        return f"System: {args.system_prompt}\nUser: {question}\nAssistant:"
    return f"User: {question}\nAssistant:"


def load_questions(args: Args) -> List[str]:
    if args.dataset.lower() in ("aime24", "aime", "aime_2024", "hf:aime_2024"):
        if not _HAVE_DATASETS:
            raise RuntimeError("datasets not installed; pip install datasets or provide --jsonl_file")
        ds = load_dataset("HuggingFaceH4/aime_2024", split=args.aime_split)
        out: List[str] = []
        for row in ds:  # type: ignore
            q = pick_first_string(row, ("question", "problem", "prompt", "input"))
            if q:
                out.append(q)
        return out

    if args.jsonl_file:
        out = []
        keys = tuple(k for k in [args.input_key, "question", "problem", "prompt", "input"] if k)
        with open(args.jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                q = pick_first_string(row, keys)
                if q:
                    out.append(q)
        return out

    raise ValueError("Unsupported dataset source. Use --dataset aime24 or provide --jsonl_file.")


def pick_first_string(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def truncate_prompt(
    tok: PreTrainedTokenizerBase,
    prompt_text: str,
    max_prompt_tokens: Optional[int],
) -> str:
    if max_prompt_tokens is None:
        return prompt_text
    ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_prompt_tokens:
        return prompt_text
    ids = ids[-max_prompt_tokens:]
    return tok.decode(ids, skip_special_tokens=False)


@torch.no_grad()
def next_token_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1].float()
    return normalized_probs_from_logits(logits, temperature=temperature, top_p=top_p, do_sample=do_sample)


@torch.no_grad()
def block_target_probs(
    model: PreTrainedModel,
    full_ids: torch.Tensor,
    prefix_len: int,
    steps: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> List[torch.Tensor]:
    out = model(input_ids=full_ids, use_cache=False)
    logits = out.logits[0].float()
    start = prefix_len - 1
    end = prefix_len - 1 + steps
    out_probs: List[torch.Tensor] = []
    for pos in range(start, end):
        out_probs.append(
            normalized_probs_from_logits(
                logits[pos],
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        )
    return out_probs


def normalized_probs_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    logits = logits.float()
    if (not do_sample) or temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices, sorted_probs)
    denom = filtered.sum()
    if denom.item() <= 0:
        return probs
    return filtered / denom


def sample_from_probs(probs: torch.Tensor) -> int:
    if probs.dim() != 1:
        raise ValueError("Expected 1D probability tensor.")
    total = probs.sum()
    if not torch.isfinite(total) or total.item() <= 0:
        return int(torch.argmax(probs).item())
    draw = torch.multinomial(probs, num_samples=1)
    return int(draw.item())


def residual_sample(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> int:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    denom = residual.sum()
    if not torch.isfinite(denom) or denom.item() <= 1e-12:
        return sample_from_probs(target_probs)
    residual = residual / denom
    return sample_from_probs(residual)


def contains_eos(token_id: int, tok: PreTrainedTokenizerBase) -> bool:
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is None:
        return False
    if isinstance(eos_id, list):
        return token_id in eos_id
    return token_id == eos_id


def append_token(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    token = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
    return torch.cat([input_ids, token], dim=1)


@dataclass
class SampleMetrics:
    prompt: str
    generated_text: str
    generated_tokens: int
    speculative_blocks: int
    drafted_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    fully_accepted_blocks: int
    correction_tokens: int
    token_acceptance_rate: float
    full_block_acceptance_rate: float
    mean_accept_length_per_block: float


@torch.no_grad()
def speculative_decode_one(
    prompt_text: str,
    target: PreTrainedModel,
    draft: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    args: Args,
) -> SampleMetrics:
    target_device = device_of(target)
    draft_device = device_of(draft)
    prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    target_ids = prompt_ids.to(target_device)
    draft_ids = prompt_ids.to(draft_device)
    prompt_len = int(prompt_ids.shape[1])

    drafted_tokens = 0
    accepted_tokens = 0
    rejected_tokens = 0
    speculative_blocks = 0
    fully_accepted_blocks = 0
    correction_tokens = 0
    generated_tokens = 0

    while generated_tokens < args.max_new_tokens:
        speculative_blocks += 1
        proposed_tokens: List[int] = []
        draft_token_probs: List[float] = []
        draft_prob_vectors: List[torch.Tensor] = []

        for _ in range(args.gamma):
            probs_q = next_token_probs(
                draft,
                draft_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            token_id = sample_from_probs(probs_q)
            proposed_tokens.append(token_id)
            draft_prob_vectors.append(probs_q.cpu())
            draft_token_probs.append(float(probs_q[token_id].item()))
            draft_ids = append_token(draft_ids, token_id)

            drafted_tokens += 1
            if contains_eos(token_id, tok):
                break
            if generated_tokens + len(proposed_tokens) >= args.max_new_tokens:
                break

        target_block = torch.tensor([proposed_tokens], dtype=torch.long, device=target_device)
        full_target_ids = torch.cat([target_ids, target_block], dim=1)
        target_prob_vectors = block_target_probs(
            target,
            full_target_ids,
            prefix_len=int(target_ids.shape[1]),
            steps=len(proposed_tokens) + 1,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

        accepted_in_block = 0
        rejection_happened = False
        for idx, token_id in enumerate(proposed_tokens):
            p_vec = target_prob_vectors[idx]
            q_vec = draft_prob_vectors[idx].to(p_vec.device)
            p_token = max(float(p_vec[token_id].item()), 0.0)
            q_token = max(float(draft_token_probs[idx]), 1e-12)
            alpha = min(1.0, p_token / q_token)
            if random.random() <= alpha:
                accepted_tokens += 1
                accepted_in_block += 1
                generated_tokens += 1
                target_ids = append_token(target_ids, token_id)
                if contains_eos(token_id, tok):
                    rejection_happened = True
                    break
                if generated_tokens >= args.max_new_tokens:
                    rejection_happened = True
                    break
                continue

            rejected_tokens += 1
            rejection_happened = True
            correction_id = residual_sample(p_vec, q_vec)
            correction_tokens += 1
            generated_tokens += 1
            target_ids = append_token(target_ids, correction_id)
            break

        if accepted_in_block == len(proposed_tokens) and not rejection_happened:
            fully_accepted_blocks += 1
            extra_target_id = sample_from_probs(target_prob_vectors[len(proposed_tokens)])
            generated_tokens += 1
            target_ids = append_token(target_ids, extra_target_id)
            if generated_tokens >= args.max_new_tokens or contains_eos(extra_target_id, tok):
                break
        elif accepted_in_block == len(proposed_tokens):
            fully_accepted_blocks += 1

        if target_ids.shape[1] > draft_ids.shape[1]:
            draft_ids = target_ids.to(draft_device)
        else:
            draft_ids = target_ids.to(draft_device)

        if contains_eos(int(target_ids[0, -1].item()), tok):
            break

    generated_seq = target_ids[0, prompt_len:]
    generated_text = tok.decode(generated_seq, skip_special_tokens=False)
    token_acceptance_rate = accepted_tokens / drafted_tokens if drafted_tokens else 0.0
    full_block_acceptance_rate = fully_accepted_blocks / speculative_blocks if speculative_blocks else 0.0
    mean_accept_length_per_block = accepted_tokens / speculative_blocks if speculative_blocks else 0.0
    return SampleMetrics(
        prompt=prompt_text,
        generated_text=generated_text,
        generated_tokens=generated_tokens,
        speculative_blocks=speculative_blocks,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        rejected_tokens=rejected_tokens,
        fully_accepted_blocks=fully_accepted_blocks,
        correction_tokens=correction_tokens,
        token_acceptance_rate=token_acceptance_rate,
        full_block_acceptance_rate=full_block_acceptance_rate,
        mean_accept_length_per_block=mean_accept_length_per_block,
    )


def summarize(samples: List[SampleMetrics]) -> Dict[str, Any]:
    total_drafted = sum(x.drafted_tokens for x in samples)
    total_accepted = sum(x.accepted_tokens for x in samples)
    total_rejected = sum(x.rejected_tokens for x in samples)
    total_blocks = sum(x.speculative_blocks for x in samples)
    total_full_blocks = sum(x.fully_accepted_blocks for x in samples)
    total_generated = sum(x.generated_tokens for x in samples)
    total_correction = sum(x.correction_tokens for x in samples)

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "num_samples": len(samples),
        "total_generated_tokens": total_generated,
        "total_drafted_tokens": total_drafted,
        "total_accepted_tokens": total_accepted,
        "total_rejected_tokens": total_rejected,
        "total_correction_tokens": total_correction,
        "total_speculative_blocks": total_blocks,
        "total_fully_accepted_blocks": total_full_blocks,
        "token_acceptance_rate": (total_accepted / total_drafted) if total_drafted else 0.0,
        "full_block_acceptance_rate": (total_full_blocks / total_blocks) if total_blocks else 0.0,
        "mean_accept_length_per_block": (total_accepted / total_blocks) if total_blocks else 0.0,
        "mean_sample_token_acceptance_rate": mean([x.token_acceptance_rate for x in samples]),
        "mean_sample_full_block_acceptance_rate": mean([x.full_block_acceptance_rate for x in samples]),
        "mean_sample_generated_tokens": mean([float(x.generated_tokens) for x in samples]),
    }


def main() -> None:
    args = parse_args()
    set_seed_all(args.seed)

    target, draft, tok = load_models_and_tokenizer(args)
    questions = load_questions(args)
    if args.max_samples is not None:
        questions = questions[: args.max_samples]

    prompt_texts = [
        truncate_prompt(tok, apply_chat_template(tok, question, args), args.max_prompt_tokens)
        for question in questions
    ]

    iterator = prompt_texts
    if _HAVE_TQDM:
        iterator = tqdm(prompt_texts, desc="spec-eval")

    sample_metrics: List[SampleMetrics] = []
    for prompt_text in iterator:
        metrics = speculative_decode_one(prompt_text, target, draft, tok, args)
        sample_metrics.append(metrics)

    summary = summarize(sample_metrics)
    payload = {
        "config": asdict(args),
        "summary": summary,
        "samples": [asdict(x) for x in sample_metrics],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for idx, sample in enumerate(sample_metrics[: max(args.print_samples, 0)]):
        print(f"[sample {idx}] acceptance={sample.token_acceptance_rate:.4f} "
              f"full_block={sample.full_block_acceptance_rate:.4f} "
              f"drafted={sample.drafted_tokens} accepted={sample.accepted_tokens}")
        print(sample.generated_text[:800])
        print("-" * 80)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
