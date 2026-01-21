"""
Megatron-LM SFT runner for OpenThoughts3 with context parallelism (CP).

This script mirrors opr-test.py data processing (tinker_cookbook renderers)
but delegates training to Megatron-LM so CP/DP can be configured via flags.

Example (8 GPUs, CP=4, DP=2):
  torchrun --nproc_per_node 8 -m off_policy_distill.opr_megatron_cp_dp \
    --context-parallel-size 4 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --model-name Qwen/Qwen3-8B-Base \
    --dataset-path /home/chuyuanlin.cyl/.cache/modelscope/hub/datasets/open-thoughts/OpenThoughts3-1___2M \
    --renderer-name qwen3 \
    --max-prompts 384000 \
    --buffer-size 384000 \
    --system-prompt ""

Note:
  - Requires Megatron-LM installed in the environment.
  - Weight loading for HF Qwen3 models depends on your Megatron fork; if
    your fork supports HF initialization, pass the corresponding flags.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import datasets
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "tinker-cookbook"))

from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights

logger = logging.getLogger(__name__)

try:
    from megatron.training import get_args, pretrain
    from megatron.training import print_rank_0
    from megatron.training.initialize import initialize_megatron
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.model import GPTModel
    from megatron.training.utils import get_ltor_masks_and_position_ids
    from megatron.core import parallel_state

    MEGATRON_AVAILABLE = True
except Exception:
    MEGATRON_AVAILABLE = False


def _build_messages_from_row(row: dict) -> list[dict]:
    conversations = row.get("conversations", [])
    messages = [
        {
            "role": "user" if msg["from"] == "human" else "assistant",
            "content": msg["value"],
        }
        for msg in conversations
    ]
    return messages


def _build_input_and_labels(
    renderer: renderers.Renderer,
    messages: list[renderers.Message],
    max_length: int,
    system_prompt: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]

    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    datum = datum_from_model_input_weights(model_input, weights, max_length=max_length)

    input_ids = torch.tensor(list(datum.model_input.to_ints()), dtype=torch.long)
    label_ids = torch.tensor(datum.loss_fn_inputs["target_tokens"].data, dtype=torch.long)
    weight_ids = torch.tensor(datum.loss_fn_inputs["weights"].data, dtype=torch.float32)
    return input_ids, label_ids, weight_ids


class StreamingSFTDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        renderer: renderers.Renderer,
        max_length: int,
        max_prompts: int,
        buffer_size: int,
        seed: int,
        system_prompt: str | None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.max_length = max_length
        self.max_prompts = max_prompts
        self.buffer_size = buffer_size
        self.seed = seed
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ds = datasets.load_dataset(self.dataset_path, split="train", streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        if MEGATRON_AVAILABLE:
            dp_size = parallel_state.get_data_parallel_world_size()
            dp_rank = parallel_state.get_data_parallel_rank()
            if dp_size > 1:
                ds = ds.shard(num_shards=dp_size, index=dp_rank)

        count = 0
        for row in ds:
            if count >= self.max_prompts:
                break
            messages = _build_messages_from_row(row)
            input_ids, labels, weights = _build_input_and_labels(
                self.renderer,
                messages,
                self.max_length,
                self.system_prompt,
            )
            yield input_ids, labels, weights
            count += 1


def _pad_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
    seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.full((len(batch), seq_length), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), seq_length), -100, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), seq_length), dtype=torch.float32)

    for i, (ids, lbls, wts) in enumerate(batch):
        seq_len = min(ids.shape[0], seq_length)
        input_ids[i, :seq_len] = ids[:seq_len]
        labels[i, :seq_len] = lbls[:seq_len]
        loss_mask[i, :seq_len] = wts[:seq_len]

    return input_ids, labels, loss_mask


def _get_batch(data_iterator):
    args = get_args()
    batch = []
    for _ in range(args.micro_batch_size):
        try:
            batch.append(next(data_iterator))
        except StopIteration:
            break
    if not batch:
        return None

    input_ids, labels, loss_mask = _pad_batch(
        batch,
        pad_id=args.pad_token_id,
        seq_length=args.seq_length,
    )

    attention_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    input_ids = input_ids.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    loss_mask = loss_mask.cuda(non_blocking=True)
    attention_mask = attention_mask.cuda(non_blocking=True)
    position_ids = position_ids.cuda(non_blocking=True)

    return input_ids, labels, loss_mask, attention_mask, position_ids


def _loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / torch.clamp_min(
        loss_mask.sum(), 1.0
    )
    return loss, {"lm loss": loss}


def _forward_step(data_iterator, model):
    batch = _get_batch(data_iterator)
    if batch is None:
        return None
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, loss_mask


def _train_valid_test_datasets_provider():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id
    args.eod_token = tokenizer.eos_token_id

    renderer = renderers.get_renderer(args.renderer_name, tokenizer)
    dataset = StreamingSFTDataset(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        renderer=renderer,
        max_length=args.max_length,
        max_prompts=args.max_prompts,
        buffer_size=args.buffer_size,
        seed=args.seed,
        system_prompt=args.system_prompt,
    )
    return dataset, None, None


def _model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )
    return model


def _add_opr_args(parser):
    group = parser.add_argument_group(title="opr")
    group.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-Base")
    group.add_argument(
        "--dataset-path",
        type=str,
        default="/home/chuyuanlin.cyl/.cache/modelscope/hub/datasets/open-thoughts/OpenThoughts3-1___2M",
    )
    group.add_argument("--renderer-name", type=str, default="qwen3")
    group.add_argument("--system-prompt", type=str, default=None)
    group.add_argument("--max-length", type=int, default=16384)
    group.add_argument("--max-prompts", type=int, default=128 * 3000)
    group.add_argument("--buffer-size", type=int, default=128 * 3000)
    return parser


def main() -> None:
    if not MEGATRON_AVAILABLE:
        raise RuntimeError(
            "Megatron-LM is not available. Install your Megatron-LM fork "
            "and ensure it is importable as 'megatron'."
        )

    initialize_megatron(extra_args_provider=_add_opr_args)
    args = get_args()

    dp_size = parallel_state.get_data_parallel_world_size()
    print_rank_0(
        f"Starting Megatron SFT | model={args.model_name} "
        f"cp={args.context_parallel_size} dp={dp_size} "
        f"tp={args.tensor_model_parallel_size} pp={args.pipeline_model_parallel_size}"
    )

    pretrain(
        _train_valid_test_datasets_provider,
        _model_provider,
        _forward_step,
        extra_args_provider=_add_opr_args,
        loss_func=_loss_func,
    )


if __name__ == "__main__":
    main()
