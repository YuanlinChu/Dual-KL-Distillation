"""
Local SFT on OpenThoughts3 without Tinker API.

Matches the hyperparameters in off_policy_reasoning.py, but runs training locally
using Transformers + Accelerate + (optional) LoRA.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import chz
import datasets
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    import swanlab  # type: ignore

    SWANLAB_AVAILABLE = True
except Exception:
    SWANLAB_AVAILABLE = False


@dataclass
class Config:
    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    renderer_name: str | None = "qwen3"
    load_checkpoint_path: str | None = None

    # Training hyperparameters (match off_policy_reasoning.py defaults)
    batch_size: int = 128  # global batch size
    learning_rate: float = 1e-3
    lr_schedule: str = "linear"
    num_epochs: int = 1
    max_length: int = 16384

    # Local training controls
    per_device_batch_size: int = 1
    grad_accum: int = 16
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Dataset configuration
    buffer_size: int = 128 * 3000
    max_prompts: int = 128 * 3000

    # Logging configuration
    log_path: str | None = None
    swanlab_project: str | None = None
    swanlab_name: str | None = None
    swanlab_mode: str = "online"

    # Checkpointing
    save_every: int = 50

    # Reproducibility
    seed: int = 42

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def init_swanlab(cfg: Config) -> None:
    init_kwargs = {"project": cfg.swanlab_project, "config": vars(cfg)}
    if cfg.swanlab_name:
        for name_key in ("experiment_name", "name", "run_name"):
            try:
                swanlab.init(**init_kwargs, **{name_key: cfg.swanlab_name})
                return
            except TypeError:
                continue
    swanlab.init(**init_kwargs)


def _build_messages_from_row(row: dict) -> list[Message]:
    conversations = row.get("conversations", [])
    messages: list[Message] = [
        {
            "role": "user" if msg["from"] == "human" else "assistant",
            "content": msg["value"],
        }
        for msg in conversations
    ]
    return messages


def _iter_datums(
    ds: datasets.IterableDataset,
    renderer: renderers.Renderer,
    max_length: int,
    train_on_what: TrainOnWhat,
    max_prompts: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    count = 0
    for row in ds:
        if count >= max_prompts:
            break
        messages = _build_messages_from_row(row)
        datum = conversation_to_datum(messages, renderer, max_length, train_on_what)
        input_ids = torch.tensor(datum.model_input.to_ints(), dtype=torch.long)
        labels = datum.loss_fn_inputs["target_tokens"].to_torch().long()
        weights = datum.loss_fn_inputs["weights"].to_torch().float()
        yield input_ids, labels, weights
        count += 1


def _collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(x[0].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, lbls, wts) in enumerate(batch):
        seq_len = ids.shape[0]
        input_ids[i, :seq_len] = ids
        labels[i, :seq_len] = lbls
        weights[i, :seq_len] = wts
        attention_mask[i, :seq_len] = 1

    return input_ids, labels, weights, attention_mask


def _compute_weighted_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    vocab = logits.shape[-1]
    loss_flat = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    weights_flat = weights.view(-1)
    loss = (loss_flat * weights_flat).sum() / torch.clamp_min(weights_flat.sum(), 1.0)
    return loss


def _append_metrics(log_path: str, metrics: dict) -> None:
    metrics_path = os.path.join(log_path, "metrics.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def _setup_logging(log_path: str) -> None:
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_path, "train.log")),
        ],
    )


def _get_log_path(cfg: Config) -> tuple[str, str]:
    if cfg.log_path is not None:
        log_path = cfg.log_path
        run_name = os.path.basename(log_path)
    else:
        model_name = cfg.model_name.replace("/", "-")
        run_name = (
            f"sft-openthoughts3-local-{model_name}-"
            f"{cfg.lora_rank}rank-{cfg.learning_rate}lr-"
            f"{cfg.batch_size}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")
    return log_path, run_name


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    datasets.set_seed(cfg.seed)

    if cfg.lr_schedule != "linear":
        raise ValueError("Only lr_schedule=linear is supported in this local script")

    log_path, run_name = _get_log_path(cfg)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    _setup_logging(log_path)

    if cfg.num_epochs != 1:
        logger.warning("num_epochs=%d with streaming data will not repeat data", cfg.num_epochs)

    if cfg.swanlab_project and not SWANLAB_AVAILABLE:
        raise RuntimeError("swanlab is not installed. Please pip install swanlab or unset swanlab_project.")

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum)
    if accelerator.is_main_process and cfg.swanlab_project:
        if cfg.swanlab_mode in ("offline", "disabled"):
            os.environ["SWANLAB_MODE"] = cfg.swanlab_mode
        init_swanlab(cfg)

    # Resolve renderer
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)

    # Tokenizer and renderer
    tokenizer = get_tokenizer(cfg.model_name)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Model
    torch_dtype = None
    if cfg.dtype.lower() == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.dtype.lower() == "fp16":
        torch_dtype = torch.float16

    load_path = cfg.load_checkpoint_path or cfg.model_name
    model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch_dtype)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.lora_rank > 0:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. Please pip install peft or set lora_rank=0.")
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)

    model, optimizer = accelerator.prepare(model, optimizer)

    # Dataset
    ds = datasets.load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.buffer_size)
    if accelerator.num_processes > 1:
        ds = ds.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

    if cfg.batch_size % accelerator.num_processes != 0:
        raise ValueError("batch_size must be divisible by number of processes")
    per_rank_batch = cfg.batch_size // accelerator.num_processes
    if per_rank_batch != cfg.per_device_batch_size * cfg.grad_accum:
        raise ValueError(
            "per_device_batch_size * grad_accum must equal batch_size / world_size"
        )

    total_steps = (cfg.max_prompts // cfg.batch_size) * cfg.num_epochs
    if total_steps <= 0:
        raise ValueError("max_prompts and batch_size imply zero steps")

    max_prompts_per_rank = total_steps * per_rank_batch

    logger.info("World size: %d", accelerator.num_processes)
    logger.info("Total steps: %d", total_steps)
    logger.info("Prompts per rank: %d", max_prompts_per_rank)

    train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    datum_iter = _iter_datums(ds, renderer, cfg.max_length, train_on_what, max_prompts_per_rank)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    start_time = time.time()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        got_any = False
        for micro_idx in range(cfg.grad_accum):
            batch = []
            for _ in range(cfg.per_device_batch_size):
                try:
                    batch.append(next(datum_iter))
                except StopIteration:
                    break
            if not batch:
                break

            got_any = True
            input_ids, labels, weights, attention_mask = _collate_batch(
                batch, pad_id=tokenizer.pad_token_id
            )
            input_ids = input_ids.to(accelerator.device)
            labels = labels.to(accelerator.device)
            weights = weights.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            sync_context = (
                accelerator.no_sync(model)
                if micro_idx < cfg.grad_accum - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                with accelerator.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = _compute_weighted_nll(outputs.logits, labels, weights)
                accelerator.backward(loss)

            step_loss += loss.detach().float().item()
            step_tokens += int(weights.sum().item())

        if not got_any:
            logger.warning("No more data available; stopping early at step %d", step)
            break

        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if accelerator.is_main_process:
            lr = scheduler.get_last_lr()[0]
            metrics = {
                "step": step,
                "train/loss": step_loss / max(cfg.grad_accum, 1),
                "train/tokens": step_tokens,
                "optim/lr": lr,
                "time/elapsed": time.time() - start_time,
            }
            logger.info(
                "step %d/%d | loss=%.4f | lr=%.6g | tokens=%d",
                step,
                total_steps,
                metrics["train/loss"],
                lr,
                step_tokens,
            )
            _append_metrics(log_path, metrics)
            if cfg.swanlab_project:
                swanlab.log(metrics, step=step)

        if accelerator.is_main_process and cfg.save_every > 0 and step % cfg.save_every == 0:
            ckpt_dir = os.path.join(log_path, f"step-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = accelerator.unwrap_model(model)
            to_save.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Saved checkpoint to %s", ckpt_dir)

    if accelerator.is_main_process:
        to_save = accelerator.unwrap_model(model)
        to_save.save_pretrained(log_path)
        tokenizer.save_pretrained(log_path)
        logger.info("Training complete. Saved model to %s", log_path)
        if cfg.swanlab_project:
            finish = getattr(swanlab, "finish", None)
            if callable(finish):
                finish()


def main() -> None:
    cfg = chz.entrypoint(Config)
    train(cfg)


if __name__ == "__main__":
    main()
