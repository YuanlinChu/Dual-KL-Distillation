"""
Local SFT on OpenThoughts3 without Tinker API or tinker_cookbook helpers.

Matches the hyperparameters in off_policy_reasoning.py, but runs training locally
using Transformers + Accelerate + (optional) LoRA.

accelerate launch --num_processes 8 --mixed_precision bf16 -m off_policy_distill.off_policy_reasoning_local \
    model_name=/home/chuyuanlin.cyl/notebook/models/Qwen/Qwen3-4B-Base \
    learning_rate=1e-3 \
    batch_size=128 \
    lora_rank=128 \
    swanlab_project=off-policy-distillation

accelerate launch --num_processes 8 --mixed_precision bf16 -m off_policy_distill.off_policy_reasoning_local \
    model_name=/home/chuyuanlin.cyl/notebook/models/Qwen/Qwen3-8B-Base \
    learning_rate=1e-3 \
    batch_size=128 \
    lora_rank=128 \
    swanlab_project=off-policy-distillation

"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import chz
import datasets
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tinker-cookbook"))

from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights

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

try:
    from tqdm.auto import tqdm  # type: ignore

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


@dataclass
class Config:
    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    load_checkpoint_path: str | None = None

    # Training hyperparameters (match off_policy_reasoning.py defaults)
    batch_size: int = 128  # global batch size
    learning_rate: float = 1e-3
    lr_schedule: str = "linear"
    num_epochs: int = 1
    max_length: int = 4096

    # Local training controls
    per_device_batch_size: int = 1
    grad_accum: int = 16
    dtype: str = "bf16"
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False

    # Dataset configuration
    buffer_size: int = 128 * 3000
    max_prompts: int = 128 * 3000

    # Logging configuration
    log_path: str | None = None
    swanlab_project: str | None = None
    swanlab_name: str | None = None
    swanlab_mode: str = "online"
    progress: bool = True

    # Checkpointing
    save_every: int = 500

    # Reproducibility
    seed: int = 42

    # Chat formatting
    system_prompt: str | None = None

    behavior_if_log_dir_exists: str = "ask"
    monitor_every: int = 1
    monitor_max_chars: int = 400


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


def _iter_datums(
    ds: datasets.IterableDataset,
    renderer: renderers.Renderer,
    max_length: int,
    max_prompts: int,
    system_prompt: str | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    count = 0
    for row in ds:
        if count >= max_prompts:
            break
        messages = _build_messages_from_row(row)
        input_ids, labels, weights = _build_input_and_labels(
            renderer,
            messages,
            max_length,
            system_prompt,
        )
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
    logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)


def _check_log_dir(log_dir: str, behavior_if_exists: str) -> None:
    if os.path.exists(log_dir):
        if behavior_if_exists == "delete":
            logger.info("Log directory %s exists, deleting", log_dir)
            for root, dirs, files in os.walk(log_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"Log directory {log_dir} exists. What to do? [delete, resume, exit]: "
                )
                if user_input == "delete":
                    return _check_log_dir(log_dir, "delete")
                if user_input == "resume":
                    return
                if user_input == "exit":
                    raise SystemExit(0)
                logger.warning("Invalid input: %s", user_input)
        elif behavior_if_exists == "resume":
            return
        elif behavior_if_exists == "raise":
            raise ValueError(f"Log directory {log_dir} already exists")
        else:
            raise ValueError(f"Unknown behavior_if_exists: {behavior_if_exists}")


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
        log_path = os.path.expanduser(f"~/out/off-policy/{run_name}")
    return log_path, run_name


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 3)] + "..."


def _summarize_params(model) -> tuple[float, bool, float, bool]:
    param_max = 0.0
    grad_max = 0.0
    param_finite = True
    grad_finite = True
    for p in model.parameters():
        data = p.detach()
        if not torch.isfinite(data).all():
            param_finite = False
        param_max = max(param_max, data.abs().max().item())
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            grad_finite = False
        grad_max = max(grad_max, g.abs().max().item())
    return param_max, param_finite, grad_max, grad_finite


def _decode_sample(
    tokenizer,
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_chars: int,
) -> tuple[str, str]:
    try:
        pred_ids = logits[0].argmax(dim=-1)
        mask = labels[0].ne(-100)
        pred = pred_ids[mask].tolist()
        gold = labels[0][mask].tolist()
        pred_text = tokenizer.decode(pred, skip_special_tokens=False)
        gold_text = tokenizer.decode(gold, skip_special_tokens=False)
        return _truncate_text(pred_text, max_chars), _truncate_text(gold_text, max_chars)
    except Exception:
        return "", ""


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    # datasets.set_seed(cfg.seed) 已移除 - 不再需要，因为 shuffle 已经使用了 seed 参数

    if cfg.lr_schedule != "linear":
        raise ValueError("Only lr_schedule=linear is supported in this local script")

    log_path, run_name = _get_log_path(cfg)
    _check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    renderer = renderers.get_renderer("qwen3", tokenizer=tokenizer)

    torch_dtype = None
    if cfg.dtype.lower() == "bf16":
        torch_dtype = torch.bfloat16
    elif cfg.dtype.lower() == "fp16":
        torch_dtype = torch.float16

    load_path = cfg.load_checkpoint_path or cfg.model_name
    attn_impl = "flash_attention_2" if cfg.use_flash_attention else None
    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=torch_dtype, attn_implementation=attn_impl
    )
    model.config.use_cache = False
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    ds = datasets.load_dataset("/home/chuyuanlin.cyl/.cache/modelscope/hub/datasets/open-thoughts/OpenThoughts3-1___2M", split="train", streaming=True)
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

    datum_iter = _iter_datums(
        ds,
        renderer,
        cfg.max_length,
        max_prompts_per_rank,
        cfg.system_prompt,
    )

    warmup_steps = max(10, int(0.03 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    start_time = time.time()
    progress_bar = None
    if accelerator.is_main_process and cfg.progress and TQDM_AVAILABLE:
        progress_bar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)
    opt_step = 0  # real optimizer steps (真实更新步数)
    while opt_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        micro_count = 0
        step_tokens = 0
        got_any = False
        did_step = False
        grad_norm = float("nan")
        monitor_this_step = (
            accelerator.is_main_process and cfg.monitor_every > 0 and (opt_step + 1) % cfg.monitor_every == 0
        )
        sample_pred = ""
        sample_gold = ""
        sample_logit_max: float | None = None
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

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    # Shift for next-token prediction (standard causal LM training)
                    logits = outputs.logits[:, :-1, :]
                    shift_labels = labels[:, 1:]
                    shift_weights = weights[:, 1:]

                    loss = _compute_weighted_nll(logits, shift_labels, shift_weights)

                    if monitor_this_step and not sample_pred:
                        sample_pred, sample_gold = _decode_sample(
                            tokenizer, logits.detach(), shift_labels.detach(), cfg.monitor_max_chars
                        )
                        try:
                            sample_logit_max = logits.detach().abs().max().item()
                        except Exception:
                            sample_logit_max = None

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    did_step = True
                    opt_step += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

            step_loss += loss.detach().float().item()
            micro_count += 1
            step_tokens += int(shift_weights.sum().item())

        if not got_any:
            logger.warning("No more data available; stopping early at step %d", step)
            break
        if got_any and not did_step:
            logger.warning(
                "Insufficient micro-batches to complete an optimizer step; stopping at opt_step %d",
                opt_step,
            )
            break

        if accelerator.is_main_process:
            lr = scheduler.get_last_lr()[0]
            metrics = {
                "step": opt_step,
                "train/loss": step_loss / max(micro_count, 1),
                "train/tokens": step_tokens,
                "optim/lr": lr,
                "train/grad_norm": float(grad_norm),
                "time/elapsed": time.time() - start_time,
            }
            if monitor_this_step:
                param_max, param_finite, grad_max, grad_finite = _summarize_params(model)
                metrics.update(
                    {
                        "train/param_max_abs": param_max,
                        "train/param_is_finite": 1.0 if param_finite else 0.0,
                        "train/grad_max_abs": grad_max,
                        "train/grad_is_finite": 1.0 if grad_finite else 0.0,
                        "train/loss_is_finite": 1.0 if math.isfinite(metrics["train/loss"]) else 0.0,
                    }
                )
                if sample_logit_max is not None:
                    metrics["train/logit_max_abs"] = float(sample_logit_max)
                if accelerator.device.type == "cuda":
                    metrics["train/gpu_mem_mb"] = (
                        torch.cuda.max_memory_allocated(accelerator.device) / 1024.0 / 1024.0
                    )
                if (
                    not math.isfinite(metrics["train/loss"])
                    or not param_finite
                    or not grad_finite
                ):
                    logger.warning(
                        "Non-finite detected | loss_finite=%s param_finite=%s grad_finite=%s",
                        math.isfinite(metrics["train/loss"]),
                        param_finite,
                        grad_finite,
                    )
            logger.info(
                "step %d/%d | loss=%.4f | lr=%.6g | tokens=%d",
                opt_step,
                total_steps,
                metrics["train/loss"],
                lr,
                step_tokens,
            )
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"loss={metrics['train/loss']:.4f} lr={lr:.6g} tok={step_tokens}"
                )
            _append_metrics(log_path, metrics)
            if cfg.swanlab_project:
                swanlab.log(metrics, step=opt_step)
            if monitor_this_step:
                if sample_pred:
                    logger.info("sample pred: %s", sample_pred)
                if sample_gold:
                    logger.info("sample gold: %s", sample_gold)

        if accelerator.is_main_process and cfg.save_every > 0 and opt_step > 0 and opt_step % cfg.save_every == 0:
            ckpt_dir = os.path.join(log_path, f"step-{opt_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            to_save = accelerator.unwrap_model(model)
            to_save.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Saved checkpoint to %s", ckpt_dir)

    if accelerator.is_main_process:
        if progress_bar is not None:
            progress_bar.close()
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
