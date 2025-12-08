from __future__ import annotations

import os
from datetime import timedelta
import torch
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize torch.distributed if env hints are present.

    Returns: (is_dist, rank, world_size, local_rank, device)
    """
    is_dist = False
    rank = 0
    world = 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", 0)))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            # Some torch versions don't expose torch.timedelta; use datetime.timedelta
            try:
                dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
            except TypeError:
                # Fallback for older torch that may not accept timeout kw
                dist.init_process_group(backend=backend)
        is_dist = True
        rank = dist.get_rank()
        world = dist.get_world_size()
    return is_dist, rank, world, local_rank, device


def shard_list(xs: list, rank: int, world_size: int) -> list:
    if world_size <= 1:
        return xs
    return [x for i, x in enumerate(xs) if (i % world_size) == rank]


def all_reduce_sum_tensor(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    return x


def cleanup_distributed() -> None:
    """Gracefully synchronize and destroy the torch.distributed process group.

    This prevents the NCCL warning on exit in recent PyTorch versions
    (>=2.4) by explicitly destroying the process group.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            # Ensure all ranks reach this point before teardown
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except Exception:
        # Be conservative: never raise on cleanup
        pass
