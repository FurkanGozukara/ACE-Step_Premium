"""Optimizer and scheduler factories for Gradio LoRA/DoRA training."""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)


OPTIMIZER_CHOICES = ("adamw", "adamw8bit", "adafactor")
SCHEDULER_CHOICES = (
    "cosine",
    "cosine_restarts",
    "linear",
    "constant",
    "constant_with_warmup",
)


def build_optimizer(
    params: Iterable,
    *,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    device_type: str,
) -> torch.optim.Optimizer:
    """Create the selected optimizer for trainable adapter parameters.

    Args:
        params: Trainable parameters.
        optimizer_type: Optimizer key from ``OPTIMIZER_CHOICES``.
        lr: Learning rate.
        weight_decay: Weight decay.
        device_type: Normalized device type such as ``"cuda"`` or ``"cpu"``.

    Returns:
        A configured PyTorch optimizer.
    """

    selected = str(optimizer_type or "adamw").lower().strip()

    if selected == "adamw8bit":
        if device_type != "cuda":
            logger.warning("AdamW 8-bit requires CUDA; falling back to AdamW.")
        else:
            try:
                from bitsandbytes.optim import AdamW8bit

                logger.info("Using optimizer: adamw8bit")
                return AdamW8bit(params, lr=lr, weight_decay=weight_decay)
            except ImportError:
                logger.warning("bitsandbytes is unavailable; falling back to AdamW.")
        selected = "adamw"

    if selected == "adafactor":
        try:
            from transformers.optimization import Adafactor

            logger.info("Using optimizer: adafactor")
            return Adafactor(
                params,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
        except ImportError:
            logger.warning("transformers Adafactor is unavailable; falling back to AdamW.")
            selected = "adamw"

    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device_type == "cuda":
        kwargs["fused"] = True
    logger.info("Using optimizer: adamw")
    return AdamW(params, **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    n_restarts: int = 4,
):
    """Create the selected learning-rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Scheduler key from ``SCHEDULER_CHOICES``.
        total_steps: Total optimizer steps in the run.
        warmup_steps: Requested warmup steps.
        lr: Base learning rate.
        n_restarts: Number of cycles for ``cosine_restarts``.

    Returns:
        A PyTorch LR scheduler.
    """

    selected = str(scheduler_type or "constant").lower().strip()
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), max(1, total_steps // 10))
    remaining = max(1, total_steps - warmup_steps)

    if selected == "constant":
        logger.info("Using scheduler: constant")
        return ConstantLR(optimizer, factor=1.0, total_iters=total_steps)

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )

    if selected == "constant_with_warmup":
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=remaining)
    elif selected == "linear":
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=remaining,
        )
    elif selected == "cosine_restarts":
        main_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, remaining // max(1, int(n_restarts))),
            T_mult=1,
            eta_min=lr * 0.01,
        )
    elif selected == "cosine":
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=remaining,
            eta_min=lr * 0.01,
        )
    else:
        logger.info("Using scheduler: constant")
        return ConstantLR(optimizer, factor=1.0, total_iters=total_steps)

    logger.info(f"Using scheduler: {selected}")
    return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_steps])
