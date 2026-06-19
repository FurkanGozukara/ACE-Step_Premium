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
OPTIMIZER_HYPERPARAMETER_KEYS = (
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "adamw8bit_min_8bit_size",
    "adamw8bit_percentile_clipping",
    "adamw8bit_block_wise",
    "adamw8bit_paged",
    "adafactor_epsilon1",
    "adafactor_epsilon2",
    "adafactor_clip_threshold",
    "adafactor_decay_rate",
    "adafactor_beta1",
    "adafactor_scale_parameter",
    "adafactor_relative_step",
    "adafactor_warmup_init",
)
SCHEDULER_CHOICES = (
    "cosine",
    "cosine_restarts",
    "linear",
    "constant",
    "constant_with_warmup",
)


def _normalize_optimizer_type(optimizer_type: str | None) -> str:
    """Return a supported optimizer key."""

    selected = str(optimizer_type or "adamw").lower().strip()
    if selected not in OPTIMIZER_CHOICES:
        return "adamw"
    return selected


def optimizer_hyperparameter_defaults(optimizer_type: str | None) -> dict[str, object]:
    """Return visible defaults for the selected optimizer.

    Defaults mirror the local optimizer constructors, except Adafactor keeps
    ``scale_parameter=False`` and ``relative_step=False`` so the UI learning
    rate and scheduler keep their current meaning.
    """

    selected = _normalize_optimizer_type(optimizer_type)
    weight_decay = 0.0 if selected == "adafactor" else 0.01
    return {
        "weight_decay": weight_decay,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "adamw8bit_min_8bit_size": 4096,
        "adamw8bit_percentile_clipping": 100,
        "adamw8bit_block_wise": True,
        "adamw8bit_paged": False,
        "adafactor_epsilon1": 1e-30,
        "adafactor_epsilon2": 1e-3,
        "adafactor_clip_threshold": 1.0,
        "adafactor_decay_rate": -0.8,
        "adafactor_beta1": 0.0,
        "adafactor_scale_parameter": False,
        "adafactor_relative_step": False,
        "adafactor_warmup_init": False,
    }


def optimizer_hyperparameter_visible(optimizer_type: object, param_key: str) -> bool:
    """Return whether an optimizer hyperparameter is relevant to the selection."""

    selected = _normalize_optimizer_type(str(optimizer_type or "adamw"))
    if param_key == "weight_decay":
        return True
    if param_key.startswith("adamw8bit_"):
        return selected == "adamw8bit"
    if param_key.startswith("adafactor_"):
        return selected == "adafactor"
    return selected in {"adamw", "adamw8bit"}


def _as_float(value: object, default: float) -> float:
    """Return ``value`` as float, or ``default`` when unset/invalid."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_positive_float(value: object, default: float) -> float:
    """Return a positive float suitable for epsilon-like parameters."""

    parsed = _as_float(value, default)
    return parsed if parsed > 0.0 else default


def _as_int(value: object, default: int) -> int:
    """Return ``value`` as int, or ``default`` when unset/invalid."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_positive_beta1(value: object) -> float | None:
    """Adafactor uses ``None`` to disable first-moment momentum."""

    parsed = _as_float(value, 0.0)
    return parsed if parsed > 0.0 else None


def build_optimizer(
    params: Iterable,
    *,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    device_type: str,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    adamw8bit_min_8bit_size: int = 4096,
    adamw8bit_percentile_clipping: int = 100,
    adamw8bit_block_wise: bool = True,
    adamw8bit_paged: bool = False,
    adafactor_epsilon1: float = 1e-30,
    adafactor_epsilon2: float = 1e-3,
    adafactor_clip_threshold: float = 1.0,
    adafactor_decay_rate: float = -0.8,
    adafactor_beta1: float = 0.0,
    adafactor_scale_parameter: bool = False,
    adafactor_relative_step: bool = False,
    adafactor_warmup_init: bool = False,
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

    selected = _normalize_optimizer_type(optimizer_type)
    defaults = optimizer_hyperparameter_defaults(selected)
    weight_decay = max(
        0.0,
        _as_float(weight_decay, float(defaults["weight_decay"])),
    )
    adam_betas = (
        min(0.999999, max(0.0, _as_float(adam_beta1, 0.9))),
        min(0.999999, max(0.0, _as_float(adam_beta2, 0.999))),
    )
    adam_epsilon = _as_positive_float(adam_epsilon, 1e-8)

    if selected == "adamw8bit":
        if device_type != "cuda":
            logger.warning("AdamW 8-bit requires CUDA; falling back to AdamW.")
        else:
            try:
                from bitsandbytes.optim import AdamW8bit

                logger.info("Using optimizer: adamw8bit")
                return AdamW8bit(
                    params,
                    lr=lr,
                    betas=adam_betas,
                    eps=adam_epsilon,
                    weight_decay=weight_decay,
                    min_8bit_size=max(
                        0,
                        _as_int(adamw8bit_min_8bit_size, 4096),
                    ),
                    percentile_clipping=min(
                        100,
                        max(1, _as_int(adamw8bit_percentile_clipping, 100)),
                    ),
                    block_wise=bool(adamw8bit_block_wise),
                    is_paged=bool(adamw8bit_paged),
                )
            except ImportError:
                logger.warning("bitsandbytes is unavailable; falling back to AdamW.")
        selected = "adamw"

    if selected == "adafactor":
        try:
            from transformers.optimization import Adafactor

            relative_step = bool(adafactor_relative_step)
            warmup_init = bool(adafactor_warmup_init)
            if warmup_init and not relative_step:
                logger.warning(
                    "Adafactor warmup_init requires relative_step; enabling relative_step."
                )
                relative_step = True

            logger.info("Using optimizer: adafactor")
            return Adafactor(
                params,
                lr=None if relative_step else lr,
                eps=(
                    _as_positive_float(adafactor_epsilon1, 1e-30),
                    _as_positive_float(adafactor_epsilon2, 1e-3),
                ),
                clip_threshold=_as_positive_float(adafactor_clip_threshold, 1.0),
                decay_rate=_as_float(adafactor_decay_rate, -0.8),
                beta1=_optional_positive_beta1(adafactor_beta1),
                weight_decay=weight_decay,
                scale_parameter=bool(adafactor_scale_parameter),
                relative_step=relative_step,
                warmup_init=warmup_init,
            )
        except ImportError:
            logger.warning("transformers Adafactor is unavailable; falling back to AdamW.")
            selected = "adamw"

    kwargs = {
        "lr": lr,
        "betas": adam_betas,
        "eps": adam_epsilon,
        "weight_decay": weight_decay,
    }
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

    if any(group.get("lr") is None for group in optimizer.param_groups):
        try:
            from transformers.optimization import AdafactorSchedule

            logger.info("Using scheduler: adafactor_internal")
            return AdafactorSchedule(optimizer, initial_lr=0.0)
        except ImportError:
            logger.warning(
                "Adafactor internal scheduler is unavailable; falling back to constant."
            )

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
