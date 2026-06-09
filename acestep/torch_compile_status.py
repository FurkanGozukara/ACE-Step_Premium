"""Console status helpers for optional ``torch.compile`` requests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from loguru import logger


def compile_request_summary(
    module: torch.nn.Module,
    *,
    device_type: str,
    backend: str,
    mode: str | None,
    fullgraph: bool,
) -> str:
    """Return concise runtime details for a model compile status line."""

    dtype = _module_dtype(module)
    return (
        f"device={device_type or 'unknown'} dtype={dtype or 'unknown'} "
        f"params={_module_parameter_count(module):,} "
        f"buffers={_module_buffer_count(module):,} "
        f"backend={backend} mode={mode or 'default'} fullgraph={fullgraph} "
        f"torch={getattr(torch, '__version__', 'unknown')} "
        f"torch_cuda={getattr(getattr(torch, 'version', None), 'cuda', None) or 'n/a'}"
    )


def module_device_type(module: torch.nn.Module) -> str:
    """Return the first parameter or buffer device type for a module."""

    for tensor in _module_parameters(module):
        return str(tensor.device.type)
    for tensor in _module_buffers(module):
        return str(tensor.device.type)
    return ""


def log_compile_status(
    label: str,
    *,
    status: str,
    detail: str,
    summary: str = "",
    warning: bool = False,
) -> None:
    """Write a standardized torch.compile status line to the command console."""

    message = "[torch_compile] {} status={} detail={}"
    args: tuple[Any, ...] = (label, status, detail)
    if summary:
        message += " ({})"
        args = (*args, summary)
    if warning:
        logger.warning(message, *args)
        return
    logger.info(message, *args)


def set_compile_attrs(
    module: torch.nn.Module,
    label: str,
    compiled: bool,
    attempts: int,
    detail: str,
    *,
    verified: bool,
) -> None:
    """Attach lightweight compile metadata to ``module``."""

    module._acestep_torch_compile_requested = True
    module._acestep_torch_compiled = compiled
    module._acestep_torch_compile_verified = verified
    module._acestep_torch_compile_label = label
    module._acestep_torch_compile_attempts = attempts
    module._acestep_torch_compile_detail = detail


def _module_dtype(module: torch.nn.Module) -> str:
    for tensor in _module_parameters(module):
        return str(tensor.dtype)
    for tensor in _module_buffers(module):
        return str(tensor.dtype)
    return ""


def _module_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in _module_parameters(module))


def _module_buffer_count(module: torch.nn.Module) -> int:
    return sum(buffer.numel() for buffer in _module_buffers(module))


def _module_parameters(module: torch.nn.Module) -> Iterable[torch.Tensor]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return ()
    return parameters(recurse=True)


def _module_buffers(module: torch.nn.Module) -> Iterable[torch.Tensor]:
    buffers = getattr(module, "buffers", None)
    if not callable(buffers):
        return ()
    return buffers(recurse=True)
