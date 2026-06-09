"""Callable compilation with status logging for optional ``torch.compile`` use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from acestep.torch_compile_status import (
    compile_request_summary,
    log_compile_status,
    module_device_type,
    set_compile_attrs,
)
from acestep.torch_compile_toolchain import ensure_compile_environment


@dataclass(frozen=True)
class TorchCompileResult:
    """Describe one optional compile request applied to a module callable."""

    requested: bool
    compiled: bool
    detail: str
    attempts: int = 0


def compile_module_callable(
    module: torch.nn.Module,
    *,
    attribute_name: str,
    label: str,
    enabled: bool = True,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
) -> TorchCompileResult:
    """Compile a module callable attribute with first-call fallback.

    Args:
        module: Module owning the callable attribute.
        attribute_name: Callable attribute to replace, such as ``"forward"`` or ``"decode"``.
        label: Human-readable label for logging and metadata.
        enabled: User-controlled switch for the compile request.
        backend: PyTorch compiler backend.
        mode: Optional PyTorch compile mode.
        fullgraph: Whether to require a single graph.

    Returns:
        Compile result metadata. A successful result means a compiled wrapper was
        installed; the wrapper restores the original callable if first execution fails.
    """
    device_type = module_device_type(module)
    summary = compile_request_summary(
        module,
        device_type=device_type,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
    )
    if not enabled:
        detail = "disabled by user option"
        log_compile_status(label, status="disabled", detail=detail, summary=summary)
        return TorchCompileResult(False, False, detail)

    original_callable = getattr(module, attribute_name, None)
    if not callable(original_callable):
        detail = f"{attribute_name} is not callable"
        log_compile_status(
            label,
            status="unavailable",
            detail=detail,
            summary=summary,
            warning=True,
        )
        return _mark_unavailable(module, label, detail)
    if not hasattr(torch, "compile"):
        detail = "torch.compile unavailable"
        log_compile_status(
            label,
            status="unavailable",
            detail=detail,
            summary=summary,
            warning=True,
        )
        return _mark_unavailable(module, label, detail)
    if device_type != "cuda":
        detail = f"device is {device_type or 'unknown'}"
        log_compile_status(label, status="skipped", detail=detail, summary=summary)
        return _mark_unavailable(module, label, detail)
    if getattr(module, "_acestep_torch_compiled", False):
        attempts = int(getattr(module, "_acestep_torch_compile_attempts", 1))
        log_compile_status(
            label,
            status="already_compiled",
            detail=f"attempts={attempts}",
            summary=summary,
        )
        return TorchCompileResult(True, True, "already compiled", attempts)

    toolchain = ensure_compile_environment()
    if not toolchain.ok:
        log_compile_status(
            label,
            status="unavailable",
            detail=toolchain.detail,
            summary=summary,
            warning=True,
        )
        return _mark_unavailable(module, label, toolchain.detail)

    attempts = int(getattr(module, "_acestep_torch_compile_attempts", 0)) + 1
    try:
        compile_kwargs = {"backend": backend, "fullgraph": fullgraph}
        if mode:
            compile_kwargs["mode"] = mode
        compiled_callable = torch.compile(original_callable, **compile_kwargs)
    except Exception as exc:
        detail = f"compile setup failed: {exc}"
        log_compile_status(
            label,
            status="setup_failed",
            detail=detail,
            summary=summary,
            warning=True,
        )
        set_compile_attrs(module, label, False, attempts, detail, verified=False)
        return TorchCompileResult(True, False, detail, attempts)

    active = {"enabled": True, "verified": False}

    def _compiled_callable_with_fallback(*args: Any, **kwargs: Any) -> Any:
        """Run compiled callable, restoring eager execution on first failure."""

        if not active["enabled"]:
            return original_callable(*args, **kwargs)
        try:
            output = compiled_callable(*args, **kwargs)
            if not active["verified"]:
                active["verified"] = True
                detail = f"first compiled {attribute_name} succeeded; {toolchain.detail}"
                set_compile_attrs(module, label, True, attempts, detail, verified=True)
                log_compile_status(
                    label,
                    status=f"first_{attribute_name}_ok",
                    detail=detail,
                    summary=summary,
                )
            return output
        except Exception as exc:
            active["enabled"] = False
            setattr(module, attribute_name, original_callable)
            detail = f"first compiled {attribute_name} failed: {exc}"
            set_compile_attrs(module, label, False, attempts, detail, verified=False)
            log_compile_status(
                label,
                status="fallback_eager",
                detail=detail,
                summary=summary,
                warning=True,
            )
            return original_callable(*args, **kwargs)

    setattr(module, attribute_name, _compiled_callable_with_fallback)
    set_compile_attrs(module, label, True, attempts, toolchain.detail, verified=False)
    log_compile_status(label, status="setup_ready", detail=toolchain.detail, summary=summary)
    return TorchCompileResult(True, True, toolchain.detail, attempts)


def _mark_unavailable(
    module: torch.nn.Module,
    label: str,
    detail: str,
) -> TorchCompileResult:
    """Mark a module as not compiled and return result metadata."""

    attempts = int(getattr(module, "_acestep_torch_compile_attempts", 0))
    set_compile_attrs(module, label, False, attempts, detail, verified=False)
    return TorchCompileResult(True, False, detail, attempts)
