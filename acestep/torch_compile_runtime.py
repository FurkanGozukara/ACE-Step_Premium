"""Runtime wrapper facade for optional ``torch.compile`` use."""

from __future__ import annotations

import torch

from acestep.torch_compile_callable import (
    TorchCompileResult,
    compile_module_callable,
)


def compile_module_forward(
    module: torch.nn.Module,
    *,
    label: str,
    enabled: bool = True,
    backend: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    disabled_detail: str | None = None,
) -> TorchCompileResult:
    """Compile a module's forward method with first-call fallback.

    Args:
        module: Module whose bound ``forward`` method should be compiled.
        label: Human-readable label for logging and metadata.
        enabled: User-controlled switch for the compile request.
        backend: PyTorch compiler backend.
        mode: Optional PyTorch compile mode.
        fullgraph: Whether to require a single graph.
        disabled_detail: Optional reason used when ``enabled`` is false.

    Returns:
        Compile result metadata. A successful result means a compiled wrapper was
        installed; the wrapper still restores eager forward if first execution fails.
    """
    return compile_module_callable(
        module,
        attribute_name="forward",
        label=label,
        enabled=enabled,
        backend=backend,
        mode=mode,
        fullgraph=fullgraph,
        disabled_detail=disabled_detail,
    )


def compile_counters_snapshot() -> dict[str, int]:
    """Return selected TorchDynamo/Inductor counters for reporting."""

    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    flattened: dict[str, int] = {}
    for group, values in counters.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, int):
                flattened[f"{group}.{key}"] = value
    return flattened
