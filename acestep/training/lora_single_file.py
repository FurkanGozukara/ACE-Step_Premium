"""Single-file PEFT LoRA artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_MODEL_BIN_FILENAME = "adapter_model.bin"
ADAPTER_MODEL_SAFETENSORS_FILENAME = "adapter_model.safetensors"
SINGLE_FILE_FORMAT = "ace_step_peft_lora_single_file"
SINGLE_FILE_FORMAT_KEY = "ace_step_format"
SINGLE_FILE_CONFIG_KEY = "ace_step_peft_config_json"


def _read_adapter_config(adapter_dir: str | os.PathLike[str]) -> dict:
    """Read a PEFT adapter config from an adapter directory."""

    config_path = Path(adapter_dir) / ADAPTER_CONFIG_FILENAME
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain a JSON object.")
    return config


def _read_adapter_weights(adapter_dir: str | os.PathLike[str]) -> dict:
    """Read adapter tensors from a PEFT adapter directory."""

    adapter_path = Path(adapter_dir)
    safetensors_path = adapter_path / ADAPTER_MODEL_SAFETENSORS_FILENAME
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")

    bin_path = adapter_path / ADAPTER_MODEL_BIN_FILENAME
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No PEFT adapter weights found in {adapter_path}")


def save_peft_lora_single_file(
    adapter_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> str:
    """Save a PEFT LoRA adapter directory as one loadable safetensors file.

    Args:
        adapter_dir: Directory containing ``adapter_config.json`` and adapter weights.
        output_path: Destination ``.safetensors`` file.

    Returns:
        Path to the written safetensors artifact.
    """

    from safetensors.torch import save_file

    config = _read_adapter_config(adapter_dir)
    state_dict = _read_adapter_weights(adapter_dir)
    metadata = {
        "format": "pt",
        SINGLE_FILE_FORMAT_KEY: SINGLE_FILE_FORMAT,
        SINGLE_FILE_CONFIG_KEY: json.dumps(config, separators=(",", ":")),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(destination), metadata=metadata)
    return str(destination)


def is_peft_lora_single_file(path: str | os.PathLike[str]) -> bool:
    """Return whether a safetensors file contains embedded PEFT LoRA config."""

    if not Path(path).is_file() or Path(path).suffix.lower() != ".safetensors":
        return False

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return False

    return (
        metadata.get(SINGLE_FILE_FORMAT_KEY) == SINGLE_FILE_FORMAT
        and isinstance(metadata.get(SINGLE_FILE_CONFIG_KEY), str)
        and bool(str(metadata.get(SINGLE_FILE_CONFIG_KEY)).strip())
    )


def _read_single_file_config(path: str | os.PathLike[str]) -> dict:
    """Read embedded PEFT config metadata from a single-file LoRA artifact."""

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get(SINGLE_FILE_CONFIG_KEY)
    if not isinstance(raw_config, str) or not raw_config.strip():
        raise ValueError("Missing embedded PEFT adapter config metadata.")

    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("Embedded PEFT adapter config must be a JSON object.")
    return config


@contextmanager
def materialize_peft_lora_single_file(
    path: str | os.PathLike[str],
) -> Iterator[str]:
    """Temporarily expand a single-file LoRA artifact into a PEFT adapter dir.

    Args:
        path: Single-file LoRA ``.safetensors`` artifact.

    Yields:
        Temporary adapter directory path containing PEFT-compatible files.
    """

    from safetensors.torch import load_file, save_file

    config = _read_single_file_config(path)
    state_dict = load_file(str(path), device="cpu")

    with tempfile.TemporaryDirectory(prefix="acestep_lora_") as tmp:
        adapter_dir = Path(tmp)
        (adapter_dir / ADAPTER_CONFIG_FILENAME).write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )
        save_file(
            state_dict,
            str(adapter_dir / ADAPTER_MODEL_SAFETENSORS_FILENAME),
            metadata={"format": "pt"},
        )
        yield str(adapter_dir)
