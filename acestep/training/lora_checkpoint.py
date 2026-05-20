"""
LoRA Checkpoint Utilities for ACE-Step

Provides functions for saving and loading LoRA checkpoints.
"""

import os
import re
import tempfile
from typing import Any, Dict, Optional

from loguru import logger

import torch
from torch.nn import Module

from acestep.training.configs import LoRAConfig
from acestep.training.lora_naming import (
    lora_safetensors_filename,
    lora_training_state_filename,
)
from acestep.training.lora_single_file import (
    is_peft_lora_single_file,
    materialize_peft_lora_single_file,
    save_peft_lora_single_file,
)
from acestep.training.path_safety import safe_path

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _save_named_lora_single_file(
    adapter_path: str,
    output_dir: str,
    artifact_name: str,
) -> str:
    """Save a PEFT adapter directory as a named single-file artifact."""

    single_file_path = os.path.join(
        output_dir,
        lora_safetensors_filename(artifact_name),
    )
    return save_peft_lora_single_file(adapter_path, single_file_path)


def save_lora_weights(
    model: Module,
    output_dir: str,
    save_full_model: bool = False,
    artifact_name: Optional[str] = None,
    save_adapter: bool = True,
) -> str:
    """Save LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_dir: Directory to save weights
        save_full_model: Whether to save the full model state dict
        artifact_name: Optional basename for a combined safetensors artifact
        save_adapter: Whether to also keep the PEFT adapter directory on disk

    Returns:
        Path to saved weights
    """
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "decoder") and hasattr(model.decoder, "save_pretrained"):
        if save_adapter:
            adapter_path = os.path.join(output_dir, "adapter")
            model.decoder.save_pretrained(adapter_path)
            logger.info(f"LoRA adapter saved to {adapter_path}")
            if artifact_name:
                single_file_path = _save_named_lora_single_file(
                    adapter_path,
                    output_dir,
                    artifact_name,
                )
                logger.info(f"Combined LoRA safetensors saved to {single_file_path}")
            return adapter_path

        if not artifact_name:
            raise ValueError("artifact_name is required when save_adapter is False")
        with tempfile.TemporaryDirectory(prefix="acestep_lora_adapter_") as tmp_dir:
            adapter_path = os.path.join(tmp_dir, "adapter")
            model.decoder.save_pretrained(adapter_path)
            single_file_path = _save_named_lora_single_file(
                adapter_path,
                output_dir,
                artifact_name,
            )
        logger.info(f"LoRA safetensors saved to {single_file_path}")
        return single_file_path
    elif save_full_model:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Full model state dict saved to {model_path}")
        return model_path
    else:
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_state_dict[name] = param.data.clone()

        if not lora_state_dict:
            logger.warning("No LoRA parameters found to save!")
            return ""

        if artifact_name:
            from safetensors.torch import save_file

            lora_path = os.path.join(
                output_dir,
                lora_safetensors_filename(artifact_name),
            )
            save_file(lora_state_dict, lora_path, metadata={"format": "pt"})
        else:
            lora_path = os.path.join(output_dir, "lora_weights.pt")
            torch.save(lora_state_dict, lora_path)
        logger.info(f"LoRA weights saved to {lora_path}")
        return lora_path


def load_lora_weights(
    model: Module,
    lora_path: str,
    _lora_config: Optional[LoRAConfig] = None,
) -> Module:
    """Load LoRA adapter weights into the model.

    Args:
        model: The base model (without LoRA)
        lora_path: Path to saved LoRA adapter directory
        lora_config: Unused; retained for API compatibility

    Returns:
        Model with LoRA weights loaded
    """
    validated = safe_path(lora_path)
    if not os.path.exists(validated):
        raise FileNotFoundError(f"LoRA weights not found: {validated}")

    if os.path.isdir(validated):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required to load adapter. Install with: pip install peft"
            )

        model.decoder = PeftModel.from_pretrained(model.decoder, validated)
        logger.info(f"LoRA adapter loaded from {validated}")

    elif is_peft_lora_single_file(validated):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required to load adapter. Install with: pip install peft"
            )

        with materialize_peft_lora_single_file(validated) as adapter_dir:
            model.decoder = PeftModel.from_pretrained(model.decoder, adapter_dir)
        logger.info(f"LoRA single-file adapter loaded from {validated}")

    elif validated.endswith(".pt"):
        raise ValueError(
            "Loading LoRA weights from .pt files is disabled for security. "
            "Use a PEFT adapter directory instead."
        )

    else:
        raise ValueError(f"Unsupported LoRA weight format: {validated}")

    return model


def save_training_checkpoint(
    model: Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    artifact_name: Optional[str] = None,
    state_suffix: str = "",
) -> str:
    """Save a flat LoRA checkpoint including weights and resume state.

    Args:
        model: Model with LoRA adapters
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        global_step: Current global step
        output_dir: Directory to save checkpoint files
        artifact_name: Optional basename for a combined safetensors artifact
        state_suffix: Optional suffix for the resume-state filename

    Returns:
        Path to saved training resume state file
    """
    output_dir = safe_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not artifact_name:
        artifact_name = f"lora-epoch-{int(epoch)}"
    lora_weights_path = save_lora_weights(
        model,
        output_dir,
        artifact_name=artifact_name,
        save_adapter=False,
    )

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict()
        if hasattr(optimizer, "state_dict")
        else {},
        "scheduler_state_dict": scheduler.state_dict()
        if hasattr(scheduler, "state_dict")
        else {},
        "lora_weights_path": os.path.basename(lora_weights_path),
        "artifact_name": artifact_name,
    }

    state_path = os.path.join(
        output_dir,
        lora_training_state_filename(epoch, suffix=state_suffix),
    )
    torch.save(training_state, state_path)

    logger.info(
        f"Training checkpoint saved to {state_path} (epoch {epoch}, step {global_step})"
    )
    return state_path


def load_training_checkpoint(
    checkpoint_dir: str,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_dir: Resume-state file, single-file LoRA, or legacy checkpoint dir
        optimizer: Optimizer instance to load state into (optional).
            When provided, loads optimizer_state_dict from the checkpoint.
        scheduler: Scheduler instance to load state into (optional).
            When provided, loads scheduler_state_dict from the checkpoint.
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint info:
        - epoch: Saved epoch number
        - global_step: Saved global step
        - adapter_path: Path to adapter weights
        - loaded_optimizer: Whether optimizer state was loaded
        - loaded_scheduler: Whether scheduler state was loaded
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "adapter_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
    }

    try:
        checkpoint_path = safe_path(checkpoint_dir)
    except ValueError:
        logger.warning(f"Rejected unsafe checkpoint path: {checkpoint_dir!r}")
        return result

    checkpoint_root = (
        os.path.dirname(checkpoint_path)
        if os.path.isfile(checkpoint_path)
        else checkpoint_path
    )
    state_path = _resolve_training_state_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors") and os.path.isfile(checkpoint_path):
        result["adapter_path"] = checkpoint_path

    if os.path.isfile(state_path):
        try:
            training_state = torch.load(
                state_path, map_location=device, weights_only=True
            )

            if "epoch" in training_state:
                try:
                    result["epoch"] = int(training_state["epoch"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse 'epoch' from resume state: {e}, using default 0"
                    )
            if "global_step" in training_state:
                try:
                    result["global_step"] = int(training_state["global_step"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse 'global_step' from resume state: {e}, using default 0"
                    )
            adapter_path = _resolve_state_lora_weights_path(
                checkpoint_root,
                state_path,
                training_state,
                result["epoch"],
            )
            if adapter_path:
                result["adapter_path"] = adapter_path

            if optimizer is not None and "optimizer_state_dict" in training_state:
                try:
                    optimizer_state = training_state["optimizer_state_dict"]
                    if device is not None:
                        for state in optimizer_state.get("state", {}).values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                    optimizer.load_state_dict(optimizer_state)
                    result["loaded_optimizer"] = True
                    logger.info("Loaded optimizer state from checkpoint")
                except (RuntimeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to load optimizer state: {e}")

            if scheduler is not None and "scheduler_state_dict" in training_state:
                try:
                    scheduler.load_state_dict(training_state["scheduler_state_dict"])
                    result["loaded_scheduler"] = True
                    logger.info("Loaded scheduler state from checkpoint")
                except (RuntimeError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to load scheduler state: {e}")

            logger.info(
                "Loaded checkpoint metadata from epoch "
                f"{result['epoch']}, step {result['global_step']}"
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to load resume state: {e}")
    else:
        adapter_path = _resolve_legacy_adapter_path(checkpoint_path)
        if adapter_path:
            result["adapter_path"] = adapter_path
        match = re.search(r"(?:epoch_|epoch-)(\d+)", checkpoint_path)
        if match:
            result["epoch"] = int(match.group(1))
            logger.info(
                f"No resume state found, extracted epoch {result['epoch']} from path"
            )

    return result


def _resolve_training_state_path(checkpoint_path: str) -> str:
    """Return a resume-state path from a file or legacy checkpoint directory."""

    if os.path.isfile(checkpoint_path):
        return checkpoint_path if checkpoint_path.endswith(".pt") else ""
    legacy_state = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.isfile(legacy_state):
        return legacy_state
    candidates = (
        [
            os.path.join(checkpoint_path, name)
            for name in os.listdir(checkpoint_path)
            if re.match(r"^epoch-\d+-training_resume_state(?:-[\w-]+)?\.pt$", name)
        ]
        if os.path.isdir(checkpoint_path)
        else []
    )
    if not candidates:
        return ""
    candidates.sort(key=lambda path: _resume_state_sort_key(os.path.basename(path)))
    return candidates[-1]


def _resume_state_sort_key(filename: str) -> tuple[int, int, str]:
    """Return sort key for resume-state files, preferring final files."""

    match = re.match(
        r"^epoch-(\d+)-training_resume_state(?:-([\w-]+))?\.pt$",
        filename,
    )
    epoch = int(match.group(1)) if match else 0
    suffix = match.group(2) if match else ""
    final_rank = 1 if suffix == "final" else 0
    return epoch, final_rank, filename


def _resolve_state_lora_weights_path(
    checkpoint_root: str,
    state_path: str,
    training_state: Dict[str, Any],
    epoch: int,
) -> str | None:
    """Return the safetensors path referenced by a flat resume state."""

    raw_path = str(training_state.get("lora_weights_path") or "").strip()
    for candidate in _state_lora_weight_candidates(
        checkpoint_root,
        state_path,
        training_state,
        epoch,
        raw_path,
    ):
        try:
            safe_candidate = (
                safe_path(candidate)
                if os.path.isabs(candidate)
                else safe_path(candidate, base=checkpoint_root)
            )
        except ValueError:
            continue
        if os.path.isfile(safe_candidate):
            return safe_candidate
    return _resolve_legacy_adapter_path(checkpoint_root)


def _state_lora_weight_candidates(
    checkpoint_root: str,
    state_path: str,
    training_state: Dict[str, Any],
    epoch: int,
    raw_path: str,
) -> list[str]:
    """Return likely LoRA weight files for a resume state."""

    candidates: list[str] = []
    if raw_path:
        candidates.append(raw_path)
    artifact_name = str(training_state.get("artifact_name") or "").strip()
    if artifact_name:
        candidates.append(lora_safetensors_filename(artifact_name))
    lora_name = str(training_state.get("lora_name") or "").strip()
    if lora_name and epoch > 0:
        final_suffix = (
            "-final" if os.path.basename(state_path).endswith("-final.pt") else ""
        )
        candidates.append(f"{lora_name}-epoch-{epoch}{final_suffix}.safetensors")
    if os.path.isdir(checkpoint_root):
        candidates.extend(
            name
            for name in os.listdir(checkpoint_root)
            if name.endswith(".safetensors")
        )
    return candidates


def _resolve_legacy_adapter_path(checkpoint_path: str) -> str | None:
    """Return an adapter path from old directory checkpoints when present."""

    if os.path.isfile(checkpoint_path):
        return checkpoint_path if checkpoint_path.endswith(".safetensors") else None
    adapter_path = os.path.join(checkpoint_path, "adapter")
    if os.path.isdir(adapter_path):
        return adapter_path
    if os.path.isdir(checkpoint_path):
        for name in sorted(os.listdir(checkpoint_path)):
            candidate = os.path.join(checkpoint_path, name)
            if os.path.isfile(candidate) and candidate.endswith(".safetensors"):
                return candidate
        return checkpoint_path
    return None
