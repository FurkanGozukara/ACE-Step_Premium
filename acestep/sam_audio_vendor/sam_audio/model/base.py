# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import os
from contextlib import contextmanager, nullcontext
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download
from safetensors.torch import load_file


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str | None = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = True,
        checkpoint_path: Optional[str] = None,
        revision: Optional[str] = None,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=cls.revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        config = cls.config_cls(**config)
        resolved_checkpoint_path = _resolve_checkpoint_path(cached_model_dir, checkpoint_path)
        meta_direct_load = _should_use_meta_direct_load(
            resolved_checkpoint_path,
            map_location,
        )
        with _fast_meta_construction(meta_direct_load):
            model = cls(config)
        state_dict = _load_checkpoint(
            resolved_checkpoint_path,
            map_location=map_location,
        )
        assign = _should_assign_loaded_tensors(resolved_checkpoint_path, map_location)
        try:
            model.load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            if not assign:
                raise
            del state_dict
            state_dict = _load_checkpoint(resolved_checkpoint_path, map_location="cpu")
            if meta_direct_load:
                model = cls(config)
            model.load_state_dict(state_dict, strict=strict)
        else:
            if meta_direct_load:
                _materialize_meta_buffers(model, map_location)
        return model


def _resolve_checkpoint_path(cached_model_dir: str, checkpoint_path: Optional[str]) -> str:
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

    for filename in ("checkpoint.pt", "checkpoint.safetensors", "model.safetensors"):
        path = os.path.join(cached_model_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {cached_model_dir}")


def _load_checkpoint(path: str, map_location: str) -> dict:
    if path.lower().endswith(".safetensors"):
        return load_file(path, device=str(map_location))
    state_dict = torch.load(path, weights_only=True, map_location=map_location)
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        return state_dict["state_dict"]
    return state_dict


def _should_assign_loaded_tensors(path: str, map_location: str) -> bool:
    """Return whether loaded checkpoint tensors should replace module tensors."""

    return path.lower().endswith(".safetensors") and str(map_location).startswith("cuda")


def _should_use_meta_direct_load(path: str, map_location: str) -> bool:
    """Return whether model construction can be skipped before CUDA assignment."""

    return _should_assign_loaded_tensors(path, map_location) and _supports_assign_load()


def _supports_assign_load() -> bool:
    """Return whether the current PyTorch build supports assign loads."""

    return "assign" in torch.nn.Module.load_state_dict.__code__.co_varnames


@contextmanager
def _fast_meta_construction(enabled: bool):
    """Construct checkpoint-backed modules on meta when CUDA tensors will be assigned."""

    if not enabled:
        yield
        return
    with _transformers_no_init_weights(), torch.device("meta"):
        yield


def _transformers_no_init_weights():
    """Return the transformers no-init context when the installed version provides it."""

    try:
        from transformers.initialization import no_init_weights
    except Exception:
        return nullcontext()
    return no_init_weights()


def _materialize_meta_buffers(model: torch.nn.Module, map_location: str) -> None:
    """Recreate non-checkpoint buffers that remain on meta after assign loading."""

    target_device = torch.device(map_location)
    for module in model.modules():
        if not _has_meta_buffers(module):
            continue
        if _materialize_rotary_buffers(module, target_device):
            continue
        if _materialize_modernbert_rotary_buffers(module, target_device):
            continue
    remaining = [
        name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)
    ]
    if remaining:
        joined = ", ".join(remaining[:8])
        suffix = "" if len(remaining) <= 8 else f", ... ({len(remaining)} total)"
        raise RuntimeError(f"Direct CUDA load left meta buffers unresolved: {joined}{suffix}")


def _has_meta_buffers(module: torch.nn.Module) -> bool:
    """Return whether a module owns any local meta buffers."""

    return any(
        buffer is not None and getattr(buffer, "is_meta", False)
        for buffer in module._buffers.values()
    )


def _materialize_rotary_buffers(module: torch.nn.Module, device: torch.device) -> bool:
    """Recreate core/SAM rotary buffers on the target device."""

    if "freqs_cis" not in module._buffers or not hasattr(module, "reset_parameters"):
        return False
    with torch.device(device):
        module.reset_parameters()
    return not _has_meta_buffers(module)


def _materialize_modernbert_rotary_buffers(
    module: torch.nn.Module,
    device: torch.device,
) -> bool:
    """Recreate ModernBERT RoPE buffers on the target device."""

    if module.__class__.__name__ != "ModernBertRotaryEmbedding":
        return False
    if not hasattr(module, "layer_types") or not hasattr(module, "compute_default_rope_parameters"):
        return False
    from transformers.models.modernbert.modeling_modernbert import ROPE_INIT_FUNCTIONS

    for layer_type in module.layer_types:
        rope_type = module.rope_type[layer_type]
        rope_init_fn = module.compute_default_rope_parameters
        if rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(
            module.config,
            device=device,
            layer_type=layer_type,
        )
        module.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        module.register_buffer(
            f"{layer_type}_original_inv_freq",
            inv_freq.clone(),
            persistent=False,
        )
        setattr(module, f"{layer_type}_attention_scaling", attention_scaling)
    return not _has_meta_buffers(module)
