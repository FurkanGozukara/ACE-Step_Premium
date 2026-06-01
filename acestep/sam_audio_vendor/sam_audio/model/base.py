# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import os
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
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
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
        model = cls(config)
        state_dict = _load_checkpoint(
            _resolve_checkpoint_path(cached_model_dir, checkpoint_path),
            map_location=map_location,
        )
        model.load_state_dict(state_dict, strict=strict)
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
