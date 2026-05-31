"""Text/span-only low-VRAM optimization for SAM-Audio."""

from __future__ import annotations

import gc
import types

import torch


def apply_text_lite_mode(model: torch.nn.Module) -> None:
    """Remove visual/ranker/span components that are unused by text-only inference."""

    vision_dim = getattr(getattr(model, "vision_encoder", None), "dim", 1024)
    if hasattr(model, "vision_encoder"):
        delattr(model, "vision_encoder")
    model._vision_encoder_dim = int(vision_dim)
    model._get_video_features = types.MethodType(_get_video_features_lite, model)
    _clear_optional_module(model, "visual_ranker")
    _clear_optional_module(model, "text_ranker")
    _clear_optional_module(model, "span_predictor")
    _clear_optional_module(model, "span_predictor_transform")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_text_lite_settings(settings) -> None:
    """Raise when low-VRAM lite mode conflicts with requested SAM features."""

    if not settings.low_vram_lite:
        return
    if settings.prompt_mode == "visual":
        raise ValueError("SAM Lite mode disables visual prompting. Turn Lite mode off.")
    if settings.predict_spans:
        raise ValueError("SAM Lite mode disables span prediction. Turn Predict spans off.")
    if settings.ranker_mode != "none":
        raise ValueError("SAM Lite mode disables rerankers. Set Ranker to Disabled.")


def _get_video_features_lite(self, video, audio_features):
    """Return zero visual features without keeping the vision encoder in memory."""

    if video is not None:
        raise ValueError("SAM Lite mode does not support visual mask prompting.")
    batch, frames, _ = audio_features.shape
    return audio_features.new_zeros(batch, self._vision_encoder_dim, frames)


def _clear_optional_module(model: torch.nn.Module, name: str) -> None:
    """Delete an optional module and leave a ``None`` placeholder."""

    if hasattr(model, name):
        try:
            delattr(model, name)
        except AttributeError:
            pass
    setattr(model, name, None)
