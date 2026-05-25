"""Custom trigger caption helpers for dataset auto-labeling."""

from __future__ import annotations

from .models import AudioSample


def normalize_custom_trigger(custom_tag: object) -> str:
    """Return a trimmed custom trigger string."""

    return str(custom_tag or "").strip()


def apply_custom_trigger_caption_only(
    sample: AudioSample,
    custom_tag: object,
) -> AudioSample:
    """Force one sample caption to exactly the custom trigger tag."""

    trigger = normalize_custom_trigger(custom_tag)
    if not trigger:
        return sample

    sample.caption = trigger
    sample.custom_tag = ""
    sample.caption_source = "custom_trigger"
    return sample


def apply_custom_trigger_caption_only_to_samples(
    samples: list[AudioSample],
    custom_tag: object,
) -> None:
    """Force every sample caption to exactly the custom trigger tag."""

    trigger = normalize_custom_trigger(custom_tag)
    if not trigger:
        return

    for sample in samples:
        apply_custom_trigger_caption_only(sample, trigger)


def custom_trigger_caption_for_builder(builder: object, enabled: bool) -> str:
    """Return the trigger caption to enforce for a builder label run."""

    if not enabled:
        return ""
    metadata = getattr(builder, "metadata", None)
    return normalize_custom_trigger(getattr(metadata, "custom_tag", ""))


def enable_custom_trigger_caption_only(builder: object, custom_tag: str) -> None:
    """Mark builder metadata so preprocessing uses only the trigger tag."""

    metadata = getattr(builder, "metadata", None)
    if metadata is None:
        return
    metadata.use_only_custom_trigger = True
    metadata.tag_position = "replace"
    metadata.custom_tag = custom_tag
