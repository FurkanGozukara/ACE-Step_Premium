from .custom_trigger_caption import (
    apply_custom_trigger_caption_only_to_samples,
    normalize_custom_trigger,
)


class MetadataMixin:
    """Dataset-level metadata helpers."""

    def set_custom_tag(self, custom_tag: str, tag_position: str = "prepend"):
        """Set or clear the custom tag for all samples."""

        normalized_tag = str(custom_tag or "").strip()
        self.metadata.custom_tag = normalized_tag
        self.metadata.tag_position = tag_position if normalized_tag else "prepend"

        for sample in self.samples:
            sample.custom_tag = normalized_tag

    def set_all_instrumental(self, is_instrumental: bool):
        """Set instrumental flag for all samples."""
        self.metadata.all_instrumental = is_instrumental

        for sample in self.samples:
            if sample.has_raw_lyrics():
                sample.is_instrumental = False
                if not sample.lyrics or sample.lyrics == "[Instrumental]":
                    sample.lyrics = sample.raw_lyrics
            else:
                sample.is_instrumental = is_instrumental
                if is_instrumental:
                    sample.lyrics = "[Instrumental]"
                    sample.language = "unknown"

    def set_use_only_custom_trigger(self, enabled: bool):
        """Store whether saved captions should contain only the custom trigger."""

        self.metadata.use_only_custom_trigger = bool(enabled)
        if not enabled:
            return

        custom_tag = normalize_custom_trigger(self.metadata.custom_tag)
        if not custom_tag:
            return

        self.metadata.tag_position = "replace"
        apply_custom_trigger_caption_only_to_samples(self.samples, custom_tag)
