from .models import AudioSample


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
