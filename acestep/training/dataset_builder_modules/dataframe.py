from typing import Any, Dict, List


class DataframeMixin:
    """Display helpers for UI."""

    def get_samples_dataframe_data(self) -> List[List[Any]]:
        """Get samples data in a format suitable for Gradio DataFrame."""
        rows = []
        for i, sample in enumerate(self.samples):
            if sample.has_raw_lyrics():
                lyrics_status = "📝"
            elif sample.has_formatted_lyrics() or (
                sample.lyrics and sample.lyrics.strip() != "[Instrumental]"
            ):
                lyrics_status = "yes"
            elif sample.is_instrumental:
                lyrics_status = "🎵"
            else:
                lyrics_status = "-"

            rows.append(
                [
                    i,
                    sample.filename,
                    f"{sample.duration:.1f}s",
                    lyrics_status,
                    "✅" if sample.labeled else "❌",
                    sample.bpm or "-",
                    sample.keyscale or "-",
                    sample.caption[:50] + "..." if len(sample.caption) > 50 else sample.caption or "-",
                ]
            )
        return rows

    def to_training_format(self) -> List[Dict[str, Any]]:
        """Convert dataset to format suitable for training."""
        training_samples = []

        tag_position = (
            "replace"
            if getattr(self.metadata, "use_only_custom_trigger", False)
            else self.metadata.tag_position
        )

        for sample in self.samples:
            if not sample.labeled:
                continue

            training_sample = {
                "audio_path": sample.audio_path,
                "caption": sample.get_full_caption(tag_position),
                "lyrics": sample.lyrics,
                "bpm": sample.bpm,
                "keyscale": sample.keyscale,
                "timesignature": sample.timesignature,
                "duration": sample.duration,
                "language": sample.language,
                "is_instrumental": sample.is_instrumental,
            }
            training_samples.append(training_sample)

        return training_samples
