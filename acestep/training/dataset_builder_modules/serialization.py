"""Serialize full datasets and processed auto-label JSON files."""

import json
import os
from datetime import datetime
from typing import List, Tuple

from loguru import logger

from acestep.training.path_safety import safe_path

from .models import AudioSample, DatasetMetadata


class SerializationMixin:
    """Save/load dataset JSON."""

    def save_dataset(self, output_path: str, dataset_name: str = None) -> str:
        """Save the dataset to a JSON file."""
        if not self.samples:
            return "❌ No samples to save"

        if dataset_name:
            self.metadata.name = dataset_name

        self.metadata.num_samples = len(self.samples)
        self.metadata.created_at = datetime.now().isoformat()

        dataset = {
            "metadata": self.metadata.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }

        try:
            validated_output = safe_path(output_path)
            parent = os.path.dirname(validated_output)
            os.makedirs(parent if parent else ".", exist_ok=True)

            with open(validated_output, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            return (
                f"✅ Dataset saved to {validated_output}\n"
                f"{len(self.samples)} samples, tag: '{self.metadata.custom_tag}'"
            )
        except Exception as e:
            logger.exception("Error saving dataset")
            return f"❌ Failed to save dataset: {str(e)}"

    def load_dataset(self, dataset_path: str) -> Tuple[List[AudioSample], str]:
        """Load a dataset from a JSON file."""
        try:
            validated_path = safe_path(dataset_path)
        except ValueError:
            return [], f"❌ Rejected unsafe dataset path: {dataset_path}"

        if not os.path.exists(validated_path):
            return [], f"❌ Dataset not found: {dataset_path}"

        try:
            if os.path.isdir(validated_path):
                return self._load_processed_label_directory(validated_path)

            with open(validated_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "samples" not in data and _is_processed_label(data):
                return self._load_processed_label_file(validated_path, data)

            if "metadata" in data:
                meta_dict = data["metadata"]
                self.metadata = DatasetMetadata(
                    name=meta_dict.get("name", "untitled"),
                    custom_tag=meta_dict.get("custom_tag", ""),
                    tag_position=meta_dict.get("tag_position", "prepend"),
                    created_at=meta_dict.get("created_at", ""),
                    num_samples=meta_dict.get("num_samples", 0),
                    all_instrumental=meta_dict.get("all_instrumental", True),
                    genre_ratio=meta_dict.get("genre_ratio", 0),
                    use_only_custom_trigger=meta_dict.get("use_only_custom_trigger", False),
                )

            self.samples = []
            for sample_dict in data.get("samples", []):
                sample = AudioSample.from_dict(sample_dict)
                if self.metadata.custom_tag and not sample.custom_tag:
                    sample.custom_tag = self.metadata.custom_tag
                self.samples.append(sample)

            if self.metadata.use_only_custom_trigger:
                self.set_use_only_custom_trigger(True)

            return self.samples, f"✅ Loaded {len(self.samples)} samples from {dataset_path}"
        except Exception as e:
            logger.exception("Error loading dataset")
            return [], f"❌ Failed to load dataset: {str(e)}"

    def _load_processed_label_file(
        self,
        label_path: str,
        data: dict,
    ) -> Tuple[List[AudioSample], str]:
        """Load one processed-label JSON file as a single-sample dataset."""

        sample = _sample_from_processed_label(data)
        if sample is None:
            return [], f"❌ Processed label has no usable audio_path: {label_path}"

        self.samples = [sample]
        self.metadata = _metadata_for_processed_labels(
            os.path.dirname(label_path),
            self.samples,
        )
        return self.samples, f"✅ Loaded 1 processed label from {label_path}"

    def _load_processed_label_directory(
        self,
        label_dir: str,
    ) -> Tuple[List[AudioSample], str]:
        """Load all processed-label JSON files in a folder as a dataset."""

        samples: list[AudioSample] = []
        skipped = 0
        for filename in sorted(os.listdir(label_dir)):
            if not filename.lower().endswith(".json"):
                continue
            label_path = os.path.join(label_dir, filename)
            try:
                with open(label_path, "r", encoding="utf-8") as file_obj:
                    data = json.load(file_obj)
            except Exception as exc:
                logger.warning(f"Failed to read processed label {label_path}: {exc}")
                skipped += 1
                continue
            sample = _sample_from_processed_label(data)
            if sample is None:
                skipped += 1
                continue
            samples.append(sample)

        if not samples:
            return [], f"❌ No processed labels with usable audio_path found in {label_dir}"

        self.samples = samples
        self.metadata = _metadata_for_processed_labels(label_dir, self.samples)
        status = f"✅ Loaded {len(samples)} processed labels from {label_dir}"
        if skipped:
            status += f" ({skipped} skipped)"
        return self.samples, status


def _is_processed_label(data: object) -> bool:
    """Return whether a JSON object looks like one processed label."""

    return isinstance(data, dict) and bool(data.get("audio_path"))


def _sample_from_processed_label(data: object) -> AudioSample | None:
    """Build an ``AudioSample`` from processed-label JSON data."""

    if not _is_processed_label(data):
        return None

    sample = AudioSample.from_dict(data)
    try:
        sample.audio_path = safe_path(sample.audio_path)
    except ValueError as exc:
        logger.warning(f"Rejected processed-label audio path {sample.audio_path!r}: {exc}")
        return None
    if not sample.filename:
        sample.filename = os.path.basename(sample.audio_path)
    if not sample.labeled and sample.caption:
        sample.labeled = True
    return sample


def _metadata_for_processed_labels(
    label_dir: str,
    samples: list[AudioSample],
) -> DatasetMetadata:
    """Return dataset metadata for a processed-label load."""

    return DatasetMetadata(
        name=os.path.basename(os.path.normpath(label_dir)) or "processed_labels",
        num_samples=len(samples),
        all_instrumental=all(sample.is_instrumental for sample in samples),
    )
