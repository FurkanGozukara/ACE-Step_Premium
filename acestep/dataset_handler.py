"""Dataset import and exploration helpers for the ACE-Step Gradio UI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Tuple

from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.training.path_inputs import normalize_user_path


_EMPTY_PREVIEW = ("", "{}", None, None, None)


class DatasetHandler:
    """Dataset Handler for Dataset Explorer functionality."""

    def __init__(self):
        """Initialize dataset handler with empty state."""

        self.dataset = None
        self.builder: DatasetBuilder | None = None
        self.dataset_imported = False

    def import_dataset(self, dataset_type: str, dataset_path: str = "") -> str:
        """Import a saved dataset JSON or scan an audio folder.

        Args:
            dataset_type: Type of dataset to import, such as ``"train"`` or ``"test"``.
            dataset_path: Saved dataset JSON path or audio folder to scan.

        Returns:
            Status message describing the imported dataset.
        """

        raw_path = normalize_user_path(dataset_path)
        if not raw_path:
            self.dataset_imported = False
            return "Select a saved dataset JSON file or an audio folder to import."

        builder = DatasetBuilder()
        if Path(raw_path).suffix.lower() == ".json":
            samples, status = builder.load_dataset(raw_path)
        else:
            builder.metadata.name = f"{dataset_type or 'train'}_dataset"
            samples, status = builder.scan_directory(raw_path)

        if not samples:
            self.dataset = None
            self.builder = None
            self.dataset_imported = False
            return status

        self.dataset = samples
        self.builder = builder
        self.dataset_imported = True
        labeled_count = builder.get_labeled_count()
        return (
            f"{status}\n"
            f"Imported as {dataset_type or 'train'} dataset.\n"
            f"Samples: {len(samples)} ({labeled_count} labeled)."
        )

    def import_dataset_for_ui(self, dataset_type: str, dataset_path: str = "") -> tuple[Any, ...]:
        """Import a dataset and return first-item preview values for the Dataset page."""

        status = self.import_dataset(dataset_type, dataset_path)
        if not self.dataset_imported:
            return (status, *_EMPTY_PREVIEW)
        return (status, *self._preview_sample(self.dataset[0], 0))

    def get_item_for_ui(self, search_type: str, search_value: str = "") -> tuple[Any, ...]:
        """Return a selected dataset item preview for the Dataset page."""

        if not self.dataset:
            return ("No dataset imported.", *_EMPTY_PREVIEW)

        index = self._resolve_sample_index(search_type, search_value)
        if index is None:
            return (f"No item found for {search_type}: {search_value}", *_EMPTY_PREVIEW)

        sample = self.dataset[index]
        status = f"Loaded item {index + 1}/{len(self.dataset)}: {sample.filename}"
        return (status, *self._preview_sample(sample, index))

    def get_item_data(self, *args, **kwargs) -> Tuple:
        """Return placeholder dataset item data for the explorer UI."""

        return (
            "",           # caption: empty string
            "",           # lyrics: empty string
            "",           # language: empty string
            "",           # bpm: empty string
            "",           # keyscale: empty string
            None,         # ref_audio: no audio file
            None,         # src_audio: no audio file
            None,         # codes: no audio codes
            "Dataset item browsing is not wired yet.",
            "",           # instruction: empty string
            0,            # duration: zero
            "",           # timesig: empty string
            None,         # audio1: no audio
            None,         # audio2: no audio
            None,         # audio3: no audio
            {},           # metadata: empty dict
            "text2music",  # task_type: default task
        )

    def _resolve_sample_index(self, search_type: str, search_value: str) -> int | None:
        """Resolve a Dataset page search request to a sample index."""

        mode = str(search_type or "random").strip().lower()
        value = str(search_value or "").strip()
        if mode == "random":
            return random.randrange(len(self.dataset))
        if mode == "idx":
            try:
                index = int(value)
            except ValueError:
                return None
            return index if 0 <= index < len(self.dataset) else None
        if mode == "keys":
            return self._find_sample_by_key(value)
        return None

    def _find_sample_by_key(self, value: str) -> int | None:
        """Return the index of a sample matching id, filename, or audio path."""

        if not value:
            return None
        needle = value.lower()
        for index, sample in enumerate(self.dataset):
            fields = (sample.id, sample.filename, sample.audio_path)
            if any(needle == str(field or "").lower() for field in fields):
                return index
        return None

    def _preview_sample(self, sample: AudioSample, index: int) -> tuple[str, str, str, None, None]:
        """Return instruction, JSON metadata, and audio preview values for a sample."""

        payload = sample.to_dict()
        payload["index"] = index
        instruction = sample.caption or sample.genre or sample.filename
        return (
            instruction,
            json.dumps(payload, indent=2, ensure_ascii=False),
            sample.audio_path,
            None,
            None,
        )
