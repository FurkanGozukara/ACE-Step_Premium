"""Unit tests for preprocess.py."""

import os
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from acestep.training.path_safety import get_safe_root, set_safe_root
from acestep.ui.gradio.events.training.preprocess import (
    load_existing_dataset_for_preprocess,
    load_training_dataset,
    preprocess_dataset,
)


class _FakeDitHandler:
    """Test double for preprocess auto-initialization."""

    def __init__(self):
        self.model = None
        self.initialize_calls = []
        self.last_init_params = None

    def initialize_service(self, **kwargs):
        """Record init arguments and mark the model as ready."""

        self.initialize_calls.append(kwargs)
        self.model = object()
        self.last_init_params = dict(kwargs)
        return "DiT ready", True


def _gpu_defaults():
    """Return GPU defaults used by preprocess auto-init tests."""

    return SimpleNamespace(
        compile_model_default=False,
        offload_to_cpu_default=True,
        offload_dit_to_cpu_default=True,
        quantization_default=False,
    )


class TestLoadTrainingDataset(unittest.TestCase):
    """Tests for load_training_dataset."""

    def setUp(self):
        """Preserve the process-wide training path-safety root."""

        self.original_safe_root = get_safe_root()

    def tearDown(self):
        """Restore the process-wide training path-safety root."""

        set_safe_root(self.original_safe_root)

    def test_empty_path(self):
        result = load_training_dataset("")
        self.assertIn("❌", result)

    def test_nonexistent_path(self):
        result = load_training_dataset("/nonexistent/path/xyz")
        self.assertIn("❌", result)

    def test_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_root(tmpdir)
            manifest = {
                "num_samples": 10,
                "metadata": {"name": "TestDataset", "custom_tag": "test"},
            }
            with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            # Create a dummy .pt file
            open(os.path.join(tmpdir, "sample_0.pt"), "w").close()

            result = load_training_dataset(tmpdir)
            self.assertIn("TestDataset", result)
            self.assertIn("10", result)

    def test_with_manifest_quoted_path_with_spaces(self):
        """Tensor directories pasted with quotes and spaces should load."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = os.path.join(tmpdir, "tensor data")
            os.makedirs(tensor_dir)
            set_safe_root(tmpdir)
            manifest = {
                "num_samples": 2,
                "metadata": {"name": "SpaceDataset", "custom_tag": "test"},
            }
            with open(os.path.join(tensor_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            open(os.path.join(tensor_dir, "sample_0.pt"), "w").close()

            pasted_path = f'"{tensor_dir.replace(os.sep, "/")}"'
            result = load_training_dataset(pasted_path)

        self.assertIn("SpaceDataset", result)
        self.assertIn("2", result)

    def test_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_root(tmpdir)
            # Create some .pt files
            for i in range(3):
                open(os.path.join(tmpdir, f"sample_{i}.pt"), "w").close()

            result = load_training_dataset(tmpdir)
            self.assertIn("3", result)
            self.assertIn("tensor files", result)

    def test_no_pt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_root(tmpdir)
            result = load_training_dataset(tmpdir)
            self.assertIn("❌", result)

    def test_load_existing_dataset_accepts_processed_label_folder(self):
        """Preprocess loader should use audio_path from processed-label JSON files."""

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = os.path.join(tmpdir, "audio")
            label_dir = os.path.join(tmpdir, "auto_label")
            os.makedirs(audio_dir)
            os.makedirs(label_dir)
            audio_path = os.path.join(audio_dir, "song.flac")
            with open(audio_path, "wb") as file_obj:
                file_obj.write(b"audio")
            label_path = os.path.join(label_dir, "song.json")
            with open(label_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "audio_path": audio_path,
                        "filename": "song.flac",
                        "caption": "loaded caption",
                        "lyrics": "[Verse]\nwords",
                        "raw_lyrics": "raw lyric from txt",
                        "language": "en",
                        "is_instrumental": False,
                        "labeled": True,
                    },
                    file_obj,
                )

            set_safe_root(tmpdir)
            result = load_existing_dataset_for_preprocess(label_dir, None)

        self.assertIn("Ready for preprocessing", result[0])
        self.assertEqual(audio_path, result[4])
        self.assertEqual("song.flac", result[5])
        self.assertEqual("loaded caption", result[6])
        self.assertEqual("raw lyric from txt", result[16]["value"])
        self.assertTrue(result[16]["visible"])
        self.assertTrue(result[17])


class TestPreprocessDataset(unittest.TestCase):
    """Tests for preprocess_dataset."""

    def test_none_builder(self):
        result = preprocess_dataset("/out", "lora", MagicMock(), None)
        self.assertIn("❌", result)

    def test_empty_samples(self):
        builder = MagicMock()
        builder.samples = []
        result = preprocess_dataset("/out", "lora", MagicMock(), builder)
        self.assertIn("❌", result)

    def test_no_labeled_samples(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_labeled_count.return_value = 0
        result = preprocess_dataset("/out", "lora", MagicMock(), builder)
        self.assertIn("❌", result)

    def test_empty_output_dir(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_labeled_count.return_value = 5
        result = preprocess_dataset("", "lora", MagicMock(), builder)
        self.assertIn("❌", result)

    def test_no_model(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_labeled_count.return_value = 5
        result = preprocess_dataset("/out", "lora", None, builder)
        self.assertIn("❌", result)

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    def test_auto_initializes_selected_model(self, _gpu_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_safe_root = get_safe_root()
            set_safe_root(tmpdir)
            builder = MagicMock()
            builder.samples = [MagicMock()]
            builder.get_labeled_count.return_value = 5
            builder.preprocess_to_tensors.return_value = (["sample.pt"], "Preprocessed")
            dit_handler = _FakeDitHandler()

            try:
                result = preprocess_dataset(
                    os.path.join(tmpdir, "out"),
                    "lora",
                    dit_handler,
                    builder,
                    model_config="model-b",
                    save_debug_text=True,
                )
            finally:
                set_safe_root(original_safe_root)

        self.assertIn("DiT service initialized automatically.", result)
        self.assertIn("Preprocessed", result)
        self.assertEqual(dit_handler.initialize_calls[0]["config_path"], "model-b")
        self.assertIn("cancel_callback", builder.preprocess_to_tensors.call_args.kwargs)
        self.assertTrue(builder.preprocess_to_tensors.call_args.kwargs["save_debug_text"])


if __name__ == "__main__":
    unittest.main()
