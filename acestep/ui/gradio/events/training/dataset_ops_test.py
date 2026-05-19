"""Unit tests for dataset_ops.py."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from acestep.training.dataset_builder import AudioSample, DatasetBuilder
from acestep.training.path_safety import get_safe_roots, set_safe_roots
from acestep.ui.gradio.events.training.dataset_ops import (
    auto_label_all,
    get_sample_preview,
    scan_directory,
    save_sample_edit,
    select_sample_from_table,
    update_settings,
    save_dataset,
)


class _FakeDitHandler:
    """Test double for dataset auto-initialization paths."""

    def __init__(self, *, model=None, init_ok=True):
        self.model = model
        self.init_ok = init_ok
        self.initialize_calls = []
        self.last_init_params = None

    def initialize_service(self, **kwargs):
        """Record service-init calls and optionally mark the model ready."""

        self.initialize_calls.append(kwargs)
        if self.init_ok:
            self.model = object()
            self.last_init_params = dict(kwargs)
        return "DiT ready", self.init_ok


def _gpu_defaults():
    """Return GPU defaults used by dataset auto-init tests."""

    return SimpleNamespace(
        compile_model_default=False,
        offload_to_cpu_default=True,
        offload_dit_to_cpu_default=True,
        quantization_default=False,
    )


def _builder_with_samples():
    """Return a builder mock with one sample and table data."""

    builder = MagicMock()
    builder.samples = [object()]
    builder.get_samples_dataframe_data.return_value = [["audio.wav"]]
    builder.label_all_samples.return_value = (builder.samples, "Labeled")
    return builder


class TestAutoLabelAll(unittest.TestCase):
    """Tests for dataset auto-label service readiness."""

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.ensure_llm_ready",
        return_value=(True, "LM ready"),
    )
    def test_auto_initializes_missing_services(self, ensure_lm_ready, _gpu_config):
        """Auto-label should initialize DiT and LM before labeling samples."""

        builder = _builder_with_samples()
        dit_handler = _FakeDitHandler()
        llm_handler = SimpleNamespace(llm_initialized=False, last_init_params=None)

        table_update, status_update, returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            label_output_dir="labels",
        )

        self.assertIs(returned_builder, builder)
        self.assertEqual(table_update["value"], [["audio.wav"]])
        self.assertIn("DiT service initialized automatically.", status_update["value"])
        self.assertIn("LM ready", status_update["value"])
        self.assertIn("Labeled", status_update["value"])
        self.assertEqual(len(dit_handler.initialize_calls), 1)
        self.assertEqual(dit_handler.initialize_calls[0]["device"], "auto")
        ensure_lm_ready.assert_called_once()
        builder.label_all_samples.assert_called_once()
        self.assertEqual(
            "unknown",
            builder.label_all_samples.call_args.kwargs["lm_lyrics_language"],
        )

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    @patch("acestep.ui.gradio.events.training.service_auto_init.ensure_llm_ready")
    def test_auto_init_failure_stops_before_labeling(self, ensure_lm_ready, _gpu_config):
        """Auto-label should not run when DiT auto-initialization fails."""

        builder = _builder_with_samples()
        dit_handler = _FakeDitHandler(init_ok=False)
        llm_handler = SimpleNamespace(llm_initialized=False, last_init_params=None)

        table_data, status, returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            label_output_dir="labels",
        )

        self.assertIs(returned_builder, builder)
        self.assertEqual(table_data, [["audio.wav"]])
        self.assertIn("DiT ready", status)
        self.assertEqual(len(dit_handler.initialize_calls), 1)
        ensure_lm_ready.assert_not_called()
        builder.label_all_samples.assert_not_called()

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.ensure_llm_ready",
        return_value=(True, ""),
    )
    def test_reuses_initialized_dit_service(self, _ensure_lm_ready, _gpu_config):
        """Auto-label should not reinitialize an already loaded DiT model."""

        builder = _builder_with_samples()
        dit_handler = _FakeDitHandler(model=object())
        dit_handler.last_init_params = {"config_path": "model-a"}
        llm_handler = SimpleNamespace(llm_initialized=True, last_init_params={})

        _table_update, status_update, _returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            model_config="model-a",
            label_output_dir="labels",
        )

        self.assertEqual(dit_handler.initialize_calls, [])
        self.assertEqual(status_update["value"], "Labeled")

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.ensure_llm_ready",
        return_value=(True, ""),
    )
    def test_selected_model_reinitializes_loaded_dit(self, _ensure_lm_ready, _gpu_config):
        """Auto-label should reinitialize DiT when the dataset model changes."""

        builder = _builder_with_samples()
        dit_handler = _FakeDitHandler(model=object())
        dit_handler.last_init_params = {"config_path": "model-a"}
        llm_handler = SimpleNamespace(llm_initialized=True, last_init_params={})

        _table_update, _status_update, _returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            model_config="model-b",
            label_output_dir="labels",
        )

        self.assertEqual(len(dit_handler.initialize_calls), 1)
        self.assertEqual(dit_handler.initialize_calls[0]["config_path"], "model-b")

    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.get_global_gpu_config",
        return_value=_gpu_defaults(),
    )
    @patch(
        "acestep.ui.gradio.events.training.service_auto_init.ensure_llm_ready",
        return_value=(True, ""),
    )
    def test_progress_and_checkpoint_save_are_wired(self, _ensure_lm_ready, _gpu_config):
        """Auto-label should report count progress and save dataset checkpoints."""

        sample = SimpleNamespace(filename="sample.wav", caption="caption", labeled=True)
        builder = _builder_with_samples()
        builder.samples = [sample]
        builder.get_samples_dataframe_data.return_value = [["sample.wav"]]
        builder.save_dataset.return_value = "\u2705 Dataset saved"

        def label_all_samples(**kwargs):
            """Exercise progress and sample callbacks without model work."""

            kwargs["progress_callback"]("Labeling 1/1; labeled 0/1; left 1: sample.wav")
            kwargs["sample_labeled_callback"](0, sample, "\u2705 Labeled: sample.wav")
            return builder.samples, "\u2705 Labeled 1/1 samples; left 0"

        builder.label_all_samples.side_effect = label_all_samples
        progress = MagicMock()
        dit_handler = _FakeDitHandler(model=object())
        llm_handler = SimpleNamespace(llm_initialized=True, last_init_params={})

        _table_update, status_update, _returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            transcribe_lyrics=True,
            lm_lyrics_language="en",
            progress=progress,
            save_path="labels/out",
            dataset_name="labels",
            label_output_dir="labels/processed",
        )

        builder.save_dataset.assert_called_once_with(os.path.normpath("labels/out.json"), "labels")
        self.assertEqual("en", builder.label_all_samples.call_args.kwargs["lm_lyrics_language"])
        expected_label_dir = os.path.normpath(os.path.realpath(os.path.abspath("labels/processed")))
        self.assertEqual(
            expected_label_dir,
            builder.label_all_samples.call_args.kwargs["label_output_dir"],
        )
        progress.assert_any_call((0, 1), desc="Labeling 1/1; labeled 0/1; left 1: sample.wav")
        self.assertIn("Labeled 1/1", status_update["value"])

    def test_requires_processed_label_folder_before_initializing_models(self):
        """Auto-label should reject blank processed-label output folders early."""

        builder = _builder_with_samples()
        dit_handler = _FakeDitHandler()
        llm_handler = SimpleNamespace(llm_initialized=False, last_init_params=None)

        _table_data, status, _returned_builder = auto_label_all(
            dit_handler,
            llm_handler,
            builder,
            label_output_dir="",
        )

        self.assertIn("processed labels folder", status)
        self.assertEqual([], dit_handler.initialize_calls)
        builder.label_all_samples.assert_not_called()


class TestScanDirectory(unittest.TestCase):
    """Tests for scan-directory UI handler behavior."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_scan_preserves_vocal_json_sidecar_with_global_instrumental_default(self):
        """The UI scan should not overwrite ACE generation sidecar vocal metadata."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            audio_path = Path(tmpdir) / "generated.mp3"
            audio_path.write_bytes(b"audio")
            audio_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "caption": "known generated caption",
                        "lyrics": "[Verse]\nknown lyric",
                        "vocal_language": "en",
                        "instrumental": False,
                        "bpm": 81,
                        "keyscale": "C major",
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "acestep.training.dataset_builder_modules.scan.get_audio_duration",
                return_value=200,
            ):
                table_data, status, _slider, builder = scan_directory(
                    tmpdir,
                    "generated_dataset",
                    "",
                    "replace",
                    True,
                    None,
                )

        self.assertIn("Found 1 audio files", status)
        self.assertEqual("yes", table_data[0][3])
        self.assertEqual("en", builder.samples[0].language)
        self.assertFalse(builder.samples[0].is_instrumental)
        self.assertEqual("[Verse]\nknown lyric", builder.samples[0].lyrics)


class TestGetSamplePreview(unittest.TestCase):
    """Tests for get_sample_preview."""

    def test_none_builder_returns_empty(self):
        result = get_sample_preview(0, None)
        # Should return the empty tuple
        self.assertIsNone(result[0])  # audio_path
        self.assertEqual(result[1], "")  # filename

    def test_empty_samples_returns_empty(self):
        builder = MagicMock()
        builder.samples = []
        result = get_sample_preview(0, builder)
        self.assertIsNone(result[0])

    def test_none_index_returns_empty(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        result = get_sample_preview(None, builder)
        self.assertIsNone(result[0])

    def test_out_of_range_index_returns_empty(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        result = get_sample_preview(5, builder)
        self.assertIsNone(result[0])

    def test_valid_sample_returns_data(self):
        sample = MagicMock()
        sample.audio_path = "/path/to/audio.wav"
        sample.filename = "audio.wav"
        sample.caption = "Test caption"
        sample.genre = "rock"
        sample.prompt_override = "genre"
        sample.lyrics = "Hello world"
        sample.formatted_lyrics = ""
        sample.bpm = 120
        sample.keyscale = "C major"
        sample.timesignature = "4/4"
        sample.duration = 30.0
        sample.language = "en"
        sample.is_instrumental = False
        sample.raw_lyrics = ""
        sample.has_raw_lyrics.return_value = False

        builder = MagicMock()
        builder.samples = [sample]

        result = get_sample_preview(0, builder)
        self.assertEqual(result[0], "/path/to/audio.wav")
        self.assertEqual(result[1], "audio.wav")
        self.assertEqual(result[4], "Genre")  # prompt_override converted


class TestSelectSampleFromTable(unittest.TestCase):
    """Tests for Found Audio Files row selection."""

    def test_selects_clicked_row_and_loads_preview(self):
        """Clicking a dataframe row should update Step 3 preview fields."""

        first = _sample(filename="first.wav", caption="First")
        second = _sample(filename="second.wav", caption="Second")
        builder = MagicMock()
        builder.samples = [first, second]
        evt = SimpleNamespace(index=(1, 0))

        result = select_sample_from_table(builder, evt)

        self.assertEqual(result[0]["value"], 1)
        self.assertEqual(result[2], "second.wav")
        self.assertEqual(result[3], "Second")

    def test_invalid_selection_returns_empty_preview(self):
        """Out-of-range selections should not load a stale sample."""

        builder = MagicMock()
        builder.samples = [_sample(filename="first.wav")]
        evt = SimpleNamespace(index=(5, 0))

        result = select_sample_from_table(builder, evt)

        self.assertIsNone(result[1])
        self.assertEqual(result[2], "")


def _sample(filename: str, caption: str = "") -> MagicMock:
    """Build a sample mock with preview fields."""

    sample = MagicMock()
    sample.audio_path = f"/path/{filename}"
    sample.filename = filename
    sample.caption = caption
    sample.genre = ""
    sample.prompt_override = None
    sample.lyrics = "[Instrumental]"
    sample.formatted_lyrics = ""
    sample.bpm = None
    sample.keyscale = ""
    sample.timesignature = ""
    sample.duration = 1.0
    sample.language = "instrumental"
    sample.is_instrumental = True
    sample.raw_lyrics = ""
    sample.has_raw_lyrics.return_value = False
    return sample


class TestUpdateSettings(unittest.TestCase):
    """Tests for update_settings."""

    def test_none_builder_returns_none(self):
        result = update_settings("tag", "prefix", False, 50, None)
        self.assertIsNone(result)

    def test_updates_genre_ratio(self):
        builder = MagicMock()
        builder.metadata = MagicMock()
        result = update_settings("", "prefix", False, 75, builder)
        self.assertEqual(result.metadata.genre_ratio, 75)

    def test_blank_custom_tag_clears_sample_tags(self):
        """Clearing the custom tag should disable tag-position behavior."""

        builder = DatasetBuilder()
        builder.samples = [
            AudioSample(caption="bright pop", genre="pop", custom_tag="oldtag")
        ]
        builder.metadata.custom_tag = "oldtag"
        builder.metadata.tag_position = "replace"

        result = update_settings("", "replace", False, 0, builder)

        self.assertEqual("", result.metadata.custom_tag)
        self.assertEqual("prepend", result.metadata.tag_position)
        self.assertEqual("", result.samples[0].custom_tag)
        self.assertEqual("bright pop", result.samples[0].get_training_prompt("replace"))


class TestSaveDataset(unittest.TestCase):
    """Tests for save_dataset."""

    def test_none_builder(self):
        status, _ = save_dataset("path.json", "name", None)
        self.assertIn("❌", status)

    def test_empty_samples(self):
        builder = MagicMock()
        builder.samples = []
        status, _ = save_dataset("path.json", "name", builder)
        self.assertIn("❌", status)

    def test_empty_path(self):
        builder = MagicMock()
        builder.samples = [MagicMock()]
        status, _ = save_dataset("", "name", builder)
        self.assertIn("❌", status)

    def test_quoted_save_path_appends_json_after_normalizing(self):
        """Quoted save paths with spaces should not keep quotes in the filename."""

        builder = MagicMock()
        builder.samples = [MagicMock()]
        builder.get_labeled_count.return_value = 1
        builder.save_dataset.return_value = "saved"

        status, update = save_dataset('"./datasets/my data"', "name", builder)

        expected_path = os.path.normpath("./datasets/my data.json")
        self.assertEqual("saved", status)
        builder.save_dataset.assert_called_once_with(expected_path, "name")
        self.assertEqual(expected_path, update["value"])


if __name__ == "__main__":
    unittest.main()
