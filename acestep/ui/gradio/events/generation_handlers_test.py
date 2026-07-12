"""Unit tests for generation input event handlers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

try:
    from acestep.ui.gradio.events import generation_handlers
    from acestep.ui.gradio.events.generation.metadata_fields import (
        LOAD_METADATA_GENERATION_OUTPUT_KEYS,
    )
    from acestep.ui.gradio.i18n import t as _t
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependency guard
    generation_handlers = None
    LOAD_METADATA_GENERATION_OUTPUT_KEYS = ()
    _t = None
    _IMPORT_ERROR = exc


def _metadata_index(key: str) -> int:
    """Return the load_metadata output index for a generation component key."""

    return list(LOAD_METADATA_GENERATION_OUTPUT_KEYS).index(key)


class _FakeDitHandler:
    """Minimal DiT handler stub for analyze-src-audio tests."""

    def __init__(self, convert_result):
        """Store a configurable conversion return value for test scenarios."""
        self._convert_result = convert_result
        self.model = MagicMock()  # Required by analyze_src_audio guard

    def convert_src_audio_to_codes(self, _src_audio):
        """Return configured conversion output."""
        return self._convert_result


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class GenerationHandlersTests(unittest.TestCase):
    """Tests for source-audio analysis validation behavior."""

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.ensure_dit_ready")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    def test_analyze_src_audio_extract_mode_shows_guidance_without_model_work(
        self,
        warning_mock,
        info_mock,
        ensure_dit_ready_mock,
    ):
        """Extract-mode Analyze should tell users to choose a track and extract."""

        message = _t("messages.extract_mode_analyze_not_useful")
        dit_handler = SimpleNamespace(convert_src_audio_to_codes=MagicMock())
        llm_handler = SimpleNamespace(llm_initialized=True)

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio=None,
            constrained_decoding_debug=False,
            task_type="extract",
        )

        self.assertEqual(result, ("", message, "", "", None, None, "", "", "", False))
        info_mock.assert_called_once_with(message)
        warning_mock.assert_not_called()
        ensure_dit_ready_mock.assert_not_called()
        dit_handler.convert_src_audio_to_codes.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_rejects_non_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Reject conversion output that has no serialized audio-code tokens."""
        dit_handler = _FakeDitHandler("ERROR: not an audio file")
        llm_handler = SimpleNamespace(llm_initialized=True)

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="fake.mp3",
            constrained_decoding_debug=False,
        )

        # When codes_string has no audio-code tokens, the function returns
        # (codes_string, warning_message, "", "", None, None, "", "", "", False)
        from acestep.ui.gradio.i18n import t
        self.assertEqual(
            result,
            ("ERROR: not an audio file", t("messages.no_audio_codes_generated"),
             "", "", None, None, "", "", "", False),
        )
        understand_music_mock.assert_not_called()
        warning_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_allows_valid_audio_code_output(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Pass valid audio codes through to LM understanding."""
        dit_handler = _FakeDitHandler("<|audio_code_123|><|audio_code_456|>")
        llm_handler = SimpleNamespace(llm_initialized=True)
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="ok",
            caption="caption",
            lyrics="lyrics",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4",
        )

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="real.mp3",
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "<|audio_code_123|><|audio_code_456|>")
        self.assertEqual(result[1], "ok")
        understand_music_mock.assert_called_once()
        warning_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_preserves_existing_text_when_update_disabled(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Analyze should not replace non-empty caption/lyrics when text updates are off."""

        dit_handler = _FakeDitHandler("<|audio_code_123|><|audio_code_456|>")
        llm_handler = SimpleNamespace(llm_initialized=True)
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="ok",
            caption="new caption",
            lyrics="new lyrics",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4",
        )

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="real.mp3",
            constrained_decoding_debug=False,
            current_caption="existing caption",
            current_lyrics="existing lyrics",
            update_caption_lyrics=False,
        )

        self.assertEqual(result[2], "existing caption")
        self.assertEqual(result[3], "existing lyrics")
        self.assertEqual(result[4], 120)
        self.assertEqual(result[6], "C major")
        warning_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.llm_analysis_actions.understand_music")
    def test_analyze_src_audio_fills_blank_text_when_update_disabled(
        self,
        understand_music_mock,
        warning_mock,
    ):
        """Analyze may populate blank caption/lyrics even when text updates are off."""

        dit_handler = _FakeDitHandler("<|audio_code_123|><|audio_code_456|>")
        llm_handler = SimpleNamespace(llm_initialized=True)
        understand_music_mock.return_value = SimpleNamespace(
            success=True,
            status_message="ok",
            caption="new caption",
            lyrics="new lyrics",
            bpm=120,
            duration=30.0,
            keyscale="C major",
            language="en",
            timesignature="4",
        )

        result = generation_handlers.analyze_src_audio(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            src_audio="real.mp3",
            constrained_decoding_debug=False,
            current_caption=" ",
            current_lyrics="",
            update_caption_lyrics=False,
        )

        self.assertEqual(result[2], "new caption")
        self.assertEqual(result[3], "new lyrics")
        warning_mock.assert_not_called()

    def test_convert_src_audio_to_codes_wrapper_uses_newest_stale_upload_list(self):
        """Audio-code conversion should tolerate Gradio stale single-file lists."""

        from acestep.ui.gradio.events.generation.ui_helpers import (
            convert_src_audio_to_codes_wrapper,
        )

        dit_handler = MagicMock()
        dit_handler.convert_src_audio_to_codes.return_value = "codes"

        result = convert_src_audio_to_codes_wrapper(
            dit_handler,
            ["old.wav", "new.mp4"],
        )

        self.assertEqual(result, "codes")
        dit_handler.convert_src_audio_to_codes.assert_called_once_with("new.mp4")

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    @patch("acestep.ui.gradio.events.generation.service_init.get_model_type_ui_settings")
    def test_init_service_wrapper_preserves_batch_size(
        self,
        get_model_type_ui_settings_mock,
        get_global_gpu_config_mock,
    ):
        """Verify that init_service_wrapper preserves current batch_size when provided."""
        # Setup mocks
        gpu_config_mock = MagicMock()
        gpu_config_mock.max_batch_size_with_lm = 8
        gpu_config_mock.max_batch_size_without_lm = 4
        gpu_config_mock.max_duration_with_lm = 600
        gpu_config_mock.max_duration_without_lm = 300
        gpu_config_mock.tier = "tier5"
        gpu_config_mock.available_lm_models = ["acestep-5Hz-lm-1.7B"]
        get_global_gpu_config_mock.return_value = gpu_config_mock

        get_model_type_ui_settings_mock.return_value = (None,) * 9

        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True
        dit_handler.initialize_service.return_value = ("Success", True)

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.initialize.return_value = ("LLM initialized", True)

        # Test with current_batch_size = 5
        result = generation_handlers.init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint=None,
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path=None,
            backend="vllm",
            use_flash_attention=True,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            mlx_dit=False,
            current_mode="Custom",
            current_batch_size=5,
        )

        # Result is a tuple: (status, btn_update, accordion, *model_settings, duration_update, batch_update, think_update)
        self.assertEqual(15, len(result))
        # batch_update is at index -2 (second to last)
        batch_update = result[-2]
        
        # Verify the Songs control preserves the value 5 without a UI maximum.
        self.assertEqual(batch_update["value"], 5)
        self.assertIsNone(batch_update["maximum"])

    @patch("acestep.ui.gradio.events.generation.service_init.get_global_gpu_config")
    @patch("acestep.ui.gradio.events.generation.service_init.get_model_type_ui_settings")
    def test_init_service_wrapper_defaults_batch_size_when_none(
        self,
        get_model_type_ui_settings_mock,
        get_global_gpu_config_mock,
    ):
        """Verify that init_service_wrapper uses default batch_size when current_batch_size is None."""
        # Setup mocks
        gpu_config_mock = MagicMock()
        gpu_config_mock.max_batch_size_with_lm = 8
        gpu_config_mock.max_batch_size_without_lm = 4
        gpu_config_mock.max_duration_with_lm = 600
        gpu_config_mock.max_duration_without_lm = 300
        gpu_config_mock.tier = "tier5"
        gpu_config_mock.available_lm_models = ["acestep-5Hz-lm-1.7B"]
        get_global_gpu_config_mock.return_value = gpu_config_mock

        get_model_type_ui_settings_mock.return_value = (None,) * 9

        dit_handler = MagicMock()
        dit_handler.model = MagicMock()
        dit_handler.is_turbo_model.return_value = True
        dit_handler.initialize_service.return_value = ("Success", True)

        llm_handler = MagicMock()
        llm_handler.llm_initialized = True
        llm_handler.initialize.return_value = ("LLM initialized", True)

        # Test with current_batch_size = None (should default to 1)
        result = generation_handlers.init_service_wrapper(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            checkpoint=None,
            config_path="acestep-v15-turbo",
            device="cuda",
            init_llm=True,
            lm_model_path=None,
            backend="vllm",
            use_flash_attention=True,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            compile_model=False,
            quantization=False,
            mlx_dit=False,
            current_mode="Custom",
            current_batch_size=None,
        )

        self.assertEqual(15, len(result))
        batch_update = result[-2]
        
        # Verify the Songs control defaults to 1 without a UI maximum.
        self.assertEqual(batch_update["value"], 1)
        self.assertIsNone(batch_update["maximum"])

    @patch("acestep.ui.gradio.events.generation.validation.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.validation.soundfile.info")
    def test_validate_uploaded_audio_file_returns_none_for_invalid_file(
        self,
        info_mock,
        warning_mock,
    ):
        """Invalid audio upload should emit warning toast and clear component value."""
        info_mock.side_effect = RuntimeError("bad file")
        result = generation_handlers.validate_uploaded_audio_file("broken.bin", "reference")
        self.assertEqual(result.get("value"), None)
        warning_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.validation.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.validation.soundfile.info")
    def test_validate_uploaded_audio_file_keeps_valid_file(
        self,
        info_mock,
        warning_mock,
    ):
        """Valid audio upload should pass through unchanged without warning."""
        result = generation_handlers.validate_uploaded_audio_file("ok.wav", "reference")
        self.assertEqual(result, {"__type__": "update"})
        info_mock.assert_called_once_with("ok.wav")
        warning_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.validation.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.validation.soundfile.info")
    def test_validate_uploaded_audio_file_keeps_supported_video_file(
        self,
        info_mock,
        warning_mock,
    ):
        """Supported video uploads should be preserved for backend audio extraction."""
        info_mock.side_effect = RuntimeError("video decoded later")
        result = generation_handlers.validate_uploaded_audio_file("clip.mp4", "source")
        self.assertEqual(result, {"__type__": "update"})
        info_mock.assert_called_once_with("clip.mp4")
        warning_mock.assert_not_called()

    @patch("acestep.ui.gradio.events.generation.validation.gr.Warning")
    @patch("acestep.ui.gradio.events.generation.validation.soundfile.info")
    def test_validate_uploaded_audio_file_shows_source_role_message(
        self,
        info_mock,
        warning_mock,
    ):
        """Source-role validation should surface a source-specific toast message."""
        info_mock.side_effect = RuntimeError("bad file")
        result = generation_handlers.validate_uploaded_audio_file("broken.bin", "source")
        self.assertEqual(result.get("value"), None)
        expected_role = _t("generation.source_audio")
        expected_message = _t("messages.audio_format_invalid", role=expected_role)
        warning_mock.assert_called_once_with(
            expected_message
        )

    @patch("acestep.ui.gradio.events.generation.llm_sample_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_sample_actions.create_sample")
    def test_handle_create_sample_success_path(
        self,
        create_sample_mock,
        info_mock,
    ):
        """Successful sample creation should populate generation fields and mode switch."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        create_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="new caption",
            lyrics="new lyrics",
            bpm=120,
            duration=42.0,
            keyscale="C major",
            language="en",
            timesignature="4",
            instrumental=False,
            status_message="ok",
        )

        result = generation_handlers.handle_create_sample(
            llm_handler=llm_handler,
            query="test prompt",
            instrumental=False,
            vocal_language="en",
            lm_temperature=0.85,
            lm_top_k=20,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "new caption")
        self.assertEqual(result[1], "new lyrics")
        self.assertEqual(result[3], 42.0)
        self.assertEqual(result[5], "en")
        self.assertEqual(result[6], "en")
        self.assertEqual(len(result), 15)
        self.assertTrue(result[10])
        self.assertEqual(result[13], "ok")
        self.assertEqual(result[14]["value"], "Custom")
        create_sample_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Warning")
    def test_handle_format_caption_lm_not_initialized_regression(self, warning_mock):
        """Caption formatting should return lm-not-initialized status when LM is unavailable."""
        llm_handler = SimpleNamespace(llm_initialized=False)

        result = generation_handlers.handle_format_caption(
            llm_handler=llm_handler,
            caption="caption",
            lyrics="lyrics",
            bpm=120,
            audio_duration=30.0,
            key_scale="C major",
            time_signature="4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[-1], _t("messages.lm_not_initialized"))
        warning_mock.assert_called_once_with(_t("messages.lm_not_initialized"))

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    def test_handle_format_lyrics_strips_quotes(self, format_sample_mock, info_mock):
        """Lyrics-only formatting should strip wrapper quotes from returned lyrics."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="unused",
            lyrics="'quoted lyrics'",
            bpm=100,
            duration=10.0,
            keyscale="D minor",
            language="en",
            timesignature="4",
            status_message="formatted",
        )

        result = generation_handlers.handle_format_lyrics(
            llm_handler=llm_handler,
            caption="caption",
            lyrics="lyrics",
            bpm=100,
            audio_duration=10.0,
            key_scale="D minor",
            time_signature="4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(result[0], "quoted lyrics")
        self.assertEqual(result[4], "en")
        self.assertEqual(len(result), 8)
        self.assertEqual(result[-1], "formatted")
        format_sample_mock.assert_called_once()
        info_mock.assert_called_once()

    @patch("acestep.ui.gradio.events.generation.llm_format_actions.gr.Info")
    @patch("acestep.ui.gradio.events.generation.llm_format_actions.format_sample")
    def test_handle_format_sample_preserves_output_contract(self, format_sample_mock, info_mock):
        """Full sample formatting should return 9 outputs with language at index 5."""
        llm_handler = SimpleNamespace(llm_initialized=True)
        format_sample_mock.return_value = SimpleNamespace(
            success=True,
            caption="caption out",
            lyrics="lyrics out",
            bpm=112,
            duration=12.0,
            keyscale="G major",
            language="en",
            timesignature="3/4",
            status_message="formatted",
        )

        result = generation_handlers.handle_format_sample(
            llm_handler=llm_handler,
            caption="caption in",
            lyrics="lyrics in",
            bpm=112,
            audio_duration=12.0,
            key_scale="G major",
            time_signature="3/4",
            lm_temperature=0.85,
            lm_top_k=0,
            lm_top_p=0.9,
            constrained_decoding_debug=False,
        )

        self.assertEqual(len(result), 9)
        self.assertEqual(result[5], "en")
        self.assertEqual(result[8], "formatted")
        format_sample_mock.assert_called_once()
        info_mock.assert_called_once()


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class LoadMetadataLmCodesTests(unittest.TestCase):
    """Tests that load_metadata sets think=False when audio_codes are present."""

    def _write_json(self, tmpdir, data):
        """Write a JSON file and return a SimpleNamespace with .name pointing to it."""
        import json, os
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return SimpleNamespace(name=path)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_think_set_false_when_audio_codes_present(self, info_mock):
        """When JSON has thinking=True AND non-empty audio_codes, think should be False."""
        import tempfile

        llm_handler = SimpleNamespace(llm_initialized=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "thinking": True,
                "audio_codes": "<|audio_code_1|><|audio_code_2|>",
            })
            result = generation_handlers.load_metadata(file_obj, llm_handler)

        think_value = result[_metadata_index("think_checkbox")]
        audio_codes_value = result[_metadata_index("text2music_audio_code_string")]
        self.assertFalse(think_value, "think should be False when audio_codes present")
        self.assertEqual(audio_codes_value, "<|audio_code_1|><|audio_code_2|>")

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_think_unchanged_when_audio_codes_empty(self, info_mock):
        """When JSON has thinking=True AND empty audio_codes, think stays True."""
        import tempfile

        llm_handler = SimpleNamespace(llm_initialized=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "thinking": True,
                "audio_codes": "",
            })
            result = generation_handlers.load_metadata(file_obj, llm_handler)

        think_value = result[_metadata_index("think_checkbox")]
        self.assertTrue(think_value, "think should remain True when audio_codes is empty")


@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class LoadMetadataFileOutputTests(unittest.TestCase):
    """Tests that metadata loading restores settings but not upload file values."""

    def _write_json(self, tmpdir, data):
        """Write a JSON file and return a SimpleNamespace with .name pointing to it."""
        import json, os
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return SimpleNamespace(name=path)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_load_metadata_clears_file_upload_outputs(self, info_mock):
        """Saved paths should not be sent back to Gradio file-upload components."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "caption": "restored caption",
                "reference_audio": r"C:\Users\Furkan\AppData\Local\Temp\gradio\old.wav",
                "src_audio": r"G:\ACE_Step_v1\ACE-Step_Premium",
                "saved_run_assets": {
                    "reference_audio_path": r"G:\ACE_Step_v1\ACE-Step_Premium",
                    "source_audio_path": r"G:\ACE_Step_v1\ACE-Step_Premium",
                },
                "audio_processing_settings": {
                    "diffpitcher": {
                        "reference_audio": r"G:\ACE_Step_v1\ACE-Step_Premium",
                        "midi_path": r"G:\ACE_Step_v1\ACE-Step_Premium",
                    },
                },
            })
            result = generation_handlers.load_metadata(file_obj, None)

        self.assertEqual(result[_metadata_index("captions")], "restored caption")
        self.assertIsNone(result[_metadata_index("reference_audio")])
        self.assertIsNone(result[_metadata_index("src_audio")])
        self.assertIsNone(result[_metadata_index("ap_diffpitcher_reference_audio")])
        self.assertIsNone(result[_metadata_index("ap_diffpitcher_midi")])




@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class LoadMetadataMp3SanitizationTests(unittest.TestCase):
    """Tests for MP3 metadata normalization and fallback behavior."""

    def _write_json(self, tmpdir, data):
        """Write a JSON file and return a SimpleNamespace with .name pointing to it."""
        import json, os
        path = os.path.join(tmpdir, "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return SimpleNamespace(name=path)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_load_metadata_normalizes_mp3_values_from_json(self, info_mock):
        """Uppercase MP3 metadata should be normalized to the UI-compatible values."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "audio_format": " MP3 ",
                "mp3_bitrate": "320K",
                "mp3_sample_rate": "44100",
            })
            result = generation_handlers.load_metadata(file_obj, None)

        self.assertEqual(result[_metadata_index("audio_format")], "mp3")
        self.assertTrue(result[_metadata_index("mp3_controls_row")]["visible"])
        self.assertEqual(result[_metadata_index("mp3_bitrate")]["value"], "320k")
        self.assertEqual(result[_metadata_index("mp3_sample_rate")]["value"], 44100)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_load_metadata_falls_back_for_invalid_mp3_values(self, info_mock):
        """Invalid MP3 metadata should fall back to the supported defaults."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "audio_format": "mp3",
                "mp3_bitrate": "999k",
                "mp3_sample_rate": "44.1",
            })
            result = generation_handlers.load_metadata(file_obj, None)

        self.assertEqual(result[_metadata_index("audio_format")], "mp3")
        self.assertTrue(result[_metadata_index("mp3_controls_row")]["visible"])
        self.assertEqual(result[_metadata_index("mp3_bitrate")]["value"], "256k")
        self.assertEqual(result[_metadata_index("mp3_sample_rate")]["value"], 48000)

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_load_metadata_restores_extract_output_format(self, info_mock):
        """Extract metadata should restore the dedicated Extract output format."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "task_type": "extract",
                "audio_format": "wav",
                "track_name": "vocals",
            })
            result = generation_handlers.load_metadata(file_obj, None)

        self.assertEqual(result[_metadata_index("track_name")], "vocals")
        self.assertEqual(result[_metadata_index("extract_output_format")], "wav")

    @patch("acestep.ui.gradio.events.generation.metadata_loading.gr.Info")
    def test_load_metadata_restores_repaint_lyric_switch_flag(self, info_mock):
        """Metadata load should restore the Repaint lyric-switch opt-out flag."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            file_obj = self._write_json(tmpdir, {
                "generation_params": {
                    "repaint_dont_switch_with_lyrics": True,
                },
            })
            result = generation_handlers.load_metadata(file_obj, None)

        self.assertTrue(result[_metadata_index("repaint_dont_switch_with_lyrics")])

@unittest.skipIf(generation_handlers is None, f"generation_handlers import unavailable: {_IMPORT_ERROR}")
class AutoCheckboxTests(unittest.TestCase):
    """Tests for optional-parameter Auto checkbox handler functions."""

    def test_on_auto_checkbox_change_checked_returns_default_and_non_interactive(self):
        """When Auto is checked, field should reset to default and become non-interactive."""
        result = generation_handlers.on_auto_checkbox_change(True, "bpm")
        # gr.update returns a dict-like object; check value and interactive
        self.assertIsNone(result["value"])
        self.assertFalse(result["interactive"])

    def test_on_auto_checkbox_change_unchecked_returns_interactive(self):
        """When Auto is unchecked, field should become interactive (no value reset)."""
        result = generation_handlers.on_auto_checkbox_change(False, "bpm")
        self.assertTrue(result["interactive"])

    def test_on_auto_checkbox_change_all_fields(self):
        """All supported field names should produce valid defaults when checked."""
        expected = {
            "bpm": None,
            "key_scale": "",
            "time_signature": "",
            "vocal_language": "unknown",
            "audio_duration": -1,
        }
        for field_name, expected_value in expected.items():
            result = generation_handlers.on_auto_checkbox_change(True, field_name)
            self.assertEqual(result["value"], expected_value, f"Field {field_name}")
            self.assertEqual(
                result["interactive"],
                field_name == "vocal_language",
                f"Field {field_name}",
            )

    def test_reset_all_auto_returns_correct_count(self):
        """reset_all_auto should return exactly 10 gr.update objects."""
        result = generation_handlers.reset_all_auto()
        self.assertEqual(len(result), 10)

    def test_reset_all_auto_checkboxes_are_true(self):
        """First 5 outputs (auto checkboxes) should all be set to True."""
        result = generation_handlers.reset_all_auto()
        for i in range(5):
            self.assertTrue(result[i]["value"], f"Auto checkbox at index {i}")

    def test_reset_all_auto_fields_are_defaults(self):
        """Last 5 outputs (fields) should be reset to auto defaults."""
        result = generation_handlers.reset_all_auto()
        self.assertIsNone(result[5]["value"])         # bpm
        self.assertEqual(result[6]["value"], "")       # key_scale
        self.assertEqual(result[7]["value"], "")       # time_signature
        self.assertEqual(result[8]["value"], "unknown") # vocal_language
        self.assertEqual(result[9]["value"], -1)       # audio_duration

    def test_uncheck_auto_for_populated_fields_all_default(self):
        """When all fields have default values, all auto checkboxes should stay checked."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=None, key_scale="", time_signature="",
            vocal_language="unknown", audio_duration=-1,
        )
        self.assertEqual(len(result), 10)
        # Auto checkboxes should be True (checked)
        for i in range(5):
            self.assertTrue(result[i]["value"], f"Auto checkbox at index {i}")
        # Language remains user-selectable so automatic metadata can still be
        # constrained; the other auto fields are non-interactive.
        for i in range(5, 10):
            self.assertEqual(
                result[i]["interactive"],
                i == 8,
                f"Field at index {i}",
            )

    def test_uncheck_auto_for_populated_fields_all_populated(self):
        """When all fields have non-default values, all auto checkboxes should be unchecked."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=120, key_scale="C major", time_signature="4",
            vocal_language="en", audio_duration=30.0,
        )
        # Auto checkboxes should be False (unchecked)
        for i in range(5):
            self.assertFalse(result[i]["value"], f"Auto checkbox at index {i}")
        # Fields should be interactive
        for i in range(5, 10):
            self.assertTrue(result[i]["interactive"], f"Field at index {i}")

    def test_uncheck_auto_for_populated_fields_mixed(self):
        """Mixed populated/default fields should only uncheck populated ones."""
        result = generation_handlers.uncheck_auto_for_populated_fields(
            bpm=120, key_scale="", time_signature="4",
            vocal_language="unknown", audio_duration=-1,
        )
        self.assertFalse(result[0]["value"])   # bpm_auto unchecked
        self.assertTrue(result[1]["value"])    # key_auto stays checked
        self.assertFalse(result[2]["value"])   # timesig_auto unchecked
        self.assertTrue(result[3]["value"])    # vocal_lang_auto stays checked
        self.assertTrue(result[4]["value"])    # duration_auto stays checked


if __name__ == "__main__":
    unittest.main()
