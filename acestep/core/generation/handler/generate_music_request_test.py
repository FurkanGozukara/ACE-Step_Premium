"""Unit tests for ``generate_music`` request helper mixin."""

import types
import unittest

import torch

from acestep.constants import TASK_INSTRUCTIONS
from acestep.core.generation.handler.generate_music_request import GenerateMusicRequestMixin
from acestep.core.generation.handler.repaint_prompt import REPAINT_LYRICS_INFO_TEXT


class _Host(GenerateMusicRequestMixin):
    """Minimal host implementing dependencies for request helper tests."""

    def __init__(self):
        """Initialize state and default stubs used by helper methods."""
        self.current_offload_cost = 0.0
        self.batch_size = 2
        self.sample_rate = 48000
        self.model = object()
        self.vae = object()
        self.text_tokenizer = object()
        self.text_encoder = object()
        self._vram_guard_reduce_batch = lambda bs, audio_duration=None: bs
        self.prepare_seeds = lambda bs, seed, use_random_seed: ([1] * bs, 1)
        self.process_reference_audio = lambda _ref: torch.zeros(2, 100)
        self.process_src_audio = lambda _src: torch.ones(2, 100)
        self.prepare_batch_data = lambda *args, **kwargs: (
            ["cap"], ["inst"], ["lyr"], ["en"], ["meta"]
        )
        self.determine_task_type = lambda task, codes: (
            False, False, task in ("cover", "cover-nofsq"), True
        )
        self.prepare_padding_info = lambda *args, **kwargs: ([0.0], [1.0], torch.zeros(1, 2, 100))


class GenerateMusicRequestMixinTests(unittest.TestCase):
    """Validate request helper behavior used by ``generate_music`` orchestration."""

    def test_resolve_task_switches_to_cover_when_audio_codes_present(self):
        """Text2music should switch to cover mode when audio code hints are supplied."""
        host = _Host()
        task, instruction = host._resolve_generate_music_task(
            task_type="text2music",
            audio_code_string="<|audio_code_1|>",
            instruction="old",
        )
        self.assertEqual(task, "cover")
        self.assertNotEqual(instruction, "old")

    def test_resolve_task_replaces_default_instruction_for_repaint(self):
        """Repaint should not keep a stale generic text2music instruction."""
        host = _Host()
        task, instruction = host._resolve_generate_music_task(
            task_type="repaint",
            audio_code_string="",
            instruction=TASK_INSTRUCTIONS["text2music"],
        )
        self.assertEqual(task, "repaint")
        self.assertEqual(instruction, TASK_INSTRUCTIONS["repaint"])

    def test_resolve_task_preserves_custom_repaint_instruction(self):
        """A user-provided repaint instruction should not be overwritten."""
        host = _Host()
        task, instruction = host._resolve_generate_music_task(
            task_type="repaint",
            audio_code_string="",
            instruction="Sing this exact new line in the masked region:",
        )
        self.assertEqual(task, "repaint")
        self.assertEqual(instruction, "Sing this exact new line in the masked region:")

    def test_prepare_runtime_normalizes_batch_and_duration(self):
        """Runtime helper should clamp batch floor and normalize invalid duration/end values."""
        host = _Host()
        out = host._prepare_generate_music_runtime(
            batch_size=0,
            audio_duration=0.0,
            repainting_end=-1.0,
            seed=7,
            use_random_seed=False,
        )
        self.assertEqual(out["actual_batch_size"], 1)
        self.assertIsNone(out["audio_duration"])
        self.assertIsNone(out["repainting_end"])

    def test_prepare_reference_and_source_audio_returns_error_for_invalid_reference(self):
        """Invalid reference audio should return a structured early error payload."""
        host = _Host()
        host.process_reference_audio = lambda _ref: None
        _, _, error = host._prepare_reference_and_source_audio(
            reference_audio="bad.wav",
            src_audio=None,
            audio_code_string="",
            actual_batch_size=1,
            task_type="cover",
        )
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertEqual(error["error"], "Invalid reference audio")

    def test_prepare_reference_and_source_audio_ignores_src_audio_for_text2music(self):
        """Text2music should ignore src audio to preserve upstream behavior."""
        host = _Host()
        called = {"process_src_audio": False}

        def _process_src_audio(_src):
            called["process_src_audio"] = True
            return torch.ones(2, 100)

        host.process_src_audio = _process_src_audio
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio="song.wav",
            audio_code_string="",
            actual_batch_size=1,
            task_type="text2music",
        )
        self.assertIsNone(error)
        self.assertIsNone(processed_src_audio)
        self.assertFalse(called["process_src_audio"])

    def test_prepare_reference_and_source_audio_returns_error_for_invalid_source(self):
        """Non-text2music source audio failure should return structured error payload."""
        host = _Host()
        host.process_src_audio = lambda _src: None
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio="bad.wav",
            audio_code_string="",
            actual_batch_size=1,
            task_type="cover",
        )
        self.assertIsNone(processed_src_audio)
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertEqual(error["error"], "Invalid source audio")


    def test_cover_no_src_audio_with_codes_succeeds(self):
        """Cover task should succeed without src_audio when audio codes are provided."""
        host = _Host()
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="<|audio_code_42|>",
            actual_batch_size=1,
            task_type="cover",
        )
        self.assertIsNone(error)
        self.assertIsNone(processed_src_audio)

    def test_cover_no_src_audio_no_codes_errors(self):
        """Cover task without src_audio and without audio codes should error."""
        host = _Host()
        _, _, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="",
            actual_batch_size=1,
            task_type="cover",
        )
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertIn("requires source audio", error["error"])

    def test_cover_nofsq_no_src_audio_no_codes_errors(self):
        """Raw cover task without src_audio and without audio codes should error."""
        host = _Host()
        _, _, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="",
            actual_batch_size=1,
            task_type="cover-nofsq",
        )
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertIn("requires source audio", error["error"])

    def test_repaint_no_src_audio_with_codes_succeeds(self):
        """Repaint task should succeed without src_audio when audio codes are provided."""
        host = _Host()
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="<|audio_code_99|>",
            actual_batch_size=1,
            task_type="repaint",
        )
        self.assertIsNone(error)
        self.assertIsNone(processed_src_audio)

    def test_cover_with_src_audio_and_codes_keeps_code_only_behavior(self):
        """Cover should keep existing code-only behavior when audio codes are supplied."""
        host = _Host()
        called = {"process_src_audio": False}

        def _process_src_audio(_src):
            called["process_src_audio"] = True
            return torch.ones(2, 100)

        host.process_src_audio = _process_src_audio
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio="source.wav",
            audio_code_string="<|audio_code_99|>",
            actual_batch_size=1,
            task_type="cover",
        )
        self.assertIsNone(error)
        self.assertIsNone(processed_src_audio)
        self.assertFalse(called["process_src_audio"])

    def test_complete_with_src_audio_and_codes_still_processes_source(self):
        """Complete must keep source audio active even when LM audio codes are present."""
        host = _Host()
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio="source.wav",
            audio_code_string="<|audio_code_99|>",
            actual_batch_size=1,
            task_type="complete",
        )
        self.assertIsNone(error)
        self.assertIsNotNone(processed_src_audio)

    def test_lego_with_src_audio_and_codes_still_processes_source(self):
        """Lego should use uploaded source audio when it is provided with LM codes."""
        host = _Host()
        _, processed_src_audio, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio="source.wav",
            audio_code_string="<|audio_code_99|>",
            actual_batch_size=1,
            task_type="lego",
        )
        self.assertIsNone(error)
        self.assertIsNotNone(processed_src_audio)

    def test_complete_no_src_audio_no_codes_errors(self):
        """Complete task without source audio should error."""
        host = _Host()
        _, _, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="",
            actual_batch_size=1,
            task_type="complete",
        )
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertIn("requires source audio", error["error"])

    def test_complete_no_src_audio_with_codes_still_errors(self):
        """Complete should require source audio even when audio codes are present."""
        host = _Host()
        _, _, error = host._prepare_reference_and_source_audio(
            reference_audio=None,
            src_audio=None,
            audio_code_string="<|audio_code_99|>",
            actual_batch_size=1,
            task_type="complete",
        )
        self.assertIsNotNone(error)
        self.assertFalse(error["success"])
        self.assertIn("requires source audio", error["error"])

    def test_should_return_intermediate_always_true(self):
        """Intermediate tensors must always be returned for LRC generation support."""
        host = _Host()
        for task in ("text2music", "cover", "cover-nofsq", "repaint"):
            inputs = host._prepare_generate_music_service_inputs(
                actual_batch_size=1,
                processed_src_audio=None,
                audio_duration=60.0,
                captions="test",
                lyrics="test",
                vocal_language="en",
                instruction="inst",
                bpm=120,
                key_scale="C major",
                time_signature="4/4",
                task_type=task,
                audio_code_string="",
                repainting_start=0.0,
                repainting_end=1.0,
            )
            self.assertTrue(
                inputs["should_return_intermediate"],
                f"should_return_intermediate must be True for task_type={task!r}",
            )

    def test_repaint_with_lyrics_strengthens_caption_and_uses_explicit_mask(self):
        """Lyric repaint should clearly condition the masked region on target words."""
        host = _Host()
        captured = {}

        def _prepare_batch_data(
            _actual_batch_size,
            _processed_src_audio,
            _audio_duration,
            captions,
            lyrics,
            vocal_language,
            instruction,
            _bpm,
            _key_scale,
            _time_signature,
        ):
            """Capture text inputs passed into batch preparation."""
            captured["captions"] = captions
            captured["lyrics"] = lyrics
            captured["vocal_language"] = vocal_language
            return (
                [captions],
                [instruction],
                [lyrics],
                [vocal_language],
                [{"duration": "60 seconds"}],
            )

        host.prepare_batch_data = _prepare_batch_data

        inputs = host._prepare_generate_music_service_inputs(
            actual_batch_size=1,
            processed_src_audio=torch.ones(2, 48000),
            audio_duration=1.0,
            captions="modern pop rap",
            lyrics="amazing song amazing tutorial you are doing great lets go",
            vocal_language="unknown",
            instruction=TASK_INSTRUCTIONS["repaint"],
            task_type="repaint",
            chunk_mask_mode="auto",
            repainting_start=0.0,
            repainting_end=1.0,
        )

        self.assertEqual(inputs["chunk_mask_modes_batch"], ["explicit"])
        self.assertIn("selected 1-second mask", captured["captions"])
        self.assertIn("clear English vocal", captured["captions"])
        self.assertIn(
            '"amazing song amazing tutorial you are doing great lets go"',
            captured["captions"],
        )
        self.assertEqual(captured["vocal_language"], "en")
        self.assertEqual(inputs["metas_batch"], [{"duration": "1 seconds"}])

    def test_repaint_helper_lyrics_are_treated_as_empty(self):
        """Repaint UI helper copy should not become target lyric conditioning."""
        host = _Host()
        captured = {}

        def _prepare_batch_data(
            _actual_batch_size,
            _processed_src_audio,
            _audio_duration,
            captions,
            lyrics,
            vocal_language,
            instruction,
            _bpm,
            _key_scale,
            _time_signature,
        ):
            """Capture sanitized lyrics passed into batch preparation."""
            captured["captions"] = captions
            captured["lyrics"] = lyrics
            captured["vocal_language"] = vocal_language
            return ([captions], [instruction], [lyrics], [vocal_language], ["meta"])

        host.prepare_batch_data = _prepare_batch_data

        inputs = host._prepare_generate_music_service_inputs(
            actual_batch_size=1,
            processed_src_audio=torch.ones(2, 48000),
            audio_duration=1.0,
            captions="rap",
            lyrics=REPAINT_LYRICS_INFO_TEXT,
            vocal_language="unknown",
            instruction=TASK_INSTRUCTIONS["repaint"],
            task_type="repaint",
            chunk_mask_mode="auto",
            repainting_start=0.0,
            repainting_end=1.0,
        )

        self.assertEqual(captured["lyrics"], "")
        self.assertEqual(captured["captions"], "rap")
        self.assertEqual(captured["vocal_language"], "unknown")
        self.assertEqual(inputs["chunk_mask_modes_batch"], ["auto"])


if __name__ == "__main__":
    unittest.main()
