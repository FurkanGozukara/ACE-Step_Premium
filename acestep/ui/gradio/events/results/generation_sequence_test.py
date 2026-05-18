"""Tests for cooperative cancellation in sequential song generation."""

import unittest
from types import SimpleNamespace

from acestep.core.generation.cancellation import (
    GenerationCancelled,
    generation_cancel_scope,
    request_generation_cancel,
)
from acestep.inference import GenerationConfig
from acestep.ui.gradio.events.results.generation_sequence import generate_sequential_songs


class GenerationSequenceCancellationTests(unittest.TestCase):
    """Verify sequential generation stops after a cancel request."""

    def tearDown(self) -> None:
        """Clear cancellation state left by each test."""

        with generation_cancel_scope():
            pass

    def test_cancel_request_stops_remaining_sequential_generations(self) -> None:
        """Cancellation after one song should prevent later songs from starting."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            """Request cancellation during the first backend call."""

            _ = params, progress
            calls.append(config.seeds)
            request_generation_cancel()
            return SimpleNamespace(success=True, audios=[], extra_outputs={})

        with generation_cancel_scope():
            with self.assertRaises(GenerationCancelled):
                generate_sequential_songs(
                    fake_generate_music,
                    None,
                    None,
                    params=object(),
                    base_config=GenerationConfig(batch_size=1, seeds=[1]),
                    generation_count=3,
                    seed="1",
                    random_seed=False,
                    progress=None,
                )

        self.assertEqual(calls, [[1]])


if __name__ == "__main__":
    unittest.main()
