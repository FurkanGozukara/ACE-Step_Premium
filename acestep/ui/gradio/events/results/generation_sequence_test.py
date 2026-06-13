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
            return SimpleNamespace(
                success=True,
                audios=[],
                extra_outputs={},
                status_message="ok",
            )

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

    def test_params_for_index_supplies_per_generation_params(self) -> None:
        """Sequential generation can customize params for each backend call."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            """Record params for each sequential call."""

            _ = config, progress
            calls.append(params)
            return SimpleNamespace(
                success=True,
                audios=[],
                extra_outputs={},
                status_message="ok",
            )

        generate_sequential_songs(
            fake_generate_music,
            None,
            None,
            params="base",
            base_config=GenerationConfig(batch_size=1, seeds=[1]),
            generation_count=3,
            seed="1",
            random_seed=False,
            progress=None,
            params_for_index=lambda params, index: f"{params}-{index}",
        )

        self.assertEqual(calls, ["base-0", "base-1", "base-2"])

    def test_reuse_fixed_seed_reuses_base_seed_for_each_generation(self) -> None:
        """All-stems Extract can compare each stem against the same fixed seed."""

        calls = []

        def fake_generate_music(_dit_handler, _llm_handler, *, params, config, progress):
            """Record seeds for each sequential call."""

            _ = params, progress
            calls.append(config.seeds)
            return SimpleNamespace(
                success=True,
                audios=[],
                extra_outputs={},
                status_message="ok",
            )

        generate_sequential_songs(
            fake_generate_music,
            None,
            None,
            params=object(),
            base_config=GenerationConfig(batch_size=1, seeds=[44]),
            generation_count=3,
            seed="44",
            random_seed=False,
            progress=None,
            reuse_fixed_seed=True,
        )

        self.assertEqual(calls, [[44], [44], [44]])


if __name__ == "__main__":
    unittest.main()
