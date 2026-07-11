"""Tests for the process-local SAM-Audio service cache."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from acestep.sam_audio_segment.service_cache import (
    cached_sam_audio_service,
    clear_cached_sam_audio_service,
)
from acestep.sam_audio_segment.settings import SamAudioSettings


class _FakeService:
    """Minimal service double that records construction and unload calls."""

    instances: list["_FakeService"] = []

    def __init__(
        self,
        settings: SamAudioSettings,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        progress_callback=None,
    ) -> None:
        self.settings = settings
        self.model_path = model_path
        self.device = device
        self.progress_callback = progress_callback
        self.unload_calls = 0
        self.instances.append(self)

    def unload(self) -> None:
        """Record that the cached service was released."""

        self.unload_calls += 1


class ServiceCacheTests(unittest.TestCase):
    """Verify cached SAM-Audio service reuse and eviction behavior."""

    def setUp(self) -> None:
        """Start each test with an empty service cache."""

        clear_cached_sam_audio_service()
        _FakeService.instances = []

    def tearDown(self) -> None:
        """Release any service left cached by a test."""

        clear_cached_sam_audio_service()

    def test_reuses_service_when_only_runtime_settings_change(self) -> None:
        """Prompt and chunk settings should not force model reloads."""

        first_progress = object()
        second_progress = object()
        settings = SamAudioSettings(subprocess=False)
        changed_runtime_settings = replace(
            settings,
            custom_prompt="vocals",
            chunk_seconds=10.0,
            output_format="wav",
        )

        with _patched_service():
            with cached_sam_audio_service(
                settings,
                progress_callback=first_progress,
            ) as first:
                pass
            with cached_sam_audio_service(
                changed_runtime_settings,
                progress_callback=second_progress,
            ) as second:
                pass

        self.assertIs(first, second)
        self.assertEqual(1, len(_FakeService.instances))
        self.assertEqual(changed_runtime_settings, second.settings)
        self.assertIs(second_progress, second.progress_callback)
        self.assertEqual(0, first.unload_calls)

    def test_evicts_service_when_model_settings_change(self) -> None:
        """Quantization changes should unload and rebuild the cached model."""

        settings = SamAudioSettings(subprocess=False, quantization="none")
        changed_model_settings = replace(settings, quantization="fp8_scaled")

        with _patched_service():
            with cached_sam_audio_service(settings) as first:
                pass
            with cached_sam_audio_service(changed_model_settings) as second:
                pass

        self.assertIsNot(first, second)
        self.assertEqual(2, len(_FakeService.instances))
        self.assertEqual(1, first.unload_calls)
        self.assertEqual(0, second.unload_calls)

    def test_evicts_service_when_compile_request_changes(self) -> None:
        """Changing the selective compile request should rebuild the model."""

        settings = SamAudioSettings(subprocess=False, compile_model=False)
        compiled_settings = replace(settings, compile_model=True)

        with _patched_service():
            with cached_sam_audio_service(settings) as first:
                pass
            with cached_sam_audio_service(compiled_settings) as second:
                pass

        self.assertIsNot(first, second)
        self.assertEqual(1, first.unload_calls)
        self.assertTrue(second.settings.compile_model)

    def test_evicts_service_when_visual_prompt_requires_full_model(self) -> None:
        """Visual mode should not reuse a text/span-only cached service."""

        settings = SamAudioSettings(subprocess=False, prompt_mode="text")
        visual_settings = replace(settings, prompt_mode="visual")

        with _patched_service():
            with cached_sam_audio_service(settings) as first:
                pass
            with cached_sam_audio_service(visual_settings) as second:
                pass

        self.assertIsNot(first, second)
        self.assertEqual(2, len(_FakeService.instances))
        self.assertEqual(1, first.unload_calls)

    def test_clear_unloads_cached_service(self) -> None:
        """Explicit cache clearing should release the retained model."""

        with _patched_service():
            with cached_sam_audio_service(SamAudioSettings(subprocess=False)) as service:
                pass
            clear_cached_sam_audio_service()

        self.assertEqual(1, service.unload_calls)


def _patched_service():
    """Patch cache dependencies for deterministic service-cache tests."""

    return patch.multiple(
        "acestep.sam_audio_segment.service_cache",
        SamAudioService=_FakeService,
        default_model_path=lambda: Path("models/SAM-Audio-Large-BF16.safetensors"),
    )


if __name__ == "__main__":
    unittest.main()
