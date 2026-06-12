"""Tests for Audio Processing stage-control metadata."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

import gradio as gr

from acestep.audio_processing.presets import STAGE_KEYS

MODULE_PATH = Path(__file__).with_name("audio_processing_stage_controls.py")


class AudioProcessingStageControlsTests(unittest.TestCase):
    """Verify stage controls expose practical help text."""

    def test_stage_sliders_use_descriptive_info_text(self) -> None:
        """Every audio-processing stage slider should expose its description."""

        stage_controls = _load_stage_controls_module()
        demo = gr.Blocks()
        try:
            with demo:
                controls: dict[str, object] = {}
                stage_controls.add_stage_controls(controls)
        finally:
            demo.close()

        self.assertEqual(set(STAGE_KEYS), set(stage_controls.STAGE_INFOS))
        self.assertEqual(
            "Check / Uncheck All Stages",
            controls["ap_toggle_audio_enhancement_btn"].value,
        )
        for key in STAGE_KEYS:
            self.assertEqual(stage_controls.STAGE_INFOS[key], controls[f"ap_{key}"].info)
            self.assertIn("Example:", controls[f"ap_{key}"].info)


def _load_stage_controls_module() -> ModuleType:
    """Load the stage-control module without importing the full Gradio app package."""

    spec = importlib.util.spec_from_file_location(
        "audio_processing_stage_controls_under_test",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load audio_processing_stage_controls module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
