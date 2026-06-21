"""Contract tests for generation runtime controls."""

from pathlib import Path
import unittest


_GENERATE_CONTROLS_PATH = (
    Path(__file__).resolve().parent / "generation_tab_generate_controls.py"
)


class GenerationTabGenerateControlsTests(unittest.TestCase):
    """Verify runtime controls expose the expected generation keys."""

    def test_seed_controls_live_after_auto_lrc_in_runtime_row(self):
        """The random seed checkbox and seed value should sit beside Auto LRC."""

        source = _GENERATE_CONTROLS_PATH.read_text(encoding="utf-8")

        self.assertIn("ace-runtime-options-row", source)
        self.assertIn('"random_seed_checkbox": random_seed_checkbox', source)
        self.assertIn('"seed": seed', source)
        self.assertLess(
            source.index("_build_right_generate_toggles"),
            source.index("_build_seed_controls()"),
        )

    def test_vocal_language_lives_next_to_think_in_runtime_row(self):
        """Advanced Vocal Language should be a primary runtime control."""

        source = _GENERATE_CONTROLS_PATH.read_text(encoding="utf-8")

        self.assertIn('"vocal_language": vocal_language', source)
        self.assertLess(
            source.index("think_checkbox = gr.Checkbox"),
            source.index("vocal_language = gr.Dropdown"),
        )
        self.assertLess(
            source.index("vocal_language = gr.Dropdown"),
            source.index("auto_score = gr.Checkbox"),
        )


if __name__ == "__main__":
    unittest.main()
