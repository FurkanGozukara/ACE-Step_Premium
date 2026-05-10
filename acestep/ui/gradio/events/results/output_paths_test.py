"""Tests for generated-output path resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.results.output_paths import (
    DEFAULT_RESULTS_DIR,
    create_generation_run_dir,
    get_results_dir,
    use_results_dir,
)


class OutputPathsTests(unittest.TestCase):
    """Verify default and scoped output directory behavior."""

    def test_default_results_dir_uses_outputs_folder(self):
        """The default generation root should be named ``outputs``."""

        self.assertEqual("outputs", DEFAULT_RESULTS_DIR.name)

    def test_scoped_results_dir_controls_run_allocation(self):
        """A scoped output directory should receive sequential run folders."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "custom_outputs"
            with use_results_dir(target):
                self.assertEqual(target.resolve(), get_results_dir())
                first = create_generation_run_dir()
                second = create_generation_run_dir()

            self.assertEqual(target / "0001", first)
            self.assertEqual(target / "0002", second)


if __name__ == "__main__":
    unittest.main()
