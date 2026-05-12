"""Tests for generated-output path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.results.output_paths import (
    DEFAULT_RESULTS_DIR,
    create_generation_run_dir,
    get_results_dir,
    use_generation_run_name,
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

    def test_scoped_run_name_controls_next_folder_name(self):
        """A scoped run name should allocate a unique named run folder."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "custom_outputs"
            with use_results_dir(target):
                with use_generation_run_name("song:name"):
                    first = create_generation_run_dir()
                with use_generation_run_name("song:name"):
                    second = create_generation_run_dir()
                third = create_generation_run_dir()

            self.assertEqual(target / "song_name", first)
            self.assertEqual(target / "song_name_002", second)
            self.assertEqual(target / "0001", third)

    def test_environment_run_name_controls_next_folder_name(self):
        """The subprocess worker should be able to name its run folder by env."""

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "custom_outputs"
            with use_results_dir(target), patch.dict(
                os.environ,
                {"ACESTEP_RUN_DIR_NAME": "battle:rap"},
            ):
                run_dir = create_generation_run_dir()

            self.assertEqual(target / "battle_rap", run_dir)


if __name__ == "__main__":
    unittest.main()
