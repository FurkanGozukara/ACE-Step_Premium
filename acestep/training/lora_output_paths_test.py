"""Tests for LoRA training output path helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

from acestep.training.lora_output_paths import (
    resolve_lora_export_root,
    resolve_lora_training_output_dir,
)
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class LoRAOutputPathTests(unittest.TestCase):
    """Verify named LoRA run directories are resolved safely."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_training_output_dir_is_named_child(self) -> None:
        """A target folder should receive a child folder named after the LoRA."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            target = os.path.join(tmpdir, "Loras")

            output_dir = resolve_lora_training_output_dir(target, "2pac_SFT-3e-04")

        self.assertEqual(
            os.path.realpath(os.path.join(target, "2pac_SFT-3e-04")),
            output_dir,
        )

    def test_training_output_dir_avoids_double_nesting_same_name(self) -> None:
        """Selecting an existing run folder should not append the same name twice."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            target = os.path.join(tmpdir, "2pac_SFT-3e-04")

            output_dir = resolve_lora_training_output_dir(target, "2pac_SFT-3e-04")

        self.assertEqual(os.path.realpath(target), output_dir)

    def test_export_root_uses_named_child_when_base_has_no_artifacts(self) -> None:
        """Export should look inside the named run folder after new trainings."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            target = os.path.join(tmpdir, "Loras")

            output_dir = resolve_lora_export_root(target, "2pac_SFT-3e-04")

        self.assertEqual(
            os.path.realpath(os.path.join(target, "2pac_SFT-3e-04")),
            output_dir,
        )

    def test_export_root_preserves_direct_legacy_output(self) -> None:
        """Older output folders with direct artifacts should still export."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            os.makedirs(os.path.join(tmpdir, "final"))

            output_dir = resolve_lora_export_root(tmpdir, "2pac_SFT-3e-04")

        self.assertEqual(os.path.realpath(tmpdir), output_dir)

    def test_export_root_preserves_flat_safetensors_output(self) -> None:
        """Flat run folders with safetensors should be treated as direct outputs."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            with open(os.path.join(tmpdir, "my-lora-epoch-3.safetensors"), "w") as handle:
                handle.write("weights")

            output_dir = resolve_lora_export_root(tmpdir, "my-lora")

        self.assertEqual(os.path.realpath(tmpdir), output_dir)


if __name__ == "__main__":
    unittest.main()
