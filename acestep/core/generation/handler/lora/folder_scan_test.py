"""Tests for project LoRA folder discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.core.generation.handler.lora.folder_scan import (
    discover_lora_folder_items,
    ensure_lora_folder,
    lora_dropdown_choices,
    resolve_loadable_lora_adapter_path,
)


def _write_peft_adapter(path: Path) -> Path:
    """Create a minimal PEFT adapter directory."""

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


class LoraFolderScanTests(unittest.TestCase):
    """Verify LoRA folder creation and case-insensitive discovery."""

    def test_ensure_lora_folder_creates_preferred_loras_folder(self) -> None:
        """The preferred folder should be named ``Loras``."""

        with tempfile.TemporaryDirectory() as tmp:
            folder = ensure_lora_folder(tmp)
            self.assertEqual(folder.name, "Loras")
            self.assertTrue(folder.is_dir())

    def test_discovers_loras_and_lora_case_insensitively(self) -> None:
        """Scanner should include both folder names regardless of case."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preferred_adapter = _write_peft_adapter(root / "Loras" / "preferred")
            legacy_adapter = _write_peft_adapter(root / "LoRA" / "legacy")

            paths = {Path(item.path).resolve() for item in discover_lora_folder_items(root)}

        self.assertIn(preferred_adapter.resolve(), paths)
        self.assertIn(legacy_adapter.resolve(), paths)

    def test_discovers_training_final_adapter_child(self) -> None:
        """A copied training ``final`` directory should list its adapter child."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_dir = _write_peft_adapter(root / "Loras" / "my_style" / "adapter")
            choices = lora_dropdown_choices(root)
            values = {Path(value).resolve() for _label, value in choices}

        self.assertIn(adapter_dir.resolve(), values)
        self.assertEqual(choices[0], ("None", ""))

    def test_discovers_standalone_safetensors_files(self) -> None:
        """Standalone safetensors files should be listed for LoKr-style artifacts."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_lora_folder(root)
            weights = root / "Loras" / "voice.safetensors"
            weights.write_bytes(b"")
            paths = {Path(item.path).resolve() for item in discover_lora_folder_items(root)}

        self.assertIn(weights.resolve(), paths)

    def test_adapter_marker_files_are_case_insensitive(self) -> None:
        """Linux should discover adapter marker files even when casing differs."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_dir = root / "loras" / "mixed_case"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "ADAPTER_CONFIG.JSON").write_text(
                '{"peft_type": "LORA"}',
                encoding="utf-8",
            )
            (adapter_dir / "ADAPTER_MODEL.SAFETENSORS").write_bytes(b"")
            paths = {Path(item.path).resolve() for item in discover_lora_folder_items(root)}

        self.assertIn(adapter_dir.resolve(), paths)

    def test_resolves_loadable_adapter_paths(self) -> None:
        """The lightweight resolver should accept PEFT dirs and standalone files."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_dir = _write_peft_adapter(root / "Loras" / "voice")
            standalone = root / "Loras" / "style.safetensors"
            standalone.write_bytes(b"")

            self.assertEqual(
                Path(resolve_loadable_lora_adapter_path(adapter_dir)).resolve(),
                adapter_dir.resolve(),
            )
            self.assertEqual(
                Path(resolve_loadable_lora_adapter_path(standalone)).resolve(),
                standalone.resolve(),
            )
            self.assertEqual(resolve_loadable_lora_adapter_path(root / "missing"), "")
            self.assertEqual(resolve_loadable_lora_adapter_path(""), "")


if __name__ == "__main__":
    unittest.main()
