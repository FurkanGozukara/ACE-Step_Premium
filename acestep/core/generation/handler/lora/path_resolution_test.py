"""Tests for robust LoRA adapter path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from acestep.core.generation.handler.lora.path_resolution import resolve_lora_input_path


def _make_adapter_dir(root: Path) -> Path:
    """Create a minimal PEFT adapter directory for path-resolution tests."""

    adapter_dir = root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    return adapter_dir


class LoraPathResolutionTests(unittest.TestCase):
    """Verify native absolute, relative, quoted, and mixed-separator paths."""

    def test_resolves_absolute_adapter_directory(self) -> None:
        """Native absolute paths should resolve directly."""

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = _make_adapter_dir(Path(tmp))
            result = resolve_lora_input_path(str(adapter_dir))
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())

    def test_resolves_quoted_adapter_directory(self) -> None:
        """Pasted paths wrapped in quotes should resolve."""

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = _make_adapter_dir(Path(tmp))
            result = resolve_lora_input_path(f'"{adapter_dir}"')
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())

    def test_resolves_relative_path_from_current_working_directory(self) -> None:
        """Relative paths should resolve from the application working directory."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            adapter_dir = _make_adapter_dir(Path(tmp))
            relative_path = os.path.relpath(adapter_dir, Path.cwd())
            result = resolve_lora_input_path(relative_path)
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())

    def test_resolves_parent_final_directory_to_adapter_child(self) -> None:
        """Training output parent directories should resolve to their adapter child."""

        with tempfile.TemporaryDirectory() as tmp:
            final_dir = Path(tmp) / "final"
            adapter_dir = _make_adapter_dir(final_dir)
            result = resolve_lora_input_path(str(final_dir))
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())

    def test_resolves_adapter_weight_file_to_parent_adapter_directory(self) -> None:
        """PEFT weight files should resolve to the containing adapter directory."""

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = _make_adapter_dir(Path(tmp))
            result = resolve_lora_input_path(str(adapter_dir / "adapter_model.safetensors"))
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())

    def test_resolves_mixed_separator_path(self) -> None:
        """Common Windows/Linux separator paste differences should resolve."""

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = _make_adapter_dir(Path(tmp))
            native_path = str(adapter_dir)
            pasted_path = (
                native_path.replace("\\", "/")
                if os.name == "nt"
                else native_path.replace("/", "\\")
            )
            result = resolve_lora_input_path(pasted_path)
            self.assertEqual(Path(result.resolved_path), adapter_dir.resolve())


if __name__ == "__main__":
    unittest.main()
