"""Tests for batch-management helper behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.ui.gradio.events.results.batch_management_helpers import (
    apply_lora_selection_for_generation,
    resolve_effective_lora_path,
)


def _write_peft_adapter(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


class _FakeDitHandler:
    def __init__(self) -> None:
        self.lora_loaded = False
        self.loaded_paths: list[str] = []
        self.unload_calls = 0
        self.scales: list[float] = []
        self.use_flags: list[bool] = []

    def load_lora(self, path: str) -> str:
        self.lora_loaded = True
        self.loaded_paths.append(path)
        return "LoRA loaded"

    def unload_lora(self) -> str:
        self.lora_loaded = False
        self.unload_calls += 1
        return "LoRA unloaded"

    def set_lora_scale(self, scale: float) -> str:
        self.scales.append(scale)
        return "scale set"

    def set_use_lora(self, use_lora: bool) -> str:
        self.use_flags.append(use_lora)
        return "use flag set"


class BatchManagementHelperTests(unittest.TestCase):
    """Verify automatic LoRA state synchronization for generation."""

    def test_resolve_effective_lora_path_prefers_manual_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = _write_peft_adapter(root / "manual")
            dropdown = _write_peft_adapter(root / "dropdown")

            resolved = resolve_effective_lora_path(str(manual), str(dropdown))

        self.assertEqual(Path(resolved).resolve(), manual.resolve())

    def test_apply_lora_selection_loads_and_enables_valid_lora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _write_peft_adapter(Path(tmp) / "voice")
            handler = _FakeDitHandler()

            resolved, use_lora, status = apply_lora_selection_for_generation(
                handler,
                "",
                str(adapter),
                0.75,
            )

        self.assertTrue(use_lora)
        self.assertEqual(Path(resolved).resolve(), adapter.resolve())
        self.assertEqual(Path(handler.loaded_paths[0]).resolve(), adapter.resolve())
        self.assertEqual(handler.scales, [0.75])
        self.assertEqual(handler.use_flags, [True])
        self.assertIn("Next run will use LoRA:", status)

    def test_apply_lora_selection_unloads_when_selection_is_empty(self) -> None:
        handler = _FakeDitHandler()
        handler.lora_loaded = True
        handler._auto_lora_path = "old"

        resolved, use_lora, status = apply_lora_selection_for_generation(
            handler,
            "",
            "",
            1.0,
        )

        self.assertEqual(resolved, "")
        self.assertFalse(use_lora)
        self.assertEqual(handler.unload_calls, 1)
        self.assertEqual(handler._auto_lora_path, "")
        self.assertEqual(status, "No LoRA will be used.")


if __name__ == "__main__":
    unittest.main()
