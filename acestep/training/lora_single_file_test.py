"""Tests for single-file PEFT LoRA artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from acestep.training.lora_checkpoint import save_lora_weights
from acestep.training.lora_single_file import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_MODEL_SAFETENSORS_FILENAME,
    is_peft_lora_single_file,
    materialize_peft_lora_single_file,
    save_peft_lora_single_file,
)
from acestep.training.path_safety import get_safe_roots, set_safe_roots


class _FakeDecoder:
    """Minimal decoder that writes PEFT-like adapter files."""

    def save_pretrained(self, path: str) -> None:
        """Write adapter config and weights to ``path``."""

        adapter_dir = Path(path)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / ADAPTER_CONFIG_FILENAME).write_text(
            json.dumps({"peft_type": "LORA", "target_modules": ["q_proj"]}),
            encoding="utf-8",
        )
        save_file(
            {"base_model.model.q_proj.lora_A.default.weight": torch.ones(1, 1)},
            str(adapter_dir / ADAPTER_MODEL_SAFETENSORS_FILENAME),
            metadata={"format": "pt"},
        )


class _FakeModel:
    """Minimal model object exposing a PEFT-like decoder."""

    decoder = _FakeDecoder()


class LoraSingleFileTests(unittest.TestCase):
    """Verify combined LoRA safetensors files include config and weights."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_single_file_round_trips_to_peft_adapter_directory(self) -> None:
        """Saved single-file artifacts should expand to PEFT-compatible files."""

        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapter"
            adapter_dir.mkdir()
            config = {
                "peft_type": "LORA",
                "base_model_name_or_path": None,
                "target_modules": ["q_proj"],
            }
            (adapter_dir / ADAPTER_CONFIG_FILENAME).write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            save_file(
                {"base_model.model.q_proj.lora_A.default.weight": torch.ones(1, 1)},
                str(adapter_dir / ADAPTER_MODEL_SAFETENSORS_FILENAME),
                metadata={"format": "pt"},
            )

            single_file = Path(tmp) / "my awesome song-32.safetensors"
            save_peft_lora_single_file(adapter_dir, single_file)

            self.assertTrue(is_peft_lora_single_file(single_file))
            with materialize_peft_lora_single_file(single_file) as materialized:
                materialized_path = Path(materialized)
                self.assertTrue((materialized_path / ADAPTER_CONFIG_FILENAME).is_file())
                self.assertTrue(
                    (materialized_path / ADAPTER_MODEL_SAFETENSORS_FILENAME).is_file()
                )
                loaded = load_file(
                    str(materialized_path / ADAPTER_MODEL_SAFETENSORS_FILENAME),
                    device="cpu",
                )

        self.assertIn("base_model.model.q_proj.lora_A.default.weight", loaded)

    def test_save_lora_weights_writes_named_single_file(self) -> None:
        """Checkpoint saves should include a named single-file artifact."""

        with tempfile.TemporaryDirectory() as tmp:
            set_safe_roots([tmp])
            output_dir = Path(tmp) / "checkpoint"

            save_lora_weights(
                _FakeModel(),
                str(output_dir),
                artifact_name="my awesome song-32",
            )

            artifact = output_dir / "my awesome song-32.safetensors"
            self.assertTrue(artifact.is_file())
            self.assertTrue(is_peft_lora_single_file(artifact))

    def test_save_lora_weights_can_skip_adapter_directory(self) -> None:
        """Flat training saves should write only the named safetensors file."""

        with tempfile.TemporaryDirectory() as tmp:
            set_safe_roots([tmp])
            output_dir = Path(tmp) / "run"

            saved_path = save_lora_weights(
                _FakeModel(),
                str(output_dir),
                artifact_name="my awesome song-epoch-32",
                save_adapter=False,
            )

            artifact = output_dir / "my awesome song-epoch-32.safetensors"
            self.assertEqual(str(artifact), saved_path)
            self.assertTrue(artifact.is_file())
            self.assertFalse((output_dir / "adapter").exists())


if __name__ == "__main__":
    unittest.main()
