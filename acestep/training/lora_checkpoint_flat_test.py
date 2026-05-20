"""Tests for flat LoRA checkpoint save and resume files."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from acestep.training.lora_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from acestep.training.lora_single_file import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_MODEL_SAFETENSORS_FILENAME,
    is_peft_lora_single_file,
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


class _Stateful:
    """Small optimizer/scheduler stand-in with loadable state."""

    def __init__(self) -> None:
        """Initialize captured load state."""

        self.loaded = None

    def state_dict(self) -> dict:
        """Return a torch-loadable state dict."""

        return {"state": {}, "param_groups": []}

    def load_state_dict(self, state: dict) -> None:
        """Capture loaded state."""

        self.loaded = state


class LoraFlatCheckpointTests(unittest.TestCase):
    """Verify new LoRA checkpoints stay flat in the named run folder."""

    def setUp(self) -> None:
        """Preserve safe-root configuration."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore safe-root configuration."""

        set_safe_roots(self._safe_roots)

    def test_checkpoint_saves_flat_safetensors_and_resume_state(self) -> None:
        """A checkpoint should not create adapter/final/checkpoints folders."""

        with tempfile.TemporaryDirectory() as tmp:
            set_safe_roots([tmp])
            optimizer = _Stateful()
            scheduler = _Stateful()

            state_path = save_training_checkpoint(
                _FakeModel(),
                optimizer,
                scheduler,
                epoch=3,
                global_step=12,
                output_dir=tmp,
                artifact_name="2pac_SFT-test1-epoch-3",
            )

            safetensors_path = os.path.join(tmp, "2pac_SFT-test1-epoch-3.safetensors")

            self.assertEqual(
                os.path.join(tmp, "epoch-3-training_resume_state.pt"),
                state_path,
            )
            self.assertTrue(os.path.isfile(safetensors_path))
            self.assertTrue(is_peft_lora_single_file(safetensors_path))
            self.assertFalse(os.path.exists(os.path.join(tmp, "adapter")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "checkpoints")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "final")))

            loaded = load_training_checkpoint(
                state_path,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
            )

        self.assertEqual(3, loaded["epoch"])
        self.assertEqual(12, loaded["global_step"])
        self.assertEqual(os.path.realpath(safetensors_path), loaded["adapter_path"])
        self.assertTrue(loaded["loaded_optimizer"])
        self.assertTrue(loaded["loaded_scheduler"])


if __name__ == "__main__":
    unittest.main()
