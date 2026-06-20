"""Tests for direct CUDA SAM-Audio Judge checkpoint loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from acestep.sam_audio_vendor import ensure_vendor_path

ensure_vendor_path()

from sam_audio.model import base as base_module  # noqa: E402
from sam_audio.model.base import BaseModel, _materialize_meta_buffers  # noqa: E402


class _TinyConfig:
    def __init__(self, size: int = 2):
        self.size = size


class _TinyCheckpointModel(BaseModel):
    """Small BaseModel subclass that records construction device placement."""

    config_cls = _TinyConfig
    revision = None
    constructed_devices: list[str] = []

    def __init__(self, config: _TinyConfig):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.size))
        self.constructed_devices.append(self.weight.device.type)


class _FakeRotary(torch.nn.Module):
    """Minimal rotary-like module with a derived non-persistent buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "freqs_cis",
            torch.empty(2, device="meta"),
            persistent=False,
        )

    def reset_parameters(self):
        self.register_buffer("freqs_cis", torch.ones(2), persistent=False)


class JudgeDirectLoadTests(unittest.TestCase):
    """Verify direct safetensors loads skip normal parameter construction."""

    def test_cuda_safetensors_constructs_model_on_meta_before_assign(self) -> None:
        """Direct CUDA safetensors should avoid real CPU/GPU init tensors."""

        _TinyCheckpointModel.constructed_devices = []
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(json.dumps({"size": 2}))
            (model_dir / "model.safetensors").write_bytes(b"placeholder")

            with patch.object(
                base_module,
                "_load_checkpoint",
                return_value={"weight": torch.ones(2)},
            ), patch.object(base_module, "_materialize_meta_buffers") as materialize:
                model = _TinyCheckpointModel.from_pretrained(
                    str(model_dir),
                    map_location="cuda",
                )

        self.assertEqual(["meta"], _TinyCheckpointModel.constructed_devices)
        self.assertFalse(model.weight.is_meta)
        self.assertTrue(torch.equal(model.weight, torch.ones(2)))
        materialize.assert_called_once()

    def test_materialize_meta_buffers_recreates_rotary_buffers(self) -> None:
        """Derived rotary buffers left by meta construction should be real tensors."""

        model = torch.nn.Sequential(_FakeRotary())

        _materialize_meta_buffers(model, "cpu")

        freqs = model[0].freqs_cis
        self.assertFalse(freqs.is_meta)
        self.assertEqual(torch.device("cpu"), freqs.device)
        self.assertTrue(torch.equal(freqs, torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
