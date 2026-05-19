"""Tests for LoRA training checkpoint sample generation helpers."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from acestep.training.sample_generation import run_sample_subprocess
from acestep.training.sample_generation_cli import _serializable_audios


class SampleGenerationTests(unittest.TestCase):
    """Verify sample generation result metadata stays JSON-safe."""

    def test_serializable_audios_drops_tensor_payloads(self) -> None:
        """Generated tensors are omitted from subprocess result JSON."""

        audios = [
            {
                "path": "sample.flac",
                "key": "abc",
                "sample_rate": 48000,
                "tensor": torch.zeros(1, 10),
            }
        ]

        result = _serializable_audios(audios)

        self.assertEqual(result, [{"path": "sample.flac", "key": "abc", "sample_rate": 48000}])
        json.dumps(result)

    def test_run_sample_subprocess_invokes_cli_module(self) -> None:
        """The parent helper launches the focused child-process CLI module."""

        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            with patch("acestep.training.sample_generation.subprocess.run", return_value=completed) as run:
                run_sample_subprocess(
                    project_root=tmp,
                    config_path="ACEStep_1_5_XL_Turbo_BF16",
                    device="cuda",
                    checkpoint_dir=tmp,
                    output_dir=tmp,
                    prompt="style",
                    lyrics="lyrics",
                    duration=10,
                    inference_steps=4,
                    seed=42,
                    offload_generation=True,
                )

        command = run.call_args.args[0]
        self.assertIn("acestep.training.sample_generation_cli", command)


if __name__ == "__main__":
    unittest.main()
