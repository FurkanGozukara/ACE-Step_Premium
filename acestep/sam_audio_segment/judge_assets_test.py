"""Tests for local SAM-Audio Judge asset preparation."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from acestep.sam_audio_segment.judge_assets import (
    JUDGE_ASSET_FILES,
    JUDGE_BF16_MODEL_NAME,
    JUDGE_MODEL_NAME,
    JUDGE_RUNTIME_CONFIG_NAME,
    JUDGE_TOKENIZER_FILES,
    local_judge_assets_available,
    prepare_local_judge_model_dir,
    resolve_local_judge_checkpoint,
)


class TestJudgeAssets(unittest.TestCase):
    """Verify local SAM-Audio Judge files are prepared without downloads."""

    def test_prepare_local_judge_model_dir_returns_bundled_assets(self):
        """Bundled metadata should be used outside the models folder."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root, assets = _asset_roots(tmpdir)
            _write_model_checkpoint(root)
            _write_bundled_assets(assets)

            model_dir = prepare_local_judge_model_dir(root, assets)

            self.assertEqual(assets, model_dir)
            self.assertTrue((model_dir / JUDGE_RUNTIME_CONFIG_NAME).is_file())
            for name in JUDGE_TOKENIZER_FILES:
                self.assertTrue((model_dir / name).is_file())

    def test_existing_bf16_checkpoint_is_enough(self):
        """Distributed installs should only need the Judge BF16 checkpoint."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bf16 = root / JUDGE_BF16_MODEL_NAME
            bf16.write_bytes(b"bf16")

            self.assertEqual(bf16, resolve_local_judge_checkpoint(root))

    def test_prepare_preserves_official_judge_config(self):
        """Official Judge config keys should be used by the runtime model folder."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root, assets = _asset_roots(tmpdir)
            _write_model_checkpoint(root)
            raw_config = _raw_official_config()
            _write_bundled_assets(assets, config=raw_config)

            model_dir = prepare_local_judge_model_dir(root, assets)
            runtime_config = json.loads(
                (model_dir / JUDGE_RUNTIME_CONFIG_NAME).read_text(encoding="utf-8")
            )

            self.assertIn("dac_vae_encoder", runtime_config)
            self.assertNotIn("audio_codec", runtime_config)
            self.assertEqual(
                64,
                runtime_config["dac_vae_encoder"]["encoder_hidden_size"],
            )
            self.assertEqual(48000, runtime_config["dac_vae_encoder"]["sampling_rate"])

    def test_local_assets_available_uses_bundled_metadata(self):
        """Model root metadata files should not be required for Judge availability."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root, assets = _asset_roots(tmpdir)
            (root / JUDGE_BF16_MODEL_NAME).write_bytes(b"bf16")
            _write_bundled_assets(assets)

            self.assertTrue(local_judge_assets_available(root, assets))

    def test_missing_bundled_metadata_disables_local_judge(self):
        """Judge should not be advertised when bundled metadata is incomplete."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root, assets = _asset_roots(tmpdir)
            (root / JUDGE_BF16_MODEL_NAME).write_bytes(b"bf16")
            (assets / JUDGE_RUNTIME_CONFIG_NAME).write_text(
                "{}",
                encoding="utf-8",
            )

            self.assertFalse(local_judge_assets_available(root, assets))

    def test_fp32_checkpoint_is_converted_to_bf16(self):
        """A local FP32 Judge checkpoint should create and use a BF16 copy."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_file(
                {
                    "float_weight": torch.ones((2, 2), dtype=torch.float32),
                    "int_weight": torch.arange(2, dtype=torch.int64),
                },
                str(root / JUDGE_MODEL_NAME),
            )

            resolved = resolve_local_judge_checkpoint(root)

            self.assertEqual(root / JUDGE_BF16_MODEL_NAME, resolved)
            with safe_open(str(resolved), framework="pt", device="cpu") as handle:
                self.assertEqual("BF16", handle.get_slice("float_weight").get_dtype())
                self.assertEqual("I64", handle.get_slice("int_weight").get_dtype())

    def test_missing_checkpoint_disables_local_judge(self):
        """Judge should not be advertised when the checkpoint is missing."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root, assets = _asset_roots(tmpdir)
            _write_bundled_assets(assets)

            self.assertFalse(local_judge_assets_available(root, assets))


def _asset_roots(tmpdir: str) -> tuple[Path, Path]:
    root = Path(tmpdir) / "models"
    assets = Path(tmpdir) / "assets"
    root.mkdir()
    assets.mkdir()
    return root, assets


def _write_model_checkpoint(root: Path) -> None:
    (root / JUDGE_MODEL_NAME).write_bytes(b"checkpoint")


def _write_bundled_assets(assets: Path, config: dict | None = None) -> None:
    config = config or {}
    (assets / JUDGE_RUNTIME_CONFIG_NAME).write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    for name in JUDGE_ASSET_FILES:
        if name != JUDGE_RUNTIME_CONFIG_NAME:
            (assets / name).write_text("{}", encoding="utf-8")


def _raw_official_config() -> dict:
    return {
        "transformer": {"hidden_size": 1792},
        "finetune_transformer": {"hidden_size": 192},
        "text_model": {"hidden_size": 1024},
        "bottleneck_dim": 256,
        "dac_vae_encoder": {
            "encoder_hidden_size": 64,
            "downsampling_ratios": [2, 8, 10, 12],
            "decoder_hidden_size": 1536,
            "n_codebooks": 16,
            "codebook_size": 1024,
            "codebook_dim": 128,
            "quantizer_dropout": 0,
            "sampling_rate": 48000,
        },
    }


if __name__ == "__main__":
    unittest.main()
