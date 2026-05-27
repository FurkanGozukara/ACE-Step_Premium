"""Tests for generated repaint-source latent persistence."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from acestep.ui.gradio.events.results.generation_progress import (
    _extract_repaint_source_latents,
    _persist_repaint_source_latents,
    _run_auto_lrc,
    _strip_extra_output_tensors,
)


class RepaintSourceLatentPersistenceTests(unittest.TestCase):
    """Verify generated audio sidecars get a reusable repaint latent pointer."""

    def test_persist_repaint_source_latents_writes_file_and_updates_params(self):
        """The helper should store generated latents beside the sidecar JSON."""
        audio_params = {}
        with tempfile.TemporaryDirectory() as tmp:
            json_path = str(Path(tmp) / "sample.json")

            _persist_repaint_source_latents(
                source_latents=torch.ones(4, 3),
                json_path=json_path,
                audio_params=audio_params,
            )

            latent_name = audio_params["repaint_source_latents_file"]
            latent_path = Path(tmp) / latent_name
            self.assertTrue(latent_path.exists())
            self.assertEqual((4, 3), np.load(latent_path).shape)

    def test_extract_repaint_source_latents_uses_pred_latents_sample(self):
        """The persisted source should come from DiT pred_latents, not audio."""
        pred_latents = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        sample = _extract_repaint_source_latents({"pred_latents": pred_latents}, 1)

        torch.testing.assert_close(sample, pred_latents[1])

    def test_strip_extra_output_tensors_preserves_metadata(self):
        """Batch queue storage should keep metadata but not large tensors."""
        stripped = _strip_extra_output_tensors({
            "pred_latents": torch.ones(1, 2, 3),
            "seed_value": "123",
            "lrcs": ["[00:00.00] hello"],
        })

        self.assertNotIn("pred_latents", stripped)
        self.assertEqual("123", stripped["seed_value"])
        self.assertEqual(["[00:00.00] hello"], stripped["lrcs"])

    def test_run_auto_lrc_writes_lrc_and_vtt_beside_sample_json(self):
        """Auto LRC should persist subtitle files in the generation run folder."""

        class _FakeDitHandler:
            """Return a deterministic LRC payload."""

            def get_lyric_timestamp(self, **_kwargs):
                return {"success": True, "lrc_text": "[00:00.00]hello"}

        extra_outputs = {
            "pred_latents": torch.ones(1, 50, 4),
            "encoder_hidden_states": torch.ones(1, 3, 4),
            "encoder_attention_mask": torch.ones(1, 3),
            "context_latents": torch.ones(1, 50, 4),
            "lyric_token_idss": torch.ones(1, 3, dtype=torch.long),
        }
        final_lrcs = [""]
        final_lrc_paths = [None]
        final_subtitles = [None]

        with tempfile.TemporaryDirectory() as tmp:
            json_path = str(Path(tmp) / "sample.json")

            _run_auto_lrc(
                _FakeDitHandler(),
                extra_outputs,
                0,
                2.0,
                "en",
                8,
                json_path,
                final_lrcs,
                final_lrc_paths,
                final_subtitles,
            )

            lrc_path = Path(tmp) / "sample.lrc"
            vtt_path = Path(tmp) / "sample.vtt"
            self.assertEqual("[00:00.00]hello", lrc_path.read_text(encoding="utf-8"))
            self.assertTrue(vtt_path.exists())
            self.assertEqual(str(lrc_path).replace("\\", "/"), final_lrc_paths[0])
            self.assertEqual(str(vtt_path).replace("\\", "/"), final_subtitles[0])


if __name__ == "__main__":
    unittest.main()
