"""Tests for SAM-Audio service model-loading helpers."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from acestep.sam_audio_segment.service import (
    SamAudioService,
    _load_checkpoint_into_model,
    _set_local_ranker_map_location,
    _should_use_meta_direct_sam_load,
)
from acestep.sam_audio_segment.settings import SamAudioSettings
from acestep.torch_compile_runtime import TorchCompileResult


class _AssignAwareModel(torch.nn.Module):
    """Minimal module that records ``load_state_dict`` placement arguments."""

    def __init__(self):
        super().__init__()
        self.assign = None
        self.strict = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Record load options without copying tensors."""

        self.strict = strict
        self.assign = assign


class _LegacyModel(torch.nn.Module):
    """Module stub that simulates older PyTorch without ``assign`` support."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Reject ``assign`` once, then record the fallback call."""

        self.calls.append({"strict": strict, **kwargs})
        if "assign" in kwargs:
            raise TypeError("assign is unsupported")


class ServiceLoadTests(unittest.TestCase):
    """Verify direct checkpoint loading and ranker placement behavior."""

    def test_load_checkpoint_into_model_assigns_cuda_safetensors(self):
        """CUDA safetensors should be loaded on CUDA and assigned into the model."""

        model = _AssignAwareModel()
        with patch(
            "acestep.sam_audio_segment.service.load_checkpoint",
            return_value={"weight": torch.zeros(1)},
        ) as load_checkpoint, patch("torch.cuda.empty_cache"):
            _load_checkpoint_into_model(
                model,
                Path("SAM-Audio-Large-BF16.safetensors"),
                torch.device("cuda"),
            )

        load_checkpoint.assert_called_once()
        self.assertEqual("cuda", load_checkpoint.call_args.kwargs["device"])
        self.assertTrue(model.assign)
        self.assertTrue(model.strict)

    def test_load_checkpoint_into_model_falls_back_when_assign_is_unavailable(self):
        """Older PyTorch builds should retry through a CPU checkpoint load."""

        model = _LegacyModel()
        with patch(
            "acestep.sam_audio_segment.service.load_checkpoint",
            return_value={"weight": torch.zeros(1)},
        ) as load_checkpoint, patch("torch.cuda.empty_cache"):
            _load_checkpoint_into_model(
                model,
                Path("SAM-Audio-Large-BF16.safetensors"),
                torch.device("cuda"),
            )

        devices = [call.kwargs["device"] for call in load_checkpoint.call_args_list]
        self.assertEqual(["cuda", "cpu"], devices)
        self.assertEqual([{"strict": True, "assign": True}, {"strict": True}], model.calls)

    def test_load_checkpoint_into_model_can_skip_direct_cuda_load(self):
        """Low-VRAM lite mode should keep checkpoint tensors on CPU before pruning."""

        model = _AssignAwareModel()
        with patch(
            "acestep.sam_audio_segment.service.load_checkpoint",
            return_value={"weight": torch.zeros(1)},
        ) as load_checkpoint, patch("torch.cuda.empty_cache"):
            _load_checkpoint_into_model(
                model,
                Path("SAM-Audio-Large-BF16.safetensors"),
                torch.device("cuda"),
                direct_device_load=False,
            )

        load_checkpoint.assert_called_once()
        self.assertEqual("cpu", load_checkpoint.call_args.kwargs["device"])
        self.assertFalse(model.assign)
        self.assertTrue(model.strict)

    def test_load_checkpoint_into_model_passes_skip_prefixes(self):
        """Text/span fast loading should forward skipped checkpoint prefixes."""

        model = _AssignAwareModel()
        with patch(
            "acestep.sam_audio_segment.service.load_checkpoint",
            return_value={"weight": torch.zeros(1)},
        ) as load_checkpoint, patch("torch.cuda.empty_cache"):
            _load_checkpoint_into_model(
                model,
                Path("SAM-Audio-Large-BF16.safetensors"),
                torch.device("cuda"),
                skip_prefixes=("vision_encoder.",),
            )

        load_checkpoint.assert_called_once()
        self.assertEqual(("vision_encoder.",), load_checkpoint.call_args.kwargs["skip_prefixes"])

    def test_should_use_meta_direct_sam_load_for_cuda_text_safetensors(self):
        """Text-mode CUDA safetensors can skip real CPU construction."""

        settings = SamAudioSettings(prompt_mode="text", ranker_mode="none")

        self.assertTrue(
            _should_use_meta_direct_sam_load(
                settings,
                model_path=Path("SAM-Audio-Large-BF16.safetensors"),
                device=torch.device("cuda"),
                skip_visual_encoder=True,
            )
        )

    def test_should_not_use_meta_direct_sam_load_for_visual_or_ranker_modes(self):
        """Modes that construct extra external components keep the existing path."""

        self.assertFalse(
            _should_use_meta_direct_sam_load(
                SamAudioSettings(prompt_mode="visual", ranker_mode="none"),
                model_path=Path("SAM-Audio-Large-BF16.safetensors"),
                device=torch.device("cuda"),
                skip_visual_encoder=False,
            )
        )
        self.assertFalse(
            _should_use_meta_direct_sam_load(
                SamAudioSettings(prompt_mode="text", ranker_mode="judge"),
                model_path=Path("SAM-Audio-Large-BF16.safetensors"),
                device=torch.device("cuda"),
                skip_visual_encoder=True,
            )
        )

    def test_set_local_ranker_map_location_updates_nested_judge_rankers(self):
        """Nested Judge rankers should load directly on the service device."""

        config = {
            "text_ranker": {
                "kind": "ensemble",
                "rankers": {
                    "judge": [
                        {"kind": "judge", "checkpoint_path": "Sam-Audio-Judge.safetensors"},
                        1.0,
                    ]
                },
            },
            "visual_ranker": {"kind": "imagebind"},
        }

        _set_local_ranker_map_location(config, torch.device("cuda:1"))

        judge_config = config["text_ranker"]["rankers"]["judge"][0]
        self.assertEqual("cuda:1", judge_config["map_location"])
        self.assertNotIn("map_location", config["visual_ranker"])

    def test_compile_metadata_reports_selective_component_targets(self):
        """Metadata should expose exactly which SAM component was selected."""

        service = SamAudioService(
            SamAudioSettings(compile_model=True),
            model_path=Path("SAM-Audio-Large-BF16.safetensors"),
            device="cpu",
        )
        service.model = torch.nn.Linear(2, 2)
        service.compile_result = TorchCompileResult(
            requested=True,
            compiled=True,
            detail="ready",
            attempts=1,
        )
        setattr(service.model, "_acestep_torch_compiled", True)
        setattr(service.model, "_acestep_torch_compile_attempts", 1)
        setattr(service.model, "_acestep_torch_compile_detail", "verified")

        metadata = service._compile_metadata()

        self.assertTrue(metadata["requested"])
        self.assertTrue(metadata["compiled"])
        self.assertEqual("verified", metadata["detail"])
        self.assertEqual(
            {
                "diffusion_forward": True,
                "codec_encoder": False,
                "codec_decoder": False,
                "text_encoder": False,
                "span_predictor": False,
                "rankers": False,
            },
            metadata["targets"],
        )


if __name__ == "__main__":
    unittest.main()
