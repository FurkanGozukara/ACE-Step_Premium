"""Tests for LoRA training run wrapper prompt handling."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.wiring import training_lora_run_wrapper as wrapper_mod


def _wrapper_args(
    *,
    sample_generation_enabled: bool = True,
    sample_prompt: str = "prompt",
    sample_lyrics: str = "lyrics",
) -> list[object]:
    """Return positional args for the training wrapper."""

    return [
        "tensor_dir",
        "lora_name",
        "lora",
        16,
        16,
        0.0,
        "all",
        0.0001,
        1,
        1,
        1,
        1,
        False,
        0,
        1,
        0.0,
        3.0,
        8,
        42,
        "adamw",
        0.01,
        0.9,
        0.999,
        1e-8,
        4096,
        100,
        True,
        False,
        1e-30,
        1e-3,
        1.0,
        -0.8,
        0.0,
        False,
        False,
        False,
        "constant",
        "uniform",
        0,
        0,
        "output",
        "",
        False,
        False,
        False,
        False,
        False,
        "none",
        0,
        sample_generation_enabled,
        10,
        sample_prompt,
        sample_lyrics,
        123,
        False,
        True,
        False,
        "model",
        "tier",
        {},
        -1.0,
        8,
        "model",
    ]


class TrainingLoraRunWrapperWildcardTests(unittest.TestCase):
    """Validate wildcard handling for checkpoint sample prompts."""

    def test_sample_prompt_wildcards_expand_before_training(self) -> None:
        captured = {}

        def fake_start_training(*_args, **kwargs):
            captured["sample_prompt"] = kwargs["sample_prompt"]
            captured["sample_lyrics"] = kwargs["sample_lyrics"]
            yield "done", "log", None, {"cancelled": False}

        training_wrapper = wrapper_mod.build_lora_training_wrapper(
            None,
            normalize_training_state=lambda _state: {"cancelled": False},
            sample_setting_keys=("audio_duration", "inference_steps", "config_path"),
        )
        args = _wrapper_args(
            sample_prompt="{warm|bright} sample",
            sample_lyrics="[Verse]\nI feel {alive|free}",
        )

        with patch.object(wrapper_mod.train_h, "start_training", fake_start_training), patch.object(
            wrapper_mod,
            "prepare_parent_runtime_for_training",
            return_value="",
        ):
            outputs = list(training_wrapper(*args))

        self.assertIn("start requested", outputs[0][0])
        self.assertEqual(outputs[-1][0], "done")
        self.assertIn(captured["sample_prompt"], {"warm sample", "bright sample"})
        self.assertIn(
            captured["sample_lyrics"],
            {"[Verse]\nI feel alive", "[Verse]\nI feel free"},
        )

    def test_invalid_sample_wildcard_warns_and_skips_training(self) -> None:
        def fake_start_training(*_args, **_kwargs):
            raise AssertionError("start_training should not run")
            yield "done", "log", None, {"cancelled": False}

        training_wrapper = wrapper_mod.build_lora_training_wrapper(
            None,
            normalize_training_state=lambda _state: {"cancelled": False},
            sample_setting_keys=("audio_duration", "inference_steps", "config_path"),
        )
        args = _wrapper_args(sample_prompt="bad {warm|bright")

        with patch.object(wrapper_mod.train_h, "start_training", fake_start_training), patch.object(
            wrapper_mod,
            "prepare_parent_runtime_for_training",
            return_value="",
        ):
            outputs = list(training_wrapper(*args))

        self.assertEqual(len(outputs), 2)
        self.assertIn("start requested", outputs[0][0])
        self.assertIn("Wildcard syntax error", outputs[1][0])
        self.assertIn("Missing closing }", outputs[1][0])

    def test_disabled_sample_generation_does_not_validate_sample_wildcards(self) -> None:
        captured = {}

        def fake_start_training(*_args, **kwargs):
            captured["sample_prompt"] = kwargs["sample_prompt"]
            yield "done", "log", None, {"cancelled": False}

        training_wrapper = wrapper_mod.build_lora_training_wrapper(
            None,
            normalize_training_state=lambda _state: {"cancelled": False},
            sample_setting_keys=("audio_duration", "inference_steps", "config_path"),
        )
        args = _wrapper_args(
            sample_generation_enabled=False,
            sample_prompt="unused {warm|bright",
        )

        with patch.object(wrapper_mod.train_h, "start_training", fake_start_training), patch.object(
            wrapper_mod,
            "prepare_parent_runtime_for_training",
            return_value="",
        ):
            outputs = list(training_wrapper(*args))

        self.assertIn("start requested", outputs[0][0])
        self.assertEqual(outputs[-1][0], "done")
        self.assertEqual(captured["sample_prompt"], "unused {warm|bright")


if __name__ == "__main__":
    unittest.main()
