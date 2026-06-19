"""Tests for LoRA/DoRA optimizer and scheduler factories."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from acestep.training.optim import (
    build_optimizer,
    build_scheduler,
    optimizer_hyperparameter_defaults,
    optimizer_hyperparameter_visible,
)


class OptimizerFactoryTests(unittest.TestCase):
    """Verify selected optimizer keys create usable optimizers."""

    def test_adamw8bit_falls_back_to_adamw_on_cpu(self) -> None:
        """The CUDA-only 8-bit optimizer should fall back cleanly on CPU."""

        param = torch.nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [param],
            optimizer_type="adamw8bit",
            lr=1e-4,
            weight_decay=0.0,
            device_type="cpu",
        )

        self.assertIsInstance(optimizer, AdamW)

    def test_adamw_uses_visible_hyperparameters(self) -> None:
        """AdamW should receive the betas, epsilon, and weight decay from UI values."""

        param = torch.nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [param],
            optimizer_type="adamw",
            lr=1e-4,
            weight_decay=0.02,
            adam_beta1=0.85,
            adam_beta2=0.995,
            adam_epsilon=1e-7,
            device_type="cpu",
        )
        group = optimizer.param_groups[0]

        self.assertEqual((0.85, 0.995), group["betas"])
        self.assertEqual(1e-7, group["eps"])
        self.assertEqual(0.02, group["weight_decay"])

    def test_optimizer_defaults_are_optimizer_specific(self) -> None:
        """Adafactor should not inherit AdamW's weight-decay default."""

        self.assertEqual(0.01, optimizer_hyperparameter_defaults("adamw")["weight_decay"])
        self.assertEqual(
            0.01,
            optimizer_hyperparameter_defaults("adamw8bit")["weight_decay"],
        )
        self.assertEqual(
            0.0,
            optimizer_hyperparameter_defaults("adafactor")["weight_decay"],
        )

    def test_optimizer_hyperparameter_visibility_is_optimizer_specific(self) -> None:
        """Shared visibility rules should match the selected optimizer."""

        self.assertTrue(optimizer_hyperparameter_visible("adamw", "weight_decay"))
        self.assertTrue(optimizer_hyperparameter_visible("adamw", "adam_beta1"))
        self.assertFalse(
            optimizer_hyperparameter_visible("adamw", "adamw8bit_min_8bit_size")
        )
        self.assertFalse(
            optimizer_hyperparameter_visible("adamw", "adafactor_relative_step")
        )
        self.assertTrue(
            optimizer_hyperparameter_visible("adamw8bit", "adamw8bit_min_8bit_size")
        )
        self.assertTrue(
            optimizer_hyperparameter_visible("adafactor", "adafactor_relative_step")
        )
        self.assertFalse(optimizer_hyperparameter_visible("adafactor", "adam_beta1"))

    def test_adafactor_choice_creates_step_capable_optimizer(self) -> None:
        """Adafactor selection should create an optimizer with a step method."""

        param = torch.nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [param],
            optimizer_type="adafactor",
            lr=1e-4,
            weight_decay=0.0,
            device_type="cpu",
        )

        self.assertTrue(callable(getattr(optimizer, "step", None)))

    def test_adafactor_uses_supplied_hyperparameters(self) -> None:
        """Adafactor should receive the values supplied by the training UI."""

        param = torch.nn.Parameter(torch.ones(1))
        expected_optimizer = object()
        with patch("transformers.optimization.Adafactor") as adafactor_cls:
            adafactor_cls.return_value = expected_optimizer
            optimizer = build_optimizer(
                [param],
                optimizer_type="adafactor",
                lr=2e-4,
                weight_decay=0.03,
                device_type="cpu",
                adafactor_epsilon1=1e-28,
                adafactor_epsilon2=0.002,
                adafactor_clip_threshold=0.75,
                adafactor_decay_rate=-0.7,
                adafactor_beta1=0.1,
                adafactor_scale_parameter=True,
                adafactor_relative_step=False,
                adafactor_warmup_init=False,
            )

        self.assertIs(expected_optimizer, optimizer)
        adafactor_cls.assert_called_once()
        kwargs = adafactor_cls.call_args.kwargs
        self.assertEqual(2e-4, kwargs["lr"])
        self.assertEqual((1e-28, 0.002), kwargs["eps"])
        self.assertEqual(0.75, kwargs["clip_threshold"])
        self.assertEqual(-0.7, kwargs["decay_rate"])
        self.assertEqual(0.1, kwargs["beta1"])
        self.assertEqual(0.03, kwargs["weight_decay"])
        self.assertTrue(kwargs["scale_parameter"])
        self.assertFalse(kwargs["relative_step"])
        self.assertFalse(kwargs["warmup_init"])


class SchedulerFactoryTests(unittest.TestCase):
    """Verify selected scheduler keys create usable schedulers."""

    def test_all_scheduler_choices_can_step(self) -> None:
        """Each exposed scheduler should step without changing the API shape."""

        for scheduler_type in [
            "cosine",
            "cosine_restarts",
            "linear",
            "constant",
            "constant_with_warmup",
        ]:
            with self.subTest(scheduler_type=scheduler_type):
                param = torch.nn.Parameter(torch.ones(1))
                optimizer = AdamW([param], lr=1e-4)
                scheduler = build_scheduler(
                    optimizer,
                    scheduler_type=scheduler_type,
                    total_steps=10,
                    warmup_steps=1,
                    lr=1e-4,
                )
                optimizer.step()
                scheduler.step()

                self.assertEqual(1, len(scheduler.get_last_lr()))

    def test_unknown_scheduler_defaults_to_constant(self) -> None:
        """Unknown scheduler keys should use the constant default."""

        param = torch.nn.Parameter(torch.ones(1))
        optimizer = AdamW([param], lr=1e-4)
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="unknown",
            total_steps=10,
            warmup_steps=1,
            lr=1e-4,
        )

        self.assertIsInstance(scheduler, ConstantLR)


if __name__ == "__main__":
    unittest.main()
