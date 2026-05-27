"""Tests for LoRA/DoRA optimizer and scheduler factories."""

from __future__ import annotations

import unittest

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from acestep.training.optim import build_optimizer, build_scheduler


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
