"""BigVGAN generator used by DiffPitcher inference."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch.nn.utils import remove_weight_norm, weight_norm

from .diffpitcher_bigvgan_aliasfree import Activation1d, SnakeBeta


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return same-length 1D convolution padding."""

    return int((kernel_size * dilation - dilation) / 2)


def _init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize convolution weights before checkpoint loading."""

    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


class AMPBlock1(torch.nn.Module):
    """BigVGAN AMP residual block with three dilation branches."""

    def __init__(
        self,
        h: SimpleNamespace,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.h = h
        self.convs1 = torch.nn.ModuleList([
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[0], padding=_get_padding(kernel_size, dilation[0]),
            )),
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[1], padding=_get_padding(kernel_size, dilation[1]),
            )),
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation[2], padding=_get_padding(kernel_size, dilation[2]),
            )),
        ])
        self.convs1.apply(_init_weights)
        self.convs2 = torch.nn.ModuleList([
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=_get_padding(kernel_size, 1),
            )),
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=_get_padding(kernel_size, 1),
            )),
            weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=_get_padding(kernel_size, 1),
            )),
        ])
        self.convs2.apply(_init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = torch.nn.ModuleList([
            Activation1d(SnakeBeta(channels, alpha_logscale=h.snake_logscale))
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the AMP residual block."""

        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = conv2(act2(conv1(act1(x))))
            x = residual + x
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all convolutions."""

        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class BigVGAN(torch.nn.Module):
    """BigVGAN generator matching the DiffPitcher 24 kHz 100-band checkpoint."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        h = SimpleNamespace(**config)
        if h.activation != "snakebeta" or h.resblock != "1":
            raise ValueError("Only the bundled DiffPitcher BigVGAN config is supported.")
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            torch.nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        self.ups = torch.nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(torch.nn.ModuleList([
                weight_norm(torch.nn.ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** index),
                    h.upsample_initial_channel // (2 ** (index + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                ))
            ]))
        self.resblocks = torch.nn.ModuleList()
        for index in range(len(self.ups)):
            channels = h.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(h, channels, kernel, tuple(dilation)))
        self.activation_post = Activation1d(
            SnakeBeta(channels, alpha_logscale=h.snake_logscale)
        )
        self.conv_post = weight_norm(torch.nn.Conv1d(channels, 1, 7, 1, padding=3))
        for layer in self.ups:
            layer.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate waveform samples from log mel features."""

        x = self.conv_pre(x)
        for index in range(self.num_upsamples):
            for upsampler in self.ups[index]:
                x = upsampler(x)
            combined = None
            for kernel_index in range(self.num_kernels):
                block = self.resblocks[index * self.num_kernels + kernel_index]
                output = block(x)
                combined = output if combined is None else combined + output
            x = combined / self.num_kernels
        return torch.tanh(self.conv_post(self.activation_post(x)))

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from inference convolutions."""

        for layer_group in self.ups:
            for layer in layer_group:
                remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
