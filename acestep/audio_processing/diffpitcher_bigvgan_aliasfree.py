"""Alias-free activation modules required by the DiffPitcher BigVGAN."""

from __future__ import annotations

import math

import torch


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """Return torch.sinc with a fallback for older torch builds."""

    if hasattr(torch, "sinc"):
        return torch.sinc(x)
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Create a Kaiser-windowed low-pass sinc filter."""

    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    a_value = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a_value > 50.0:
        beta = 0.1102 * (a_value - 8.7)
    elif a_value >= 21.0:
        beta = 0.5842 * (a_value - 21) ** 0.4 + 0.07886 * (a_value - 21)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(torch.nn.Module):
    """Depthwise low-pass filter used for anti-aliased downsampling."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Low-pass filter a channel-first 1D signal."""

        _, channels, _ = x.shape
        x = torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return torch.nn.functional.conv1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class UpSample1d(torch.nn.Module):
    """Alias-free 1D upsampler."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample a channel-first 1D signal."""

        _, channels, _ = x.shape
        x = torch.nn.functional.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * torch.nn.functional.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(torch.nn.Module):
    """Alias-free 1D downsampler."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        kernel = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample a channel-first 1D signal."""

        return self.lowpass(x)


class SnakeBeta(torch.nn.Module):
    """SnakeBeta periodic activation used by the bundled BigVGAN."""

    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        initializer = torch.zeros if alpha_logscale else torch.ones
        self.alpha = torch.nn.Parameter(initializer(in_features) * alpha)
        self.beta = torch.nn.Parameter(initializer(in_features) * alpha)
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SnakeBeta to a channel-first 1D signal."""

        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.sin(x * alpha).pow(2)


class Activation1d(torch.nn.Module):
    """Alias-free wrapper around the periodic activation."""

    def __init__(self, activation: torch.nn.Module) -> None:
        super().__init__()
        self.up_ratio = 2
        self.down_ratio = 2
        self.act = activation
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsample, activation, and downsample."""

        return self.downsample(self.act(self.upsample(x)))
