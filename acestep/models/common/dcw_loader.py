"""Lazy loader for ``pytorch_wavelets`` DWT1D modules used by DCW.

Splitting this out keeps :mod:`acestep.models.common.dcw_correction` under
the project's 200-LOC module cap.

We import ``pytorch_wavelets`` lazily so the dependency stays optional —
DCW is opt-in, and users who don't enable it never touch this code path.
If the import fails when DCW *is* enabled, we log a single warning and
return ``None`` so callers can short-circuit to a no-op.
"""

from __future__ import annotations

import importlib.resources
import sys
from types import ModuleType
from typing import Optional, Tuple

import torch
from loguru import logger


def _resource_stream(package: str, resource_name: str):
    """Provide the one legacy ``pkg_resources`` API used by pytorch_wavelets."""

    return importlib.resources.files(package).joinpath(resource_name).open("rb")


def _install_pkg_resources_compat() -> ModuleType | None:
    """Temporarily bridge pytorch_wavelets on setuptools without pkg_resources."""

    try:
        __import__("pkg_resources")
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
    else:
        return None

    compat = ModuleType("pkg_resources")
    compat.resource_stream = _resource_stream  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = compat
    return compat


def _import_dwt1d_modules():
    """Import DWT1D while accommodating pytorch_wavelets' legacy loader."""

    compat = _install_pkg_resources_compat()
    try:
        from pytorch_wavelets import DWT1DForward, DWT1DInverse
    finally:
        if compat is not None and sys.modules.get("pkg_resources") is compat:
            sys.modules.pop("pkg_resources", None)
    return DWT1DForward, DWT1DInverse


class _LazyWavelet:
    """Lazy loader for ``pytorch_wavelets`` DWT1D modules.

    We cache one ``DWT1DForward`` / ``DWT1DInverse`` pair per
    ``(device, dtype, wavelet)`` triple so repeated sampler steps don't
    keep rebuilding the filter banks.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        self._import_failed = False

    def _try_import(self):
        if self._import_failed:
            return None
        try:
            return _import_dwt1d_modules()
        except ImportError as exc:
            self._import_failed = True
            logger.warning(
                "DCW is enabled but pytorch_wavelets could not be imported "
                "({}: {}). Install or repair `pytorch_wavelets PyWavelets` "
                "to use Differential Correction in Wavelet domain. Falling "
                "back to no-op for this generation.",
                type(exc).__name__,
                exc,
            )
            return None

    def get(
        self,
        device: torch.device,
        dtype: torch.dtype,
        wavelet: str,
    ) -> Optional[Tuple["torch.nn.Module", "torch.nn.Module"]]:
        """Return ``(dwt, iwt)`` for the requested device/dtype/wavelet.

        Returns ``None`` when ``pytorch_wavelets`` is missing — callers
        must treat this as a "skip the correction" signal.
        """
        modules = self._try_import()
        if modules is None:
            return None
        DWT1DForward, DWT1DInverse = modules
        key = (str(device), str(dtype), wavelet)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # DCW's math is numerically sensitive; always run the DWT in fp32 on
        # the latent's device and cast results back to the caller's dtype.
        dwt = DWT1DForward(J=1, mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        iwt = DWT1DInverse(mode="zero", wave=wavelet).to(device=device, dtype=torch.float32)
        self._cache[key] = (dwt, iwt)
        # One-shot confirmation that the requested basis actually got built.
        # Log filter length too so users can see haar(2-tap) vs db4(8-tap)
        # vs sym8(16-tap) — this is where the "different wavelets, same
        # output?" claim gets falsified.
        try:
            # pytorch_wavelets stores the low-pass analysis filter as `h0`
            # with shape [1, 1, N]; its length `N` is what differs between
            # haar (2), db4 (8), sym8 (16) etc.
            h0 = getattr(dwt, "h0", None)
            ntap = int(h0.shape[-1]) if h0 is not None else -1
        except Exception:
            ntap = -1
        logger.info(
            "[DCW] Built DWT1D for wavelet={!r} (low-pass filter taps={}, device={}, dtype={}).",
            wavelet, ntap, str(device), str(dtype),
        )
        return dwt, iwt


# Module-level singleton — one cache per process is fine.
WAVELET_CACHE = _LazyWavelet()
