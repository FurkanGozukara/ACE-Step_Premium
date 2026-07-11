"""Tests for the optional pytorch_wavelets compatibility loader."""

from __future__ import annotations

import importlib.util
import sys
import unittest

import torch

from acestep.models.common.dcw_loader import _import_dwt1d_modules


_HAS_WAVELET_DEPS = bool(
    importlib.util.find_spec("pytorch_wavelets")
    and importlib.util.find_spec("pywt")
)


class WaveletLoaderTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_WAVELET_DEPS, "optional wavelet packages are absent")
    def test_dwt1d_import_and_roundtrip(self) -> None:
        pkg_resources_was_loaded = "pkg_resources" in sys.modules

        dwt_type, iwt_type = _import_dwt1d_modules()
        dwt = dwt_type(J=1, mode="zero", wave="haar")
        iwt = iwt_type(mode="zero", wave="haar")
        value = torch.randn(1, 2, 32)
        restored = iwt(dwt(value))

        torch.testing.assert_close(restored, value)
        if not pkg_resources_was_loaded:
            self.assertNotIn("pkg_resources", sys.modules)


if __name__ == "__main__":
    unittest.main()
