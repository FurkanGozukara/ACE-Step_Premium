"""Tests for user-entered path normalization."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import get_safe_roots, safe_path, set_safe_roots


class PathInputNormalizationTests(unittest.TestCase):
    """Verify pasted paths normalize before safety checks."""

    def setUp(self) -> None:
        """Preserve process-wide safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore process-wide safe roots."""

        set_safe_roots(self._safe_roots)

    def test_normalize_user_path_strips_quotes_and_preserves_spaces(self) -> None:
        """Quoted relative paths with spaces should keep the intended filename."""

        result = normalize_user_path('  "./data set/file name.json"  ')

        self.assertEqual(os.path.normpath("./data set/file name.json"), result)

    def test_normalize_user_path_accepts_file_uri(self) -> None:
        """File-manager URIs should normalize to local filesystem paths."""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "folder with spaces"
            uri = path.as_uri()

        result = normalize_user_path(uri)

        self.assertEqual(
            os.path.normcase(os.path.normpath(str(path))),
            os.path.normcase(os.path.normpath(result)),
        )

    def test_safe_path_accepts_quoted_full_path_with_forward_slashes(self) -> None:
        """Full paths pasted with quotes and slash variants should validate."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            child = Path(tmpdir) / "tensor data" / "sample.pt"
            pasted = f'"{str(child).replace(os.sep, "/")}"'

            result = safe_path(pasted)

        self.assertEqual(os.path.realpath(str(child)), result)

    def test_safe_path_accepts_quoted_relative_path_with_spaces(self) -> None:
        """Relative paths with shell quotes should resolve under the base."""

        with tempfile.TemporaryDirectory() as tmpdir:
            set_safe_roots([tmpdir])
            result = safe_path("'tensor data/sample.pt'", base=tmpdir)

        self.assertEqual(
            os.path.realpath(os.path.join(tmpdir, "tensor data", "sample.pt")),
            result,
        )


if __name__ == "__main__":
    unittest.main()
