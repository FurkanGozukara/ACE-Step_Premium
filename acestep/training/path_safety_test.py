"""Tests for default filesystem-safe path roots."""

import os
import tempfile
import unittest

from acestep.training.path_safety import (
    discover_default_safe_roots,
    get_safe_roots,
    safe_path,
    set_safe_roots,
)


class DefaultSafeRootTests(unittest.TestCase):
    """Verify broad default roots still constrain relative traversal."""

    def setUp(self) -> None:
        """Preserve process-wide safe roots."""

        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore process-wide safe roots."""

        set_safe_roots(self._safe_roots)

    def test_default_safe_roots_allow_absolute_local_paths(self) -> None:
        """Default roots should accept absolute folders on local filesystems."""

        with tempfile.TemporaryDirectory() as directory:
            set_safe_roots(discover_default_safe_roots())

            result = safe_path(directory)

        self.assertEqual(os.path.realpath(directory), result)

    def test_relative_paths_stay_under_primary_safe_root(self) -> None:
        """Relative traversal should not escape through a broad filesystem root."""

        with tempfile.TemporaryDirectory() as directory:
            drive, _ = os.path.splitdrive(os.path.realpath(directory))
            filesystem_root = drive + os.sep if drive else os.sep
            set_safe_roots([directory, filesystem_root])

            with self.assertRaises(ValueError):
                safe_path("..")


if __name__ == "__main__":
    unittest.main()
