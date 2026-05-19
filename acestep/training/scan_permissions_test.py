"""Tests for dataset scan filesystem permissions."""

import os
import tempfile
import unittest

from acestep.training.path_safety import get_safe_roots, safe_path, set_safe_roots
from acestep.training.scan_permissions import (
    configure_scan_permissions,
    discover_local_filesystem_roots,
)


class ScanPermissionsTests(unittest.TestCase):
    """Verify scan permission setup for local filesystems."""

    def setUp(self) -> None:
        """Remember global path safety state."""
        self._safe_roots = get_safe_roots()

    def tearDown(self) -> None:
        """Restore global path safety state."""
        set_safe_roots(self._safe_roots)

    def test_discovers_at_least_current_filesystem_root(self) -> None:
        roots = discover_local_filesystem_roots()

        self.assertTrue(roots)
        self.assertTrue(all(os.path.isabs(root) for root in roots))

    def test_configure_scan_permissions_allows_extra_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configured = configure_scan_permissions([directory])

            self.assertIn(os.path.realpath(directory), configured)
            self.assertEqual(safe_path(directory), os.path.realpath(directory))

    def test_configure_scan_permissions_ignores_missing_extra_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing")
            configured = configure_scan_permissions([missing])

            self.assertNotIn(os.path.realpath(missing), configured)


if __name__ == "__main__":
    unittest.main()
