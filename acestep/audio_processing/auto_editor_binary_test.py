"""Tests for Auto-Editor binary download fallback handling."""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from acestep.audio_processing.auto_editor_binary import (
    HUGGINGFACE_AUTO_EDITOR_URL,
    _asset_name,
    ensure_auto_editor_binary,
)


class AutoEditorBinaryTests(unittest.TestCase):
    """Verify Auto-Editor executable discovery and mirrored fallback downloads."""

    @patch("acestep.audio_processing.auto_editor_binary.urllib.request.urlretrieve")
    @patch("acestep.audio_processing.auto_editor_binary._asset_name")
    @patch("acestep.audio_processing.auto_editor_binary._auto_editor_package_info")
    def test_download_uses_huggingface_after_github_failure(
        self,
        package_mock,
        asset_mock,
        urlretrieve_mock,
    ) -> None:
        """GitHub download failure should retry the mirrored Hugging Face asset."""

        with tempfile.TemporaryDirectory() as temp_dir:
            package_mock.return_value = (Path(temp_dir), "30.4.0")
            asset_mock.return_value = ("auto-editor-windows-x86_64.exe", "auto-editor.exe")

            def fake_urlretrieve(url: str, filename: str) -> tuple[str, None]:
                if "github.com" in url:
                    raise urllib.error.HTTPError(url, 504, "Gateway Time-out", None, None)
                Path(filename).write_bytes(b"binary")
                return filename, None

            urlretrieve_mock.side_effect = fake_urlretrieve

            binary = ensure_auto_editor_binary()

        urls = [call.args[0] for call in urlretrieve_mock.call_args_list]
        self.assertEqual("auto-editor.exe", binary.name)
        self.assertEqual(2, len(urls))
        self.assertIn("github.com/WyattBlue/auto-editor", urls[0])
        self.assertEqual(
            HUGGINGFACE_AUTO_EDITOR_URL.format(asset="auto-editor-windows-x86_64.exe"),
            urls[1],
        )

    @patch("acestep.audio_processing.auto_editor_binary.urllib.request.urlretrieve")
    @patch("acestep.audio_processing.auto_editor_binary._binary_version")
    @patch("acestep.audio_processing.auto_editor_binary._asset_name")
    @patch("acestep.audio_processing.auto_editor_binary._auto_editor_package_info")
    def test_existing_matching_binary_skips_download(
        self,
        package_mock,
        asset_mock,
        version_mock,
        urlretrieve_mock,
    ) -> None:
        """A cached binary with the package version should be reused."""

        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            binary = package_dir / "bin" / "auto-editor.exe"
            binary.parent.mkdir()
            binary.write_bytes(b"binary")
            package_mock.return_value = (package_dir, "30.4.0")
            asset_mock.return_value = ("auto-editor-windows-x86_64.exe", "auto-editor.exe")
            version_mock.return_value = "30.4.0"

            resolved = ensure_auto_editor_binary()

        self.assertEqual(binary, resolved)
        urlretrieve_mock.assert_not_called()

    def test_asset_names_cover_supported_platforms(self) -> None:
        """Platform detection should preserve Auto-Editor's supported asset names."""

        self.assertEqual(
            ("auto-editor-windows-x86_64.exe", "auto-editor.exe"),
            _asset_name("windows", "AMD64"),
        )
        self.assertEqual(
            ("auto-editor-windows-aarch64.exe", "auto-editor.exe"),
            _asset_name("windows", "aarch64"),
        )
        self.assertEqual(
            ("auto-editor-linux-x86_64", "auto-editor"),
            _asset_name("linux", "x86_64"),
        )
        self.assertEqual(
            ("auto-editor-linux-aarch64", "auto-editor"),
            _asset_name("linux", "aarch64"),
        )
        self.assertEqual(
            ("auto-editor-linux-armv7", "auto-editor"),
            _asset_name("linux", "armv7l"),
        )
        self.assertEqual(
            ("auto-editor-macos-arm64", "auto-editor"),
            _asset_name("darwin", "arm64"),
        )


if __name__ == "__main__":
    unittest.main()
