"""Tests for generated-song library scanning."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta, timezone
from pathlib import Path

from acestep.ui.gradio.generated_library import (
    _records_to_table,
    filter_library_by_date,
    scan_generated_songs,
    select_library_item,
    select_library_table_item,
)


class _FakeSelectEvent:
    """Small SelectData-like object for table selection tests."""

    def __init__(self, index: tuple[int, int]) -> None:
        """Store the clicked Dataframe cell index."""

        self.index = index


class GeneratedLibraryTests(unittest.TestCase):
    """Verify generated-song metadata is read from manifests and sidecars."""

    def test_scans_manifest_samples_with_full_metadata(self) -> None:
        """It returns a selectable record with audio, lyrics, and metadata."""

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "0001"
            run_dir.mkdir()
            audio_path = run_dir / "song.flac"
            audio_path.write_bytes(b"audio")
            sidecar_path = run_dir / "song.json"
            sidecar_path.write_text(
                json.dumps({"lyrics": "sidecar lyrics", "audio_codes": "abc"}),
                encoding="utf-8",
            )
            manifest_path = run_dir / "generation_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "_meta": {"finished_at_utc": "2026-04-15T19:12:00+00:00"},
                        "audio_format": "flac",
                        "request": {
                            "caption": "cinematic synth pop",
                            "lyrics": "request lyrics",
                            "audio_duration": 12,
                            "runtime": {
                                "dit_quantization": "fp8_scaled",
                                "dit_last_init_params": {"config_path": "ace-xl"},
                            },
                        },
                        "samples": [
                            {
                                "sample_index": 1,
                                "key": "song",
                                "audio_path": str(audio_path),
                                "metadata_path": str(sidecar_path),
                                "audio_format": "flac",
                                "score": "Done!",
                                "params": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            local_timezone = timezone(timedelta(hours=-4))
            records = scan_generated_songs(tmp, local_timezone=local_timezone)
            audio, details, lyrics, metadata_json = select_library_item(
                records[0]["id"], records
            )
            table = _records_to_table(records)

        self.assertEqual(1, len(records))
        self.assertEqual(str(audio_path), audio)
        self.assertEqual("15 April 2026", records[0]["created_day"])
        self.assertEqual("15 April 2026, 3:12 PM", table[0][0])
        self.assertIn("Created: `15 April 2026, 3:12 PM`", details)
        self.assertIn("ace-xl / fp8_scaled", details)
        self.assertEqual("request lyrics", lyrics)
        self.assertIn('"audio_codes": "abc"', metadata_json)

    def test_filters_library_by_date_and_loads_first_song(self) -> None:
        """Selecting a day filters the table and loads that day's first song."""

        records = [
            {
                "id": "first",
                "created_day": "15 April 2026",
                "created_display": "15 April 2026, 3:12 PM",
                "title": "first song",
                "audio_path": "first.flac",
                "lyrics": "first lyrics",
                "metadata": {"id": 1},
            },
            {
                "id": "second",
                "created_day": "16 April 2026",
                "created_display": "16 April 2026, 9:01 AM",
                "title": "second song",
                "audio_path": "second.flac",
                "lyrics": "second lyrics",
                "metadata": {"id": 2},
            },
        ]

        filtered, table, audio, details, lyrics, metadata_json = filter_library_by_date(
            "16 April 2026",
            records,
        )

        self.assertEqual([records[1]], filtered)
        self.assertEqual("16 April 2026, 9:01 AM", table[0][0])
        self.assertEqual("second.flac", audio)
        self.assertIn("second song", details)
        self.assertEqual("second lyrics", lyrics)
        self.assertIn('"id": 2', metadata_json)

    def test_table_click_loads_clicked_song(self) -> None:
        """Clicking a filtered table row should load that row's song details."""

        records = [
            {"id": "first", "title": "first song", "audio_path": "first.flac"},
            {
                "id": "second",
                "title": "second song",
                "audio_path": "second.flac",
                "lyrics": "second lyrics",
            },
        ]

        audio, details, lyrics, _metadata = select_library_table_item(
            records,
            _FakeSelectEvent((1, 1)),
        )

        self.assertEqual("second.flac", audio)
        self.assertIn("second song", details)
        self.assertEqual("second lyrics", lyrics)


if __name__ == "__main__":
    unittest.main()
