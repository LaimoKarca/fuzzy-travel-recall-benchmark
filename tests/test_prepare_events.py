from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.prepare_events import (
    HIGH_LOAD_EVENTS_FILENAME,
    MAIN_EVENTS_FILENAME,
    REMOVED_TRAILS_FILENAME,
    EventPreparationError,
    prepare_events,
)


def _trail(
    user_id: str,
    trail_number: int,
    gap: timedelta = timedelta(hours=1),
    names: tuple[str, str] = ("起點", "終點"),
) -> list[dict[str, str]]:
    start = datetime(2018, 1, 1) + timedelta(days=trail_number)
    trail_id = f"trail-{user_id}-{trail_number:02d}"
    return [
        {
            "trail_id": trail_id,
            "user_id": user_id,
            "timestamp": start.strftime("%Y-%m-%d %H:%M:%S"),
            "name": names[0],
            "venue_category": "Station",
            "venue_city": "Tokyo",
        },
        {
            "trail_id": trail_id,
            "user_id": user_id,
            "timestamp": (start + gap).strftime("%Y-%m-%d %H:%M:%S"),
            "name": names[1],
            "venue_category": "Restaurant",
            "venue_city": "Tokyo",
        },
    ]


def _user(user_id: str, trail_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trail_number in range(trail_count):
        rows.extend(_trail(user_id, trail_number))
    return rows


class PrepareEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.csv"
        self.output_dir = self.root / "prepared"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write(self, rows: list[dict[str, str]]) -> None:
        pd.DataFrame(rows).to_csv(self.input_path, index=False, encoding="utf-8")

    def test_gap_boundaries_remove_entire_trail_and_reapply_threshold(self) -> None:
        rows = _user("under-eight", 10)
        rows[1]["timestamp"] = (
            datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
            + timedelta(hours=7, minutes=59, seconds=59)
        ).strftime("%Y-%m-%d %H:%M:%S")
        rows.extend(_user("equal-eight", 10))
        equal_start = len(rows) - 20
        rows[equal_start + 1]["timestamp"] = (
            datetime.strptime(
                rows[equal_start]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            + timedelta(hours=8)
        ).strftime("%Y-%m-%d %H:%M:%S")
        rows.extend(_user("over-eight-stays", 11))
        over_start = len(rows) - 22
        rows[over_start + 1]["timestamp"] = (
            datetime.strptime(
                rows[over_start]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            + timedelta(hours=8, seconds=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        rows.extend(_user("over-eight-drops", 10))
        drop_start = len(rows) - 20
        rows[drop_start + 1]["timestamp"] = (
            datetime.strptime(
                rows[drop_start]["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            + timedelta(hours=8, seconds=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        self._write(rows)

        summary = prepare_events(self.input_path, self.output_dir)

        main = pd.read_csv(
            self.output_dir / MAIN_EVENTS_FILENAME, dtype=str, keep_default_na=False
        )
        audit = pd.read_csv(self.output_dir / REMOVED_TRAILS_FILENAME, dtype=str)
        self.assertEqual(
            set(main["user_id"]),
            {"under-eight", "equal-eight", "over-eight-stays"},
        )
        self.assertNotIn("trail-over-eight-stays-00", set(main["trail_id"]))
        self.assertEqual(len(audit), 2)
        self.assertEqual(summary.removed_trails, 2)
        self.assertEqual(summary.main_counts.users, 3)

    def test_multiple_invalid_gaps_produce_one_audit_row(self) -> None:
        rows = _user("user-a", 11)
        first_trail = rows[:2]
        first_time = datetime.strptime(
            first_trail[0]["timestamp"], "%Y-%m-%d %H:%M:%S"
        )
        first_trail[1]["timestamp"] = (first_time + timedelta(hours=9)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        third_event = first_trail[1].copy()
        third_event["timestamp"] = (first_time + timedelta(hours=18)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows.insert(2, third_event)
        self._write(rows)

        summary = prepare_events(self.input_path, self.output_dir)
        audit = pd.read_csv(self.output_dir / REMOVED_TRAILS_FILENAME)

        self.assertEqual(summary.removed_trails, 1)
        self.assertEqual(summary.removed_checkins, 3)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0]["checkin_count"], 3)

    def test_links_are_ordered_and_do_not_cross_trails(self) -> None:
        rows = _user("user-a", 10)
        rows[0], rows[1] = rows[1], rows[0]
        rows[0]["name"] = "晚"
        rows[1]["name"] = "早"
        self._write(rows)

        prepare_events(self.input_path, self.output_dir)
        main = pd.read_csv(
            self.output_dir / MAIN_EVENTS_FILENAME, dtype=str, keep_default_na=False
        )
        first_trail = main.loc[main["trail_id"] == "trail-user-a-00"]

        self.assertEqual(first_trail["event_position"].tolist(), ["1", "2"])
        self.assertEqual(first_trail.iloc[0]["previous_event_id"], "")
        self.assertEqual(first_trail.iloc[0]["next_place"], "晚")
        self.assertEqual(first_trail.iloc[1]["previous_place"], "早")
        self.assertEqual(first_trail.iloc[1]["next_event_id"], "")
        self.assertEqual(main["event_id"].nunique(), len(main))

    def test_ineligible_event_is_retained_and_flagged(self) -> None:
        rows = _user("user-a", 10)
        rows[0]["name"] = ""
        self._write(rows)

        prepare_events(self.input_path, self.output_dir)
        main = pd.read_csv(
            self.output_dir / MAIN_EVENTS_FILENAME, dtype=str, keep_default_na=False
        )

        self.assertEqual(len(main), len(rows))
        self.assertEqual(main.iloc[0]["is_target_eligible"], "False")

    def test_high_load_is_subset_of_main(self) -> None:
        rows = _user("main-only", 10) + _user("high-load", 20)
        self._write(rows)

        prepare_events(self.input_path, self.output_dir)
        main = pd.read_csv(self.output_dir / MAIN_EVENTS_FILENAME, dtype=str)
        high = pd.read_csv(self.output_dir / HIGH_LOAD_EVENTS_FILENAME, dtype=str)

        self.assertEqual(set(high["user_id"]), {"high-load"})
        self.assertTrue(set(high["event_id"]).issubset(set(main["event_id"])))

    def test_invalid_timestamp_leaves_no_outputs(self) -> None:
        rows = _user("user-a", 10)
        rows[0]["timestamp"] = "not-a-time"
        self._write(rows)

        with self.assertRaisesRegex(EventPreparationError, "invalid timestamp"):
            prepare_events(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_cli_returns_nonzero_for_invalid_timestamp(self) -> None:
        rows = _user("user-a", 10)
        rows[0]["timestamp"] = "not-a-time"
        self._write(rows)

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "main.py"),
                "prepare-events",
                "--input",
                str(self.input_path),
                "--output-dir",
                str(self.output_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        self.assertFalse(self.output_dir.exists())

    def test_trail_shared_by_users_is_rejected(self) -> None:
        rows = _user("user-a", 10)
        duplicate_owner = rows[0].copy()
        duplicate_owner["user_id"] = "user-b"
        rows.append(duplicate_owner)
        self._write(rows)

        with self.assertRaisesRegex(EventPreparationError, "shared by multiple users"):
            prepare_events(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
