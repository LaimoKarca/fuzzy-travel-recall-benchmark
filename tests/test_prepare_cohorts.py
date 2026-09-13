from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.prepare_cohorts import (
    HIGH_LOAD_COHORT_FILENAME,
    MAIN_COHORT_FILENAME,
    USER_STATS_FILENAME,
    CohortPreparationError,
    prepare_cohorts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows_for_user(user_id: str, trail_count: int) -> list[dict[str, str]]:
    return [
        {
            "trail_id": f"{user_id}-trail-{trail_number}",
            "user_id": user_id,
            "name": f"地點 {trail_number}",
        }
        for trail_number in range(trail_count)
    ]


class PrepareCohortsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.csv"
        self.output_dir = self.root / "prepared"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        pd.DataFrame(rows, columns=["trail_id", "user_id", "name"]).to_csv(
            self.input_path, index=False, encoding="utf-8"
        )

    def test_thresholds_distinct_trails_and_preserved_rows(self) -> None:
        rows = []
        rows.extend(_rows_for_user("user-9", 9))
        rows.extend(_rows_for_user("user-10", 10))
        rows.append(
            {
                "trail_id": "user-10-trail-0",
                "user_id": "user-10",
                "name": "重複 check-in，不是新 trail",
            }
        )
        rows.extend(_rows_for_user("user-19", 19))
        rows.extend(_rows_for_user("user-20", 20))
        self._write_rows(rows)

        summary = prepare_cohorts(self.input_path, self.output_dir)

        main = pd.read_csv(
            self.output_dir / MAIN_COHORT_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        high_load = pd.read_csv(
            self.output_dir / HIGH_LOAD_COHORT_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        stats = pd.read_csv(self.output_dir / USER_STATS_FILENAME, dtype=str)

        self.assertEqual(list(main.columns), ["trail_id", "user_id", "name"])
        self.assertEqual(
            list(main["user_id"].unique()), ["user-10", "user-19", "user-20"]
        )
        self.assertEqual(list(high_load["user_id"].unique()), ["user-20"])
        self.assertIn("重複 check-in，不是新 trail", main["name"].tolist())
        self.assertTrue(set(high_load["user_id"]).issubset(set(main["user_id"])))
        self.assertEqual(
            stats.loc[stats["user_id"] == "user-10", "trail_count"].item(), "10"
        )
        self.assertEqual(summary["Main (>=10)"].users, 3)
        self.assertEqual(summary["High load (>=20)"].users, 1)

    def test_missing_required_column_leaves_no_outputs(self) -> None:
        pd.DataFrame([{"user_id": "user-1"}]).to_csv(
            self.input_path, index=False, encoding="utf-8"
        )

        with self.assertRaisesRegex(CohortPreparationError, "trail_id"):
            prepare_cohorts(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_missing_input_leaves_no_outputs(self) -> None:
        with self.assertRaisesRegex(CohortPreparationError, "does not exist"):
            prepare_cohorts(self.root / "missing.csv", self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_blank_required_value_leaves_no_outputs(self) -> None:
        self._write_rows([{"trail_id": "", "user_id": "user-1", "name": "x"}])

        with self.assertRaisesRegex(CohortPreparationError, "blank value"):
            prepare_cohorts(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_cli_returns_nonzero_for_invalid_csv(self) -> None:
        self.input_path.write_text('user_id,trail_id\n"unterminated', encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "prepare-cohorts",
                "--input",
                str(self.input_path),
                "--output-dir",
                str(self.output_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error:", result.stderr)
        self.assertFalse(self.output_dir.exists())

    def test_written_csv_is_utf8_and_keeps_unicode(self) -> None:
        rows = _rows_for_user("user-10", 10)
        rows[0]["name"] = "JR 橋本駅"
        self._write_rows(rows)

        prepare_cohorts(self.input_path, self.output_dir)

        with (self.output_dir / MAIN_COHORT_FILENAME).open(
            encoding="utf-8", newline=""
        ) as stream:
            written_rows = list(csv.DictReader(stream))
        self.assertEqual(written_rows[0]["name"], "JR 橋本駅")


if __name__ == "__main__":
    unittest.main()
