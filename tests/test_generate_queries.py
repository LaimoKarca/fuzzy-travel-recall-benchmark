from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.generate_queries import (
    AUDIT_FILENAME,
    BETWEEN_HIGH_LOAD_FILENAME,
    BETWEEN_MAIN_FILENAME,
    CORE_HIGH_LOAD_FILENAME,
    CORE_MAIN_FILENAME,
    QUERY_COLUMNS,
    SUMMARY_FILENAME,
    QueryGenerationError,
    generate_queries,
)


def _row(
    *,
    user_id: str,
    trail_id: str,
    event_id: str,
    venue_id: str,
    timestamp: str,
    name: str,
    category: str,
    city: str = "Hachiōji",
    previous_event_id: str = "",
    previous_place: str = "",
    next_event_id: str = "",
    next_place: str = "",
    high_load: bool = False,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "trail_id": trail_id,
        "event_id": event_id,
        "venue_id": venue_id,
        "timestamp": timestamp,
        "name": name,
        "venue_category": category,
        "venue_city": city,
        "previous_event_id": previous_event_id,
        "previous_place": previous_place,
        "next_event_id": next_event_id,
        "next_place": next_place,
        "is_high_memory_load": high_load,
        "is_target_eligible": eligible,
    }


def _three_event_trail(
    user_id: str = "user-a", high_load: bool = True
) -> list[dict[str, object]]:
    return [
        _row(
            user_id=user_id,
            trail_id="trail-1",
            event_id="event-1",
            venue_id="venue-1",
            timestamp="2017-10-01 10:00:00",
            name="新宿御苑",
            category="",
            next_event_id="event-2",
            next_place="一蘭ラーメン",
            high_load=high_load,
            eligible=False,
        ),
        _row(
            user_id=user_id,
            trail_id="trail-1",
            event_id="event-2",
            venue_id="venue-2",
            timestamp="2017-10-01 11:00:00",
            name="一蘭ラーメン",
            category="Ramen Restaurant",
            previous_event_id="event-1",
            previous_place="新宿御苑",
            next_event_id="event-3",
            next_place="東京駅",
            high_load=high_load,
        ),
        _row(
            user_id=user_id,
            trail_id="trail-1",
            event_id="event-3",
            venue_id="venue-3",
            timestamp="2017-10-01 12:00:00",
            name="東京駅",
            category="",
            previous_event_id="event-2",
            previous_place="一蘭ラーメン",
            high_load=high_load,
            eligible=False,
        ),
    ]


class GenerateQueriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "events.csv"
        self.output_dir = self.root / "stage_03_queries"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write(self, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(self.input_path, index=False, encoding="utf-8")

    def test_generates_all_templates_and_high_load_subsets(self) -> None:
        self._write(_three_event_trail())

        summary = generate_queries(self.input_path, self.output_dir)

        core = pd.read_csv(
            self.output_dir / CORE_MAIN_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        core_high = pd.read_csv(
            self.output_dir / CORE_HIGH_LOAD_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        stress = pd.read_csv(
            self.output_dir / BETWEEN_MAIN_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        stress_high = pd.read_csv(
            self.output_dir / BETWEEN_HIGH_LOAD_FILENAME,
            dtype=str,
            keep_default_na=False,
        )

        self.assertEqual(tuple(core.columns), QUERY_COLUMNS)
        self.assertEqual(summary.core_main, {
            "semantic": 1,
            "temporal_spatial": 1,
            "relational_after": 1,
        })
        self.assertEqual(summary.stress_main, {"relational_between": 1})
        self.assertEqual(len(core_high), len(core))
        self.assertEqual(len(stress_high), len(stress))

        queries = dict(zip(core["query_type"], core["query_text"], strict=True))
        self.assertEqual(
            queries["semantic"],
            "Which Ramen Restaurant did I visit in Hachiōji?",
        )
        self.assertEqual(
            queries["temporal_spatial"],
            "Which Ramen Restaurant did I visit in Hachiōji in October 2017?",
        )
        self.assertEqual(
            queries["relational_after"],
            "Which Ramen Restaurant did I visit after 新宿御苑?",
        )
        self.assertEqual(
            stress.iloc[0]["query_text"],
            "Which Ramen Restaurant did I visit between 新宿御苑 and 東京駅?",
        )
        self.assertEqual(set(core["query_language"]), {"dataset_native_mixed"})
        self.assertNotIn("一蘭ラーメン", " ".join(core["query_text"]))
        self.assertNotIn("一蘭ラーメン", stress.iloc[0]["query_text"])

    def test_same_venue_multiple_events_is_not_unique_event(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows.append(
            _row(
                user_id="user-a",
                trail_id="trail-2",
                event_id="event-4",
                venue_id="venue-2",
                timestamp="2017-11-01 11:00:00",
                name="一蘭ラーメン",
                category="Ramen Restaurant",
            )
        )
        self._write(rows)

        generate_queries(self.input_path, self.output_dir)
        audit = pd.read_csv(
            self.output_dir / AUDIT_FILENAME, dtype=str, keep_default_na=False
        )
        semantic = audit.loc[audit["query_type"].eq("semantic")].iloc[0]

        self.assertEqual(semantic["status"], "skipped")
        self.assertEqual(semantic["skip_reason"], "no_unique_event_candidate")

    def test_ineligible_event_still_makes_cue_ambiguous(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows.append(
            _row(
                user_id="user-a",
                trail_id="trail-2",
                event_id="event-4",
                venue_id="venue-4",
                timestamp="2017-11-01 11:00:00",
                name="",
                category="Ramen Restaurant",
                eligible=False,
            )
        )
        self._write(rows)

        generate_queries(self.input_path, self.output_dir)
        audit = pd.read_csv(
            self.output_dir / AUDIT_FILENAME, dtype=str, keep_default_na=False
        )
        semantic = audit.loc[audit["query_type"].eq("semantic")].iloc[0]

        self.assertEqual(semantic["status"], "skipped")
        self.assertEqual(semantic["unique_event_candidate_count"], "0")

    def test_target_name_leakage_blocks_relational_queries(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[0]["name"] = "一蘭ラーメン"
        rows[1]["previous_place"] = "一蘭ラーメン"
        self._write(rows)

        generate_queries(self.input_path, self.output_dir)
        audit = pd.read_csv(
            self.output_dir / AUDIT_FILENAME, dtype=str, keep_default_na=False
        ).set_index("query_type")

        self.assertEqual(audit.loc["relational_after", "status"], "skipped")
        self.assertEqual(
            audit.loc["relational_after", "skip_reason"], "target_name_leakage"
        )
        self.assertEqual(audit.loc["relational_between", "status"], "skipped")

    def test_hash_selection_is_independent_of_input_row_order(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows.extend(
            [
                _row(
                    user_id="user-a",
                    trail_id="trail-2",
                    event_id="event-4",
                    venue_id="venue-4",
                    timestamp="2017-11-01 10:00:00",
                    name="上野公園",
                    category="Park",
                ),
                _row(
                    user_id="user-b",
                    trail_id="trail-3",
                    event_id="event-5",
                    venue_id="venue-5",
                    timestamp="2018-01-01 10:00:00",
                    name="喫茶店",
                    category="Café",
                    high_load=True,
                ),
            ]
        )
        self._write(rows)
        first_dir = self.root / "first"
        second_dir = self.root / "second"

        generate_queries(self.input_path, first_dir)
        pd.DataFrame(list(reversed(rows))).to_csv(
            self.input_path, index=False, encoding="utf-8"
        )
        generate_queries(self.input_path, second_dir)

        for filename in (
            CORE_MAIN_FILENAME,
            CORE_HIGH_LOAD_FILENAME,
            BETWEEN_MAIN_FILENAME,
            BETWEEN_HIGH_LOAD_FILENAME,
            AUDIT_FILENAME,
            SUMMARY_FILENAME,
        ):
            self.assertEqual(
                (first_dir / filename).read_bytes(),
                (second_dir / filename).read_bytes(),
            )

    def test_reused_target_venue_is_allowed_and_audited(self) -> None:
        self._write(_three_event_trail(high_load=False))

        generate_queries(self.input_path, self.output_dir)
        audit = pd.read_csv(
            self.output_dir / AUDIT_FILENAME, dtype=str, keep_default_na=False
        )
        selected = audit.loc[audit["selected_venue_id"].eq("venue-2")]

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            set(selected["same_user_target_venue_query_count"]), {"4"}
        )
        self.assertEqual(
            set(selected["target_venue_reused_across_query_types"]), {"True"}
        )

    def test_broken_event_relation_leaves_no_outputs(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[1]["previous_event_id"] = "missing-event"
        self._write(rows)

        with self.assertRaisesRegex(QueryGenerationError, "unknown previous_event_id"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_invalid_boolean_leaves_no_outputs(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[0]["is_target_eligible"] = "maybe"
        self._write(rows)

        with self.assertRaisesRegex(QueryGenerationError, "invalid"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_missing_required_column_leaves_no_outputs(self) -> None:
        frame = pd.DataFrame(_three_event_trail(high_load=False)).drop(
            columns="venue_id"
        )
        frame.to_csv(self.input_path, index=False, encoding="utf-8")

        with self.assertRaisesRegex(QueryGenerationError, "missing required"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_duplicate_event_id_leaves_no_outputs(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[2]["event_id"] = rows[1]["event_id"]
        self._write(rows)

        with self.assertRaisesRegex(QueryGenerationError, "duplicate event_id"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_invalid_timestamp_leaves_no_outputs(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[1]["timestamp"] = "not-a-time"
        self._write(rows)

        with self.assertRaisesRegex(QueryGenerationError, "invalid timestamp"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())

    def test_cross_trail_link_leaves_no_outputs(self) -> None:
        rows = _three_event_trail(high_load=False)
        rows[0]["trail_id"] = "other-trail"
        self._write(rows)

        with self.assertRaisesRegex(QueryGenerationError, "across user or trail"):
            generate_queries(self.input_path, self.output_dir)

        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
