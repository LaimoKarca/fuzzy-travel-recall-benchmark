from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.retrieve_flat import (
    BETWEEN_RANKINGS_FILENAME,
    CORE_RANKINGS_FILENAME,
    DOCUMENT_FILENAME,
    FlatRetrievalError,
    mixed_language_tokenize,
    retrieve_flat,
)


def _event(
    event_id: str,
    user_id: str,
    name: str,
    category: str,
    city: str,
    *,
    previous_event_id: str = "",
    previous_place: str = "",
    next_event_id: str = "",
    next_place: str = "",
    high: str = "False",
) -> dict[str, str]:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "trail_id": f"trail-{user_id}",
        "venue_id": f"venue-{event_id}",
        "timestamp": "2017-10-04 14:15:00",
        "name": name,
        "venue_category": category,
        "venue_city": city,
        "address": "東京都中央区",
        "previous_event_id": previous_event_id,
        "previous_place": previous_place,
        "next_event_id": next_event_id,
        "next_place": next_place,
        "is_high_memory_load": high,
    }


def _query(
    query_id: str,
    user_id: str,
    tier: str,
    query_type: str,
    text: str,
    high: str = "False",
) -> dict[str, str]:
    return {
        "query_id": query_id,
        "user_id": user_id,
        "benchmark_tier": tier,
        "query_type": query_type,
        "query_text": text,
        "is_high_memory_load": high,
    }


class MixedLanguageTokenizerTests(unittest.TestCase):
    def test_normalizes_latin_and_emits_japanese_ngrams(self) -> None:
        tokens = mixed_language_tokenize("  HACHIŌJI／新宿御苑  ")

        self.assertIn("hachiōji", tokens)
        self.assertEqual(
            [token for token in tokens if token != "hachiōji"],
            ["新宿御苑", "新宿", "宿御", "御苑", "新宿御", "宿御苑"],
        )

    def test_nfkc_normalizes_width_and_katakana(self) -> None:
        self.assertEqual(
            mixed_language_tokenize("ＲＡＭＥＮ　ﾗｰﾒﾝ"),
            ["ramen", "ラーメン", "ラー", "ーメ", "メン", "ラーメ", "ーメン"],
        )


class FlatRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.events_path = self.root / "events.csv"
        self.core_path = self.root / "core.csv"
        self.stress_path = self.root / "stress.csv"
        self.output_dir = self.root / "output"

        self.events = pd.DataFrame(
            [
                _event(
                    "event-001",
                    "u1",
                    "京都駅",
                    "Train Station",
                    "Kyoto",
                    next_event_id="event-002",
                    next_place="一蘭ラーメン",
                ),
                _event(
                    "event-002",
                    "u1",
                    "一蘭ラーメン",
                    "Ramen Restaurant",
                    "Tokyo",
                    previous_event_id="event-001",
                    previous_place="京都駅",
                    next_event_id="event-003",
                    next_place="上野公園",
                ),
                _event(
                    "event-003",
                    "u1",
                    "上野公園",
                    "Park",
                    "Tokyo",
                    previous_event_id="event-002",
                    previous_place="一蘭ラーメン",
                ),
                _event("event-101", "u2", "別の店", "Ramen Restaurant", "Tokyo"),
            ]
        )
        self.core = pd.DataFrame(
            [
                _query(
                    "q-core",
                    "u1",
                    "core",
                    "semantic",
                    "Which Ramen Restaurant did I visit in Tokyo?",
                ),
                _query(
                    "q-zero", "u1", "core", "semantic", "completely unmatched cue"
                ),
            ]
        )
        self.stress = pd.DataFrame(
            [
                _query(
                    "q-between",
                    "u1",
                    "stress_test",
                    "relational_between",
                    "Which Ramen Restaurant did I visit between 京都駅 and 上野公園?",
                )
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_inputs(self) -> None:
        self.events.to_csv(self.events_path, index=False, encoding="utf-8")
        self.core.to_csv(self.core_path, index=False, encoding="utf-8")
        self.stress.to_csv(self.stress_path, index=False, encoding="utf-8")

    def test_builds_documents_and_complete_user_scoped_rankings(self) -> None:
        self._write_inputs()

        summary = retrieve_flat(
            self.events_path, self.core_path, self.stress_path, self.output_dir
        )

        self.assertEqual(summary.document_count, 4)
        self.assertEqual(summary.core_ranking_count, 6)
        self.assertEqual(summary.stress_ranking_count, 3)

        documents = pd.read_csv(
            self.output_dir / "common" / DOCUMENT_FILENAME,
            dtype=str,
            keep_default_na=False,
        )
        text = documents.set_index("event_id").at["event-002", "event_text"]
        self.assertIn("On October 4, 2017 at 14:15", text)
        self.assertIn("Previous place: 京都駅.", text)
        self.assertIn("Next place: 上野公園.", text)
        self.assertNotIn("nan", text.casefold())
        boundary_text = documents.set_index("event_id").at["event-001", "event_text"]
        self.assertNotIn("Previous place:", boundary_text)

        core = pd.read_csv(
            self.output_dir / "flat" / CORE_RANKINGS_FILENAME,
            dtype={"rank": int},
        )
        q_core = core.loc[core["query_id"].eq("q-core")]
        self.assertEqual(q_core.iloc[0]["retrieved_event_id"], "event-002")
        self.assertEqual(set(q_core["user_id"]), {"u1"})
        self.assertEqual(set(q_core["retrieved_event_id"]), {
            "event-001", "event-002", "event-003"
        })
        self.assertEqual(q_core["rank"].tolist(), [1, 2, 3])

        q_zero = core.loc[core["query_id"].eq("q-zero")]
        self.assertEqual(
            q_zero["retrieved_event_id"].tolist(),
            ["event-001", "event-002", "event-003"],
        )
        self.assertTrue(q_zero["score"].eq(0.0).all())

        between = pd.read_csv(
            self.output_dir / "flat" / BETWEEN_RANKINGS_FILENAME
        )
        self.assertEqual(len(between), 3)
        self.assertEqual(set(between["retrieved_event_id"]), {
            "event-001", "event-002", "event-003"
        })

    def test_output_is_independent_of_input_row_order(self) -> None:
        self._write_inputs()
        retrieve_flat(
            self.events_path, self.core_path, self.stress_path, self.output_dir
        )
        first_documents = (self.output_dir / "common" / DOCUMENT_FILENAME).read_bytes()
        first_core = (self.output_dir / "flat" / CORE_RANKINGS_FILENAME).read_bytes()

        self.events.iloc[::-1].to_csv(self.events_path, index=False, encoding="utf-8")
        self.core.iloc[::-1].to_csv(self.core_path, index=False, encoding="utf-8")
        retrieve_flat(
            self.events_path, self.core_path, self.stress_path, self.output_dir
        )

        self.assertEqual(
            (self.output_dir / "common" / DOCUMENT_FILENAME).read_bytes(),
            first_documents,
        )
        self.assertEqual(
            (self.output_dir / "flat" / CORE_RANKINGS_FILENAME).read_bytes(),
            first_core,
        )

    def test_missing_column_fails_before_creating_outputs(self) -> None:
        self.events = self.events.drop(columns="event_id")
        self._write_inputs()

        with self.assertRaisesRegex(FlatRetrievalError, "event_id"):
            retrieve_flat(
                self.events_path, self.core_path, self.stress_path, self.output_dir
            )

        self.assertFalse(self.output_dir.exists())

    def test_duplicate_event_id_is_rejected(self) -> None:
        self.events.loc[1, "event_id"] = "event-001"
        self._write_inputs()

        with self.assertRaisesRegex(FlatRetrievalError, "duplicate event_id"):
            retrieve_flat(
                self.events_path, self.core_path, self.stress_path, self.output_dir
            )

    def test_duplicate_or_blank_query_id_is_rejected(self) -> None:
        for value, message in (("q-core", "duplicate query_id"), ("", "blank")):
            with self.subTest(value=value):
                changed = pd.concat([self.core, self.core.iloc[[0]]], ignore_index=True)
                changed.loc[len(changed) - 1, "query_id"] = value
                self.events.to_csv(self.events_path, index=False, encoding="utf-8")
                changed.to_csv(self.core_path, index=False, encoding="utf-8")
                self.stress.to_csv(self.stress_path, index=False, encoding="utf-8")
                with self.assertRaisesRegex(FlatRetrievalError, message):
                    retrieve_flat(
                        self.events_path,
                        self.core_path,
                        self.stress_path,
                        self.output_dir,
                    )

    def test_ground_truth_columns_do_not_change_ranking(self) -> None:
        first = self.core.copy()
        first["target_event_id"] = ["event-001", "event-002"]
        self.events.to_csv(self.events_path, index=False, encoding="utf-8")
        first.to_csv(self.core_path, index=False, encoding="utf-8")
        self.stress.to_csv(self.stress_path, index=False, encoding="utf-8")
        retrieve_flat(
            self.events_path, self.core_path, self.stress_path, self.output_dir
        )
        original = (self.output_dir / "flat" / CORE_RANKINGS_FILENAME).read_bytes()

        first["target_event_id"] = ["event-003", "event-001"]
        first.to_csv(self.core_path, index=False, encoding="utf-8")
        retrieve_flat(
            self.events_path, self.core_path, self.stress_path, self.output_dir
        )
        self.assertEqual(
            (self.output_dir / "flat" / CORE_RANKINGS_FILENAME).read_bytes(),
            original,
        )

    def test_invalid_timestamp_boolean_and_unknown_user_are_rejected(self) -> None:
        cases = (
            ("timestamp", "not-a-date", "invalid timestamp"),
            ("is_high_memory_load", "maybe", "invalid boolean"),
        )
        for column, value, message in cases:
            with self.subTest(column=column):
                changed = self.events.copy()
                changed.loc[0, column] = value
                changed.to_csv(self.events_path, index=False, encoding="utf-8")
                self.core.to_csv(self.core_path, index=False, encoding="utf-8")
                self.stress.to_csv(self.stress_path, index=False, encoding="utf-8")
                with self.assertRaisesRegex(FlatRetrievalError, message):
                    retrieve_flat(
                        self.events_path,
                        self.core_path,
                        self.stress_path,
                        self.output_dir,
                    )

        self._write_inputs()
        unknown = self.core.copy()
        unknown.loc[0, "user_id"] = "missing-user"
        unknown.to_csv(self.core_path, index=False, encoding="utf-8")
        with self.assertRaisesRegex(FlatRetrievalError, "absent from events"):
            retrieve_flat(
                self.events_path, self.core_path, self.stress_path, self.output_dir
            )


if __name__ == "__main__":
    unittest.main()
