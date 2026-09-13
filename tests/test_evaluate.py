from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate import EvaluationError, _holm_adjust, evaluate
from src.model_download import sha256_file


class EvaluationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
        events = pd.DataFrame(
            [
                {
                    "event_id": f"e{position}",
                    "user_id": "u1",
                    "venue_id": "v1" if position in (1, 4) else f"v{position}",
                    "is_high_memory_load": False,
                }
                for position in range(1, 7)
            ]
        )
        events_path = root / "events.csv"
        events.to_csv(events_path, index=False)

        def query(
            query_id: str, tier: str, query_type: str, target: str
        ) -> dict[str, object]:
            target_number = int(target.removeprefix("e"))
            return {
                "query_id": query_id,
                "user_id": "u1",
                "benchmark_tier": tier,
                "query_type": query_type,
                "query_text": f"query {query_id}",
                "target_event_id": target,
                "target_venue_id": "v1" if target_number in (1, 4) else f"v{target_number}",
                "target_name": f"場所{target_number}",
                "cue_previous_event_id": "",
                "cue_previous_place": "",
                "cue_next_event_id": "",
                "cue_next_place": "",
                "is_high_memory_load": False,
            }

        core = pd.DataFrame(
            [
                query("q1", "core", "semantic", "e1"),
                query("q2", "core", "temporal_spatial", "e2"),
                query("q3", "core", "relational_after", "e6"),
            ]
        )
        between = pd.DataFrame(
            [query("q4", "stress_test", "relational_between", "e4")]
        )
        core_path = root / "core.csv"
        between_path = root / "between.csv"
        core.to_csv(core_path, index=False)
        between.to_csv(between_path, index=False)

        flat_dir = root / "flat"
        vector_dir = root / "vector"
        graph_dir = root / "graph"
        for directory in (flat_dir, vector_dir, graph_dir):
            directory.mkdir()

        orders = {
            "flat": {
                "q1": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q2": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q3": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q4": ["e1", "e2", "e3", "e4", "e5", "e6"],
            },
            "vector": {
                "q1": ["e2", "e1", "e3", "e4", "e5", "e6"],
                "q2": ["e2", "e1", "e3", "e4", "e5", "e6"],
                "q3": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q4": ["e1", "e2", "e3", "e4", "e5", "e6"],
            },
            "graph": {
                "q1": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q2": ["e1", "e2", "e3", "e4", "e5", "e6"],
                "q3": ["e1", "e2", "e3", "e4", "e6", "e5"],
                "q4": ["e1", "e2", "e3", "e4", "e5", "e6"],
            },
        }
        query_lookup = pd.concat([core, between]).set_index("query_id")

        def ranking(method: str, query_ids: list[str]) -> pd.DataFrame:
            rows = []
            for query_id in query_ids:
                query_row = query_lookup.loc[query_id]
                for rank, event_id in enumerate(orders[method][query_id], start=1):
                    rows.append(
                        {
                            "query_id": query_id,
                            "user_id": "u1",
                            "benchmark_tier": query_row["benchmark_tier"],
                            "query_type": query_row["query_type"],
                            "method": method,
                            "rank": rank,
                            "retrieved_event_id": event_id,
                            "score": 7 - rank,
                            "is_high_memory_load": False,
                        }
                    )
            return pd.DataFrame(rows)

        filenames: dict[str, list[str]] = {"vector": [], "graph": []}
        for method, directory in (
            ("flat", flat_dir),
            ("vector", vector_dir),
            ("graph", graph_dir),
        ):
            for tier, query_ids, suffix in (
                ("core", ["q1", "q2", "q3"], "core"),
                ("stress_test", ["q4"], "between"),
            ):
                filename = f"tokyo_{method}_{suffix}_rankings.csv"
                ranking(method, query_ids).to_csv(directory / filename, index=False)
                if method in filenames:
                    filenames[method].append(filename)

        for tier, query_ids, suffix in (
            ("core", ["q1", "q2", "q3"], "core"),
            ("stress_test", ["q4"], "between"),
        ):
            audit_rows = []
            for query_id in query_ids:
                target = query_lookup.loc[query_id, "target_event_id"]
                audit_rows.append(
                    {
                        "query_id": query_id,
                        "user_id": "u1",
                        "benchmark_tier": tier,
                        "query_type": query_lookup.loc[query_id, "query_type"],
                        "seed_rank": 1,
                        "seed_event_id": target,
                        "relation": "SELF",
                        "expanded_event_id": target,
                        "is_first_discovery": True,
                        "expansion_rank": 1,
                        "is_high_memory_load": False,
                    }
                )
            filename = f"tokyo_graph_{suffix}_expansion_audit.csv"
            pd.DataFrame(audit_rows).to_csv(graph_dir / filename, index=False)
            filenames["graph"].append(filename)

        for method, directory in (("vector", vector_dir), ("graph", graph_dir)):
            manifest = {
                "method": method,
                "outputs": {
                    filename: sha256_file(directory / filename)
                    for filename in filenames[method]
                },
            }
            (directory / f"tokyo_{method}_run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        return events_path, core_path, between_path, flat_dir, vector_dir, graph_dir

    def test_evaluates_metrics_boundaries_and_graph_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "output"
            summary = evaluate(*fixture, output)

            self.assertEqual(summary.per_query_count, 12)
            per_query = pd.read_csv(output / "tokyo_per_query_metrics.csv")
            flat = per_query.loc[per_query["method"].eq("flat")].set_index("query_id")
            self.assertEqual(flat.loc["q1", "target_rank"], 1)
            self.assertEqual(flat.loc["q2", "target_rank"], 2)
            self.assertEqual(flat.loc["q3", "target_rank"], 6)
            self.assertTrue(flat.loc["q1", "success_at_1"])
            self.assertTrue(flat.loc["q2", "success_at_5"])
            self.assertFalse(flat.loc["q3", "success_at_5"])
            self.assertAlmostEqual(flat.loc["q3", "reciprocal_rank"], 1 / 6)

            diagnostics = pd.read_csv(
                output / "tokyo_graph_query_diagnostics.csv"
            ).set_index("query_id")
            self.assertEqual(diagnostics.loc["q1", "graph_outcome_vs_vector"], "improved")
            self.assertEqual(diagnostics.loc["q2", "graph_outcome_vs_vector"], "worsened")
            self.assertEqual(diagnostics.loc["q4", "graph_outcome_vs_vector"], "unchanged")
            self.assertEqual(diagnostics.loc["q1", "target_venue_visit_count"], 2)
            self.assertTrue(diagnostics.loc["q1", "is_repeated_target_venue"])

    def test_manifest_mismatch_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            vector_core = fixture[4] / "tokyo_vector_core_rankings.csv"
            with vector_core.open("a", encoding="utf-8") as target:
                target.write("modified")
            output = root / "output"
            with self.assertRaisesRegex(EvaluationError, "hash mismatch"):
                evaluate(*fixture, output)
            self.assertFalse(output.exists())

    def test_incomplete_candidate_set_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            flat_core = fixture[3] / "tokyo_flat_core_rankings.csv"
            ranking = pd.read_csv(flat_core)
            ranking = ranking.loc[~((ranking["query_id"] == "q1") & (ranking["rank"] == 6))]
            ranking.to_csv(flat_core, index=False)
            output = root / "output"
            with self.assertRaisesRegex(EvaluationError, "incomplete candidate coverage"):
                evaluate(*fixture, output)
            self.assertFalse(output.exists())

    def test_holm_adjustment_is_monotone(self) -> None:
        self.assertEqual(_holm_adjust([0.04, 0.01]), [0.04, 0.02])


if __name__ == "__main__":
    unittest.main()
