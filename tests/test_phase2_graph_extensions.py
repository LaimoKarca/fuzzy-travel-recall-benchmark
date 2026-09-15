from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.diagnose_structured_next import (
    CANDIDATES_FILENAME,
    diagnose_structured_next,
)
from src.evaluate_graph_extensions import (
    OVERALL_FILENAME,
    PPR_DIAGNOSTICS_FILENAME,
    STRUCTURED_EVALUATION_FILENAME,
    evaluate_graph_extensions,
)
from src.model_specs import E5_SMALL_SPEC
from src.retrieve_graph import retrieve_graph
from src.retrieve_graph_ppr import (
    EVENT_SCORES_FILENAME,
    RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME,
    retrieve_graph_ppr,
)
from tests.test_retrieve_graph import GraphRetrievalTests


class PhaseTwoGraphExtensionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        events, vector_dir = GraphRetrievalTests()._fixture(
            root, E5_SMALL_SPEC, "relational_after"
        )
        queries = root / "queries.csv"
        pd.DataFrame([{
            "query_id": "q-core",
            "user_id": "u1",
            "benchmark_tier": "core",
            "query_type": "relational_after",
            "query_text": "Which Park did I visit after 場所3?",
            "target_event_id": "e4",
            "target_venue_id": "v4",
            "target_name": "場所4",
            "cue_previous_event_id": "e3",
            "cue_previous_place": "場所3",
            "cue_next_event_id": "e5",
            "cue_next_place": "場所5",
            "cue_category": "Park",
            "is_high_memory_load": False,
        }]).to_csv(queries, index=False, encoding="utf-8")
        return events, vector_dir, queries

    def test_ppr_uses_top_five_and_writes_complete_rrf_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir, _ = self._fixture(root)
            output = root / "ppr"
            summary = retrieve_graph_ppr(events, vector_dir, output)

            self.assertEqual(summary.query_count, 1)
            self.assertEqual(summary.ranking_count, 6)
            rankings = pd.read_csv(output / RANKINGS_FILENAME)
            scores = pd.read_csv(output / EVENT_SCORES_FILENAME)
            self.assertEqual(rankings["rank"].tolist(), list(range(1, 7)))
            self.assertEqual(set(rankings["method"]), {"graph_ppr_rrf"})
            self.assertEqual(int(scores["is_personalization_seed"].sum()), 5)
            self.assertAlmostEqual(scores["personalization_weight"].sum(), 1.0)
            expected = 1 / (60 + scores["dense_rank"]) + 1 / (60 + scores["ppr_rank"])
            self.assertTrue((expected - scores["rrf_score"]).abs().max() < 1e-12)
            manifest = json.loads((output / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertIsNone(manifest["parameters"]["ppr_weight"])
            self.assertNotIn("HAS", manifest["parameters"]["included_relations"])

    def test_structured_next_does_not_use_previous_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, _, queries = self._fixture(root)
            first = root / "first"
            diagnose_structured_next(events, queries, first)
            original = pd.read_csv(first / CANDIDATES_FILENAME)

            changed = pd.read_csv(queries, dtype=str, keep_default_na=False)
            changed["cue_previous_event_id"] = "deliberately-wrong"
            changed["target_event_id"] = "also-not-used"
            changed.to_csv(queries, index=False, encoding="utf-8")
            second = root / "second"
            diagnose_structured_next(events, queries, second)
            rerun = pd.read_csv(second / CANDIDATES_FILENAME)
            pd.testing.assert_frame_equal(original, rerun)
            self.assertEqual(rerun.iloc[0]["uniquely_selected_event_id"], "e4")

    def test_extended_evaluation_preserves_four_candidate_sets_and_distances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir, queries = self._fixture(root)
            onehop = root / "onehop"
            ppr = root / "ppr"
            oracle = root / "oracle"
            retrieve_graph(events, vector_dir, onehop)
            retrieve_graph_ppr(events, vector_dir, ppr)
            diagnose_structured_next(events, queries, oracle)

            flat = root / "flat"
            flat.mkdir()
            vector_ranking = pd.read_csv(vector_dir / "tokyo_vector_core_rankings.csv")
            vector_ranking["method"] = "flat"
            vector_ranking.to_csv(flat / "tokyo_flat_core_rankings.csv", index=False)

            output = root / "evaluation"
            summary = evaluate_graph_extensions(
                events, queries, flat, vector_dir, onehop, ppr, oracle, output
            )
            self.assertEqual(summary.query_count, 1)
            self.assertEqual(summary.per_query_count, 4)
            overall = pd.read_csv(output / OVERALL_FILENAME)
            self.assertEqual(set(overall["method"]), {
                "flat", "vector", "graph", "graph_ppr_rrf"
            })
            diagnostics = pd.read_csv(output / PPR_DIAGNOSTICS_FILENAME)
            self.assertIn("shortest_any_relation_distance", diagnostics)
            self.assertIn("shortest_next_only_distance", diagnostics)
            structured = pd.read_csv(output / STRUCTURED_EVALUATION_FILENAME)
            self.assertTrue(bool(structured.iloc[0]["target_reachable_before_category_filter"]))
            self.assertTrue(bool(structured.iloc[0]["uniquely_resolved_target"]))


if __name__ == "__main__":
    unittest.main()
