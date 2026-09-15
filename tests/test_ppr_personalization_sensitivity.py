from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import networkx as nx

from src.evaluate_ppr_personalization_sensitivity import (
    DIAGNOSTICS_FILENAME,
    OVERALL_FILENAME,
    evaluate_ppr_personalization_sensitivity,
)
from src.model_specs import E5_SMALL_SPEC
from src.retrieve_graph_ppr import retrieve_graph_ppr
from src.retrieve_graph_ppr_all_event import (
    COMPONENT_AUDIT_FILENAME,
    EVENT_SCORES_FILENAME,
    QUERY_AUDIT_FILENAME,
    RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME,
    GraphPPRAllEventRetrievalError,
    _retrieve,
    retrieve_graph_ppr_all_event,
)
from tests.test_retrieve_graph import GraphRetrievalTests


class PPRPersonalizationSensitivityTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        events, vector_dir = GraphRetrievalTests()._fixture(
            root, E5_SMALL_SPEC, "relational_after"
        )
        queries = root / "queries.csv"
        pd.DataFrame([{
            "query_id": "q-core", "user_id": "u1", "benchmark_tier": "core",
            "query_type": "relational_after", "query_text": "Which Park came after place 3?",
            "target_event_id": "e4", "target_venue_id": "v4", "target_name": "place 4",
            "cue_previous_event_id": "e3", "cue_previous_place": "place 3",
            "cue_next_event_id": "e5", "cue_next_place": "place 5",
            "is_high_memory_load": False,
        }]).to_csv(queries, index=False, encoding="utf-8")
        return events, vector_dir, queries

    def test_all_event_personalization_and_mass_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir, _ = self._fixture(root)
            output = root / "all"
            summary = retrieve_graph_ppr_all_event(events, vector_dir, output)

            self.assertEqual(summary.query_count, 1)
            self.assertEqual(summary.ranking_count, 6)
            rankings = pd.read_csv(output / RANKINGS_FILENAME)
            scores = pd.read_csv(output / EVENT_SCORES_FILENAME)
            audit = pd.read_csv(output / QUERY_AUDIT_FILENAME)
            components = pd.read_csv(output / COMPONENT_AUDIT_FILENAME)
            self.assertEqual(rankings["rank"].tolist(), list(range(1, 7)))
            self.assertEqual(set(rankings["method"]), {"graph_ppr_rrf_all_event"})
            self.assertTrue(scores["has_positive_personalization"].all())
            self.assertAlmostEqual(scores["personalization_weight"].sum(), 1.0)
            self.assertAlmostEqual(audit.iloc[0]["full_graph_ppr_score_sum"], 1.0)
            self.assertLessEqual(audit.iloc[0]["event_only_ppr_score_sum"], 1.0)
            self.assertAlmostEqual(components["component_personalization_mass"].sum(), 1.0)
            self.assertAlmostEqual(components["component_ppr_mass"].sum(), 1.0)
            expected = 1 / (60 + scores["dense_rank"]) + 1 / (60 + scores["ppr_rank"])
            self.assertTrue((expected - scores["rrf_score"]).abs().max() < 1e-12)
            manifest = json.loads((output / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["analysis_role"], "post_hoc_personalization_sensitivity")

    def test_all_event_handles_multiple_components_and_rejects_zero_mass(self) -> None:
        graph = nx.Graph()
        graph.add_edge("event::e1", "trail::t1")
        graph.add_edge("event::e2", "trail::t2")
        query_index = pd.DataFrame([{
            "query_id": "q", "user_id": "u", "benchmark_tier": "core",
            "query_type": "semantic", "is_high_memory_load": "false",
        }])
        rankings = pd.DataFrame([
            {"query_id": "q", "user_id": "u", "rank": 1,
             "retrieved_event_id": "e1", "score": 0.8},
            {"query_id": "q", "user_id": "u", "rank": 2,
             "retrieved_event_id": "e2", "score": 0.2},
        ])
        _, _, audit, components = _retrieve({"u": graph}, query_index, rankings)
        self.assertEqual(int(audit.iloc[0]["component_count"]), 2)
        self.assertAlmostEqual(components["component_personalization_mass"].sum(), 1.0)
        self.assertAlmostEqual(components["component_ppr_mass"].sum(), 1.0)
        self.assertTrue((components["component_personalization_mass"] > 0).all())

        rankings["score"] = -1.0
        with self.assertRaises(GraphPPRAllEventRetrievalError):
            _retrieve({"u": graph}, query_index, rankings)

    def test_sensitivity_evaluation_keeps_all_event_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir, queries = self._fixture(root)
            top5 = root / "top5"
            all_event = root / "all"
            retrieve_graph_ppr(events, vector_dir, top5)
            retrieve_graph_ppr_all_event(events, vector_dir, all_event)

            output = root / "evaluation"
            summary = evaluate_ppr_personalization_sensitivity(
                events, queries, vector_dir, top5, all_event, output
            )
            self.assertEqual(summary.query_count, 1)
            overall = pd.read_csv(output / OVERALL_FILENAME)
            self.assertEqual(set(overall["configuration"]), {
                "e5_dense", "top5_raw_ppr", "top5_ppr_rrf",
                "all_event_raw_ppr", "all_event_ppr_rrf",
            })
            self.assertEqual(
                set(overall.loc[
                    overall["configuration"].str.startswith("all_event"), "analysis_role"
                ]), {"post_hoc_sensitivity"}
            )
            diagnostics = pd.read_csv(output / DIAGNOSTICS_FILENAME)
            self.assertEqual(set(diagnostics["stratum"]), {
                "overall", "target_in_dense_top5", "target_outside_dense_top5"
            })


if __name__ == "__main__":
    unittest.main()
