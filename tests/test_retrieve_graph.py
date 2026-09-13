from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_download import sha256_file
from src.model_specs import E5_SMALL_SPEC, MINILM_SPEC
from src.retrieve_graph import GraphRetrievalError, retrieve_graph
from src.retrieve_graph_next_only import (
    AUDIT_FILENAME as NEXT_ONLY_AUDIT_FILENAME,
    RANKINGS_FILENAME as NEXT_ONLY_RANKINGS_FILENAME,
    retrieve_graph_next_only,
)
from src.evaluate_graph_direction import (
    DIAGNOSTICS_FILENAME as DIRECTION_DIAGNOSTICS_FILENAME,
    GraphDirectionEvaluationError,
    METRICS_FILENAME as DIRECTION_METRICS_FILENAME,
    evaluate_graph_direction,
)
from src.retrieve_vector import (
    BETWEEN_EMBEDDINGS_FILENAME,
    BETWEEN_INDEX_FILENAME,
    BETWEEN_RANKINGS_FILENAME,
    CORE_EMBEDDINGS_FILENAME,
    CORE_INDEX_FILENAME,
    CORE_RANKINGS_FILENAME,
    EVENT_EMBEDDINGS_FILENAME,
    EVENT_INDEX_FILENAME,
    SUMMARY_FILENAME,
)


class GraphRetrievalTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        model_spec=MINILM_SPEC,
        core_query_type: str = "semantic",
    ) -> tuple[Path, Path]:
        event_rows = []
        for position in range(1, 7):
            event_rows.append(
                {
                    "user_id": "u1",
                    "event_id": f"e{position}",
                    "trail_id": "t1",
                    "venue_id": f"v{position}",
                    "name": f"場所{position}",
                    "venue_category": "Park",
                    "venue_city": "Tokyo",
                    "address": f"住所{position}",
                    "latitude": str(35 + position / 100),
                    "longitude": str(139 + position / 100),
                    "timestamp": f"2018-01-01 0{position}:00:00",
                    "previous_event_id": "" if position == 1 else f"e{position - 1}",
                    "next_event_id": "" if position == 6 else f"e{position + 1}",
                    "is_high_memory_load": "False",
                }
            )
        events = pd.DataFrame(event_rows)
        events_path = root / "events.csv"
        events.to_csv(events_path, index=False, encoding="utf-8")

        vector_dir = root / "vector"
        vector_dir.mkdir()
        event_embeddings = np.zeros((6, 384), dtype=np.float32)
        for index in range(6):
            event_embeddings[index, index] = 1.0
        query = np.zeros((1, 384), dtype=np.float32)
        query[0, :6] = np.asarray([6, 3, 5, 2, 4, 1], dtype=np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        np.save(vector_dir / EVENT_EMBEDDINGS_FILENAME, event_embeddings, allow_pickle=False)
        np.save(vector_dir / CORE_EMBEDDINGS_FILENAME, query, allow_pickle=False)
        np.save(vector_dir / BETWEEN_EMBEDDINGS_FILENAME, query, allow_pickle=False)

        event_index = pd.DataFrame(
            {
                "row_index": range(6),
                "event_id": [f"e{x}" for x in range(1, 7)],
                "user_id": ["u1"] * 6,
                "is_high_memory_load": [False] * 6,
            }
        )
        core_index = pd.DataFrame(
            [{"row_index": 0, "query_id": "q-core", "user_id": "u1", "benchmark_tier": "core", "query_type": core_query_type, "is_high_memory_load": False}]
        )
        between_index = pd.DataFrame(
            [{"row_index": 0, "query_id": "q-between", "user_id": "u1", "benchmark_tier": "stress_test", "query_type": "relational_between", "is_high_memory_load": False}]
        )
        event_index.to_csv(vector_dir / EVENT_INDEX_FILENAME, index=False)
        core_index.to_csv(vector_dir / CORE_INDEX_FILENAME, index=False)
        between_index.to_csv(vector_dir / BETWEEN_INDEX_FILENAME, index=False)

        def rankings(query_id: str, tier: str, query_type: str) -> pd.DataFrame:
            scores = event_embeddings @ query[0]
            order = sorted(range(6), key=lambda i: (-float(scores[i]), f"e{i + 1}"))
            return pd.DataFrame(
                [
                    {
                        "query_id": query_id,
                        "user_id": "u1",
                        "benchmark_tier": tier,
                        "query_type": query_type,
                        "method": "vector",
                        "rank": rank,
                        "retrieved_event_id": f"e{index + 1}",
                        "score": float(scores[index]),
                        "is_high_memory_load": False,
                    }
                    for rank, index in enumerate(order, start=1)
                ]
            )

        rankings("q-core", "core", core_query_type).to_csv(
            vector_dir / CORE_RANKINGS_FILENAME, index=False
        )
        rankings("q-between", "stress_test", "relational_between").to_csv(
            vector_dir / BETWEEN_RANKINGS_FILENAME, index=False
        )
        pd.DataFrame([{"method": "vector"}]).to_csv(
            vector_dir / SUMMARY_FILENAME, index=False
        )

        output_names = (
            EVENT_EMBEDDINGS_FILENAME,
            EVENT_INDEX_FILENAME,
            CORE_EMBEDDINGS_FILENAME,
            CORE_INDEX_FILENAME,
            BETWEEN_EMBEDDINGS_FILENAME,
            BETWEEN_INDEX_FILENAME,
            CORE_RANKINGS_FILENAME,
            BETWEEN_RANKINGS_FILENAME,
            SUMMARY_FILENAME,
        )
        manifest = {
            "method": "vector",
            "model": {
                "id": model_spec.model_id,
                "revision": model_spec.revision,
                "manifest_sha256": "0" * 64,
            },
            "outputs": {
                name: sha256_file(vector_dir / name) for name in output_names
            },
        }
        (vector_dir / "tokyo_vector_run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return events_path, vector_dir

    def test_retrieves_complete_rankings_and_audits_bfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir = self._fixture(root)
            output = root / "graph"
            summary = retrieve_graph(events, vector_dir, output)

            self.assertEqual(summary.node_count, 15)
            self.assertEqual(summary.edge_count, 29)
            self.assertEqual(summary.next_edge_count, 5)
            self.assertEqual(summary.core_ranking_count, 6)
            self.assertEqual(summary.stress_ranking_count, 6)
            self.assertEqual(summary.core_audit_count, 14)
            self.assertEqual(summary.core_unique_expanded_count, 6)

            audit = pd.read_csv(output / "tokyo_graph_core_expansion_audit.csv")
            first = audit.loc[audit["is_first_discovery"]]
            self.assertEqual(first["expanded_event_id"].tolist(), ["e1", "e2", "e3", "e4", "e5", "e6"])
            self.assertEqual(first["expansion_rank"].astype(int).tolist(), list(range(1, 7)))
            rankings = pd.read_csv(output / "tokyo_graph_core_rankings.csv")
            self.assertEqual(rankings["rank"].tolist(), list(range(1, 7)))
            self.assertEqual(set(rankings["method"]), {"graph"})
            self.assertTrue(rankings["score"].is_monotonic_decreasing)
            self.assertAlmostEqual(rankings.iloc[0]["score"], 2 / 61)
            graph_manifest = json.loads(
                (output / "tokyo_graph_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                graph_manifest["dense_seed_model"]["id"], MINILM_SPEC.model_id
            )

    def test_vector_hash_mismatch_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir = self._fixture(root)
            with (vector_dir / CORE_INDEX_FILENAME).open("a", encoding="utf-8") as target:
                target.write("modified")
            output = root / "graph"
            with self.assertRaisesRegex(GraphRetrievalError, "hash mismatch"):
                retrieve_graph(events, vector_dir, output)
            self.assertFalse(output.exists())

    def test_next_only_ablation_uses_self_and_next_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir = self._fixture(
                root, E5_SMALL_SPEC, "relational_after"
            )
            output = root / "next-only"
            summary = retrieve_graph_next_only(events, vector_dir, output)

            self.assertEqual(summary.query_count, 1)
            self.assertEqual(summary.ranking_count, 6)
            audit = pd.read_csv(output / NEXT_ONLY_AUDIT_FILENAME)
            self.assertEqual(set(audit["relation"]), {"SELF", "NEXT"})
            self.assertNotIn("PREVIOUS", set(audit["relation"]))
            rankings = pd.read_csv(output / NEXT_ONLY_RANKINGS_FILENAME)
            self.assertEqual(rankings["rank"].tolist(), list(range(1, 7)))

    def test_evaluates_e5_graph_direction_on_relational_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events, vector_dir = self._fixture(
                root, E5_SMALL_SPEC, "relational_after"
            )
            symmetric = root / "symmetric"
            next_only = root / "next-only"
            retrieve_graph(events, vector_dir, symmetric)
            retrieve_graph_next_only(events, vector_dir, next_only)
            queries = root / "queries.csv"
            pd.DataFrame(
                [
                    {
                        "query_id": "q-core",
                        "user_id": "u1",
                        "benchmark_tier": "core",
                        "query_type": "relational_after",
                        "query_text": "Which Park did I visit after ?湔?3?",
                        "target_event_id": "e4",
                        "target_venue_id": "v4",
                        "target_name": "?湔?4",
                        "cue_previous_event_id": "e3",
                        "cue_previous_place": "?湔?3",
                        "cue_next_event_id": "e5",
                        "cue_next_place": "?湔?5",
                        "is_high_memory_load": False,
                    }
                ]
            ).to_csv(queries, index=False, encoding="utf-8")

            output = root / "evaluation"
            summary = evaluate_graph_direction(
                queries, vector_dir, symmetric, next_only, output
            )
            self.assertEqual(summary.query_count, 1)
            metrics = pd.read_csv(output / DIRECTION_METRICS_FILENAME)
            self.assertEqual(metrics["configuration"].tolist(), [
                "E5 Dense", "E5 Graph Symmetric", "E5 Graph NEXT-only"
            ])
            diagnostics = pd.read_csv(output / DIRECTION_DIAGNOSTICS_FILENAME)
            self.assertEqual(
                diagnostics.loc[
                    diagnostics["configuration"].eq("E5 Graph NEXT-only"),
                    "target_first_reached_as_previous",
                ].iloc[0],
                0,
            )

            with (next_only / NEXT_ONLY_AUDIT_FILENAME).open("a", encoding="utf-8") as stream:
                stream.write("modified")
            failed_output = root / "failed-evaluation"
            with self.assertRaisesRegex(
                GraphDirectionEvaluationError, "hash mismatch"
            ):
                evaluate_graph_direction(
                    queries, vector_dir, symmetric, next_only, failed_output
                )
            self.assertFalse(failed_output.exists())


if __name__ == "__main__":
    unittest.main()
