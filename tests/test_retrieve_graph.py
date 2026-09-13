from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_download import sha256_file
from src.retrieve_graph import GraphRetrievalError, retrieve_graph
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
    def _fixture(self, root: Path) -> tuple[Path, Path]:
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
            [{"row_index": 0, "query_id": "q-core", "user_id": "u1", "benchmark_tier": "core", "query_type": "semantic", "is_high_memory_load": False}]
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

        rankings("q-core", "core", "semantic").to_csv(
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


if __name__ == "__main__":
    unittest.main()
