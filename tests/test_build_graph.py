from __future__ import annotations

import unittest

import pandas as pd

from src.build_graph import GraphBuildError, build_travel_graph


def sample_events() -> pd.DataFrame:
    rows = []
    for position in range(1, 4):
        rows.append(
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
                "next_event_id": "" if position == 3 else f"e{position + 1}",
                "is_high_memory_load": "False",
            }
        )
    return pd.DataFrame(rows)


class BuildGraphTests(unittest.TestCase):
    def test_builds_typed_nodes_edges_and_reverse_traversal(self) -> None:
        built = build_travel_graph(sample_events())

        self.assertEqual(len(built.nodes), 9)  # 1 user + 3 events + 3 POIs + 1 category + 1 trail
        self.assertEqual(len(built.edges), 14)  # 3 HAS + 3 AT + 3 IN_TRAIL + 3 CATEGORY + 2 NEXT
        self.assertEqual(built.nodes["node_id"].nunique(), 9)
        self.assertEqual(
            built.edges["relation"].value_counts().to_dict(),
            {"HAS": 3, "AT": 3, "CATEGORY": 3, "IN_TRAIL": 3, "NEXT": 2},
        )
        self.assertEqual(
            built.graph.edges["event::e1", "event::e2"]["relation"], "NEXT"
        )
        self.assertIn("event::e1", list(built.graph.predecessors("event::e2")))

    def test_rejects_nonreciprocal_and_cross_trail_links(self) -> None:
        nonreciprocal = sample_events()
        nonreciprocal.loc[1, "previous_event_id"] = ""
        with self.assertRaisesRegex(GraphBuildError, "not reciprocal"):
            build_travel_graph(nonreciprocal)

        cross_trail = sample_events()
        cross_trail.loc[1, "trail_id"] = "t2"
        with self.assertRaisesRegex(GraphBuildError, "crosses"):
            build_travel_graph(cross_trail)

    def test_rejects_inconsistent_poi_metadata_and_long_gap(self) -> None:
        inconsistent = pd.concat([sample_events(), sample_events().iloc[[0]]], ignore_index=True)
        inconsistent.loc[3, "event_id"] = "e4"
        inconsistent.loc[3, "name"] = "Different"
        inconsistent.loc[3, "previous_event_id"] = ""
        inconsistent.loc[3, "next_event_id"] = ""
        with self.assertRaisesRegex(GraphBuildError, "inconsistent metadata"):
            build_travel_graph(inconsistent)

        long_gap = sample_events()
        long_gap.loc[1, "timestamp"] = "2018-01-01 10:00:01"
        with self.assertRaisesRegex(GraphBuildError, "timestamp gap"):
            build_travel_graph(long_gap)


if __name__ == "__main__":
    unittest.main()
