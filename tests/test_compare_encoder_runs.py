from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compare_encoder_runs import EncoderComparisonError, compare_encoder_runs
from src.model_download import sha256_file
from src.model_specs import E5_SMALL_SPEC, MINILM_SPEC


class EncoderComparisonTests(unittest.TestCase):
    def _run(self, root: Path, label: str, model_id: str, dense_rr: float) -> Path:
        evaluation = root / label / "evaluation"
        vector_dir = root / label / "vector"
        graph_dir = root / label / "graph"
        evaluation.mkdir(parents=True)
        vector_dir.mkdir()
        graph_dir.mkdir()

        vector_manifest = {
            "method": "vector",
            "model": {"id": model_id, "revision": "fixed", "manifest_sha256": "0" * 64},
            "outputs": {},
        }
        vector_manifest_path = vector_dir / "tokyo_vector_run_manifest.json"
        vector_manifest_path.write_text(json.dumps(vector_manifest), encoding="utf-8")
        graph_manifest = {
            "method": "graph",
            "inputs": {
                "vector_run_manifest": {
                    "path": str(vector_manifest_path.resolve()),
                    "sha256": sha256_file(vector_manifest_path),
                }
            },
            "dense_seed_model": {"id": model_id},
            "outputs": {},
        }
        graph_manifest_path = graph_dir / "tokyo_graph_run_manifest.json"
        graph_manifest_path.write_text(json.dumps(graph_manifest), encoding="utf-8")

        metrics = []
        diagnostics = []
        vector_rankings = []
        graph_rankings = []
        query_types = ("semantic", "temporal_spatial", "relational_after")
        for index in range(320):
            query_id = f"q{index:03d}"
            query_type = query_types[index % 3]
            target = f"e{index:03d}"
            for method, rr in (
                ("flat", 1.0),
                ("vector", dense_rr),
                ("graph", dense_rr - (0.01 if index == 0 else 0.0)),
            ):
                metrics.append(
                    {
                        "query_id": query_id,
                        "user_id": f"u{index:03d}",
                        "benchmark_tier": "core",
                        "query_type": query_type,
                        "method": method,
                        "target_event_id": target,
                        "reciprocal_rank": rr,
                        "success_at_1": int(rr == 1.0),
                        "success_at_5": 1,
                    }
                )
            relation = "NEXT" if index == 0 else "SELF"
            diagnostics.append(
                {
                    "query_id": query_id,
                    "benchmark_tier": "core",
                    "graph_outcome_vs_vector": "worsened" if index == 0 else "unchanged",
                    "target_in_expansion": True,
                    "target_first_discovery_relation": relation,
                }
            )
            vector_rankings.append(
                {"query_id": query_id, "method": "vector", "rank": 1, "retrieved_event_id": target}
            )
            graph_rankings.append(
                {
                    "query_id": query_id,
                    "method": "graph",
                    "rank": 1,
                    "retrieved_event_id": target if index else "other",
                }
            )

        outputs = {
            "tokyo_per_query_metrics.csv": pd.DataFrame(metrics),
            "tokyo_core_metrics_by_query_type.csv": pd.DataFrame(
                [{"method": "vector"}]
            ),
            "tokyo_graph_query_diagnostics.csv": pd.DataFrame(diagnostics),
        }
        for filename, frame in outputs.items():
            frame.to_csv(evaluation / filename, index=False)
        pd.DataFrame(vector_rankings).to_csv(
            vector_dir / "tokyo_vector_core_rankings.csv", index=False
        )
        pd.DataFrame(graph_rankings).to_csv(
            graph_dir / "tokyo_graph_core_rankings.csv", index=False
        )
        evaluation_manifest = {
            "inputs": {
                "vector_run_manifest": {
                    "path": str(vector_manifest_path.resolve()),
                    "sha256": sha256_file(vector_manifest_path),
                },
                "graph_run_manifest": {
                    "path": str(graph_manifest_path.resolve()),
                    "sha256": sha256_file(graph_manifest_path),
                },
                "vector_core_rankings": {
                    "path": str((vector_dir / "tokyo_vector_core_rankings.csv").resolve()),
                    "sha256": sha256_file(vector_dir / "tokyo_vector_core_rankings.csv"),
                },
                "graph_core_rankings": {
                    "path": str((graph_dir / "tokyo_graph_core_rankings.csv").resolve()),
                    "sha256": sha256_file(graph_dir / "tokyo_graph_core_rankings.csv"),
                },
            },
            "outputs": {
                filename: sha256_file(evaluation / filename) for filename in outputs
            },
        }
        (evaluation / "tokyo_evaluation_run_manifest.json").write_text(
            json.dumps(evaluation_manifest), encoding="utf-8"
        )
        return evaluation

    def test_compares_metrics_diagnostics_and_top1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minilm = self._run(root, "minilm", MINILM_SPEC.model_id, 0.7)
            e5 = self._run(root, "e5", E5_SMALL_SPEC.model_id, 0.8)
            output = root / "comparison"

            summary = compare_encoder_runs(minilm, e5, output)

            self.assertAlmostEqual(summary.e5_dense_mrr, 0.8)
            metrics = pd.read_csv(output / "tokyo_minilm_vs_e5_core_metrics.csv")
            self.assertEqual(
                metrics["configuration"].tolist(),
                ["MiniLM Dense", "E5 Dense", "MiniLM Graph", "E5 Graph"],
            )
            self.assertAlmostEqual(metrics.iloc[1]["mrr_delta_vs_minilm"], 0.1)
            self.assertAlmostEqual(metrics.iloc[3]["mrr_delta_vs_minilm"], 0.1)
            diagnostics = pd.read_csv(
                output / "tokyo_minilm_vs_e5_graph_diagnostics.csv"
            )
            self.assertEqual(diagnostics["encoder"].tolist(), ["MiniLM", "E5", "E5_minus_MiniLM"])
            self.assertEqual(diagnostics.iloc[0]["target_first_reached_as_next"], 1)
            self.assertEqual(diagnostics.iloc[0]["structural_first_reach_count"], 1)
            self.assertAlmostEqual(
                diagnostics.iloc[0]["structural_first_reach_rate"], 1 / 320
            )
            self.assertEqual(diagnostics.iloc[0]["vector_graph_top1_identical_count"], 319)
            report = (output / "tokyo_minilm_vs_e5_validation_report.md").read_text(encoding="utf-8")
            self.assertIn("E5 Dense 相對 MiniLM Dense", report)

    def test_ground_truth_mismatch_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minilm = self._run(root, "minilm", MINILM_SPEC.model_id, 0.7)
            e5 = self._run(root, "e5", E5_SMALL_SPEC.model_id, 0.8)
            per_query = e5 / "tokyo_per_query_metrics.csv"
            frame = pd.read_csv(per_query)
            frame.loc[0, "target_event_id"] = "changed"
            frame.to_csv(per_query, index=False)
            manifest_path = e5 / "tokyo_evaluation_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["tokyo_per_query_metrics.csv"] = sha256_file(per_query)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "comparison"

            with self.assertRaisesRegex(EncoderComparisonError, "Ground Truth"):
                compare_encoder_runs(minilm, e5, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
