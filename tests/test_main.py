from __future__ import annotations

import unittest

from main import (
    DEFAULT_COHORT_INPUT,
    DEFAULT_COHORT_OUTPUT_DIR,
    DEFAULT_EVENT_OUTPUT_DIR,
    DEFAULT_QUERY_INPUT,
    DEFAULT_QUERY_OUTPUT_DIR,
    DEFAULT_RETRIEVAL_OUTPUT_DIR,
    DEFAULT_DOCUMENTS_INPUT,
    DEFAULT_MODEL_DIR,
    DEFAULT_VECTOR_OUTPUT_DIR,
    DEFAULT_GRAPH_OUTPUT_DIR,
    DEFAULT_FLAT_OUTPUT_DIR,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_CORE_QUERIES_INPUT,
    DEFAULT_STRESS_QUERIES_INPUT,
    build_parser,
)


class MainCliDefaultsTests(unittest.TestCase):
    def test_prepare_cohorts_uses_stage_one_directory(self) -> None:
        args = build_parser().parse_args(["prepare-cohorts"])

        self.assertEqual(args.output_dir, DEFAULT_COHORT_OUTPUT_DIR)
        self.assertEqual(args.output_dir.name, "stage_01_cohorts")

    def test_prepare_events_connects_stage_one_to_stage_two(self) -> None:
        args = build_parser().parse_args(["prepare-events"])

        self.assertEqual(args.input, DEFAULT_COHORT_INPUT)
        self.assertEqual(args.input.parent, DEFAULT_COHORT_OUTPUT_DIR)
        self.assertEqual(args.output_dir, DEFAULT_EVENT_OUTPUT_DIR)
        self.assertEqual(args.output_dir.name, "stage_02_canonical_events")

    def test_generate_queries_connects_stage_two_to_stage_three(self) -> None:
        args = build_parser().parse_args(["generate-queries"])

        self.assertEqual(args.input, DEFAULT_QUERY_INPUT)
        self.assertEqual(args.input.parent, DEFAULT_EVENT_OUTPUT_DIR)
        self.assertEqual(args.output_dir, DEFAULT_QUERY_OUTPUT_DIR)
        self.assertEqual(args.output_dir.name, "stage_03_queries")

    def test_retrieve_flat_connects_stage_two_and_three_to_stage_four(self) -> None:
        args = build_parser().parse_args(["retrieve-flat"])

        self.assertEqual(args.events, DEFAULT_QUERY_INPUT)
        self.assertEqual(args.core_queries, DEFAULT_CORE_QUERIES_INPUT)
        self.assertEqual(args.stress_queries, DEFAULT_STRESS_QUERIES_INPUT)
        self.assertEqual(args.output_dir, DEFAULT_RETRIEVAL_OUTPUT_DIR)
        self.assertEqual(args.output_dir.name, "stage_04_retrieval")

    def test_download_models_uses_project_model_directory(self) -> None:
        args = build_parser().parse_args(["download-models"])

        self.assertEqual(args.output_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.output_dir.name, "paraphrase-multilingual-MiniLM-L12-v2")

    def test_retrieve_vector_uses_shared_documents_and_local_model(self) -> None:
        args = build_parser().parse_args(["retrieve-vector"])

        self.assertEqual(args.documents, DEFAULT_DOCUMENTS_INPUT)
        self.assertEqual(args.core_queries, DEFAULT_CORE_QUERIES_INPUT)
        self.assertEqual(args.stress_queries, DEFAULT_STRESS_QUERIES_INPUT)
        self.assertEqual(args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.output_dir, DEFAULT_VECTOR_OUTPUT_DIR)

    def test_retrieve_graph_uses_canonical_events_and_vector_outputs(self) -> None:
        args = build_parser().parse_args(["retrieve-graph"])

        self.assertEqual(args.events, DEFAULT_QUERY_INPUT)
        self.assertEqual(args.vector_dir, DEFAULT_VECTOR_OUTPUT_DIR)
        self.assertEqual(args.output_dir, DEFAULT_GRAPH_OUTPUT_DIR)

    def test_evaluate_uses_stage_five_output_directory(self) -> None:
        args = build_parser().parse_args(["evaluate"])

        self.assertEqual(args.events, DEFAULT_QUERY_INPUT)
        self.assertEqual(args.core_queries, DEFAULT_CORE_QUERIES_INPUT)
        self.assertEqual(args.between_queries, DEFAULT_STRESS_QUERIES_INPUT)
        self.assertEqual(args.flat_dir, DEFAULT_FLAT_OUTPUT_DIR)
        self.assertEqual(args.vector_dir, DEFAULT_VECTOR_OUTPUT_DIR)
        self.assertEqual(args.graph_dir, DEFAULT_GRAPH_OUTPUT_DIR)
        self.assertEqual(args.output_dir, DEFAULT_EVALUATION_OUTPUT_DIR)
        self.assertEqual(args.output_dir.name, "stage_05_evaluation")


if __name__ == "__main__":
    unittest.main()
