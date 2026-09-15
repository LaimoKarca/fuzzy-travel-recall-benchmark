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
    DEFAULT_MINILM_EVALUATION_DIR,
    DEFAULT_CORE_QUERIES_INPUT,
    DEFAULT_STRESS_QUERIES_INPUT,
    DEFAULT_E5_EVALUATION_DIR,
    DEFAULT_ENCODER_COMPARISON_DIR,
    DEFAULT_E5_VECTOR_OUTPUT_DIR,
    DEFAULT_E5_GRAPH_OUTPUT_DIR,
    DEFAULT_E5_NEXT_ONLY_GRAPH_OUTPUT_DIR,
    DEFAULT_GRAPH_DIRECTION_EVALUATION_DIR,
    DEFAULT_PPR_OUTPUT_DIR,
    DEFAULT_STRUCTURED_NEXT_OUTPUT_DIR,
    DEFAULT_PHASE2_EVALUATION_DIR,
    DEFAULT_PPR_ALL_EVENT_OUTPUT_DIR,
    DEFAULT_PPR_SENSITIVITY_OUTPUT_DIR,
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

        self.assertEqual(args.model, "multilingual-e5-small")
        self.assertIsNone(args.output_dir)

        e5 = build_parser().parse_args(
            ["download-models", "--model", "multilingual-e5-small"]
        )
        self.assertEqual(e5.model, "multilingual-e5-small")

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

    def test_compare_encoder_runs_uses_frozen_and_formal_directories(self) -> None:
        args = build_parser().parse_args(["compare-encoder-runs"])

        self.assertEqual(args.minilm_evaluation_dir, DEFAULT_MINILM_EVALUATION_DIR)
        self.assertEqual(args.e5_evaluation_dir, DEFAULT_E5_EVALUATION_DIR)
        self.assertEqual(args.output_dir, DEFAULT_ENCODER_COMPARISON_DIR)

    def test_next_only_graph_uses_e5_candidate_directories(self) -> None:
        args = build_parser().parse_args(["retrieve-graph-next-only"])

        self.assertEqual(args.events, DEFAULT_QUERY_INPUT)
        self.assertEqual(args.vector_dir, DEFAULT_E5_VECTOR_OUTPUT_DIR)
        self.assertEqual(args.output_dir, DEFAULT_E5_NEXT_ONLY_GRAPH_OUTPUT_DIR)

    def test_graph_direction_evaluation_uses_candidate_directories(self) -> None:
        args = build_parser().parse_args(["evaluate-graph-direction"])

        self.assertEqual(args.queries, DEFAULT_CORE_QUERIES_INPUT)
        self.assertEqual(args.e5_vector_dir, DEFAULT_E5_VECTOR_OUTPUT_DIR)
        self.assertEqual(args.symmetric_graph_dir, DEFAULT_E5_GRAPH_OUTPUT_DIR)
        self.assertEqual(
            args.next_only_graph_dir, DEFAULT_E5_NEXT_ONLY_GRAPH_OUTPUT_DIR
        )
        self.assertEqual(args.output_dir, DEFAULT_GRAPH_DIRECTION_EVALUATION_DIR)

    def test_phase_two_graph_extension_defaults_are_isolated(self) -> None:
        ppr = build_parser().parse_args(["retrieve-graph-ppr"])
        self.assertEqual(ppr.events, DEFAULT_QUERY_INPUT)
        self.assertEqual(ppr.vector_dir, DEFAULT_E5_VECTOR_OUTPUT_DIR)
        self.assertEqual(ppr.output_dir, DEFAULT_PPR_OUTPUT_DIR)

        oracle = build_parser().parse_args(["diagnose-structured-next"])
        self.assertEqual(oracle.queries, DEFAULT_CORE_QUERIES_INPUT)
        self.assertEqual(oracle.output_dir, DEFAULT_STRUCTURED_NEXT_OUTPUT_DIR)

        evaluation = build_parser().parse_args(["evaluate-graph-extensions"])
        self.assertEqual(evaluation.ppr_dir, DEFAULT_PPR_OUTPUT_DIR)
        self.assertEqual(evaluation.output_dir, DEFAULT_PHASE2_EVALUATION_DIR)

        all_event = build_parser().parse_args(["retrieve-graph-ppr-all-event"])
        self.assertEqual(all_event.vector_dir, DEFAULT_E5_VECTOR_OUTPUT_DIR)
        self.assertEqual(all_event.output_dir, DEFAULT_PPR_ALL_EVENT_OUTPUT_DIR)

        sensitivity = build_parser().parse_args(
            ["evaluate-ppr-personalization-sensitivity"]
        )
        self.assertEqual(sensitivity.top5_ppr_dir, DEFAULT_PPR_OUTPUT_DIR)
        self.assertEqual(
            sensitivity.all_event_ppr_dir, DEFAULT_PPR_ALL_EVENT_OUTPUT_DIR
        )
        self.assertEqual(
            sensitivity.output_dir, DEFAULT_PPR_SENSITIVITY_OUTPUT_DIR
        )


if __name__ == "__main__":
    unittest.main()
