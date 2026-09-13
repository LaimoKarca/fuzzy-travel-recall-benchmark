"""Command-line entry point for the fuzzy travel recall benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.prepare_cohorts import CohortPreparationError, prepare_cohorts
from src.prepare_events import EventPreparationError, prepare_events
from src.generate_queries import QueryGenerationError, generate_queries
from src.retrieve_flat import FlatRetrievalError, retrieve_flat
from src.model_download import (
    MODEL_DIRECTORY_NAME,
    MODEL_ID,
    MODEL_REVISION,
    ModelDownloadError,
    download_models,
)
from src.retrieve_vector import VectorRetrievalError, retrieve_vector
from src.retrieve_graph import GraphRetrievalError, retrieve_graph
from src.evaluate import EvaluationError, evaluate


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "tokyo_checkins.csv"
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "data" / "prepared"
DEFAULT_COHORT_OUTPUT_DIR = DEFAULT_PREPARED_ROOT / "stage_01_cohorts"
DEFAULT_EVENT_OUTPUT_DIR = DEFAULT_PREPARED_ROOT / "stage_02_canonical_events"
DEFAULT_QUERY_OUTPUT_DIR = DEFAULT_PREPARED_ROOT / "stage_03_queries"
DEFAULT_RETRIEVAL_OUTPUT_DIR = DEFAULT_PREPARED_ROOT / "stage_04_retrieval"
DEFAULT_DOCUMENTS_INPUT = (
    DEFAULT_RETRIEVAL_OUTPUT_DIR / "common" / "tokyo_event_documents.csv"
)
DEFAULT_VECTOR_OUTPUT_DIR = DEFAULT_RETRIEVAL_OUTPUT_DIR / "vector"
DEFAULT_GRAPH_OUTPUT_DIR = DEFAULT_RETRIEVAL_OUTPUT_DIR / "graph"
DEFAULT_EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "stage_05_evaluation"
DEFAULT_FLAT_OUTPUT_DIR = DEFAULT_RETRIEVAL_OUTPUT_DIR / "flat"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "sentence-transformers" / MODEL_DIRECTORY_NAME
)
DEFAULT_COHORT_INPUT = (
    DEFAULT_COHORT_OUTPUT_DIR
    / "tokyo_checkins_users_with_at_least_10_trails.csv"
)
DEFAULT_QUERY_INPUT = (
    DEFAULT_EVENT_OUTPUT_DIR
    / "tokyo_canonical_events_users_with_at_least_10_clean_trails.csv"
)
DEFAULT_CORE_QUERIES_INPUT = DEFAULT_QUERY_OUTPUT_DIR / "tokyo_recall_queries_main.csv"
DEFAULT_STRESS_QUERIES_INPUT = (
    DEFAULT_QUERY_OUTPUT_DIR / "tokyo_relational_between_queries_main.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the fuzzy travel recall benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-cohorts",
        help="Create the fixed >=10-trail and >=20-trail Tokyo cohorts.",
    )
    prepare_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source CSV (default: {DEFAULT_INPUT})",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COHORT_OUTPUT_DIR,
        help=f"Stage 1 output directory (default: {DEFAULT_COHORT_OUTPUT_DIR})",
    )

    events_parser = subparsers.add_parser(
        "prepare-events",
        help="Remove trails with >8-hour gaps and build canonical events.",
    )
    events_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_COHORT_INPUT,
        help=f"Preliminary main-cohort CSV (default: {DEFAULT_COHORT_INPUT})",
    )

    queries_parser = subparsers.add_parser(
        "generate-queries",
        help="Generate deterministic core and relational-between recall queries.",
    )
    queries_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_QUERY_INPUT,
        help=f"Canonical main-cohort CSV (default: {DEFAULT_QUERY_INPUT})",
    )
    queries_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_QUERY_OUTPUT_DIR,
        help=f"Stage 3 output directory (default: {DEFAULT_QUERY_OUTPUT_DIR})",
    )
    events_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVENT_OUTPUT_DIR,
        help=f"Stage 2 output directory (default: {DEFAULT_EVENT_OUTPUT_DIR})",
    )

    flat_parser = subparsers.add_parser(
        "retrieve-flat",
        help="Build shared event documents and run per-user Flat BM25 retrieval.",
    )
    flat_parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_QUERY_INPUT,
        help=f"Canonical Main events CSV (default: {DEFAULT_QUERY_INPUT})",
    )
    flat_parser.add_argument(
        "--core-queries",
        type=Path,
        default=DEFAULT_CORE_QUERIES_INPUT,
        help=f"Core queries CSV (default: {DEFAULT_CORE_QUERIES_INPUT})",
    )
    flat_parser.add_argument(
        "--stress-queries",
        type=Path,
        default=DEFAULT_STRESS_QUERIES_INPUT,
        help=f"Between stress queries CSV (default: {DEFAULT_STRESS_QUERIES_INPUT})",
    )
    flat_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RETRIEVAL_OUTPUT_DIR,
        help=f"Stage 4 retrieval root (default: {DEFAULT_RETRIEVAL_OUTPUT_DIR})",
    )

    model_parser = subparsers.add_parser(
        "download-models",
        help="Download and validate the pinned multilingual MiniLM model.",
    )
    model_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local model directory (default: {DEFAULT_MODEL_DIR})",
    )

    vector_parser = subparsers.add_parser(
        "retrieve-vector",
        help="Run per-user dense retrieval with the local multilingual MiniLM model.",
    )
    vector_parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_INPUT,
        help=f"Shared event documents CSV (default: {DEFAULT_DOCUMENTS_INPUT})",
    )
    vector_parser.add_argument(
        "--core-queries",
        type=Path,
        default=DEFAULT_CORE_QUERIES_INPUT,
        help=f"Core queries CSV (default: {DEFAULT_CORE_QUERIES_INPUT})",
    )
    vector_parser.add_argument(
        "--stress-queries",
        type=Path,
        default=DEFAULT_STRESS_QUERIES_INPUT,
        help=f"Between stress queries CSV (default: {DEFAULT_STRESS_QUERIES_INPUT})",
    )
    vector_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Offline model directory (default: {DEFAULT_MODEL_DIR})",
    )
    vector_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_VECTOR_OUTPUT_DIR,
        help=f"Vector output directory (default: {DEFAULT_VECTOR_OUTPUT_DIR})",
    )

    graph_parser = subparsers.add_parser(
        "retrieve-graph",
        help="Build the heterogeneous graph and run dense-seed graph retrieval.",
    )
    graph_parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_QUERY_INPUT,
        help=f"Canonical Main events CSV (default: {DEFAULT_QUERY_INPUT})",
    )
    graph_parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_OUTPUT_DIR,
        help=f"Validated Stage 4.2 artifact directory (default: {DEFAULT_VECTOR_OUTPUT_DIR})",
    )
    graph_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GRAPH_OUTPUT_DIR,
        help=f"Graph output directory (default: {DEFAULT_GRAPH_OUTPUT_DIR})",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate Flat, Vector, and Graph rankings and audit Graph cases.",
    )
    evaluate_parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_QUERY_INPUT,
        help=f"Canonical Main events CSV (default: {DEFAULT_QUERY_INPUT})",
    )
    evaluate_parser.add_argument(
        "--core-queries",
        type=Path,
        default=DEFAULT_CORE_QUERIES_INPUT,
        help=f"Core queries CSV (default: {DEFAULT_CORE_QUERIES_INPUT})",
    )
    evaluate_parser.add_argument(
        "--between-queries",
        type=Path,
        default=DEFAULT_STRESS_QUERIES_INPUT,
        help=f"Between stress queries CSV (default: {DEFAULT_STRESS_QUERIES_INPUT})",
    )
    evaluate_parser.add_argument(
        "--flat-dir",
        type=Path,
        default=DEFAULT_FLAT_OUTPUT_DIR,
        help=f"Flat ranking directory (default: {DEFAULT_FLAT_OUTPUT_DIR})",
    )
    evaluate_parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_OUTPUT_DIR,
        help=f"Validated Vector artifact directory (default: {DEFAULT_VECTOR_OUTPUT_DIR})",
    )
    evaluate_parser.add_argument(
        "--graph-dir",
        type=Path,
        default=DEFAULT_GRAPH_OUTPUT_DIR,
        help=f"Validated Graph artifact directory (default: {DEFAULT_GRAPH_OUTPUT_DIR})",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_DIR,
        help=f"Stage 5 output directory (default: {DEFAULT_EVALUATION_OUTPUT_DIR})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare-cohorts":
        try:
            summary = prepare_cohorts(args.input, args.output_dir)
        except CohortPreparationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Cohort preparation completed.")
        for label, counts in summary.items():
            print(
                f"{label}: {counts.users:,} users, "
                f"{counts.trails:,} trails, {counts.checkins:,} check-ins"
            )
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "prepare-events":
        try:
            summary = prepare_events(args.input, args.output_dir)
        except EventPreparationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Canonical event preparation completed.")
        print(
            f"Removed: {summary.removed_trails:,} trails, "
            f"{summary.removed_checkins:,} check-ins"
        )
        for label, counts in (
            ("Input", summary.input_counts),
            ("After trail cleaning", summary.cleaned_counts),
            ("Main (>=10 clean trails)", summary.main_counts),
            ("High load (>=20 clean trails)", summary.high_load_counts),
        ):
            print(
                f"{label}: {counts.users:,} users, "
                f"{counts.trails:,} trails, {counts.checkins:,} events"
            )
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "generate-queries":
        try:
            summary = generate_queries(args.input, args.output_dir)
        except QueryGenerationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Benchmark query generation completed.")
        for label, counts in (
            ("Core main", summary.core_main),
            ("Core high load", summary.core_high_load),
            ("Between stress main", summary.stress_main),
            ("Between stress high load", summary.stress_high_load),
        ):
            details = ", ".join(
                f"{query_type}={count:,}"
                for query_type, count in sorted(counts.items())
            )
            print(f"{label}: {sum(counts.values()):,} queries ({details})")
        print(f"Audit: {summary.audit_rows:,} user/query-type rows")
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "retrieve-flat":
        try:
            summary = retrieve_flat(
                args.events,
                args.core_queries,
                args.stress_queries,
                args.output_dir,
            )
        except FlatRetrievalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Flat BM25 retrieval completed.")
        print(f"Event documents: {summary.document_count:,}")
        print(
            f"Core: {summary.core_query_count:,} queries, "
            f"{summary.core_ranking_count:,} ranking rows "
            f"({summary.high_load_core_query_count:,} high-load queries)"
        )
        print(
            f"Between stress: {summary.stress_query_count:,} queries, "
            f"{summary.stress_ranking_count:,} ranking rows "
            f"({summary.high_load_stress_query_count:,} high-load queries)"
        )
        print(f"All-zero-score queries: {summary.all_zero_score_query_count:,}")
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "download-models":
        try:
            summary = download_models(args.output_dir)
        except ModelDownloadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        status = "downloaded and validated" if summary.downloaded else "already valid"
        print(f"Model {status}.")
        print(f"Model: {MODEL_ID}")
        print(f"Revision: {MODEL_REVISION}")
        print(f"Required files: {summary.file_count}")
        print(f"Model directory: {summary.model_dir.resolve()}")
        print(f"Manifest: {summary.manifest_path.resolve()}")
        print(f"Reference runtime: {summary.runtime_manifest_path.resolve()}")
        return 0

    if args.command == "retrieve-vector":
        try:
            summary = retrieve_vector(
                args.documents,
                args.core_queries,
                args.stress_queries,
                args.model_dir,
                args.output_dir,
            )
        except VectorRetrievalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Vector retrieval completed.")
        print(
            f"Embeddings: events={summary.document_count:,}, "
            f"core={summary.core_query_count:,}, between={summary.stress_query_count:,}"
        )
        print(
            f"Rankings: core={summary.core_ranking_count:,} rows "
            f"({summary.high_load_core_query_count:,} high-load queries), "
            f"between={summary.stress_ranking_count:,} rows "
            f"({summary.high_load_stress_query_count:,} high-load queries)"
        )
        print(
            "Tokenizer truncations: "
            f"documents={summary.truncated_document_count:,}, "
            f"core={summary.truncated_core_query_count:,}, "
            f"between={summary.truncated_stress_query_count:,}"
        )
        print(f"All-equal-score queries: {summary.all_equal_score_query_count:,}")
        print(f"Device: cpu; output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "retrieve-graph":
        try:
            summary = retrieve_graph(
                args.events,
                args.vector_dir,
                args.output_dir,
            )
        except GraphRetrievalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Graph retrieval completed.")
        print(
            f"Graph: {summary.node_count:,} nodes, {summary.edge_count:,} edges "
            f"({summary.next_edge_count:,} NEXT edges)"
        )
        print(
            f"Core: {summary.core_query_count:,} queries, "
            f"{summary.core_ranking_count:,} ranking rows, "
            f"{summary.core_audit_count:,} expansion audit rows, "
            f"{summary.core_unique_expanded_count:,} unique expanded candidates"
        )
        print(
            f"Between: {summary.stress_query_count:,} queries, "
            f"{summary.stress_ranking_count:,} ranking rows, "
            f"{summary.stress_audit_count:,} expansion audit rows, "
            f"{summary.stress_unique_expanded_count:,} unique expanded candidates"
        )
        print(
            f"High-load queries: core={summary.high_load_core_query_count:,}, "
            f"between={summary.high_load_stress_query_count:,}"
        )
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    if args.command == "evaluate":
        try:
            summary = evaluate(
                args.events,
                args.core_queries,
                args.between_queries,
                args.flat_dir,
                args.vector_dir,
                args.graph_dir,
                args.output_dir,
            )
        except EvaluationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print("Formal evaluation completed.")
        print(
            f"Per-query metrics: {summary.per_query_count:,} rows "
            f"(core={summary.core_query_count:,} queries, "
            f"between={summary.between_query_count:,} queries)"
        )
        print(
            f"High-load queries: core={summary.high_load_core_query_count:,}, "
            f"between={summary.high_load_between_query_count:,}"
        )
        print(
            "Graph vs Vector: "
            f"improved={summary.graph_improved_count:,}, "
            f"unchanged={summary.graph_unchanged_count:,}, "
            f"worsened={summary.graph_worsened_count:,}"
        )
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
