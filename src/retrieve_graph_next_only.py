"""Run the fixed E5 relational-after NEXT-only Graph ablation."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.build_graph import GraphBuildError, build_travel_graph
from src.model_download import sha256_file
from src.model_specs import E5_SMALL_SPEC
from src.retrieve_graph import (
    AUDIT_COLUMNS,
    DENSE_SEED_COUNT,
    EVENT_INDEX_COLUMNS,
    EXPANSION_HOPS,
    GRAPH_RANKING_COLUMNS,
    NEXT_ONLY_EXPANSION_DIRECTION,
    QUERY_INDEX_COLUMNS,
    RRF_K,
    SUMMARY_COLUMNS,
    GraphRetrievalError,
    _load_embeddings,
    _read_csv,
    _retrieve_dataset,
    _summary_rows,
    _validate_event_mapping,
    _validate_index,
    _validate_manifest,
    _validate_query_mapping,
    _validate_vector_rankings,
    _write_outputs_atomically,
)
from src.retrieve_vector import (
    CORE_EMBEDDINGS_FILENAME,
    CORE_INDEX_FILENAME,
    CORE_RANKINGS_FILENAME as VECTOR_CORE_RANKINGS_FILENAME,
    EVENT_EMBEDDINGS_FILENAME,
    EVENT_INDEX_FILENAME,
)


QUERY_TYPE = "relational_after"
RANKINGS_FILENAME = "tokyo_graph_next_only_relational_after_rankings.csv"
AUDIT_FILENAME = "tokyo_graph_next_only_relational_after_expansion_audit.csv"
SUMMARY_FILENAME = "tokyo_graph_next_only_relational_after_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_graph_next_only_relational_after_run_manifest.json"


@dataclass(frozen=True)
class NextOnlyGraphSummary:
    node_count: int
    edge_count: int
    next_edge_count: int
    query_count: int
    high_load_query_count: int
    ranking_count: int
    audit_count: int
    unique_expanded_count: int


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def retrieve_graph_next_only(
    events_path: str | Path,
    vector_dir: str | Path,
    output_dir: str | Path,
) -> NextOnlyGraphSummary:
    """Validate E5 artifacts and run SELF+NEXT retrieval for relational-after."""

    events_path = Path(events_path)
    vector_dir = Path(vector_dir)
    output_dir = Path(output_dir)
    vector_manifest, vector_manifest_path = _validate_manifest(vector_dir)
    model = vector_manifest["model"]
    if (
        model.get("id") != E5_SMALL_SPEC.model_id
        or model.get("revision") != E5_SMALL_SPEC.revision
    ):
        raise GraphRetrievalError(
            "NEXT-only ablation requires the pinned multilingual-e5-small Vector run"
        )

    events_source = _read_csv(events_path, "events")
    try:
        built = build_travel_graph(events_source)
    except GraphBuildError as exc:
        raise GraphRetrievalError(str(exc)) from exc

    event_index = _validate_index(
        vector_dir / EVENT_INDEX_FILENAME,
        EVENT_INDEX_COLUMNS,
        "event_id",
        "event embedding index",
    )
    core_index = _validate_index(
        vector_dir / CORE_INDEX_FILENAME,
        QUERY_INDEX_COLUMNS,
        "query_id",
        "core query embedding index",
    )
    allowed_query_types = {
        "semantic",
        "temporal_spatial",
        "relational_after",
    }
    if not set(core_index["query_type"]).issubset(allowed_query_types):
        raise GraphRetrievalError("core query index has unexpected query types")
    relational_index = core_index.loc[
        core_index["query_type"].eq(QUERY_TYPE)
    ].copy()
    if relational_index.empty:
        raise GraphRetrievalError("core query index has no relational_after queries")

    _validate_event_mapping(built.events, event_index)
    _validate_query_mapping((relational_index,), event_index)
    event_embeddings = _load_embeddings(
        vector_dir / EVENT_EMBEDDINGS_FILENAME, len(event_index), "event"
    )
    core_embeddings = _load_embeddings(
        vector_dir / CORE_EMBEDDINGS_FILENAME, len(core_index), "core query"
    )
    core_vector = _validate_vector_rankings(
        vector_dir / VECTOR_CORE_RANKINGS_FILENAME,
        core_index,
        event_index,
        event_embeddings,
        core_embeddings,
        "core",
        "core Vector rankings",
    )
    relational_vector = core_vector.loc[
        core_vector["query_id"].isin(relational_index["query_id"])
    ].copy()

    rankings, audit, statistics = _retrieve_dataset(
        built.graph,
        relational_index,
        relational_vector,
        NEXT_ONLY_EXPANSION_DIRECTION,
    )
    if set(audit["relation"]) - {"SELF", "NEXT"}:
        raise GraphRetrievalError("NEXT-only traversal produced an unexpected relation")

    summary_rows: list[dict[str, object]] = []
    for cohort in ("main", "high_memory_load"):
        summary_rows.extend(
            _summary_rows(
                built.nodes,
                built.edges,
                relational_index,
                rankings,
                audit,
                statistics,
                cohort,
                NEXT_ONLY_EXPANSION_DIRECTION,
            )
        )
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    frames = {
        output_dir / RANKINGS_FILENAME: rankings.loc[:, GRAPH_RANKING_COLUMNS],
        output_dir / AUDIT_FILENAME: audit.loc[:, AUDIT_COLUMNS],
        output_dir / SUMMARY_FILENAME: summary_frame,
    }
    input_paths = {
        "canonical_events": events_path,
        "vector_run_manifest": vector_manifest_path,
        "event_embeddings": vector_dir / EVENT_EMBEDDINGS_FILENAME,
        "event_index": vector_dir / EVENT_INDEX_FILENAME,
        "core_query_embeddings": vector_dir / CORE_EMBEDDINGS_FILENAME,
        "core_query_index": vector_dir / CORE_INDEX_FILENAME,
        "core_vector_rankings": vector_dir / VECTOR_CORE_RANKINGS_FILENAME,
    }
    manifest_base = {
        "ablation": "relational_after_next_only",
        "dense_seed_model": {
            "id": model["id"],
            "manifest_sha256": model["manifest_sha256"],
            "revision": model["revision"],
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "method": "graph",
        "parameters": {
            "dense_seed_count": DENSE_SEED_COUNT,
            "expansion_direction": NEXT_ONLY_EXPANSION_DIRECTION,
            "expansion_hops": EXPANSION_HOPS,
            "expansion_order": "seed rank; SELF; NEXT neighbors by event_id; first discovery",
            "query_type": QUERY_TYPE,
            "ranking_depth": "all events belonging to the query user",
            "rrf_k": RRF_K,
            "rrf_weights": "equal",
        },
        "runtime_versions": {
            "networkx": _package_version("networkx"),
            "numpy": _package_version("numpy"),
            "os": platform.platform(),
            "pandas": _package_version("pandas"),
            "python": sys.version.split()[0],
        },
    }
    _write_outputs_atomically(frames, output_dir / RUN_MANIFEST_FILENAME, manifest_base)

    return NextOnlyGraphSummary(
        node_count=len(built.nodes),
        edge_count=len(built.edges),
        next_edge_count=int(built.edges["relation"].eq("NEXT").sum()),
        query_count=len(relational_index),
        high_load_query_count=int(relational_index["_is_high_memory_load"].sum()),
        ranking_count=len(rankings),
        audit_count=len(audit),
        unique_expanded_count=int(statistics["unique_expanded_count"].sum()),
    )
