"""Run deterministic dense-seed graph expansion and RRF retrieval."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.build_graph import GraphBuildError, build_travel_graph
from src.model_download import sha256_file
from src.retrieve_vector import (
    BETWEEN_EMBEDDINGS_FILENAME,
    BETWEEN_INDEX_FILENAME,
    BETWEEN_RANKINGS_FILENAME as VECTOR_BETWEEN_RANKINGS_FILENAME,
    CORE_EMBEDDINGS_FILENAME,
    CORE_INDEX_FILENAME,
    CORE_RANKINGS_FILENAME as VECTOR_CORE_RANKINGS_FILENAME,
    EVENT_EMBEDDINGS_FILENAME,
    EVENT_INDEX_FILENAME,
    RUN_MANIFEST_FILENAME as VECTOR_RUN_MANIFEST_FILENAME,
    SUMMARY_FILENAME as VECTOR_SUMMARY_FILENAME,
)


METHOD = "graph"
DENSE_SEED_COUNT = 5
EXPANSION_HOPS = 1
EXPANSION_DIRECTION = "symmetric_previous_next"
NEXT_ONLY_EXPANSION_DIRECTION = "next_only"
VALID_EXPANSION_DIRECTIONS = {
    EXPANSION_DIRECTION,
    NEXT_ONLY_EXPANSION_DIRECTION,
}
RRF_K = 60

NODES_FILENAME = "tokyo_graph_nodes.csv"
EDGES_FILENAME = "tokyo_graph_edges.csv"
CORE_AUDIT_FILENAME = "tokyo_graph_core_expansion_audit.csv"
BETWEEN_AUDIT_FILENAME = "tokyo_graph_between_expansion_audit.csv"
CORE_RANKINGS_FILENAME = "tokyo_graph_core_rankings.csv"
BETWEEN_RANKINGS_FILENAME = "tokyo_graph_between_rankings.csv"
SUMMARY_FILENAME = "tokyo_graph_retrieval_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_graph_run_manifest.json"

EVENT_INDEX_COLUMNS = ("row_index", "event_id", "user_id", "is_high_memory_load")
QUERY_INDEX_COLUMNS = (
    "row_index",
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "is_high_memory_load",
)
VECTOR_RANKING_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "method",
    "rank",
    "retrieved_event_id",
    "score",
    "is_high_memory_load",
)
GRAPH_RANKING_COLUMNS = VECTOR_RANKING_COLUMNS
AUDIT_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "seed_rank",
    "seed_event_id",
    "relation",
    "expanded_event_id",
    "is_first_discovery",
    "expansion_rank",
    "is_high_memory_load",
)
SUMMARY_COLUMNS = (
    "method",
    "benchmark_tier",
    "cohort",
    "query_type",
    "query_count",
    "ranked_candidate_count",
    "graph_node_count",
    "graph_edge_count",
    "user_node_count",
    "event_node_count",
    "poi_node_count",
    "category_node_count",
    "trail_node_count",
    "next_edge_count",
    "dense_seed_count",
    "expansion_hops",
    "expansion_direction",
    "rrf_k",
    "expansion_traversal_count",
    "unique_expanded_candidate_count",
    "min_expanded_candidates_per_query",
    "max_expanded_candidates_per_query",
    "mean_expanded_candidates_per_query",
    "queries_with_no_new_neighbor",
)
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}


class GraphRetrievalError(RuntimeError):
    """Raised when Graph retrieval cannot complete safely."""


@dataclass(frozen=True)
class GraphRetrievalSummary:
    node_count: int
    edge_count: int
    next_edge_count: int
    core_query_count: int
    stress_query_count: int
    core_ranking_count: int
    stress_ranking_count: int
    core_audit_count: int
    stress_audit_count: int
    core_unique_expanded_count: int
    stress_unique_expanded_count: int
    high_load_core_query_count: int
    high_load_stress_query_count: int


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise GraphRetrievalError(f"{label} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise GraphRetrievalError(f"could not parse {label} CSV {path}: {exc}") from exc
    if frame.empty:
        raise GraphRetrievalError(f"{label} CSV contains no rows: {path}")
    return frame


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:3]
        raise GraphRetrievalError(
            f"{label} contains invalid boolean value(s): {', '.join(examples)}"
        )
    return normalized.isin(TRUE_VALUES)


def _validate_manifest(vector_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = vector_dir / VECTOR_RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise GraphRetrievalError(f"Vector run manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphRetrievalError(f"could not parse Vector run manifest: {exc}") from exc
    if manifest.get("method") != "vector" or not isinstance(manifest.get("outputs"), dict):
        raise GraphRetrievalError("Vector run manifest has an invalid structure")
    model = manifest.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("id"), str)
        or not isinstance(model.get("revision"), str)
        or not isinstance(model.get("manifest_sha256"), str)
    ):
        raise GraphRetrievalError("Vector run manifest has invalid model metadata")

    required = {
        EVENT_EMBEDDINGS_FILENAME,
        EVENT_INDEX_FILENAME,
        CORE_EMBEDDINGS_FILENAME,
        CORE_INDEX_FILENAME,
        BETWEEN_EMBEDDINGS_FILENAME,
        BETWEEN_INDEX_FILENAME,
        VECTOR_CORE_RANKINGS_FILENAME,
        VECTOR_BETWEEN_RANKINGS_FILENAME,
        VECTOR_SUMMARY_FILENAME,
    }
    missing_manifest_entries = sorted(required - set(manifest["outputs"]))
    if missing_manifest_entries:
        raise GraphRetrievalError(
            "Vector run manifest is missing output hash(es): "
            + ", ".join(missing_manifest_entries)
        )
    for filename in sorted(required):
        path = vector_dir / filename
        if not path.is_file():
            raise GraphRetrievalError(f"Vector artifact does not exist: {path}")
        if sha256_file(path) != manifest["outputs"][filename]:
            raise GraphRetrievalError(f"Vector artifact hash mismatch: {filename}")
    return manifest, manifest_path


def _validate_index(
    path: Path,
    required: tuple[str, ...],
    id_column: str,
    label: str,
) -> pd.DataFrame:
    frame = _read_csv(path, label)
    missing = [column for column in required if column not in frame]
    if missing:
        raise GraphRetrievalError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )
    frame = frame.loc[:, required].copy()
    for column in ("row_index", id_column, "user_id"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise GraphRetrievalError(f"{label} contains blank {column} value(s)")
    if frame[id_column].duplicated().any():
        raise GraphRetrievalError(f"{label} contains duplicate {id_column} values")
    try:
        numeric_row_indexes = pd.to_numeric(frame["row_index"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise GraphRetrievalError(f"{label} contains invalid row_index values") from exc
    if (
        not np.isfinite(numeric_row_indexes.to_numpy(dtype=float)).all()
        or not np.equal(numeric_row_indexes, np.floor(numeric_row_indexes)).all()
    ):
        raise GraphRetrievalError(f"{label} contains invalid row_index values")
    row_indexes = numeric_row_indexes.astype(int)
    if row_indexes.tolist() != list(range(len(frame))):
        raise GraphRetrievalError(f"{label} row_index must be continuous and 0-based")
    frame["row_index"] = row_indexes
    frame["_is_high_memory_load"] = _parse_boolean(
        frame["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    return frame


def _load_embeddings(path: Path, rows: int, label: str) -> np.ndarray:
    try:
        embeddings = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GraphRetrievalError(f"could not load {label} embeddings: {exc}") from exc
    if embeddings.shape != (rows, 384):
        raise GraphRetrievalError(
            f"{label} embeddings have shape {embeddings.shape}, expected {(rows, 384)}"
        )
    if embeddings.dtype != np.float32:
        raise GraphRetrievalError(f"{label} embeddings are not float32")
    if not np.isfinite(embeddings).all():
        raise GraphRetrievalError(f"{label} embeddings contain non-finite values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise GraphRetrievalError(f"{label} embeddings are not L2-normalized")
    return embeddings


def _validate_event_mapping(events: pd.DataFrame, index: pd.DataFrame) -> None:
    if set(events["event_id"]) != set(index["event_id"]):
        raise GraphRetrievalError("event embedding index does not match canonical events")
    event_lookup = events.set_index("event_id")
    for _, row in index.iterrows():
        event = event_lookup.loc[row["event_id"]]
        if event["user_id"] != row["user_id"]:
            raise GraphRetrievalError("event embedding index has an incorrect user_id")
        if bool(event["_is_high_memory_load"]) != bool(row["_is_high_memory_load"]):
            raise GraphRetrievalError("event embedding index has an incorrect High-load flag")


def _validate_query_mapping(
    query_indexes: tuple[pd.DataFrame, ...], event_index: pd.DataFrame
) -> None:
    user_high_load = event_index.groupby("user_id", sort=False)[
        "_is_high_memory_load"
    ].agg(["min", "max"])
    if (user_high_load["min"] != user_high_load["max"]).any():
        raise GraphRetrievalError(
            "event embedding index has inconsistent High-load flags within a user"
        )
    expected = user_high_load["min"].to_dict()
    for query_index in query_indexes:
        for _, query in query_index.iterrows():
            user_id = query["user_id"]
            if user_id not in expected:
                raise GraphRetrievalError(
                    "query embedding index references a user absent from events"
                )
            if bool(query["_is_high_memory_load"]) != bool(expected[user_id]):
                raise GraphRetrievalError(
                    "query embedding index has an incorrect High-load flag"
                )


def _validate_vector_rankings(
    path: Path,
    query_index: pd.DataFrame,
    event_index: pd.DataFrame,
    event_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    expected_tier: str,
    label: str,
) -> pd.DataFrame:
    rankings = _read_csv(path, label)
    missing = [column for column in VECTOR_RANKING_COLUMNS if column not in rankings]
    if missing:
        raise GraphRetrievalError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )
    rankings = rankings.loc[:, VECTOR_RANKING_COLUMNS].copy()
    if set(rankings["method"]) != {"vector"}:
        raise GraphRetrievalError(f"{label} contains an unexpected method")
    if set(rankings["benchmark_tier"]) != {expected_tier}:
        raise GraphRetrievalError(f"{label} contains an unexpected benchmark tier")
    try:
        rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
        rankings["score"] = pd.to_numeric(rankings["score"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise GraphRetrievalError(f"{label} contains invalid rank or score values") from exc
    rankings["_is_high_memory_load"] = _parse_boolean(
        rankings["is_high_memory_load"], f"{label}.is_high_memory_load"
    )

    event_rows = {
        user_id: group["row_index"].to_numpy(dtype=int)
        for user_id, group in event_index.groupby("user_id", sort=False)
    }
    query_ids = set(query_index["query_id"])
    if set(rankings["query_id"]) != query_ids:
        raise GraphRetrievalError(f"{label} query coverage does not match its index")

    ranking_groups = {query_id: group for query_id, group in rankings.groupby("query_id", sort=False)}
    for query_row, query in query_index.iterrows():
        if query["benchmark_tier"] != expected_tier:
            raise GraphRetrievalError(f"{label} query index contains an unexpected tier")
        if query["user_id"] not in event_rows:
            raise GraphRetrievalError(f"{label} references a user absent from events")
        candidate_rows = event_rows[query["user_id"]]
        scores = event_embeddings[candidate_rows] @ query_embeddings[query_row]
        order = sorted(
            range(len(candidate_rows)),
            key=lambda position: (
                -float(scores[position]),
                event_index.at[int(candidate_rows[position]), "event_id"],
            ),
        )
        expected_ids = [
            event_index.at[int(candidate_rows[position]), "event_id"] for position in order
        ]
        expected_scores = np.asarray([scores[position] for position in order], dtype=float)
        actual = ranking_groups[query["query_id"]].sort_values("rank", kind="stable")
        if actual["rank"].tolist() != list(range(1, len(expected_ids) + 1)):
            raise GraphRetrievalError(f"{label} has non-continuous ranks")
        if actual["retrieved_event_id"].tolist() != expected_ids:
            raise GraphRetrievalError(f"{label} does not match recomputed dense ranking")
        if not np.allclose(actual["score"].to_numpy(float), expected_scores, rtol=1e-7, atol=1e-7):
            raise GraphRetrievalError(f"{label} scores do not match recomputed cosine scores")
        if set(actual["user_id"]) != {query["user_id"]}:
            raise GraphRetrievalError(f"{label} crosses a user boundary")
        if actual["_is_high_memory_load"].ne(query["_is_high_memory_load"]).any():
            raise GraphRetrievalError(f"{label} has an inconsistent High-load flag")
    return rankings.sort_values(["user_id", "query_id", "rank"], kind="stable").reset_index(drop=True)


def _traverse_neighbors(
    graph: Any,
    event_id: str,
    expansion_direction: str = EXPANSION_DIRECTION,
) -> list[tuple[str, str]]:
    if expansion_direction not in VALID_EXPANSION_DIRECTIONS:
        raise GraphRetrievalError(
            f"unsupported expansion direction: {expansion_direction}"
        )
    node_id = f"event::{event_id}"
    neighbors: list[tuple[str, str]] = []
    if expansion_direction == EXPANSION_DIRECTION:
        for predecessor in graph.predecessors(node_id):
            if graph.edges[predecessor, node_id].get("relation") == "NEXT":
                neighbors.append((predecessor.removeprefix("event::"), "PREVIOUS"))
    for successor in graph.successors(node_id):
        if graph.edges[node_id, successor].get("relation") == "NEXT":
            neighbors.append((successor.removeprefix("event::"), "NEXT"))
    return sorted(neighbors, key=lambda item: item[0])


def _retrieve_dataset(
    graph: Any,
    query_index: pd.DataFrame,
    vector_rankings: pd.DataFrame,
    expansion_direction: str = EXPANSION_DIRECTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if expansion_direction not in VALID_EXPANSION_DIRECTIONS:
        raise GraphRetrievalError(
            f"unsupported expansion direction: {expansion_direction}"
        )
    audit_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    statistic_rows: list[dict[str, object]] = []
    ranking_groups = {
        query_id: group.sort_values("rank", kind="stable")
        for query_id, group in vector_rankings.groupby("query_id", sort=False)
    }

    for _, query in query_index.iterrows():
        dense = ranking_groups[query["query_id"]]
        seed_rows = dense.head(min(DENSE_SEED_COUNT, len(dense)))
        expansion_ranks: dict[str, int] = {}
        traversal_count = 0

        def record(seed_rank: int, seed_event_id: str, relation: str, candidate: str) -> None:
            nonlocal traversal_count
            traversal_count += 1
            first = candidate not in expansion_ranks
            if first:
                expansion_ranks[candidate] = len(expansion_ranks) + 1
            audit_rows.append(
                {
                    "query_id": query["query_id"],
                    "user_id": query["user_id"],
                    "benchmark_tier": query["benchmark_tier"],
                    "query_type": query["query_type"],
                    "seed_rank": seed_rank,
                    "seed_event_id": seed_event_id,
                    "relation": relation,
                    "expanded_event_id": candidate,
                    "is_first_discovery": first,
                    "expansion_rank": expansion_ranks[candidate] if first else "",
                    "is_high_memory_load": bool(query["_is_high_memory_load"]),
                }
            )

        seed_ids: set[str] = set()
        for seed in seed_rows.itertuples(index=False):
            seed_event_id = str(seed.retrieved_event_id)
            seed_ids.add(seed_event_id)
            record(int(seed.rank), seed_event_id, "SELF", seed_event_id)
            for neighbor, relation in _traverse_neighbors(
                graph, seed_event_id, expansion_direction
            ):
                record(int(seed.rank), seed_event_id, relation, neighbor)

        scored: list[tuple[str, float]] = []
        for row in dense.itertuples(index=False):
            event_id = str(row.retrieved_event_id)
            score = 1.0 / (RRF_K + int(row.rank))
            if event_id in expansion_ranks:
                score += 1.0 / (RRF_K + expansion_ranks[event_id])
            scored.append((event_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        for rank, (event_id, score) in enumerate(scored, start=1):
            ranking_rows.append(
                {
                    "query_id": query["query_id"],
                    "user_id": query["user_id"],
                    "benchmark_tier": query["benchmark_tier"],
                    "query_type": query["query_type"],
                    "method": METHOD,
                    "rank": rank,
                    "retrieved_event_id": event_id,
                    "score": score,
                    "is_high_memory_load": bool(query["_is_high_memory_load"]),
                }
            )
        statistic_rows.append(
            {
                "query_id": query["query_id"],
                "traversal_count": traversal_count,
                "unique_expanded_count": len(expansion_ranks),
                "no_new_neighbor": set(expansion_ranks).issubset(seed_ids),
            }
        )

    return (
        pd.DataFrame(ranking_rows, columns=GRAPH_RANKING_COLUMNS),
        pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
        pd.DataFrame(statistic_rows),
    )


def _summary_rows(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    queries: pd.DataFrame,
    rankings: pd.DataFrame,
    audit: pd.DataFrame,
    statistics: pd.DataFrame,
    cohort: str,
    expansion_direction: str = EXPANSION_DIRECTION,
) -> list[dict[str, object]]:
    selected = queries
    if cohort == "high_memory_load":
        selected = queries.loc[queries["_is_high_memory_load"]]
    node_counts = nodes["node_type"].value_counts().to_dict()
    next_edge_count = int(edges["relation"].eq("NEXT").sum())
    rows: list[dict[str, object]] = []
    for query_type in sorted(selected["query_type"].unique()):
        query_ids = set(selected.loc[selected["query_type"].eq(query_type), "query_id"])
        selected_rankings = rankings.loc[rankings["query_id"].isin(query_ids)]
        selected_audit = audit.loc[audit["query_id"].isin(query_ids)]
        selected_statistics = statistics.loc[statistics["query_id"].isin(query_ids)]
        rows.append(
            {
                "method": METHOD,
                "benchmark_tier": selected["benchmark_tier"].iloc[0],
                "cohort": cohort,
                "query_type": query_type,
                "query_count": len(query_ids),
                "ranked_candidate_count": len(selected_rankings),
                "graph_node_count": len(nodes),
                "graph_edge_count": len(edges),
                "user_node_count": node_counts.get("user", 0),
                "event_node_count": node_counts.get("event", 0),
                "poi_node_count": node_counts.get("poi", 0),
                "category_node_count": node_counts.get("category", 0),
                "trail_node_count": node_counts.get("trail", 0),
                "next_edge_count": next_edge_count,
                "dense_seed_count": DENSE_SEED_COUNT,
                "expansion_hops": EXPANSION_HOPS,
                "expansion_direction": expansion_direction,
                "rrf_k": RRF_K,
                "expansion_traversal_count": len(selected_audit),
                "unique_expanded_candidate_count": int(selected_statistics["unique_expanded_count"].sum()),
                "min_expanded_candidates_per_query": int(selected_statistics["unique_expanded_count"].min()),
                "max_expanded_candidates_per_query": int(selected_statistics["unique_expanded_count"].max()),
                "mean_expanded_candidates_per_query": float(selected_statistics["unique_expanded_count"].mean()),
                "queries_with_no_new_neighbor": int(selected_statistics["no_new_neighbor"].sum()),
            }
        )
    return rows


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_outputs_atomically(
    frames: dict[Path, pd.DataFrame],
    manifest_destination: Path,
    manifest_base: dict[str, Any],
) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for destination, frame in frames.items():
            temp = _temporary_path(destination)
            temporary[destination] = temp
            frame.to_csv(temp, index=False, encoding="utf-8")
        manifest = dict(manifest_base)
        manifest["outputs"] = {
            destination.name: sha256_file(temp)
            for destination, temp in sorted(temporary.items(), key=lambda item: item[0].name)
        }
        manifest_temp = _temporary_path(manifest_destination)
        temporary[manifest_destination] = manifest_temp
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for destination, temp in temporary.items():
            temp.replace(destination)
    except (OSError, ValueError) as exc:
        raise GraphRetrievalError(f"could not write Graph retrieval outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def retrieve_graph(
    events_path: str | Path,
    vector_dir: str | Path,
    output_dir: str | Path,
) -> GraphRetrievalSummary:
    """Build the graph, validate dense artifacts, expand, fuse, and rank."""

    events_path = Path(events_path)
    vector_dir = Path(vector_dir)
    output_dir = Path(output_dir)
    vector_manifest, vector_manifest_path = _validate_manifest(vector_dir)
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
    between_index = _validate_index(
        vector_dir / BETWEEN_INDEX_FILENAME,
        QUERY_INDEX_COLUMNS,
        "query_id",
        "between query embedding index",
    )
    if set(core_index["query_id"]) & set(between_index["query_id"]):
        raise GraphRetrievalError("core and between indexes contain duplicate query IDs")
    _validate_event_mapping(built.events, event_index)
    _validate_query_mapping((core_index, between_index), event_index)

    event_embeddings = _load_embeddings(
        vector_dir / EVENT_EMBEDDINGS_FILENAME, len(event_index), "event"
    )
    core_embeddings = _load_embeddings(
        vector_dir / CORE_EMBEDDINGS_FILENAME, len(core_index), "core query"
    )
    between_embeddings = _load_embeddings(
        vector_dir / BETWEEN_EMBEDDINGS_FILENAME, len(between_index), "between query"
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
    between_vector = _validate_vector_rankings(
        vector_dir / VECTOR_BETWEEN_RANKINGS_FILENAME,
        between_index,
        event_index,
        event_embeddings,
        between_embeddings,
        "stress_test",
        "between Vector rankings",
    )

    core_rankings, core_audit, core_statistics = _retrieve_dataset(
        built.graph, core_index, core_vector
    )
    between_rankings, between_audit, between_statistics = _retrieve_dataset(
        built.graph, between_index, between_vector
    )
    summary_rows: list[dict[str, object]] = []
    for cohort in ("main", "high_memory_load"):
        summary_rows.extend(
            _summary_rows(
                built.nodes,
                built.edges,
                core_index,
                core_rankings,
                core_audit,
                core_statistics,
                cohort,
            )
        )
        summary_rows.extend(
            _summary_rows(
                built.nodes,
                built.edges,
                between_index,
                between_rankings,
                between_audit,
                between_statistics,
                cohort,
            )
        )
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    frames = {
        output_dir / NODES_FILENAME: built.nodes,
        output_dir / EDGES_FILENAME: built.edges,
        output_dir / CORE_AUDIT_FILENAME: core_audit,
        output_dir / BETWEEN_AUDIT_FILENAME: between_audit,
        output_dir / CORE_RANKINGS_FILENAME: core_rankings,
        output_dir / BETWEEN_RANKINGS_FILENAME: between_rankings,
        output_dir / SUMMARY_FILENAME: summary_frame,
    }
    input_paths = {
        "canonical_events": events_path,
        "vector_run_manifest": vector_manifest_path,
        "event_embeddings": vector_dir / EVENT_EMBEDDINGS_FILENAME,
        "event_index": vector_dir / EVENT_INDEX_FILENAME,
        "core_query_embeddings": vector_dir / CORE_EMBEDDINGS_FILENAME,
        "core_query_index": vector_dir / CORE_INDEX_FILENAME,
        "between_query_embeddings": vector_dir / BETWEEN_EMBEDDINGS_FILENAME,
        "between_query_index": vector_dir / BETWEEN_INDEX_FILENAME,
        "core_vector_rankings": vector_dir / VECTOR_CORE_RANKINGS_FILENAME,
        "between_vector_rankings": vector_dir / VECTOR_BETWEEN_RANKINGS_FILENAME,
    }
    manifest_base = {
        "dense_seed_model": {
            "id": vector_manifest["model"]["id"],
            "manifest_sha256": vector_manifest["model"]["manifest_sha256"],
            "revision": vector_manifest["model"]["revision"],
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "method": METHOD,
        "parameters": {
            "dense_seed_count": DENSE_SEED_COUNT,
            "expansion_direction": EXPANSION_DIRECTION,
            "expansion_hops": EXPANSION_HOPS,
            "expansion_order": "seed rank; SELF; neighbors by event_id; first discovery",
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
    _write_outputs_atomically(
        frames,
        output_dir / RUN_MANIFEST_FILENAME,
        manifest_base,
    )

    return GraphRetrievalSummary(
        node_count=len(built.nodes),
        edge_count=len(built.edges),
        next_edge_count=int(built.edges["relation"].eq("NEXT").sum()),
        core_query_count=len(core_index),
        stress_query_count=len(between_index),
        core_ranking_count=len(core_rankings),
        stress_ranking_count=len(between_rankings),
        core_audit_count=len(core_audit),
        stress_audit_count=len(between_audit),
        core_unique_expanded_count=int(core_statistics["unique_expanded_count"].sum()),
        stress_unique_expanded_count=int(between_statistics["unique_expanded_count"].sum()),
        high_load_core_query_count=int(core_index["_is_high_memory_load"].sum()),
        high_load_stress_query_count=int(between_index["_is_high_memory_load"].sum()),
    )
