"""Run deterministic E5-seeded heterogeneous PPR retrieval with RRF."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.build_graph import GraphBuildError, build_travel_graph
from src.model_download import sha256_file
from src.retrieve_graph import (
    CORE_EMBEDDINGS_FILENAME,
    CORE_INDEX_FILENAME,
    EVENT_EMBEDDINGS_FILENAME,
    EVENT_INDEX_FILENAME,
    EVENT_INDEX_COLUMNS,
    QUERY_INDEX_COLUMNS,
    VECTOR_CORE_RANKINGS_FILENAME,
    GraphRetrievalError,
    _load_embeddings,
    _read_csv,
    _validate_event_mapping,
    _validate_index,
    _validate_manifest,
    _validate_query_mapping,
    _validate_vector_rankings,
)


METHOD = "graph_ppr_rrf"
DENSE_SEED_COUNT = 5
PPR_ALPHA = 0.5
PPR_MAX_ITER = 1000
PPR_TOL = 1e-12
RRF_K = 60
SCORE_ATOL = 1e-12
SCORE_RTOL = 1e-10

RANKINGS_FILENAME = "tokyo_graph_ppr_rrf_core_rankings.csv"
EVENT_SCORES_FILENAME = "tokyo_graph_ppr_event_scores.csv"
QUERY_AUDIT_FILENAME = "tokyo_graph_ppr_query_audit.csv"
PROJECTION_SUMMARY_FILENAME = "tokyo_graph_ppr_projection_summary.csv"
RETRIEVAL_SUMMARY_FILENAME = "tokyo_graph_ppr_retrieval_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_graph_ppr_run_manifest.json"

RANKING_COLUMNS = (
    "query_id", "user_id", "benchmark_tier", "query_type", "method", "rank",
    "retrieved_event_id", "score", "is_high_memory_load",
)
EVENT_SCORE_COLUMNS = (
    "query_id", "user_id", "event_id", "dense_rank", "dense_score",
    "is_personalization_seed", "personalization_weight", "ppr_score",
    "ppr_rank", "rrf_score", "final_rank",
)
QUERY_AUDIT_COLUMNS = (
    "query_id", "user_id", "query_type", "candidate_count", "seed_count",
    "seed_event_ids", "positive_seed_count", "personalization_sum",
    "ppr_event_score_sum", "ppr_all_node_score_sum", "nonzero_ppr_event_count",
    "projected_node_count", "projected_edge_count", "connected_component_count",
    "seeded_component_count", "unseeded_component_count",
)
PROJECTION_SUMMARY_COLUMNS = (
    "user_id", "event_node_count", "poi_node_count", "category_node_count",
    "trail_node_count", "projected_node_count", "projected_edge_count",
    "connected_component_count",
)
SUMMARY_COLUMNS = (
    "method", "benchmark_tier", "cohort", "query_type", "query_count",
    "ranked_candidate_count", "min_candidates_per_query",
    "max_candidates_per_query", "mean_candidates_per_query", "dense_seed_count",
    "ppr_alpha", "ppr_max_iter", "ppr_tol", "ppr_weight", "rrf_k",
    "rrf_weights", "ranking_depth",
)


class GraphPPRRetrievalError(RuntimeError):
    """Raised when Phase 2 PPR retrieval cannot complete safely."""


@dataclass(frozen=True)
class GraphPPRRetrievalSummary:
    query_count: int
    ranking_count: int
    event_score_count: int
    user_count: int
    output_dir: Path


def _project_user_graphs(events: pd.DataFrame) -> tuple[dict[str, nx.Graph], pd.DataFrame]:
    graphs: dict[str, nx.Graph] = {}
    rows: list[dict[str, object]] = []
    for user_id, user_events in events.groupby("user_id", sort=True):
        graph = nx.Graph()
        for event in user_events.sort_values("event_id", kind="stable").itertuples():
            event_node = f"event::{event.event_id}"
            poi_node = f"poi::{event.venue_id}"
            category_node = f"category::{event.venue_category}"
            trail_node = f"trail::{event.trail_id}"
            graph.add_edge(event_node, poi_node)
            graph.add_edge(poi_node, category_node)
            graph.add_edge(event_node, trail_node)
            if event.next_event_id:
                graph.add_edge(event_node, f"event::{event.next_event_id}")
        graphs[user_id] = graph
        node_types = pd.Series(
            [node_id.split("::", 1)[0] for node_id in graph.nodes], dtype=str
        ).value_counts()
        rows.append({
            "user_id": user_id,
            "event_node_count": int(node_types.get("event", 0)),
            "poi_node_count": int(node_types.get("poi", 0)),
            "category_node_count": int(node_types.get("category", 0)),
            "trail_node_count": int(node_types.get("trail", 0)),
            "projected_node_count": graph.number_of_nodes(),
            "projected_edge_count": graph.number_of_edges(),
            "connected_component_count": nx.number_connected_components(graph),
        })
    return graphs, pd.DataFrame(rows, columns=PROJECTION_SUMMARY_COLUMNS)


def _retrieve(
    graphs: dict[str, nx.Graph],
    query_index: pd.DataFrame,
    vector_rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    dense_groups = {
        query_id: group.sort_values("rank", kind="stable")
        for query_id, group in vector_rankings.groupby("query_id", sort=False)
    }

    for query in query_index.itertuples(index=False):
        is_high_memory_load = str(query.is_high_memory_load).strip().casefold() in {
            "true", "1"
        }
        dense = dense_groups[query.query_id]
        seed_rows = dense.head(min(DENSE_SEED_COUNT, len(dense)))
        seed_scores = np.maximum(seed_rows["score"].to_numpy(dtype=float), 0.0)
        seed_total = float(seed_scores.sum())
        if not np.isfinite(seed_total) or seed_total <= 0:
            raise GraphPPRRetrievalError(
                f"query {query.query_id} has no positive Top-5 personalization mass"
            )
        seed_weights = seed_scores / seed_total
        graph = graphs.get(query.user_id)
        if graph is None:
            raise GraphPPRRetrievalError(
                f"query {query.query_id} references a user absent from the PPR projection"
            )

        personalization = {node_id: 0.0 for node_id in graph.nodes}
        seed_ids: list[str] = []
        for seed, weight in zip(seed_rows.itertuples(index=False), seed_weights, strict=True):
            node_id = f"event::{seed.retrieved_event_id}"
            if node_id not in graph:
                raise GraphPPRRetrievalError(
                    f"PPR seed is absent from its user graph: {seed.retrieved_event_id}"
                )
            seed_ids.append(str(seed.retrieved_event_id))
            personalization[node_id] = float(weight)

        try:
            ppr = nx.pagerank(
                graph,
                alpha=PPR_ALPHA,
                personalization=personalization,
                max_iter=PPR_MAX_ITER,
                tol=PPR_TOL,
                weight=None,
                dangling=personalization,
            )
        except nx.PowerIterationFailedConvergence as exc:
            raise GraphPPRRetrievalError(
                f"PPR failed to converge for query {query.query_id}"
            ) from exc

        dense_records = list(dense.itertuples(index=False))
        ppr_order = sorted(
            (str(row.retrieved_event_id) for row in dense_records),
            key=lambda event_id: (-float(ppr[f"event::{event_id}"]), event_id),
        )
        ppr_ranks = {event_id: rank for rank, event_id in enumerate(ppr_order, start=1)}
        dense_by_id = {str(row.retrieved_event_id): row for row in dense_records}
        scored = []
        for event_id in ppr_order:
            dense_row = dense_by_id[event_id]
            score = 1.0 / (RRF_K + int(dense_row.rank)) + 1.0 / (
                RRF_K + ppr_ranks[event_id]
            )
            scored.append((event_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        final_ranks = {event_id: rank for rank, (event_id, _) in enumerate(scored, start=1)}
        final_scores = dict(scored)

        for event_id, final_score in scored:
            dense_row = dense_by_id[event_id]
            ranking_rows.append({
                "query_id": query.query_id,
                "user_id": query.user_id,
                "benchmark_tier": query.benchmark_tier,
                "query_type": query.query_type,
                "method": METHOD,
                "rank": final_ranks[event_id],
                "retrieved_event_id": event_id,
                "score": final_score,
                "is_high_memory_load": is_high_memory_load,
            })
            score_rows.append({
                "query_id": query.query_id,
                "user_id": query.user_id,
                "event_id": event_id,
                "dense_rank": int(dense_row.rank),
                "dense_score": float(dense_row.score),
                "is_personalization_seed": event_id in seed_ids,
                "personalization_weight": personalization[f"event::{event_id}"],
                "ppr_score": float(ppr[f"event::{event_id}"]),
                "ppr_rank": ppr_ranks[event_id],
                "rrf_score": final_scores[event_id],
                "final_rank": final_ranks[event_id],
            })

        components = list(nx.connected_components(graph))
        seeded_components = sum(
            any(f"event::{seed_id}" in component for seed_id in seed_ids)
            for component in components
        )
        event_scores = [
            score for node_id, score in ppr.items() if node_id.startswith("event::")
        ]
        audit_rows.append({
            "query_id": query.query_id,
            "user_id": query.user_id,
            "query_type": query.query_type,
            "candidate_count": len(dense),
            "seed_count": len(seed_ids),
            "seed_event_ids": json.dumps(seed_ids, ensure_ascii=False, separators=(",", ":")),
            "positive_seed_count": int(np.count_nonzero(seed_scores > 0)),
            "personalization_sum": float(sum(personalization.values())),
            "ppr_event_score_sum": float(sum(event_scores)),
            "ppr_all_node_score_sum": float(sum(ppr.values())),
            "nonzero_ppr_event_count": int(np.count_nonzero(np.asarray(event_scores) > 0)),
            "projected_node_count": graph.number_of_nodes(),
            "projected_edge_count": graph.number_of_edges(),
            "connected_component_count": len(components),
            "seeded_component_count": seeded_components,
            "unseeded_component_count": len(components) - seeded_components,
        })

    rankings = pd.DataFrame(ranking_rows, columns=RANKING_COLUMNS).sort_values(
        ["user_id", "query_id", "rank"], kind="stable"
    ).reset_index(drop=True)
    event_scores = pd.DataFrame(score_rows, columns=EVENT_SCORE_COLUMNS).sort_values(
        ["user_id", "query_id", "final_rank"], kind="stable"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows, columns=QUERY_AUDIT_COLUMNS).sort_values(
        ["user_id", "query_id"], kind="stable"
    ).reset_index(drop=True)
    return rankings, event_scores, audit


def _summary_rows(query_index: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ("main", "high_memory_load"):
        selected = query_index
        if cohort == "high_memory_load":
            selected = selected.loc[selected["_is_high_memory_load"]]
        for query_type in sorted(selected["query_type"].unique()):
            query_ids = set(selected.loc[selected["query_type"].eq(query_type), "query_id"])
            counts = rankings.loc[rankings["query_id"].isin(query_ids)].groupby("query_id").size()
            rows.append({
                "method": METHOD,
                "benchmark_tier": "core",
                "cohort": cohort,
                "query_type": query_type,
                "query_count": len(query_ids),
                "ranked_candidate_count": int(counts.sum()),
                "min_candidates_per_query": int(counts.min()),
                "max_candidates_per_query": int(counts.max()),
                "mean_candidates_per_query": float(counts.mean()),
                "dense_seed_count": DENSE_SEED_COUNT,
                "ppr_alpha": PPR_ALPHA,
                "ppr_max_iter": PPR_MAX_ITER,
                "ppr_tol": PPR_TOL,
                "ppr_weight": "None",
                "rrf_k": RRF_K,
                "rrf_weights": "equal",
                "ranking_depth": "all events belonging to the query user",
            })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_outputs(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    manifest_base: dict[str, Any],
) -> None:
    temporary: dict[str, Path] = {}
    try:
        for filename, frame in frames.items():
            temp = _temporary_path(output_dir / filename)
            frame.to_csv(temp, index=False, encoding="utf-8", float_format="%.15g")
            temporary[filename] = temp
        manifest = dict(manifest_base)
        manifest["outputs"] = {
            filename: sha256_file(temp) for filename, temp in sorted(temporary.items())
        }
        manifest_temp = _temporary_path(output_dir / RUN_MANIFEST_FILENAME)
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for filename, temp in temporary.items():
            temp.replace(output_dir / filename)
        manifest_temp.replace(output_dir / RUN_MANIFEST_FILENAME)
    except (OSError, ValueError) as exc:
        raise GraphPPRRetrievalError(f"could not write PPR outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)
        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def retrieve_graph_ppr(
    events_path: str | Path,
    vector_dir: str | Path,
    output_dir: str | Path,
) -> GraphPPRRetrievalSummary:
    """Validate frozen E5 artifacts, run per-user PPR, and fuse with Dense."""

    events_path = Path(events_path)
    vector_dir = Path(vector_dir)
    output_dir = Path(output_dir)
    project_root = Path(__file__).resolve().parents[1]
    try:
        vector_manifest, vector_manifest_path = _validate_manifest(vector_dir)
        if vector_manifest["model"]["id"] != "intfloat/multilingual-e5-small":
            raise GraphPPRRetrievalError("PPR retrieval requires the frozen multilingual E5 run")
        events_source = _read_csv(events_path, "events")
        built = build_travel_graph(events_source)
        event_index = _validate_index(
            vector_dir / EVENT_INDEX_FILENAME, EVENT_INDEX_COLUMNS,
            "event_id", "event embedding index",
        )
        query_index = _validate_index(
            vector_dir / CORE_INDEX_FILENAME, QUERY_INDEX_COLUMNS,
            "query_id", "core query embedding index",
        )
        _validate_event_mapping(built.events, event_index)
        _validate_query_mapping((query_index,), event_index)
        event_embeddings = _load_embeddings(
            vector_dir / EVENT_EMBEDDINGS_FILENAME, len(event_index), "event"
        )
        query_embeddings = _load_embeddings(
            vector_dir / CORE_EMBEDDINGS_FILENAME, len(query_index), "core query"
        )
        vector_rankings = _validate_vector_rankings(
            vector_dir / VECTOR_CORE_RANKINGS_FILENAME,
            query_index, event_index, event_embeddings, query_embeddings,
            "core", "core Vector rankings",
        )
    except (GraphRetrievalError, GraphBuildError) as exc:
        raise GraphPPRRetrievalError(str(exc)) from exc

    graphs, projection_summary = _project_user_graphs(built.events)
    rankings, event_scores, query_audit = _retrieve(
        graphs, query_index, vector_rankings
    )
    retrieval_summary = _summary_rows(query_index, rankings)

    input_paths = {
        "canonical_events": events_path,
        "vector_run_manifest": vector_manifest_path,
        "event_embeddings": vector_dir / EVENT_EMBEDDINGS_FILENAME,
        "event_index": vector_dir / EVENT_INDEX_FILENAME,
        "core_query_embeddings": vector_dir / CORE_EMBEDDINGS_FILENAME,
        "core_query_index": vector_dir / CORE_INDEX_FILENAME,
        "core_vector_rankings": vector_dir / VECTOR_CORE_RANKINGS_FILENAME,
    }
    manifest = {
        "method": METHOD,
        "dense_seed_model": {
            "id": vector_manifest["model"]["id"],
            "revision": vector_manifest["model"]["revision"],
            "manifest_sha256": vector_manifest["model"]["manifest_sha256"],
        },
        "inputs": {
            key: {"path": _portable_path(path, project_root), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "parameters": {
            "candidate_truncation": "none",
            "dense_seed_count": DENSE_SEED_COUNT,
            "graph_projection": "per_user_undirected_unweighted",
            "included_relations": ["AT", "CATEGORY", "IN_TRAIL", "NEXT"],
            "excluded_nodes": ["User"],
            "excluded_relations": ["HAS"],
            "personalization": "positive cosine normalized over frozen E5 Top-5 seeds",
            "ppr_alpha": PPR_ALPHA,
            "ppr_max_iter": PPR_MAX_ITER,
            "ppr_tol": PPR_TOL,
            "ppr_weight": None,
            "ppr_output_nodes": "Event only",
            "ranking_depth": "all events belonging to the query user",
            "rrf_formula": "1/(60+dense_rank)+1/(60+ppr_rank)",
            "rrf_k": RRF_K,
            "rrf_ranks_are_one_based": True,
            "score_comparison_atol": SCORE_ATOL,
            "score_comparison_rtol": SCORE_RTOL,
            "tie_break": "event_id ascending",
        },
        "runtime_versions": {
            "networkx": _version("networkx"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "python": sys.version.split()[0],
        },
    }
    _write_outputs(output_dir, {
        RANKINGS_FILENAME: rankings,
        EVENT_SCORES_FILENAME: event_scores,
        QUERY_AUDIT_FILENAME: query_audit,
        PROJECTION_SUMMARY_FILENAME: projection_summary,
        RETRIEVAL_SUMMARY_FILENAME: retrieval_summary,
    }, manifest)
    return GraphPPRRetrievalSummary(
        query_count=len(query_index), ranking_count=len(rankings),
        event_score_count=len(event_scores), user_count=len(graphs), output_dir=output_dir,
    )
