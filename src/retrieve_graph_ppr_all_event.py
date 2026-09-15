"""Run the fixed post-hoc all-event PPR personalization sensitivity."""

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
    EVENT_INDEX_COLUMNS,
    EVENT_INDEX_FILENAME,
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
from src.retrieve_graph_ppr import (
    PPR_ALPHA,
    PPR_MAX_ITER,
    PPR_TOL,
    RRF_K,
    SCORE_ATOL,
    SCORE_RTOL,
    _project_user_graphs,
)


METHOD = "graph_ppr_rrf_all_event"
RANKINGS_FILENAME = "tokyo_graph_ppr_all_event_rrf_core_rankings.csv"
EVENT_SCORES_FILENAME = "tokyo_graph_ppr_all_event_event_scores.csv"
QUERY_AUDIT_FILENAME = "tokyo_graph_ppr_all_event_query_audit.csv"
COMPONENT_AUDIT_FILENAME = "tokyo_graph_ppr_all_event_component_audit.csv"
PROJECTION_SUMMARY_FILENAME = "tokyo_graph_ppr_all_event_projection_summary.csv"
RETRIEVAL_SUMMARY_FILENAME = "tokyo_graph_ppr_all_event_retrieval_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_graph_ppr_all_event_run_manifest.json"

RANKING_COLUMNS = (
    "query_id", "user_id", "benchmark_tier", "query_type", "method", "rank",
    "retrieved_event_id", "score", "is_high_memory_load",
)
EVENT_SCORE_COLUMNS = (
    "query_id", "user_id", "event_id", "dense_rank", "dense_score",
    "has_positive_personalization", "personalization_weight", "ppr_score",
    "ppr_rank", "rrf_score", "final_rank",
)
QUERY_AUDIT_COLUMNS = (
    "query_id", "user_id", "query_type", "candidate_count",
    "positive_personalization_event_count", "zero_personalization_event_count",
    "personalization_event_weight_sum", "full_graph_ppr_score_sum",
    "event_only_ppr_score_sum", "component_count",
)
COMPONENT_AUDIT_COLUMNS = (
    "query_id", "user_id", "component_id", "component_event_count",
    "component_personalization_mass", "component_ppr_mass",
)


class GraphPPRAllEventRetrievalError(RuntimeError):
    """Raised when the all-event PPR sensitivity cannot complete safely."""


@dataclass(frozen=True)
class GraphPPRAllEventSummary:
    query_count: int
    ranking_count: int
    event_score_count: int
    component_audit_count: int
    user_count: int
    output_dir: Path


def _retrieve(
    graphs: dict[str, nx.Graph],
    query_index: pd.DataFrame,
    vector_rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    dense_groups = {
        query_id: group.sort_values("rank", kind="stable")
        for query_id, group in vector_rankings.groupby("query_id", sort=False)
    }

    for query in query_index.itertuples(index=False):
        dense = dense_groups[query.query_id]
        scores = np.maximum(dense["score"].to_numpy(dtype=float), 0.0)
        total = float(scores.sum())
        if not np.isfinite(total) or total <= 0:
            raise GraphPPRAllEventRetrievalError(
                f"query {query.query_id} has no positive all-event personalization mass"
            )
        weights = scores / total
        graph = graphs.get(query.user_id)
        if graph is None:
            raise GraphPPRAllEventRetrievalError(
                f"query {query.query_id} references a user absent from the PPR projection"
            )
        personalization = {node_id: 0.0 for node_id in graph.nodes}
        dense_records = list(dense.itertuples(index=False))
        for row, weight in zip(dense_records, weights, strict=True):
            node_id = f"event::{row.retrieved_event_id}"
            if node_id not in graph:
                raise GraphPPRAllEventRetrievalError(
                    f"PPR event is absent from its user graph: {row.retrieved_event_id}"
                )
            personalization[node_id] = float(weight)

        try:
            ppr = nx.pagerank(
                graph, alpha=PPR_ALPHA, personalization=personalization,
                max_iter=PPR_MAX_ITER, tol=PPR_TOL, weight=None,
                dangling=personalization,
            )
        except nx.PowerIterationFailedConvergence as exc:
            raise GraphPPRAllEventRetrievalError(
                f"PPR failed to converge for query {query.query_id}"
            ) from exc

        personalization_sum = float(sum(personalization.values()))
        full_sum = float(sum(ppr.values()))
        event_sum = float(sum(
            value for node_id, value in ppr.items() if node_id.startswith("event::")
        ))
        if not np.isclose(personalization_sum, 1.0, atol=SCORE_ATOL, rtol=SCORE_RTOL):
            raise GraphPPRAllEventRetrievalError("personalization Event weights do not sum to one")
        if not np.isclose(full_sum, 1.0, atol=SCORE_ATOL, rtol=SCORE_RTOL):
            raise GraphPPRAllEventRetrievalError("full-graph PPR scores do not sum to one")
        if event_sum > 1.0 + SCORE_ATOL:
            raise GraphPPRAllEventRetrievalError("Event-only PPR mass exceeds one")

        event_ids = [str(row.retrieved_event_id) for row in dense_records]
        ppr_order = sorted(
            event_ids, key=lambda event_id: (-float(ppr[f"event::{event_id}"]), event_id)
        )
        ppr_ranks = {event_id: rank for rank, event_id in enumerate(ppr_order, start=1)}
        dense_by_id = {str(row.retrieved_event_id): row for row in dense_records}
        fused = [
            (
                event_id,
                1.0 / (RRF_K + int(dense_by_id[event_id].rank))
                + 1.0 / (RRF_K + ppr_ranks[event_id]),
            )
            for event_id in ppr_order
        ]
        fused.sort(key=lambda item: (-item[1], item[0]))
        final_ranks = {event_id: rank for rank, (event_id, _) in enumerate(fused, start=1)}
        final_scores = dict(fused)
        high_load = str(query.is_high_memory_load).strip().casefold() in {"true", "1"}

        for event_id, final_score in fused:
            dense_row = dense_by_id[event_id]
            weight = personalization[f"event::{event_id}"]
            ranking_rows.append({
                "query_id": query.query_id, "user_id": query.user_id,
                "benchmark_tier": query.benchmark_tier, "query_type": query.query_type,
                "method": METHOD, "rank": final_ranks[event_id],
                "retrieved_event_id": event_id, "score": final_score,
                "is_high_memory_load": high_load,
            })
            score_rows.append({
                "query_id": query.query_id, "user_id": query.user_id,
                "event_id": event_id, "dense_rank": int(dense_row.rank),
                "dense_score": float(dense_row.score),
                "has_positive_personalization": weight > 0,
                "personalization_weight": weight,
                "ppr_score": float(ppr[f"event::{event_id}"]),
                "ppr_rank": ppr_ranks[event_id], "rrf_score": final_scores[event_id],
                "final_rank": final_ranks[event_id],
            })

        components = sorted(
            nx.connected_components(graph), key=lambda nodes: min(str(node) for node in nodes)
        )
        for component_index, component in enumerate(components, start=1):
            component_rows.append({
                "query_id": query.query_id, "user_id": query.user_id,
                "component_id": component_index,
                "component_event_count": sum(
                    str(node).startswith("event::") for node in component
                ),
                "component_personalization_mass": float(sum(
                    personalization[node] for node in component
                )),
                "component_ppr_mass": float(sum(ppr[node] for node in component)),
            })
        query_rows.append({
            "query_id": query.query_id, "user_id": query.user_id,
            "query_type": query.query_type, "candidate_count": len(dense),
            "positive_personalization_event_count": int(np.count_nonzero(scores > 0)),
            "zero_personalization_event_count": int(np.count_nonzero(scores <= 0)),
            "personalization_event_weight_sum": personalization_sum,
            "full_graph_ppr_score_sum": full_sum,
            "event_only_ppr_score_sum": event_sum,
            "component_count": len(components),
        })

    def frame(rows: list[dict[str, object]], columns: tuple[str, ...], order: list[str]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=columns).sort_values(order, kind="stable").reset_index(drop=True)

    return (
        frame(ranking_rows, RANKING_COLUMNS, ["user_id", "query_id", "rank"]),
        frame(score_rows, EVENT_SCORE_COLUMNS, ["user_id", "query_id", "final_rank"]),
        frame(query_rows, QUERY_AUDIT_COLUMNS, ["user_id", "query_id"]),
        frame(component_rows, COMPONENT_AUDIT_COLUMNS, ["user_id", "query_id", "component_id"]),
    )


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write(output_dir: Path, frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temps: dict[str, Path] = {}
    try:
        for filename, value in frames.items():
            descriptor, name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output_dir)
            os.close(descriptor)
            temp = Path(name)
            value.to_csv(temp, index=False, encoding="utf-8", float_format="%.15g")
            temps[filename] = temp
        materialized = dict(manifest)
        materialized["outputs"] = {
            filename: sha256_file(temp) for filename, temp in sorted(temps.items())
        }
        descriptor, name = tempfile.mkstemp(prefix=f".{RUN_MANIFEST_FILENAME}.", suffix=".tmp", dir=output_dir)
        os.close(descriptor)
        manifest_temp = Path(name)
        manifest_temp.write_text(
            json.dumps(materialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for filename, temp in temps.items():
            temp.replace(output_dir / filename)
        manifest_temp.replace(output_dir / RUN_MANIFEST_FILENAME)
    except (OSError, ValueError) as exc:
        raise GraphPPRAllEventRetrievalError(f"could not write all-event PPR outputs: {exc}") from exc
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)


def retrieve_graph_ppr_all_event(
    events_path: str | Path,
    vector_dir: str | Path,
    output_dir: str | Path,
) -> GraphPPRAllEventSummary:
    """Run all-event dense-weighted PPR without changing the frozen E5 run."""

    events_path, vector_dir, output_dir = map(Path, (events_path, vector_dir, output_dir))
    project_root = Path(__file__).resolve().parents[1]
    try:
        vector_manifest, vector_manifest_path = _validate_manifest(vector_dir)
        if vector_manifest["model"]["id"] != "intfloat/multilingual-e5-small":
            raise GraphPPRAllEventRetrievalError("all-event PPR requires the frozen multilingual E5 run")
        source = _read_csv(events_path, "events")
        built = build_travel_graph(source)
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
        raise GraphPPRAllEventRetrievalError(str(exc)) from exc

    graphs, projection = _project_user_graphs(built.events)
    rankings, scores, query_audit, component_audit = _retrieve(
        graphs, query_index, vector_rankings
    )
    counts = rankings.groupby("query_id").size()
    retrieval_summary = pd.DataFrame([{
        "method": METHOD, "benchmark_tier": "core", "cohort": "main",
        "query_type": "overall", "query_count": len(query_index),
        "ranked_candidate_count": len(rankings),
        "min_candidates_per_query": int(counts.min()),
        "max_candidates_per_query": int(counts.max()),
        "mean_candidates_per_query": float(counts.mean()),
        "personalization_scope": "all same-user Events with positive cosine mass",
        "ppr_alpha": PPR_ALPHA, "ppr_max_iter": PPR_MAX_ITER,
        "ppr_tol": PPR_TOL, "ppr_weight": "None", "rrf_k": RRF_K,
        "rrf_weights": "equal", "ranking_depth": "all same-user Events",
    }])
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
        "analysis_role": "post_hoc_personalization_sensitivity",
        "dense_seed_model": {
            "id": vector_manifest["model"]["id"],
            "revision": vector_manifest["model"]["revision"],
            "manifest_sha256": vector_manifest["model"]["manifest_sha256"],
        },
        "inputs": {
            key: {"path": _portable(path, project_root), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "parameters": {
            "personalization": "max(cosine,0) normalized over all same-user Events",
            "candidate_truncation": "none", "graph_projection": "per_user_undirected_unweighted",
            "included_relations": ["AT", "CATEGORY", "IN_TRAIL", "NEXT"],
            "excluded_nodes": ["User"], "excluded_relations": ["HAS"],
            "ppr_alpha": PPR_ALPHA, "ppr_max_iter": PPR_MAX_ITER,
            "ppr_tol": PPR_TOL, "ppr_weight": None,
            "ppr_output_nodes": "Event only", "rrf_k": RRF_K,
            "rrf_formula": "1/(60+dense_rank)+1/(60+ppr_rank)",
            "rrf_ranks_are_one_based": True, "tie_break": "event_id ascending",
            "score_comparison_atol": SCORE_ATOL, "score_comparison_rtol": SCORE_RTOL,
        },
        "runtime_versions": {
            "networkx": importlib.metadata.version("networkx"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "python": sys.version.split()[0],
        },
    }
    _write(output_dir, {
        RANKINGS_FILENAME: rankings,
        EVENT_SCORES_FILENAME: scores,
        QUERY_AUDIT_FILENAME: query_audit,
        COMPONENT_AUDIT_FILENAME: component_audit,
        PROJECTION_SUMMARY_FILENAME: projection,
        RETRIEVAL_SUMMARY_FILENAME: retrieval_summary,
    }, manifest)
    return GraphPPRAllEventSummary(
        len(query_index), len(rankings), len(scores), len(component_audit),
        len(graphs), output_dir,
    )
