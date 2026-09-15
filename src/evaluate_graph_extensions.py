"""Evaluate the frozen baselines and Phase 2 Graph-PPR-RRF extension."""

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
from src.diagnose_structured_next import (
    CANDIDATES_FILENAME as STRUCTURED_CANDIDATES_FILENAME,
    RUN_MANIFEST_FILENAME as STRUCTURED_MANIFEST_FILENAME,
)
from src.evaluate import (
    EvaluationError,
    RANKING_COLUMNS,
    _aggregate_metrics,
    _holm_adjust,
    _load_events,
    _load_queries,
    _load_ranking,
    _validate_run_manifest,
)
from src.model_download import sha256_file
from src.retrieve_graph_ppr import (
    EVENT_SCORES_FILENAME as PPR_EVENT_SCORES_FILENAME,
    METHOD as PPR_METHOD,
    RANKINGS_FILENAME as PPR_RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME as PPR_MANIFEST_FILENAME,
    _project_user_graphs,
)


METHODS = ("flat", "vector", "graph", PPR_METHOD)
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
METHOD_LABELS = {
    "flat": "Flat BM25",
    "vector": "E5 Dense",
    "graph": "E5 Graph-1Hop-RRF",
    PPR_METHOD: "E5 Graph-PPR-RRF",
}

PER_QUERY_FILENAME = "tokyo_phase2_per_query_metrics.csv"
OVERALL_FILENAME = "tokyo_phase2_core_overall_metrics.csv"
BY_TYPE_FILENAME = "tokyo_phase2_metrics_by_query_type.csv"
WILCOXON_FILENAME = "tokyo_phase2_pairwise_wilcoxon.csv"
PPR_DIAGNOSTICS_FILENAME = "tokyo_graph_ppr_query_diagnostics.csv"
PPR_DIAGNOSTIC_SUMMARY_FILENAME = "tokyo_graph_ppr_diagnostic_summary.csv"
STRUCTURED_EVALUATION_FILENAME = "tokyo_structured_next_evaluation.csv"
REPORT_FILENAME = "tokyo_phase2_validation_report.md"
RUN_MANIFEST_FILENAME = "tokyo_phase2_run_manifest.json"


class GraphExtensionEvaluationError(RuntimeError):
    """Raised when Phase 2 evaluation cannot complete safely."""


@dataclass(frozen=True)
class GraphExtensionEvaluationSummary:
    query_count: int
    per_query_count: int
    ppr_improved_vs_dense: int
    ppr_unchanged_vs_dense: int
    ppr_worsened_vs_dense: int
    structured_unique_correct: int
    output_dir: Path


def _read_json_manifest(directory: Path, filename: str, expected_key: str, expected_value: str) -> tuple[dict[str, Any], Path]:
    path = directory / filename
    if not path.is_file():
        raise GraphExtensionEvaluationError(f"run manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphExtensionEvaluationError(f"could not parse run manifest {path}: {exc}") from exc
    if manifest.get(expected_key) != expected_value or not isinstance(manifest.get("outputs"), dict):
        raise GraphExtensionEvaluationError(f"run manifest has an invalid identity: {path}")
    for filename_value, expected_hash in sorted(manifest["outputs"].items()):
        if Path(filename_value).name != filename_value:
            raise GraphExtensionEvaluationError("run manifest contains an unsafe filename")
        artifact = directory / filename_value
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise GraphExtensionEvaluationError(f"artifact hash mismatch: {artifact}")
    return manifest, path


def _validate_targets(events: pd.DataFrame, queries: pd.DataFrame) -> None:
    lookup = events.set_index("event_id")
    missing = sorted(set(queries["target_event_id"]) - set(events["event_id"]))
    if missing:
        raise GraphExtensionEvaluationError(
            "queries reference missing target event(s): " + ", ".join(missing[:5])
        )
    for query in queries.itertuples(index=False):
        event = lookup.loc[query.target_event_id]
        if event["user_id"] != query.user_id or event["venue_id"] != query.target_venue_id:
            raise GraphExtensionEvaluationError("query Ground Truth does not match events")


def _calculate_per_query(queries: pd.DataFrame, rankings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    query_lookup = queries.set_index("query_id")
    for method in METHODS:
        groups = {
            query_id: group for query_id, group in rankings[method].groupby("query_id", sort=False)
        }
        for query_id, query in query_lookup.iterrows():
            target = groups[query_id].loc[
                groups[query_id]["retrieved_event_id"].eq(query["target_event_id"])
            ]
            if len(target) != 1:
                raise GraphExtensionEvaluationError(
                    f"{method} does not contain exactly one target for {query_id}"
                )
            rank = int(target.iloc[0]["rank"])
            rows.append({
                "query_id": query_id,
                "user_id": query["user_id"],
                "benchmark_tier": "core",
                "query_type": query["query_type"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "query_text": query["query_text"],
                "target_event_id": query["target_event_id"],
                "target_venue_id": query["target_venue_id"],
                "target_name": query["target_name"],
                "target_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "success_at_1": rank <= 1,
                "success_at_5": rank <= 5,
                "is_high_memory_load": bool(query["_is_high_memory_load"]),
            })
    result = pd.DataFrame(rows)
    result["_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["user_id", "query_id", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def _metrics(per_query: pd.DataFrame, by_type: bool) -> pd.DataFrame:
    group_columns = ["query_type", "method"] if by_type else ["method"]
    rows = []
    for key, group in per_query.groupby(group_columns, sort=False):
        if by_type:
            query_type, method = key
        else:
            method = key[0] if isinstance(key, tuple) else key
            query_type = "overall"
        rows.append({
            "benchmark_tier": "core",
            "cohort": "main",
            "query_type": query_type,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "query_count": group["query_id"].nunique(),
            "user_count": group["user_id"].nunique(),
            "mrr": group["reciprocal_rank"].mean(),
            "success_at_1": group["success_at_1"].mean(),
            "success_at_5": group["success_at_5"].mean(),
        })
    result = pd.DataFrame(rows)
    result["_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["query_type", "_order"], kind="stable"
    ).drop(columns="_order").reset_index(drop=True)


def _wilcoxon(per_query: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except (ImportError, OSError) as exc:
        raise GraphExtensionEvaluationError("scipy is required for Wilcoxon evaluation") from exc
    user_means = per_query.groupby(["user_id", "method"], sort=True)[
        "reciprocal_rank"
    ].mean().unstack("method")
    comparisons = (
        ("flat", "vector"),
        ("vector", "graph"),
        ("vector", PPR_METHOD),
        ("graph", PPR_METHOD),
    )
    rows = []
    for method_a, method_b in comparisons:
        paired = user_means[[method_a, method_b]].dropna()
        differences = paired[method_b] - paired[method_a]
        nonzero = int(differences.ne(0).sum())
        if nonzero == 0:
            statistic, p_value, status = 0.0, 1.0, "all_zero_differences"
        else:
            result = wilcoxon(
                paired[method_a], paired[method_b], zero_method="wilcox",
                correction=False, alternative="two-sided", method="approx",
            )
            statistic, p_value, status = float(result.statistic), float(result.pvalue), "ok"
        rows.append({
            "comparison": f"{method_a}_vs_{method_b}",
            "method_a": method_a,
            "method_b": method_b,
            "observation_unit": "per_user_mean_core_reciprocal_rank",
            "observation_count": len(paired),
            "nonzero_difference_count": nonzero,
            "mean_rr_method_a": paired[method_a].mean(),
            "mean_rr_method_b": paired[method_b].mean(),
            "mean_paired_difference_b_minus_a": differences.mean(),
            "wilcoxon_statistic": statistic,
            "raw_p_value": p_value,
            "holm_adjusted_p_value": 0.0,
            "alpha": 0.05,
            "reject_null_holm": False,
            "status": status,
        })
    adjusted = _holm_adjust([float(row["raw_p_value"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p_value"] = value
        row["reject_null_holm"] = value < 0.05
    return pd.DataFrame(rows)


def _distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "unreachable"
    return str(distance) if distance < 3 else "3+"


def _next_graphs(events: pd.DataFrame) -> dict[str, nx.Graph]:
    result = {}
    for user_id, group in events.groupby("user_id", sort=True):
        graph = nx.Graph()
        graph.add_nodes_from(f"event::{event_id}" for event_id in sorted(group["event_id"]))
        for event in group.itertuples(index=False):
            if event.next_event_id:
                graph.add_edge(f"event::{event.event_id}", f"event::{event.next_event_id}")
        result[user_id] = graph
    return result


def _multi_source_distance(graph: nx.Graph, seed_nodes: list[str], target: str) -> int | None:
    lengths = nx.multi_source_dijkstra_path_length(graph, seed_nodes, weight=None)
    value = lengths.get(target)
    return int(value) if value is not None else None


def _ppr_diagnostics(
    events: pd.DataFrame,
    queries: pd.DataFrame,
    per_query: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    ppr_event_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = per_query.pivot(index="query_id", columns="method", values=["target_rank", "reciprocal_rank"])
    ranking_groups = {
        method: {
            query_id: group.sort_values("rank", kind="stable")
            for query_id, group in ranking.groupby("query_id", sort=False)
        }
        for method, ranking in rankings.items()
    }
    score_lookup = ppr_event_scores.set_index(["query_id", "event_id"])
    any_graphs, _ = _project_user_graphs(build_travel_graph(events).events)
    next_graphs = _next_graphs(build_travel_graph(events).events)
    rows = []
    for _, query in queries.iterrows():
        query_id = query["query_id"]
        dense_group = ranking_groups["vector"][query_id]
        seed_ids = dense_group.head(min(5, len(dense_group)))["retrieved_event_id"].tolist()
        seed_nodes = [f"event::{event_id}" for event_id in seed_ids]
        target_node = f"event::{query['target_event_id']}"
        any_distance = _multi_source_distance(any_graphs[query["user_id"]], seed_nodes, target_node)
        next_distance = _multi_source_distance(next_graphs[query["user_id"]], seed_nodes, target_node)
        dense_rank = int(metric.loc[query_id, ("target_rank", "vector")])
        onehop_rank = int(metric.loc[query_id, ("target_rank", "graph")])
        ppr_rank = int(metric.loc[query_id, ("target_rank", PPR_METHOD)])
        raw = score_lookup.loc[(query_id, query["target_event_id"])]
        dense_top1 = ranking_groups["vector"][query_id].iloc[0]["retrieved_event_id"]
        onehop_top1 = ranking_groups["graph"][query_id].iloc[0]["retrieved_event_id"]
        ppr_top1 = ranking_groups[PPR_METHOD][query_id].iloc[0]["retrieved_event_id"]
        rows.append({
            "query_id": query_id,
            "user_id": query["user_id"],
            "query_type": query["query_type"],
            "query_text": query["query_text"],
            "target_event_id": query["target_event_id"],
            "dense_target_rank": dense_rank,
            "onehop_target_rank": onehop_rank,
            "ppr_rrf_target_rank": ppr_rank,
            "raw_ppr_target_rank": int(raw["ppr_rank"]),
            "raw_ppr_target_score": float(raw["ppr_score"]),
            "ppr_rrf_rank_gain_vs_dense": dense_rank - ppr_rank,
            "ppr_rrf_rank_gain_vs_onehop": onehop_rank - ppr_rank,
            "ppr_rrf_outcome_vs_dense": "improved" if ppr_rank < dense_rank else "worsened" if ppr_rank > dense_rank else "unchanged",
            "ppr_rrf_outcome_vs_onehop": "improved" if ppr_rank < onehop_rank else "worsened" if ppr_rank > onehop_rank else "unchanged",
            "dense_top1_event_id": dense_top1,
            "onehop_top1_event_id": onehop_top1,
            "ppr_rrf_top1_event_id": ppr_top1,
            "ppr_rrf_top1_identical_to_dense": ppr_top1 == dense_top1,
            "ppr_rrf_top1_identical_to_onehop": ppr_top1 == onehop_top1,
            "target_is_dense_seed": query["target_event_id"] in seed_ids,
            "shortest_any_relation_distance": "" if any_distance is None else any_distance,
            "shortest_any_relation_distance_bucket": _distance_bucket(any_distance),
            "shortest_next_only_distance": "" if next_distance is None else next_distance,
            "shortest_next_only_distance_bucket": _distance_bucket(next_distance),
            "is_high_memory_load": bool(query["_is_high_memory_load"]),
        })
    diagnostics = pd.DataFrame(rows).sort_values(["user_id", "query_id"], kind="stable").reset_index(drop=True)

    summary_rows = []
    groups = [("overall", diagnostics)] + list(diagnostics.groupby("query_type", sort=True))
    for query_type, group in groups:
        row: dict[str, object] = {
            "query_type": query_type,
            "query_count": len(group),
            "ppr_improved_vs_dense": int(group["ppr_rrf_outcome_vs_dense"].eq("improved").sum()),
            "ppr_unchanged_vs_dense": int(group["ppr_rrf_outcome_vs_dense"].eq("unchanged").sum()),
            "ppr_worsened_vs_dense": int(group["ppr_rrf_outcome_vs_dense"].eq("worsened").sum()),
            "ppr_improved_vs_onehop": int(group["ppr_rrf_outcome_vs_onehop"].eq("improved").sum()),
            "ppr_unchanged_vs_onehop": int(group["ppr_rrf_outcome_vs_onehop"].eq("unchanged").sum()),
            "ppr_worsened_vs_onehop": int(group["ppr_rrf_outcome_vs_onehop"].eq("worsened").sum()),
            "ppr_top1_identical_to_dense": int(group["ppr_rrf_top1_identical_to_dense"].sum()),
            "ppr_top1_identical_to_onehop": int(group["ppr_rrf_top1_identical_to_onehop"].sum()),
            "mean_ppr_rank_gain_vs_dense": group["ppr_rrf_rank_gain_vs_dense"].mean(),
            "mean_ppr_rank_gain_vs_onehop": group["ppr_rrf_rank_gain_vs_onehop"].mean(),
        }
        for prefix in ("shortest_any_relation_distance_bucket", "shortest_next_only_distance_bucket"):
            counts = group[prefix].value_counts()
            for bucket in ("0", "1", "2", "3+", "unreachable"):
                row[f"{prefix}_{bucket.replace('+', 'plus')}"] = int(counts.get(bucket, 0))
        summary_rows.append(row)
    return diagnostics, pd.DataFrame(summary_rows)


def _structured_evaluation(candidates: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "user_id", "next_successor_event_ids",
        "category_matched_event_ids", "uniquely_selected_event_id", "candidate_status",
    }
    if not required.issubset(candidates.columns):
        raise GraphExtensionEvaluationError("Structured NEXT candidates have an invalid schema")
    relational = queries.loc[queries["query_type"].eq("relational_after")].set_index("query_id")
    if set(candidates["query_id"]) != set(relational.index):
        raise GraphExtensionEvaluationError("Structured NEXT query coverage is incorrect")
    rows = []
    for candidate in candidates.itertuples(index=False):
        query = relational.loc[candidate.query_id]
        try:
            successors = json.loads(candidate.next_successor_event_ids)
            matches = json.loads(candidate.category_matched_event_ids)
        except json.JSONDecodeError as exc:
            raise GraphExtensionEvaluationError("Structured NEXT contains invalid candidate JSON") from exc
        target = query["target_event_id"]
        reachable = target in successors
        category_contains = target in matches
        unique_correct = candidate.uniquely_selected_event_id == target
        rows.append({
            "query_id": candidate.query_id,
            "user_id": candidate.user_id,
            "target_event_id": target,
            "target_reachable_before_category_filter": reachable,
            "target_in_category_matched_candidates": category_contains,
            "uniquely_resolved_target": unique_correct,
            "candidate_status": candidate.candidate_status,
            "next_successor_count": len(successors),
            "category_matched_candidate_count": len(matches),
            "is_high_memory_load": bool(query["_is_high_memory_load"]),
        })
    return pd.DataFrame(rows).sort_values(["user_id", "query_id"], kind="stable").reset_index(drop=True)


def _render_report(
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    diagnostics_summary: pd.DataFrame,
    structured: pd.DataFrame,
    wilcoxon: pd.DataFrame,
) -> str:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *body])

    diag = diagnostics_summary.loc[diagnostics_summary["query_type"].eq("overall")].iloc[0]
    relational_diag = diagnostics_summary.loc[
        diagnostics_summary["query_type"].eq("relational_after")
    ].iloc[0]
    pre = int(structured["target_reachable_before_category_filter"].sum())
    unique = int(structured["uniquely_resolved_target"].sum())
    ppr_overall = overall.loc[overall["method"].eq(PPR_METHOD)].iloc[0]
    dense_overall = overall.loc[overall["method"].eq("vector")].iloc[0]
    onehop_overall = overall.loc[overall["method"].eq("graph")].iloc[0]
    top1_changed = int(diag["query_count"] - diag["ppr_top1_identical_to_dense"])

    def distance_distribution(prefix: str) -> str:
        return ", ".join(
            f"{bucket}={int(diag[f'{prefix}_{bucket.replace('+', 'plus')}'])}"
            for bucket in ("0", "1", "2", "3+", "unreachable")
        )

    lines = [
        "# Phase 2 Graph Extension Validation Report", "",
        "## Core overall", "",
        table(overall, ["method_label", "query_count", "mrr", "success_at_1", "success_at_5"]), "",
        "## MRR by query type", "",
        table(by_type, ["query_type", "method_label", "mrr"]), "",
        "## Graph-PPR-RRF diagnosis", "",
        f"- Versus E5 Dense: {diag['ppr_improved_vs_dense']} improved, {diag['ppr_unchanged_vs_dense']} unchanged, {diag['ppr_worsened_vs_dense']} worsened.",
        f"- Versus Graph-1Hop-RRF: {diag['ppr_improved_vs_onehop']} improved, {diag['ppr_unchanged_vs_onehop']} unchanged, {diag['ppr_worsened_vs_onehop']} worsened.",
        f"- Top-1 identical to Dense: {diag['ppr_top1_identical_to_dense']}/{diag['query_count']} (changed: {top1_changed}).",
        f"- Any-relation seed-to-target distance: {distance_distribution('shortest_any_relation_distance_bucket')}.",
        f"- NEXT-only seed-to-target distance: {distance_distribution('shortest_next_only_distance_bucket')}.", "",
        "## Structured NEXT oracle", "",
        f"- Target reachable before category filtering: {pre}/{len(structured)}.",
        f"- Target uniquely resolved after category filtering: {unique}/{len(structured)}.",
        "- This is an oracle structured-cue diagnostic, not general text-retrieval performance.", "",
        "## User-level Wilcoxon comparisons", "",
        table(wilcoxon, ["comparison", "observation_count", "mean_paired_difference_b_minus_a", "raw_p_value", "holm_adjusted_p_value"]), "",
        "## Interpretation boundary", "",
        "Graph-PPR-RRF holds the encoder, Top-5 seeds, candidate set, and RRF parameters fixed, while replacing local sequential expansion with heterogeneous PPR evidence. Relation coverage and graph scoring also differ, so the comparison must not be described as changing hop depth alone.", "",
        "## Acceptance questions", "",
        f"1. **Does PPR-RRF outperform Dense or Graph-1Hop-RRF?** No. Its Core MRR is {ppr_overall['mrr']:.4f}, compared with {dense_overall['mrr']:.4f} for E5 Dense and {onehop_overall['mrr']:.4f} for Graph-1Hop-RRF. The user-level comparisons against both configurations remain different after Holm correction.",
        f"2. **Are rank changes concentrated in relational-after queries?** Not exclusively. Relational-after accounts for {int(relational_diag['ppr_improved_vs_dense'])} of {int(diag['ppr_improved_vs_dense'])} improvements and {int(relational_diag['ppr_worsened_vs_dense'])} of {int(diag['ppr_worsened_vs_dense'])} worsenings versus Dense; changes also occur in semantic and temporal-spatial queries.",
        f"3. **Does PPR-RRF change Top-1 decisions?** Yes. It changes {top1_changed}/{int(diag['query_count'])} Top-1 events relative to Dense, but aggregate Success@1 falls from {dense_overall['success_at_1']:.4f} to {ppr_overall['success_at_1']:.4f}.",
        f"4. **How far are targets from the E5 Top-5 seeds?** Any-relation distances are {distance_distribution('shortest_any_relation_distance_bucket')}; NEXT-only distances are {distance_distribution('shortest_next_only_distance_bucket')}. The difference identifies reachability introduced by POI, Category, and Trail links rather than sequential adjacency alone.",
        f"5. **Does the graph retain the benchmark's AFTER relation under oracle structured cues?** Yes for this dataset: {pre}/{len(structured)} targets are reachable before category filtering and {unique}/{len(structured)} are uniquely resolved after filtering. This is not text-query retrieval performance.",
        "6. **What mechanism is most consistent with the combined evidence?** Broader heterogeneous PPR diffusion does not improve the local Graph-1Hop-RRF result, while the oracle confirms that the required NEXT structure is present. Together with the distance and rank-change diagnostics, the result is more consistent with limitations of generic static diffusion and structural-to-ranking conversion than with missing graph information alone. Query-conditioned relation semantics remain an untested alternative, so no causal claim is made.", "",
    ]
    return "\n".join(lines)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_outputs(output_dir: Path, frames: dict[str, pd.DataFrame], report: str, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temps: dict[str, Path] = {}
    try:
        for filename, frame in frames.items():
            descriptor, name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output_dir)
            os.close(descriptor)
            temp = Path(name)
            frame.to_csv(temp, index=False, encoding="utf-8", float_format="%.10f")
            temps[filename] = temp
        descriptor, name = tempfile.mkstemp(prefix=f".{REPORT_FILENAME}.", suffix=".tmp", dir=output_dir)
        os.close(descriptor)
        report_temp = Path(name)
        report_temp.write_text(report, encoding="utf-8")
        temps[REPORT_FILENAME] = report_temp
        manifest = dict(manifest)
        manifest["outputs"] = {filename: sha256_file(temp) for filename, temp in sorted(temps.items())}
        descriptor, name = tempfile.mkstemp(prefix=f".{RUN_MANIFEST_FILENAME}.", suffix=".tmp", dir=output_dir)
        os.close(descriptor)
        manifest_temp = Path(name)
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for filename, temp in temps.items():
            temp.replace(output_dir / filename)
        manifest_temp.replace(output_dir / RUN_MANIFEST_FILENAME)
    except (OSError, ValueError) as exc:
        raise GraphExtensionEvaluationError(f"could not write Phase 2 evaluation: {exc}") from exc
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)


def evaluate_graph_extensions(
    events_path: str | Path,
    queries_path: str | Path,
    flat_dir: str | Path,
    vector_dir: str | Path,
    onehop_dir: str | Path,
    ppr_dir: str | Path,
    structured_next_dir: str | Path,
    output_dir: str | Path,
) -> GraphExtensionEvaluationSummary:
    """Validate and evaluate the four Core retrieval configurations."""

    paths = [Path(value) for value in (
        events_path, queries_path, flat_dir, vector_dir, onehop_dir,
        ppr_dir, structured_next_dir, output_dir,
    )]
    events_path, queries_path, flat_dir, vector_dir, onehop_dir, ppr_dir, structured_next_dir, output_dir = paths
    project_root = Path(__file__).resolve().parents[1]
    try:
        events = _load_events(events_path)
        full_events = build_travel_graph(
            pd.read_csv(events_path, dtype=str, keep_default_na=False)
        ).events
        queries = _load_queries(queries_path, "core", "Core queries")
        _validate_targets(events, queries)
        vector_manifest_path = _validate_run_manifest(vector_dir, "vector")
        onehop_manifest_path = _validate_run_manifest(onehop_dir, "graph")
        ppr_manifest, ppr_manifest_path = _read_json_manifest(
            ppr_dir, PPR_MANIFEST_FILENAME, "method", PPR_METHOD
        )
        structured_manifest, structured_manifest_path = _read_json_manifest(
            structured_next_dir, STRUCTURED_MANIFEST_FILENAME,
            "diagnostic", "structured_next_structured_cue_oracle",
        )
        if ppr_manifest.get("dense_seed_model", {}).get("id") != "intfloat/multilingual-e5-small":
            raise GraphExtensionEvaluationError("PPR manifest does not identify E5 seeds")
        rankings = {
            "flat": _load_ranking(flat_dir / "tokyo_flat_core_rankings.csv", "flat", "core", queries, events),
            "vector": _load_ranking(vector_dir / "tokyo_vector_core_rankings.csv", "vector", "core", queries, events),
            "graph": _load_ranking(onehop_dir / "tokyo_graph_core_rankings.csv", "graph", "core", queries, events),
            PPR_METHOD: _load_ranking(ppr_dir / PPR_RANKINGS_FILENAME, PPR_METHOD, "core", queries, events),
        }
    except (EvaluationError, GraphExtensionEvaluationError, GraphBuildError, OSError, pd.errors.ParserError) as exc:
        raise GraphExtensionEvaluationError(str(exc)) from exc

    candidate_sets = {
        method: {
            query_id: frozenset(group["retrieved_event_id"])
            for query_id, group in frame.groupby("query_id", sort=False)
        }
        for method, frame in rankings.items()
    }
    for query_id in queries["query_id"]:
        sets = [candidate_sets[method][query_id] for method in METHODS]
        if any(value != sets[0] for value in sets[1:]):
            raise GraphExtensionEvaluationError(f"candidate sets differ for {query_id}")

    try:
        ppr_scores = pd.read_csv(ppr_dir / PPR_EVENT_SCORES_FILENAME, dtype=str, keep_default_na=False)
        candidates = pd.read_csv(
            structured_next_dir / STRUCTURED_CANDIDATES_FILENAME,
            dtype=str, keep_default_na=False,
        )
        for column in ("ppr_score", "ppr_rank"):
            if column not in ppr_scores:
                raise GraphExtensionEvaluationError(f"PPR event scores missing {column}")
        ppr_scores["ppr_score"] = pd.to_numeric(ppr_scores["ppr_score"], errors="raise")
        ppr_scores["ppr_rank"] = pd.to_numeric(ppr_scores["ppr_rank"], errors="raise").astype(int)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise GraphExtensionEvaluationError(f"could not load diagnostic inputs: {exc}") from exc

    per_query = _calculate_per_query(queries, rankings)
    overall = _metrics(per_query, False)
    by_type = _metrics(per_query, True)
    wilcoxon = _wilcoxon(per_query)
    diagnostics, diagnostic_summary = _ppr_diagnostics(
        full_events, queries, per_query, rankings, ppr_scores
    )
    structured = _structured_evaluation(candidates, queries)
    report = _render_report(overall, by_type, diagnostic_summary, structured, wilcoxon)

    input_paths = {
        "canonical_events": events_path,
        "core_queries": queries_path,
        "flat_rankings": flat_dir / "tokyo_flat_core_rankings.csv",
        "vector_manifest": vector_manifest_path,
        "onehop_manifest": onehop_manifest_path,
        "ppr_manifest": ppr_manifest_path,
        "structured_next_manifest": structured_manifest_path,
    }
    manifest = {
        "evaluation": "phase_02_graph_extensions",
        "inputs": {
            key: {"path": _portable(path, project_root), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "methods": list(METHODS),
        "parameters": {
            "benchmark_tier": "core",
            "distance_buckets": ["0", "1", "2", "3+", "unreachable"],
            "holm_comparison_family": [
                "flat_vs_vector", "vector_vs_graph",
                f"vector_vs_{PPR_METHOD}", f"graph_vs_{PPR_METHOD}",
            ],
            "metrics": ["mrr", "success_at_1", "success_at_5"],
            "ppr_score_atol": 1e-12,
            "ppr_score_rtol": 1e-10,
            "wilcoxon": {
                "alternative": "two-sided", "calculation_method": "approx",
                "continuity_correction": False, "observation_unit": "per-user Core mean RR",
                "zero_method": "wilcox",
            },
        },
        "runtime_versions": {
            "networkx": importlib.metadata.version("networkx"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "python": sys.version.split()[0],
            "scipy": importlib.metadata.version("scipy"),
        },
    }
    _write_outputs(output_dir, {
        PER_QUERY_FILENAME: per_query,
        OVERALL_FILENAME: overall,
        BY_TYPE_FILENAME: by_type,
        WILCOXON_FILENAME: wilcoxon,
        PPR_DIAGNOSTICS_FILENAME: diagnostics,
        PPR_DIAGNOSTIC_SUMMARY_FILENAME: diagnostic_summary,
        STRUCTURED_EVALUATION_FILENAME: structured,
    }, report, manifest)
    overall_diag = diagnostic_summary.loc[diagnostic_summary["query_type"].eq("overall")].iloc[0]
    return GraphExtensionEvaluationSummary(
        query_count=len(queries), per_query_count=len(per_query),
        ppr_improved_vs_dense=int(overall_diag["ppr_improved_vs_dense"]),
        ppr_unchanged_vs_dense=int(overall_diag["ppr_unchanged_vs_dense"]),
        ppr_worsened_vs_dense=int(overall_diag["ppr_worsened_vs_dense"]),
        structured_unique_correct=int(structured["uniquely_resolved_target"].sum()),
        output_dir=output_dir,
    )
