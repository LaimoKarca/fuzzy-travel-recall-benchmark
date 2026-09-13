"""Evaluate Flat, Vector, and Graph retrieval with deterministic outputs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.model_download import sha256_file


METHODS = ("flat", "vector", "graph")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}

PER_QUERY_FILENAME = "tokyo_per_query_metrics.csv"
CORE_OVERALL_FILENAME = "tokyo_core_overall_metrics.csv"
BY_QUERY_TYPE_FILENAME = "tokyo_core_metrics_by_query_type.csv"
HIGH_LOAD_FILENAME = "tokyo_high_memory_load_metrics.csv"
BETWEEN_FILENAME = "tokyo_between_stress_metrics.csv"
WILCOXON_FILENAME = "tokyo_pairwise_wilcoxon.csv"
GRAPH_DIAGNOSTICS_FILENAME = "tokyo_graph_query_diagnostics.csv"
GRAPH_DIAGNOSTIC_SUMMARY_FILENAME = "tokyo_graph_diagnostic_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_evaluation_run_manifest.json"

RANKING_FILENAMES = {
    ("flat", "core"): "tokyo_flat_core_rankings.csv",
    ("flat", "stress_test"): "tokyo_flat_between_rankings.csv",
    ("vector", "core"): "tokyo_vector_core_rankings.csv",
    ("vector", "stress_test"): "tokyo_vector_between_rankings.csv",
    ("graph", "core"): "tokyo_graph_core_rankings.csv",
    ("graph", "stress_test"): "tokyo_graph_between_rankings.csv",
}
MANIFEST_FILENAMES = {
    "vector": "tokyo_vector_run_manifest.json",
    "graph": "tokyo_graph_run_manifest.json",
}
GRAPH_AUDIT_FILENAMES = {
    "core": "tokyo_graph_core_expansion_audit.csv",
    "stress_test": "tokyo_graph_between_expansion_audit.csv",
}

EVENT_COLUMNS = (
    "event_id",
    "user_id",
    "venue_id",
    "is_high_memory_load",
)
QUERY_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "query_text",
    "target_event_id",
    "target_venue_id",
    "target_name",
    "cue_previous_event_id",
    "cue_previous_place",
    "cue_next_event_id",
    "cue_next_place",
    "is_high_memory_load",
)
RANKING_COLUMNS = (
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
PER_QUERY_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "method",
    "query_text",
    "target_event_id",
    "target_venue_id",
    "target_name",
    "target_rank",
    "reciprocal_rank",
    "success_at_1",
    "success_at_5",
    "is_high_memory_load",
)
METRIC_COLUMNS = (
    "benchmark_tier",
    "cohort",
    "query_type",
    "method",
    "query_count",
    "user_count",
    "mrr",
    "success_at_1",
    "success_at_5",
)


class EvaluationError(RuntimeError):
    """Raised when formal evaluation cannot complete safely."""


@dataclass(frozen=True)
class EvaluationSummary:
    per_query_count: int
    core_query_count: int
    between_query_count: int
    high_load_core_query_count: int
    high_load_between_query_count: int
    graph_improved_count: int
    graph_unchanged_count: int
    graph_worsened_count: int


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise EvaluationError(f"{label} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"could not parse {label} CSV {path}: {exc}") from exc
    if frame.empty:
        raise EvaluationError(f"{label} CSV contains no rows: {path}")
    return frame


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> pd.DataFrame:
    missing = [column for column in required if column not in frame]
    if missing:
        raise EvaluationError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )
    return frame.loc[:, required].copy()


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:3]
        raise EvaluationError(
            f"{label} contains invalid boolean value(s): {', '.join(examples)}"
        )
    return normalized.isin(TRUE_VALUES)


def _strict_integer(series: pd.Series, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} contains invalid integer value(s)") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise EvaluationError(f"{label} contains invalid integer value(s)")
    return numeric.astype(int)


def _load_events(path: Path) -> pd.DataFrame:
    events = _require_columns(_read_csv(path, "events"), EVENT_COLUMNS, "events")
    for column in ("event_id", "user_id", "venue_id"):
        if events[column].str.strip().eq("").any():
            raise EvaluationError(f"events contains blank {column} value(s)")
    if events["event_id"].duplicated().any():
        raise EvaluationError("events contains duplicate event_id values")
    events["_is_high_memory_load"] = _parse_boolean(
        events["is_high_memory_load"], "events.is_high_memory_load"
    )
    if events.groupby("user_id")["_is_high_memory_load"].nunique().gt(1).any():
        raise EvaluationError("events has inconsistent High-load flags within a user")
    return events.sort_values(["user_id", "event_id"], kind="stable").reset_index(
        drop=True
    )


def _load_queries(path: Path, tier: str, label: str) -> pd.DataFrame:
    queries = _require_columns(_read_csv(path, label), QUERY_COLUMNS, label)
    for column in (
        "query_id",
        "user_id",
        "benchmark_tier",
        "query_type",
        "query_text",
        "target_event_id",
        "target_venue_id",
        "target_name",
    ):
        if queries[column].str.strip().eq("").any():
            raise EvaluationError(f"{label} contains blank {column} value(s)")
    if queries["query_id"].duplicated().any():
        raise EvaluationError(f"{label} contains duplicate query_id values")
    if set(queries["benchmark_tier"]) != {tier}:
        raise EvaluationError(f"{label} contains an unexpected benchmark tier")
    allowed_types = (
        {"semantic", "temporal_spatial", "relational_after"}
        if tier == "core"
        else {"relational_between"}
    )
    if not set(queries["query_type"]).issubset(allowed_types):
        raise EvaluationError(f"{label} contains an unexpected query type")
    queries["_is_high_memory_load"] = _parse_boolean(
        queries["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    return queries.sort_values(["user_id", "query_id"], kind="stable").reset_index(
        drop=True
    )


def _validate_query_targets(
    events: pd.DataFrame, core: pd.DataFrame, between: pd.DataFrame
) -> pd.DataFrame:
    if set(core["query_id"]) & set(between["query_id"]):
        raise EvaluationError("Core and Between queries contain duplicate query IDs")
    queries = pd.concat([core, between], ignore_index=True)
    event_lookup = events.set_index("event_id")
    missing = sorted(set(queries["target_event_id"]) - set(events["event_id"]))
    if missing:
        raise EvaluationError(
            "queries reference missing target event(s): " + ", ".join(missing[:5])
        )
    for _, query in queries.iterrows():
        event = event_lookup.loc[query["target_event_id"]]
        if event["user_id"] != query["user_id"]:
            raise EvaluationError("query target crosses a user boundary")
        if event["venue_id"] != query["target_venue_id"]:
            raise EvaluationError("query target_venue_id does not match its event")
        if bool(event["_is_high_memory_load"]) != bool(query["_is_high_memory_load"]):
            raise EvaluationError("query has an incorrect High-load flag")
    return queries


def _validate_run_manifest(directory: Path, method: str) -> Path:
    manifest_path = directory / MANIFEST_FILENAMES[method]
    if not manifest_path.is_file():
        raise EvaluationError(f"{method} run manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"could not parse {method} run manifest: {exc}") from exc
    if manifest.get("method") != method or not isinstance(manifest.get("outputs"), dict):
        raise EvaluationError(f"{method} run manifest has an invalid structure")
    required = {
        RANKING_FILENAMES[(method, "core")],
        RANKING_FILENAMES[(method, "stress_test")],
    }
    if method == "graph":
        required |= set(GRAPH_AUDIT_FILENAMES.values())
    missing = sorted(required - set(manifest["outputs"]))
    if missing:
        raise EvaluationError(
            f"{method} run manifest is missing output hash(es): " + ", ".join(missing)
        )
    for filename, expected_hash in sorted(manifest["outputs"].items()):
        if Path(filename).name != filename:
            raise EvaluationError(f"{method} run manifest contains an unsafe filename")
        artifact = directory / filename
        if not artifact.is_file():
            raise EvaluationError(f"{method} artifact does not exist: {artifact}")
        if sha256_file(artifact) != expected_hash:
            raise EvaluationError(f"{method} artifact hash mismatch: {filename}")
    return manifest_path


def _load_ranking(
    path: Path,
    method: str,
    tier: str,
    queries: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    label = f"{method} {tier} rankings"
    ranking = _require_columns(_read_csv(path, label), RANKING_COLUMNS, label)
    if set(ranking["method"]) != {method}:
        raise EvaluationError(f"{label} contains an unexpected method")
    if set(ranking["benchmark_tier"]) != {tier}:
        raise EvaluationError(f"{label} contains an unexpected benchmark tier")
    ranking["rank"] = _strict_integer(ranking["rank"], f"{label}.rank")
    try:
        ranking["score"] = pd.to_numeric(ranking["score"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} contains invalid score value(s)") from exc
    if not np.isfinite(ranking["score"].to_numpy(dtype=float)).all():
        raise EvaluationError(f"{label} contains non-finite score value(s)")
    ranking["_is_high_memory_load"] = _parse_boolean(
        ranking["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    if set(ranking["query_id"]) != set(queries["query_id"]):
        raise EvaluationError(f"{label} query coverage does not match its query CSV")

    event_users = events.set_index("event_id")["user_id"].to_dict()
    events_by_user = {
        user_id: set(group["event_id"])
        for user_id, group in events.groupby("user_id", sort=False)
    }
    query_lookup = queries.set_index("query_id")
    for query_id, group in ranking.groupby("query_id", sort=False):
        query = query_lookup.loc[query_id]
        actual = group.sort_values("rank", kind="stable")
        if actual["rank"].tolist() != list(range(1, len(actual) + 1)):
            raise EvaluationError(f"{label} has non-continuous ranks for {query_id}")
        if actual["retrieved_event_id"].duplicated().any():
            raise EvaluationError(f"{label} has duplicate candidates for {query_id}")
        expected_candidates = events_by_user.get(query["user_id"])
        if expected_candidates is None or set(actual["retrieved_event_id"]) != expected_candidates:
            raise EvaluationError(f"{label} has incomplete candidate coverage for {query_id}")
        if any(event_users[event_id] != query["user_id"] for event_id in actual["retrieved_event_id"]):
            raise EvaluationError(f"{label} crosses a user boundary for {query_id}")
        for column in ("user_id", "benchmark_tier", "query_type"):
            if set(actual[column]) != {query[column]}:
                raise EvaluationError(f"{label} has inconsistent {column} for {query_id}")
        if actual["_is_high_memory_load"].ne(query["_is_high_memory_load"]).any():
            raise EvaluationError(f"{label} has an incorrect High-load flag for {query_id}")
        scores = actual["score"].to_numpy(dtype=float)
        ids = actual["retrieved_event_id"].tolist()
        if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
            raise EvaluationError(f"{label} scores are not descending for {query_id}")
        if any(
            scores[index] == scores[index + 1] and ids[index] > ids[index + 1]
            for index in range(len(scores) - 1)
        ):
            raise EvaluationError(f"{label} has an invalid tie order for {query_id}")
    return ranking.sort_values(["user_id", "query_id", "rank"], kind="stable").reset_index(drop=True)


def _calculate_per_query(
    queries: pd.DataFrame, rankings: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    query_lookup = queries.set_index("query_id")
    for method in METHODS:
        ranking = rankings[method]
        groups = {query_id: group for query_id, group in ranking.groupby("query_id", sort=False)}
        for query_id, query in query_lookup.iterrows():
            target_rows = groups[query_id].loc[
                groups[query_id]["retrieved_event_id"].eq(query["target_event_id"])
            ]
            if len(target_rows) != 1:
                raise EvaluationError(
                    f"{method} ranking does not contain exactly one target for {query_id}"
                )
            target_rank = int(target_rows.iloc[0]["rank"])
            rows.append(
                {
                    "query_id": query_id,
                    "user_id": query["user_id"],
                    "benchmark_tier": query["benchmark_tier"],
                    "query_type": query["query_type"],
                    "method": method,
                    "query_text": query["query_text"],
                    "target_event_id": query["target_event_id"],
                    "target_venue_id": query["target_venue_id"],
                    "target_name": query["target_name"],
                    "target_rank": target_rank,
                    "reciprocal_rank": 1.0 / target_rank,
                    "success_at_1": target_rank <= 1,
                    "success_at_5": target_rank <= 5,
                    "is_high_memory_load": bool(query["_is_high_memory_load"]),
                }
            )
    result = pd.DataFrame(rows, columns=PER_QUERY_COLUMNS)
    result["_method_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["benchmark_tier", "user_id", "query_id", "_method_order"], kind="stable"
    ).drop(columns="_method_order").reset_index(drop=True)


def _aggregate_metrics(
    per_query: pd.DataFrame,
    tier: str,
    cohort: str,
    by_query_type: bool,
) -> pd.DataFrame:
    selected = per_query.loc[per_query["benchmark_tier"].eq(tier)].copy()
    if cohort == "high_memory_load":
        selected = selected.loc[selected["is_high_memory_load"]]
    if selected.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    group_columns = ["query_type", "method"] if by_query_type else ["method"]
    grouped = selected.groupby(group_columns, sort=False, observed=True)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        if by_query_type:
            query_type, method = key
        else:
            method = key[0] if isinstance(key, tuple) else key
            query_type = "overall"
        rows.append(
            {
                "benchmark_tier": tier,
                "cohort": cohort,
                "query_type": query_type,
                "method": method,
                "query_count": group["query_id"].nunique(),
                "user_count": group["user_id"].nunique(),
                "mrr": group["reciprocal_rank"].mean(),
                "success_at_1": group["success_at_1"].mean(),
                "success_at_5": group["success_at_5"].mean(),
            }
        )
    result = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    result["_method_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["query_type", "_method_order"], kind="stable"
    ).drop(columns="_method_order").reset_index(drop=True)


def _high_load_comparison(per_query: pd.DataFrame) -> pd.DataFrame:
    main = _aggregate_metrics(per_query, "core", "main", False)
    high = _aggregate_metrics(per_query, "core", "high_memory_load", False)
    result = pd.concat(
        [frame for frame in (main, high) if not frame.empty], ignore_index=True
    )
    main_lookup = main.set_index("method")
    for metric in ("mrr", "success_at_1", "success_at_5"):
        result[f"{metric}_change_from_main"] = result.apply(
            lambda row: row[metric] - main_lookup.loc[row["method"], metric], axis=1
        )
    result["_cohort_order"] = result["cohort"].map({"main": 0, "high_memory_load": 1})
    result["_method_order"] = result["method"].map(METHOD_ORDER)
    return result.sort_values(
        ["_cohort_order", "_method_order"], kind="stable"
    ).drop(columns=["_cohort_order", "_method_order"]).reset_index(drop=True)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for position, index in enumerate(order):
        candidate = min(1.0, p_values[index] * (count - position))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _wilcoxon_results(per_query: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except (ImportError, OSError) as exc:
        raise EvaluationError(
            "scipy is not installed; run 'python -m pip install -r requirements.txt'"
        ) from exc

    core = per_query.loc[per_query["benchmark_tier"].eq("core")]
    user_means = core.groupby(["user_id", "method"], sort=True)[
        "reciprocal_rank"
    ].mean().unstack("method")
    comparisons = (("flat", "vector"), ("vector", "graph"))
    rows: list[dict[str, object]] = []
    for method_a, method_b in comparisons:
        paired = user_means[[method_a, method_b]].dropna()
        differences = paired[method_b] - paired[method_a]
        nonzero = int(differences.ne(0).sum())
        if nonzero == 0:
            statistic = 0.0
            p_value = 1.0
            status = "all_zero_differences"
        else:
            result = wilcoxon(
                paired[method_a],
                paired[method_b],
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
            status = "ok"
        rows.append(
            {
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
                "alternative": "two-sided",
                "zero_method": "wilcox",
                "calculation_method": "approx",
                "continuity_correction": False,
                "status": status,
            }
        )
    adjusted = _holm_adjust([float(row["raw_p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p_value"] = adjusted_p
        row["reject_null_holm"] = adjusted_p < float(row["alpha"])
    return pd.DataFrame(rows)


def _load_graph_audit(
    path: Path, tier: str, queries: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    label = f"Graph {tier} expansion audit"
    audit = _require_columns(_read_csv(path, label), AUDIT_COLUMNS, label)
    audit["seed_rank"] = _strict_integer(audit["seed_rank"], f"{label}.seed_rank")
    if audit["seed_rank"].lt(1).any() or audit["seed_rank"].gt(5).any():
        raise EvaluationError(f"{label} contains seed ranks outside 1..5")
    audit["_is_first_discovery"] = _parse_boolean(
        audit["is_first_discovery"], f"{label}.is_first_discovery"
    )
    audit["_is_high_memory_load"] = _parse_boolean(
        audit["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    if not set(audit["relation"]).issubset({"SELF", "PREVIOUS", "NEXT"}):
        raise EvaluationError(f"{label} contains an unexpected relation")
    if set(audit["query_id"]) != set(queries["query_id"]):
        raise EvaluationError(f"{label} query coverage does not match its query CSV")
    query_lookup = queries.set_index("query_id")
    event_users = events.set_index("event_id")["user_id"].to_dict()
    for column in ("seed_event_id", "expanded_event_id"):
        if audit[column].str.strip().eq("").any():
            raise EvaluationError(f"{label} contains blank {column} value(s)")
        missing_events = sorted(set(audit[column]) - set(event_users))
        if missing_events:
            raise EvaluationError(
                f"{label} references missing {column} value(s): "
                + ", ".join(missing_events[:5])
            )
    first = audit.loc[audit["_is_first_discovery"]].copy()
    if first.duplicated(["query_id", "expanded_event_id"]).any():
        raise EvaluationError(f"{label} contains duplicate first discoveries")
    first["_expansion_rank"] = _strict_integer(
        first["expansion_rank"], f"{label}.expansion_rank"
    )
    nonfirst = audit.loc[~audit["_is_first_discovery"]]
    if nonfirst["expansion_rank"].str.strip().ne("").any():
        raise EvaluationError(f"{label} assigns expansion ranks after first discovery")
    for query_id, group in audit.groupby("query_id", sort=False):
        query = query_lookup.loc[query_id]
        for column in ("user_id", "benchmark_tier", "query_type"):
            if set(group[column]) != {query[column]}:
                raise EvaluationError(f"{label} has inconsistent {column} for {query_id}")
        if group["_is_high_memory_load"].ne(query["_is_high_memory_load"]).any():
            raise EvaluationError(f"{label} has an incorrect High-load flag for {query_id}")
        if any(
            event_users[event_id] != query["user_id"]
            for event_id in pd.concat(
                [group["seed_event_id"], group["expanded_event_id"]],
                ignore_index=True,
            )
        ):
            raise EvaluationError(f"{label} crosses a user boundary for {query_id}")
        ranks = first.loc[first["query_id"].eq(query_id), "_expansion_rank"].tolist()
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise EvaluationError(f"{label} has invalid expansion ranks for {query_id}")
    return audit


def _graph_diagnostics(
    queries: pd.DataFrame,
    per_query: pd.DataFrame,
    events: pd.DataFrame,
    audits: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metric_pivot = per_query.pivot(
        index="query_id", columns="method", values=["target_rank", "reciprocal_rank"]
    )
    first = audits.loc[audits["_is_first_discovery"]].copy()
    expansion_lookup = {
        (row["query_id"], row["expanded_event_id"]): row
        for _, row in first.iterrows()
    }
    event_venue = events.set_index("event_id")["venue_id"].to_dict()
    visit_counts = events.groupby(["user_id", "venue_id"], sort=False).size().to_dict()
    ranking_groups = {
        method: {
            query_id: group.sort_values("rank", kind="stable")
            for query_id, group in frame.groupby("query_id", sort=False)
        }
        for method, frame in rankings.items()
    }

    rows: list[dict[str, object]] = []
    for _, query in queries.sort_values(
        ["benchmark_tier", "user_id", "query_id"], kind="stable"
    ).iterrows():
        vector_rank = int(metric_pivot.loc[query["query_id"], ("target_rank", "vector")])
        graph_rank = int(metric_pivot.loc[query["query_id"], ("target_rank", "graph")])
        flat_rank = int(metric_pivot.loc[query["query_id"], ("target_rank", "flat")])
        graph_gain_vector = vector_rank - graph_rank
        graph_gain_flat = flat_rank - graph_rank
        expansion = expansion_lookup.get((query["query_id"], query["target_event_id"]))
        row: dict[str, object] = {
            "query_id": query["query_id"],
            "user_id": query["user_id"],
            "benchmark_tier": query["benchmark_tier"],
            "query_type": query["query_type"],
            "query_text": query["query_text"],
            "target_event_id": query["target_event_id"],
            "target_venue_id": query["target_venue_id"],
            "target_name": query["target_name"],
            "cue_previous_event_id": query["cue_previous_event_id"],
            "cue_previous_place": query["cue_previous_place"],
            "cue_next_event_id": query["cue_next_event_id"],
            "cue_next_place": query["cue_next_place"],
            "flat_target_rank": flat_rank,
            "vector_target_rank": vector_rank,
            "graph_target_rank": graph_rank,
            "flat_reciprocal_rank": float(metric_pivot.loc[query["query_id"], ("reciprocal_rank", "flat")]),
            "vector_reciprocal_rank": float(metric_pivot.loc[query["query_id"], ("reciprocal_rank", "vector")]),
            "graph_reciprocal_rank": float(metric_pivot.loc[query["query_id"], ("reciprocal_rank", "graph")]),
            "graph_rank_gain_vs_vector": graph_gain_vector,
            "graph_rank_gain_vs_flat": graph_gain_flat,
            "graph_outcome_vs_vector": "improved" if graph_gain_vector > 0 else "worsened" if graph_gain_vector < 0 else "unchanged",
            "graph_outcome_vs_flat": "improved" if graph_gain_flat > 0 else "worsened" if graph_gain_flat < 0 else "unchanged",
            "target_in_expansion": expansion is not None,
            "target_first_discovery_relation": expansion["relation"] if expansion is not None else "",
            "target_first_discovery_seed_rank": int(expansion["seed_rank"]) if expansion is not None else "",
            "target_expansion_rank": int(expansion["expansion_rank"]) if expansion is not None else "",
            "target_venue_visit_count": int(visit_counts[(query["user_id"], query["target_venue_id"])]),
            "is_repeated_target_venue": visit_counts[(query["user_id"], query["target_venue_id"])] > 1,
            "is_high_memory_load": bool(query["_is_high_memory_load"]),
        }
        for method in METHODS:
            group = ranking_groups[method][query["query_id"]]
            preceding = group.loc[group["rank"].lt({"flat": flat_rank, "vector": vector_rank, "graph": graph_rank}[method])]
            wrong_same = preceding.loc[
                preceding["retrieved_event_id"].map(event_venue).eq(query["target_venue_id"])
                & preceding["retrieved_event_id"].ne(query["target_event_id"])
            ]
            row[f"{method}_wrong_same_venue_above_target_count"] = len(wrong_same)
            row[f"{method}_best_wrong_same_venue_rank"] = (
                int(wrong_same["rank"].min()) if not wrong_same.empty else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _graph_diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    grouping_sets: list[tuple[str, pd.DataFrame]] = [("overall", diagnostics)]
    grouping_sets.extend(
        (query_type, group)
        for query_type, group in diagnostics.groupby("query_type", sort=True)
    )
    rows: list[dict[str, object]] = []
    for query_type, group in grouping_sets:
        rows.append(
            {
                "query_type": query_type,
                "query_count": len(group),
                "graph_improved_vs_vector_count": int(group["graph_outcome_vs_vector"].eq("improved").sum()),
                "graph_unchanged_vs_vector_count": int(group["graph_outcome_vs_vector"].eq("unchanged").sum()),
                "graph_worsened_vs_vector_count": int(group["graph_outcome_vs_vector"].eq("worsened").sum()),
                "graph_improved_vs_flat_count": int(group["graph_outcome_vs_flat"].eq("improved").sum()),
                "graph_unchanged_vs_flat_count": int(group["graph_outcome_vs_flat"].eq("unchanged").sum()),
                "graph_worsened_vs_flat_count": int(group["graph_outcome_vs_flat"].eq("worsened").sum()),
                "target_in_expansion_count": int(group["target_in_expansion"].sum()),
                "target_first_discovered_as_self_count": int(group["target_first_discovery_relation"].eq("SELF").sum()),
                "target_first_discovered_as_previous_count": int(group["target_first_discovery_relation"].eq("PREVIOUS").sum()),
                "target_first_discovered_as_next_count": int(group["target_first_discovery_relation"].eq("NEXT").sum()),
                "repeated_target_venue_query_count": int(group["is_repeated_target_venue"].sum()),
                "flat_wrong_same_venue_above_target_query_count": int(group["flat_wrong_same_venue_above_target_count"].gt(0).sum()),
                "vector_wrong_same_venue_above_target_query_count": int(group["vector_wrong_same_venue_above_target_count"].gt(0).sum()),
                "graph_wrong_same_venue_above_target_query_count": int(group["graph_wrong_same_venue_above_target_count"].gt(0).sum()),
                "mean_graph_rank_gain_vs_vector": group["graph_rank_gain_vs_vector"].mean(),
                "mean_graph_rank_gain_vs_flat": group["graph_rank_gain_vs_flat"].mean(),
            }
        )
    return pd.DataFrame(rows)


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
            frame.to_csv(
                temp,
                index=False,
                encoding="utf-8",
                float_format="%.10f",
            )
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
        raise EvaluationError(f"could not write evaluation outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def evaluate(
    events_path: str | Path,
    core_queries_path: str | Path,
    between_queries_path: str | Path,
    flat_dir: str | Path,
    vector_dir: str | Path,
    graph_dir: str | Path,
    output_dir: str | Path,
) -> EvaluationSummary:
    """Validate Stage 2-4 artifacts and write the formal Stage 5 evaluation."""

    events_path = Path(events_path)
    core_queries_path = Path(core_queries_path)
    between_queries_path = Path(between_queries_path)
    method_dirs = {
        "flat": Path(flat_dir),
        "vector": Path(vector_dir),
        "graph": Path(graph_dir),
    }
    output_dir = Path(output_dir)

    vector_manifest = _validate_run_manifest(method_dirs["vector"], "vector")
    graph_manifest = _validate_run_manifest(method_dirs["graph"], "graph")
    events = _load_events(events_path)
    core_queries = _load_queries(core_queries_path, "core", "Core queries")
    between_queries = _load_queries(
        between_queries_path, "stress_test", "Between queries"
    )
    all_queries = _validate_query_targets(events, core_queries, between_queries)

    rankings_by_tier: dict[str, dict[str, pd.DataFrame]] = {
        "core": {},
        "stress_test": {},
    }
    for tier, queries in (("core", core_queries), ("stress_test", between_queries)):
        for method in METHODS:
            rankings_by_tier[tier][method] = _load_ranking(
                method_dirs[method] / RANKING_FILENAMES[(method, tier)],
                method,
                tier,
                queries,
                events,
            )
        for query_id in queries["query_id"]:
            candidate_sets = [
                set(
                    rankings_by_tier[tier][method].loc[
                        rankings_by_tier[tier][method]["query_id"].eq(query_id),
                        "retrieved_event_id",
                    ]
                )
                for method in METHODS
            ]
            if not all(candidates == candidate_sets[0] for candidates in candidate_sets[1:]):
                raise EvaluationError(
                    f"retrieval methods have different candidates for {query_id}"
                )

    core_metrics = _calculate_per_query(core_queries, rankings_by_tier["core"])
    between_metrics = _calculate_per_query(
        between_queries, rankings_by_tier["stress_test"]
    )
    per_query = pd.concat([core_metrics, between_metrics], ignore_index=True)

    core_overall = _aggregate_metrics(per_query, "core", "main", False)
    by_query_type = _aggregate_metrics(per_query, "core", "main", True)
    high_load = _high_load_comparison(per_query)
    between_main = _aggregate_metrics(per_query, "stress_test", "main", False)
    between_high = _aggregate_metrics(
        per_query, "stress_test", "high_memory_load", False
    )
    between_results = pd.concat(
        [frame for frame in (between_main, between_high) if not frame.empty],
        ignore_index=True,
    )
    between_results["_cohort_order"] = between_results["cohort"].map(
        {"main": 0, "high_memory_load": 1}
    )
    between_results["_method_order"] = between_results["method"].map(METHOD_ORDER)
    between_results = between_results.sort_values(
        ["_cohort_order", "_method_order"], kind="stable"
    ).drop(columns=["_cohort_order", "_method_order"]).reset_index(drop=True)
    wilcoxon_results = _wilcoxon_results(per_query)

    core_audit = _load_graph_audit(
        method_dirs["graph"] / GRAPH_AUDIT_FILENAMES["core"],
        "core",
        core_queries,
        events,
    )
    between_audit = _load_graph_audit(
        method_dirs["graph"] / GRAPH_AUDIT_FILENAMES["stress_test"],
        "stress_test",
        between_queries,
        events,
    )
    audits = pd.concat([core_audit, between_audit], ignore_index=True)
    all_rankings = {
        method: pd.concat(
            [rankings_by_tier["core"][method], rankings_by_tier["stress_test"][method]],
            ignore_index=True,
        )
        for method in METHODS
    }
    diagnostics = _graph_diagnostics(
        all_queries, per_query, events, audits, all_rankings
    )
    diagnostic_summary = _graph_diagnostic_summary(diagnostics)

    frames = {
        output_dir / PER_QUERY_FILENAME: per_query,
        output_dir / CORE_OVERALL_FILENAME: core_overall,
        output_dir / BY_QUERY_TYPE_FILENAME: by_query_type,
        output_dir / HIGH_LOAD_FILENAME: high_load,
        output_dir / BETWEEN_FILENAME: between_results,
        output_dir / WILCOXON_FILENAME: wilcoxon_results,
        output_dir / GRAPH_DIAGNOSTICS_FILENAME: diagnostics,
        output_dir / GRAPH_DIAGNOSTIC_SUMMARY_FILENAME: diagnostic_summary,
    }
    input_paths = {
        "canonical_events": events_path,
        "core_queries": core_queries_path,
        "between_queries": between_queries_path,
        "vector_run_manifest": vector_manifest,
        "graph_run_manifest": graph_manifest,
        "flat_core_rankings": method_dirs["flat"] / RANKING_FILENAMES[("flat", "core")],
        "flat_between_rankings": method_dirs["flat"] / RANKING_FILENAMES[("flat", "stress_test")],
        "vector_core_rankings": method_dirs["vector"] / RANKING_FILENAMES[("vector", "core")],
        "vector_between_rankings": method_dirs["vector"] / RANKING_FILENAMES[("vector", "stress_test")],
        "graph_core_rankings": method_dirs["graph"] / RANKING_FILENAMES[("graph", "core")],
        "graph_between_rankings": method_dirs["graph"] / RANKING_FILENAMES[("graph", "stress_test")],
        "graph_core_expansion_audit": method_dirs["graph"] / GRAPH_AUDIT_FILENAMES["core"],
        "graph_between_expansion_audit": method_dirs["graph"] / GRAPH_AUDIT_FILENAMES["stress_test"],
    }
    manifest_base = {
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "metrics": {
            "ground_truth": "unique target_event_id",
            "mrr": "mean(1 / target_rank)",
            "success_at_1": "mean(target_rank <= 1)",
            "success_at_5": "mean(target_rank <= 5)",
            "core_overall_weighting": "query_weighted",
            "float_output_precision": 10,
        },
        "statistical_test": {
            "comparisons": ["flat_vs_vector", "vector_vs_graph"],
            "observation_unit": "per_user_mean_core_reciprocal_rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": "approx",
            "continuity_correction": False,
            "multiple_testing_correction": "Holm",
            "alpha": 0.05,
        },
        "runtime_versions": {
            "numpy": _package_version("numpy"),
            "os": platform.platform(),
            "pandas": _package_version("pandas"),
            "python": platform.python_version(),
            "scipy": _package_version("scipy"),
        },
    }
    _write_outputs_atomically(
        frames, output_dir / RUN_MANIFEST_FILENAME, manifest_base
    )

    overall_diagnostic = diagnostic_summary.loc[
        diagnostic_summary["query_type"].eq("overall")
    ].iloc[0]
    return EvaluationSummary(
        per_query_count=len(per_query),
        core_query_count=core_queries["query_id"].nunique(),
        between_query_count=between_queries["query_id"].nunique(),
        high_load_core_query_count=int(core_queries["_is_high_memory_load"].sum()),
        high_load_between_query_count=int(between_queries["_is_high_memory_load"].sum()),
        graph_improved_count=int(overall_diagnostic["graph_improved_vs_vector_count"]),
        graph_unchanged_count=int(overall_diagnostic["graph_unchanged_vs_vector_count"]),
        graph_worsened_count=int(overall_diagnostic["graph_worsened_vs_vector_count"]),
    )
