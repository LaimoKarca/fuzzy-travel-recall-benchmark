"""Evaluate the fixed E5 symmetric and NEXT-only Graph configurations."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate import (
    EvaluationError,
    _load_events,
    _load_graph_audit,
    _load_queries,
    _load_ranking,
)
from src.model_download import sha256_file
from src.model_specs import E5_SMALL_SPEC
from src.retrieve_graph import (
    CORE_AUDIT_FILENAME as SYMMETRIC_AUDIT_FILENAME,
    CORE_RANKINGS_FILENAME as SYMMETRIC_RANKINGS_FILENAME,
    EXPANSION_DIRECTION,
    RUN_MANIFEST_FILENAME as SYMMETRIC_MANIFEST_FILENAME,
)
from src.retrieve_graph_next_only import (
    AUDIT_FILENAME as NEXT_ONLY_AUDIT_FILENAME,
    NEXT_ONLY_EXPANSION_DIRECTION,
    RANKINGS_FILENAME as NEXT_ONLY_RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME as NEXT_ONLY_MANIFEST_FILENAME,
)
from src.retrieve_vector import (
    CORE_RANKINGS_FILENAME as VECTOR_RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME as VECTOR_MANIFEST_FILENAME,
)


PER_QUERY_FILENAME = "tokyo_e5_graph_directional_per_query.csv"
METRICS_FILENAME = "tokyo_e5_graph_directional_metrics.csv"
DIAGNOSTICS_FILENAME = "tokyo_e5_graph_directional_diagnostics.csv"
REPORT_FILENAME = "tokyo_e5_graph_directional_validation_report.md"
RUN_MANIFEST_FILENAME = "tokyo_e5_graph_directional_run_manifest.json"
QUERY_TYPE = "relational_after"
CONFIGURATIONS = ("E5 Dense", "E5 Graph Symmetric", "E5 Graph NEXT-only")


class GraphDirectionEvaluationError(RuntimeError):
    """Raised when the directional ablation cannot be evaluated safely."""


@dataclass(frozen=True)
class GraphDirectionEvaluationSummary:
    query_count: int
    high_load_query_count: int
    dense_mrr: float
    symmetric_mrr: float
    next_only_mrr: float
    next_only_improved_vs_symmetric: int
    next_only_unchanged_vs_symmetric: int
    next_only_worsened_vs_symmetric: int
    output_dir: Path


def _read_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise GraphDirectionEvaluationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphDirectionEvaluationError(f"could not parse {label}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("outputs"), dict):
        raise GraphDirectionEvaluationError(f"{label} has an invalid structure")
    for filename, expected_hash in sorted(value["outputs"].items()):
        if Path(filename).name != filename:
            raise GraphDirectionEvaluationError(f"{label} contains an unsafe filename")
        artifact = path.parent / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise GraphDirectionEvaluationError(f"artifact hash mismatch: {artifact}")
    return value


def _manifest_input(manifest: dict[str, Any], key: str, label: str) -> Path:
    inputs = manifest.get("inputs")
    entry = inputs.get(key) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise GraphDirectionEvaluationError(f"{label} lacks input {key}")
    path = Path(entry["path"])
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise GraphDirectionEvaluationError(f"{label} input hash mismatch: {path}")
    return path


def _validate_provenance(
    vector_dir: Path, symmetric_dir: Path, next_only_dir: Path
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    vector_path = vector_dir / VECTOR_MANIFEST_FILENAME
    symmetric_path = symmetric_dir / SYMMETRIC_MANIFEST_FILENAME
    next_only_path = next_only_dir / NEXT_ONLY_MANIFEST_FILENAME
    vector = _read_manifest(vector_path, "E5 Vector manifest")
    symmetric = _read_manifest(symmetric_path, "symmetric Graph manifest")
    next_only = _read_manifest(next_only_path, "NEXT-only Graph manifest")

    model = vector.get("model")
    if not isinstance(model, dict) or (
        model.get("id") != E5_SMALL_SPEC.model_id
        or model.get("revision") != E5_SMALL_SPEC.revision
    ):
        raise GraphDirectionEvaluationError("Vector run is not the pinned E5 model")
    vector_hash = sha256_file(vector_path)
    for manifest, label, direction in (
        (symmetric, "symmetric Graph", EXPANSION_DIRECTION),
        (next_only, "NEXT-only Graph", NEXT_ONLY_EXPANSION_DIRECTION),
    ):
        seed_model = manifest.get("dense_seed_model")
        parameters = manifest.get("parameters")
        if not isinstance(seed_model, dict) or seed_model.get("id") != E5_SMALL_SPEC.model_id:
            raise GraphDirectionEvaluationError(f"{label} did not use E5 seeds")
        if not isinstance(parameters, dict) or parameters.get("expansion_direction") != direction:
            raise GraphDirectionEvaluationError(f"{label} direction metadata mismatch")
        consumed = _manifest_input(manifest, "vector_run_manifest", label)
        if consumed.resolve() != vector_path.resolve() or sha256_file(consumed) != vector_hash:
            raise GraphDirectionEvaluationError(
                f"{label} did not consume the selected E5 Vector manifest"
            )
    if next_only.get("ablation") != "relational_after_next_only":
        raise GraphDirectionEvaluationError("NEXT-only manifest has the wrong ablation label")

    symmetric_events = _manifest_input(symmetric, "canonical_events", "symmetric Graph")
    next_events = _manifest_input(next_only, "canonical_events", "NEXT-only Graph")
    if (
        symmetric_events.resolve() != next_events.resolve()
        or sha256_file(symmetric_events) != sha256_file(next_events)
    ):
        raise GraphDirectionEvaluationError("Graph configurations used different events")
    return symmetric_events, vector, symmetric, next_only


def _target_rank(rankings: pd.DataFrame, queries: pd.DataFrame) -> pd.Series:
    target = queries.set_index("query_id")["target_event_id"]
    selected = rankings.loc[
        rankings.apply(
            lambda row: row["retrieved_event_id"] == target[row["query_id"]], axis=1
        ),
        ["query_id", "rank"],
    ]
    if len(selected) != len(queries) or selected["query_id"].duplicated().any():
        raise GraphDirectionEvaluationError(
            "a ranking does not contain exactly one target per query"
        )
    return selected.set_index("query_id")["rank"].astype(int)


def _top1(rankings: pd.DataFrame) -> pd.Series:
    selected = rankings.loc[rankings["rank"].eq(1), ["query_id", "retrieved_event_id"]]
    if selected["query_id"].duplicated().any():
        raise GraphDirectionEvaluationError("a ranking contains duplicate Top-1 rows")
    return selected.set_index("query_id")["retrieved_event_id"]


def _first_reach(audit: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    targets = queries.set_index("query_id")["target_event_id"]
    first = audit.loc[audit["_is_first_discovery"]].copy()
    matched = first.loc[
        first.apply(
            lambda row: row["expanded_event_id"] == targets[row["query_id"]], axis=1
        )
    ].set_index("query_id")
    rows = []
    for query_id in queries["query_id"]:
        if query_id in matched.index:
            row = matched.loc[query_id]
            rows.append(
                {
                    "query_id": query_id,
                    "relation": row["relation"],
                    "seed_rank": int(row["seed_rank"]),
                    "expansion_rank": int(row["expansion_rank"]),
                }
            )
        else:
            rows.append(
                {"query_id": query_id, "relation": "", "seed_rank": "", "expansion_rank": ""}
            )
    return pd.DataFrame(rows).set_index("query_id")


def _outcome(left: pd.Series, right: pd.Series) -> pd.Series:
    difference = left - right
    return difference.map(lambda value: "improved" if value > 0 else "worsened" if value < 0 else "unchanged")


def _metric_row(configuration: str, ranks: pd.Series, dense_rr: pd.Series) -> dict[str, object]:
    rr = 1.0 / ranks
    return {
        "configuration": configuration,
        "query_count": len(ranks),
        "mrr": rr.mean(),
        "success_at_1": ranks.le(1).mean(),
        "success_at_5": ranks.le(5).mean(),
        "mrr_delta_vs_dense": (rr - dense_rr).mean(),
    }


def _diagnostic_row(
    configuration: str,
    dense_rank: pd.Series,
    graph_rank: pd.Series,
    symmetric_rank: pd.Series,
    dense_top1: pd.Series,
    graph_top1: pd.Series,
    reach: pd.DataFrame,
) -> dict[str, object]:
    versus_dense = _outcome(dense_rank, graph_rank)
    versus_symmetric = _outcome(symmetric_rank, graph_rank)
    relations = reach["relation"].value_counts().to_dict()
    structural = int(relations.get("PREVIOUS", 0) + relations.get("NEXT", 0))
    return {
        "configuration": configuration,
        "query_count": len(graph_rank),
        "improved_vs_dense": int(versus_dense.eq("improved").sum()),
        "unchanged_vs_dense": int(versus_dense.eq("unchanged").sum()),
        "worsened_vs_dense": int(versus_dense.eq("worsened").sum()),
        "improved_vs_symmetric": int(versus_symmetric.eq("improved").sum()),
        "unchanged_vs_symmetric": int(versus_symmetric.eq("unchanged").sum()),
        "worsened_vs_symmetric": int(versus_symmetric.eq("worsened").sum()),
        "target_first_reached_as_self": int(relations.get("SELF", 0)),
        "target_first_reached_as_previous": int(relations.get("PREVIOUS", 0)),
        "target_first_reached_as_next": int(relations.get("NEXT", 0)),
        "target_not_expanded": int(relations.get("", 0)),
        "structural_first_reach_count": structural,
        "structural_first_reach_rate": structural / len(graph_rank),
        "dense_graph_top1_identical_count": int(dense_top1.eq(graph_top1).sum()),
        "dense_graph_top1_changed_count": int(dense_top1.ne(graph_top1).sum()),
        "mean_rr_delta_vs_dense": ((1.0 / graph_rank) - (1.0 / dense_rank)).mean(),
        "mean_rr_delta_vs_symmetric": ((1.0 / graph_rank) - (1.0 / symmetric_rank)).mean(),
    }


def _render_report(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    metric = metrics.set_index("configuration")
    diagnostic = diagnostics.set_index("configuration")
    symmetric = diagnostic.loc["E5 Graph Symmetric"]
    next_only = diagnostic.loc["E5 Graph NEXT-only"]
    query_count = int(metric.loc["E5 Dense", "query_count"])
    delta = float(metric.loc["E5 Graph NEXT-only", "mrr"] - metric.loc["E5 Graph Symmetric", "mrr"])
    if delta > 0:
        interpretation = "NEXT-only 的 MRR 較高，結果支持對稱鄰居升位是部分干擾來源。"
    elif delta < 0:
        interpretation = "NEXT-only 的 MRR 較低，PREVIOUS 鄰居不能被視為純雜訊。"
    else:
        interpretation = "兩種方向的 MRR 相同，方向過濾未改變整體表現。"
    lines = [
        "# E5 Graph 方向性消融驗收報告",
        "",
        f"本報告只分析 {query_count} 題 relational-after；所有設定除展開方向外均固定。",
        "",
        "## Metrics",
        "",
        "| Configuration | MRR | S@1 | S@5 | ΔMRR vs Dense |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            f"| {row['configuration']} | {row['mrr']:.4f} | {row['success_at_1']:.4f} | {row['success_at_5']:.4f} | {row['mrr_delta_vs_dense']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Graph diagnosis",
            "",
            "| Configuration | Improved | Unchanged | Worsened | SELF | PREVIOUS | NEXT | Not expanded | Structural reach | Top-1 changed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.to_dict("records"):
        lines.append(
            f"| {row['configuration']} | {row['improved_vs_dense']} | {row['unchanged_vs_dense']} | {row['worsened_vs_dense']} | {row['target_first_reached_as_self']} | {row['target_first_reached_as_previous']} | {row['target_first_reached_as_next']} | {row['target_not_expanded']} | {row['structural_first_reach_count']} ({row['structural_first_reach_rate']:.2%}) | {row['dense_graph_top1_changed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Validation answers",
            "",
            f"1. NEXT-only 相對 symmetric 的 MRR 差為 {delta:+.4f}。{interpretation}",
            f"2. NEXT-only 相對 symmetric：改善 {int(next_only['improved_vs_symmetric'])}、不變 {int(next_only['unchanged_vs_symmetric'])}、惡化 {int(next_only['worsened_vs_symmetric'])} 題。",
            f"3. Symmetric 的 structural first reach 為 {int(symmetric['structural_first_reach_count'])}/{query_count}；NEXT-only 為 {int(next_only['structural_first_reach_count'])}/{query_count}。可達性是否轉成排序收益須與 MRR 與逐題 delta 一起解讀。",
            f"4. NEXT-only 改變 Dense Top-1 的題數為 {int(next_only['dense_graph_top1_changed_count'])}/{query_count}；本結果僅適用於 E5 Top-5、一跳、等權 RRF(k=60)。",
            "",
        ]
    )
    return "\n".join(lines)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(name)


def _write_outputs(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    report: str,
    manifest_base: dict[str, Any],
) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for filename, frame in frames.items():
            destination = output_dir / filename
            temp = _temporary_path(destination)
            temporary[destination] = temp
            frame.to_csv(temp, index=False, encoding="utf-8", float_format="%.10f")
        report_destination = output_dir / REPORT_FILENAME
        report_temp = _temporary_path(report_destination)
        temporary[report_destination] = report_temp
        report_temp.write_text(report, encoding="utf-8")
        manifest = dict(manifest_base)
        manifest["outputs"] = {
            destination.name: sha256_file(temp)
            for destination, temp in sorted(temporary.items(), key=lambda item: item[0].name)
        }
        manifest_destination = output_dir / RUN_MANIFEST_FILENAME
        manifest_temp = _temporary_path(manifest_destination)
        temporary[manifest_destination] = manifest_temp
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for destination, temp in temporary.items():
            temp.replace(destination)
    except (OSError, ValueError) as exc:
        raise GraphDirectionEvaluationError(f"could not write outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def evaluate_graph_direction(
    queries_path: str | Path,
    vector_dir: str | Path,
    symmetric_graph_dir: str | Path,
    next_only_graph_dir: str | Path,
    output_dir: str | Path,
) -> GraphDirectionEvaluationSummary:
    """Compare E5 Dense, symmetric Graph, and NEXT-only Graph."""

    queries_path = Path(queries_path)
    vector_dir = Path(vector_dir)
    symmetric_graph_dir = Path(symmetric_graph_dir)
    next_only_graph_dir = Path(next_only_graph_dir)
    output_dir = Path(output_dir)
    events_path, _, _, _ = _validate_provenance(
        vector_dir, symmetric_graph_dir, next_only_graph_dir
    )
    try:
        events = _load_events(events_path)
        all_queries = _load_queries(queries_path, "core", "Core queries")
        relational = all_queries.loc[all_queries["query_type"].eq(QUERY_TYPE)].copy()
        if relational.empty:
            raise GraphDirectionEvaluationError("Core queries contain no relational_after rows")
        event_lookup = events.set_index("event_id")
        for _, query in relational.iterrows():
            if query["target_event_id"] not in event_lookup.index:
                raise GraphDirectionEvaluationError("query references a missing target event")
            target = event_lookup.loc[query["target_event_id"]]
            if target["user_id"] != query["user_id"] or target["venue_id"] != query["target_venue_id"]:
                raise GraphDirectionEvaluationError("query Ground Truth does not match its event")

        vector_all = _load_ranking(
            vector_dir / VECTOR_RANKINGS_FILENAME, "vector", "core", all_queries, events
        )
        symmetric_all = _load_ranking(
            symmetric_graph_dir / SYMMETRIC_RANKINGS_FILENAME,
            "graph",
            "core",
            all_queries,
            events,
        )
        query_ids = set(relational["query_id"])
        vector = vector_all.loc[vector_all["query_id"].isin(query_ids)].copy()
        symmetric = symmetric_all.loc[symmetric_all["query_id"].isin(query_ids)].copy()
        next_only = _load_ranking(
            next_only_graph_dir / NEXT_ONLY_RANKINGS_FILENAME,
            "graph",
            "core",
            relational,
            events,
        )
        for query_id in query_ids:
            candidates = [
                set(frame.loc[frame["query_id"].eq(query_id), "retrieved_event_id"])
                for frame in (vector, symmetric, next_only)
            ]
            if not (candidates[0] == candidates[1] == candidates[2]):
                raise GraphDirectionEvaluationError(
                    f"configurations have different candidates for {query_id}"
                )

        symmetric_audit_all = _load_graph_audit(
            symmetric_graph_dir / SYMMETRIC_AUDIT_FILENAME,
            "core",
            all_queries,
            events,
        )
        symmetric_audit = symmetric_audit_all.loc[
            symmetric_audit_all["query_id"].isin(query_ids)
        ].copy()
        next_audit = _load_graph_audit(
            next_only_graph_dir / NEXT_ONLY_AUDIT_FILENAME,
            "core",
            relational,
            events,
        )
    except EvaluationError as exc:
        raise GraphDirectionEvaluationError(str(exc)) from exc
    if set(next_audit["relation"]) - {"SELF", "NEXT"}:
        raise GraphDirectionEvaluationError("NEXT-only audit contains PREVIOUS traversal")

    query_order = relational.sort_values(["user_id", "query_id"], kind="stable")["query_id"]
    dense_rank = _target_rank(vector, relational).reindex(query_order)
    symmetric_rank = _target_rank(symmetric, relational).reindex(query_order)
    next_rank = _target_rank(next_only, relational).reindex(query_order)
    dense_top1 = _top1(vector).reindex(query_order)
    symmetric_top1 = _top1(symmetric).reindex(query_order)
    next_top1 = _top1(next_only).reindex(query_order)
    symmetric_reach = _first_reach(symmetric_audit, relational).reindex(query_order)
    next_reach = _first_reach(next_audit, relational).reindex(query_order)
    dense_rr = 1.0 / dense_rank

    query_lookup = relational.set_index("query_id")
    per_query_rows = []
    for query_id in query_order:
        query = query_lookup.loc[query_id]
        per_query_rows.append(
            {
                "query_id": query_id,
                "user_id": query["user_id"],
                "query_text": query["query_text"],
                "target_event_id": query["target_event_id"],
                "is_high_memory_load": bool(query["_is_high_memory_load"]),
                "dense_target_rank": dense_rank[query_id],
                "symmetric_target_rank": symmetric_rank[query_id],
                "next_only_target_rank": next_rank[query_id],
                "dense_reciprocal_rank": dense_rr[query_id],
                "symmetric_reciprocal_rank": 1.0 / symmetric_rank[query_id],
                "next_only_reciprocal_rank": 1.0 / next_rank[query_id],
                "symmetric_rr_delta_vs_dense": 1.0 / symmetric_rank[query_id] - dense_rr[query_id],
                "next_only_rr_delta_vs_dense": 1.0 / next_rank[query_id] - dense_rr[query_id],
                "next_only_rr_delta_vs_symmetric": 1.0 / next_rank[query_id] - 1.0 / symmetric_rank[query_id],
                "symmetric_outcome_vs_dense": _outcome(dense_rank, symmetric_rank)[query_id],
                "next_only_outcome_vs_dense": _outcome(dense_rank, next_rank)[query_id],
                "next_only_outcome_vs_symmetric": _outcome(symmetric_rank, next_rank)[query_id],
                "symmetric_first_reach_relation": symmetric_reach.at[query_id, "relation"],
                "next_only_first_reach_relation": next_reach.at[query_id, "relation"],
                "dense_top1_event_id": dense_top1[query_id],
                "symmetric_top1_event_id": symmetric_top1[query_id],
                "next_only_top1_event_id": next_top1[query_id],
                "symmetric_top1_identical_to_dense": symmetric_top1[query_id] == dense_top1[query_id],
                "next_only_top1_identical_to_dense": next_top1[query_id] == dense_top1[query_id],
            }
        )
    per_query = pd.DataFrame(per_query_rows)
    metrics = pd.DataFrame(
        [
            _metric_row("E5 Dense", dense_rank, dense_rr),
            _metric_row("E5 Graph Symmetric", symmetric_rank, dense_rr),
            _metric_row("E5 Graph NEXT-only", next_rank, dense_rr),
        ]
    )
    diagnostics = pd.DataFrame(
        [
            _diagnostic_row(
                "E5 Graph Symmetric", dense_rank, symmetric_rank, symmetric_rank,
                dense_top1, symmetric_top1, symmetric_reach,
            ),
            _diagnostic_row(
                "E5 Graph NEXT-only", dense_rank, next_rank, symmetric_rank,
                dense_top1, next_top1, next_reach,
            ),
        ]
    )
    if int(diagnostics.loc[diagnostics["configuration"].eq("E5 Graph NEXT-only"), "target_first_reached_as_previous"].iloc[0]) != 0:
        raise GraphDirectionEvaluationError("NEXT-only PREVIOUS first reach must be zero")

    report = _render_report(metrics, diagnostics)
    input_paths = {
        "queries": queries_path,
        "e5_vector_manifest": vector_dir / VECTOR_MANIFEST_FILENAME,
        "symmetric_graph_manifest": symmetric_graph_dir / SYMMETRIC_MANIFEST_FILENAME,
        "next_only_graph_manifest": next_only_graph_dir / NEXT_ONLY_MANIFEST_FILENAME,
        "e5_vector_rankings": vector_dir / VECTOR_RANKINGS_FILENAME,
        "symmetric_graph_rankings": symmetric_graph_dir / SYMMETRIC_RANKINGS_FILENAME,
        "next_only_graph_rankings": next_only_graph_dir / NEXT_ONLY_RANKINGS_FILENAME,
        "symmetric_graph_audit": symmetric_graph_dir / SYMMETRIC_AUDIT_FILENAME,
        "next_only_graph_audit": next_only_graph_dir / NEXT_ONLY_AUDIT_FILENAME,
    }
    manifest_base = {
        "ablation": "e5_relational_after_graph_direction",
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in sorted(input_paths.items())
        },
        "parameters": {
            "dense_seed_count": 5,
            "expansion_hops": 1,
            "query_type": QUERY_TYPE,
            "rrf_k": 60,
            "rrf_weights": "equal",
        },
        "runtime_versions": {
            "os": platform.platform(),
            "pandas": importlib.metadata.version("pandas"),
            "python": platform.python_version(),
        },
    }
    _write_outputs(
        output_dir,
        {
            PER_QUERY_FILENAME: per_query,
            METRICS_FILENAME: metrics,
            DIAGNOSTICS_FILENAME: diagnostics,
        },
        report,
        manifest_base,
    )

    next_vs_symmetric = _outcome(symmetric_rank, next_rank)
    return GraphDirectionEvaluationSummary(
        query_count=len(relational),
        high_load_query_count=int(relational["_is_high_memory_load"].sum()),
        dense_mrr=float(metrics.iloc[0]["mrr"]),
        symmetric_mrr=float(metrics.iloc[1]["mrr"]),
        next_only_mrr=float(metrics.iloc[2]["mrr"]),
        next_only_improved_vs_symmetric=int(next_vs_symmetric.eq("improved").sum()),
        next_only_unchanged_vs_symmetric=int(next_vs_symmetric.eq("unchanged").sum()),
        next_only_worsened_vs_symmetric=int(next_vs_symmetric.eq("worsened").sum()),
        output_dir=output_dir,
    )
