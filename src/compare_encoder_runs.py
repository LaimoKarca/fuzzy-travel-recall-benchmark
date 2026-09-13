"""Compare frozen MiniLM and candidate E5 evaluation runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.evaluate import (
    BY_QUERY_TYPE_FILENAME,
    GRAPH_DIAGNOSTICS_FILENAME,
    PER_QUERY_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from src.model_specs import E5_SMALL_SPEC, MINILM_SPEC
from src.model_download import sha256_file


CORE_METRICS_FILENAME = "tokyo_minilm_vs_e5_core_metrics.csv"
GRAPH_DIAGNOSTICS_COMPARISON_FILENAME = (
    "tokyo_minilm_vs_e5_graph_diagnostics.csv"
)
VALIDATION_REPORT_FILENAME = "tokyo_minilm_vs_e5_validation_report.md"

CORE_METRIC_COLUMNS = (
    "configuration",
    "mrr",
    "success_at_1",
    "success_at_5",
    "semantic_mrr",
    "temporal_spatial_mrr",
    "relational_after_mrr",
    "mrr_delta_vs_minilm",
    "success_at_1_delta_vs_minilm",
    "success_at_5_delta_vs_minilm",
    "semantic_mrr_delta_vs_minilm",
    "temporal_spatial_mrr_delta_vs_minilm",
    "relational_after_mrr_delta_vs_minilm",
)
GRAPH_COMPARISON_COLUMNS = (
    "encoder",
    "core_query_count",
    "graph_improved_vs_dense",
    "graph_unchanged_vs_dense",
    "graph_worsened_vs_dense",
    "target_first_reached_as_self",
    "target_first_reached_as_previous",
    "target_first_reached_as_next",
    "target_not_expanded",
    "structural_first_reach_count",
    "structural_first_reach_rate",
    "vector_graph_top1_identical_count",
)


class EncoderComparisonError(RuntimeError):
    """Raised when encoder evaluation runs cannot be safely compared."""


@dataclass(frozen=True)
class EncoderComparisonSummary:
    core_query_count: int
    minilm_dense_mrr: float
    e5_dense_mrr: float
    minilm_graph_mrr: float
    e5_graph_mrr: float
    minilm_top1_identical_count: int
    e5_top1_identical_count: int
    output_dir: Path


@dataclass(frozen=True)
class _EvaluationRun:
    label: str
    directory: Path
    manifest: dict[str, Any]
    manifest_path: Path
    encoder_id: str
    per_query: pd.DataFrame
    diagnostics: pd.DataFrame
    vector_core_rankings: pd.DataFrame
    graph_core_rankings: pd.DataFrame


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise EncoderComparisonError(f"{label} does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise EncoderComparisonError(f"could not parse {label}: {exc}") from exc
    if frame.empty:
        raise EncoderComparisonError(f"{label} contains no rows: {path}")
    return frame


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EncoderComparisonError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EncoderComparisonError(f"could not parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EncoderComparisonError(f"{label} is not a JSON object")
    return value


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise EncoderComparisonError(
            f"{label} is missing required column(s): {', '.join(missing)}"
        )


def _validate_evaluation_outputs(
    directory: Path, manifest: dict[str, Any]
) -> None:
    required = {
        PER_QUERY_FILENAME,
        BY_QUERY_TYPE_FILENAME,
        GRAPH_DIAGNOSTICS_FILENAME,
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise EncoderComparisonError("evaluation manifest has no outputs mapping")
    missing = required - set(outputs)
    if missing:
        raise EncoderComparisonError(
            "evaluation manifest is missing output hash(es): " + ", ".join(sorted(missing))
        )
    for filename in sorted(required):
        path = directory / filename
        if not path.is_file() or sha256_file(path) != outputs[filename]:
            raise EncoderComparisonError(f"evaluation output hash mismatch: {path}")


def _validated_input_path(
    manifest: dict[str, Any], key: str, label: str
) -> Path:
    inputs = manifest.get("inputs")
    entry = inputs.get(key) if isinstance(inputs, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise EncoderComparisonError(f"evaluation manifest lacks {key}")
    path = Path(entry["path"])
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise EncoderComparisonError(f"{label} input hash mismatch: {path}")
    return path


def _load_run(directory: Path, label: str, expected_encoder: str) -> _EvaluationRun:
    manifest_path = directory / RUN_MANIFEST_FILENAME
    manifest = _read_json(manifest_path, f"{label} evaluation manifest")
    _validate_evaluation_outputs(directory, manifest)

    vector_manifest_path = _validated_input_path(
        manifest, "vector_run_manifest", f"{label} Vector manifest"
    )
    graph_manifest_path = _validated_input_path(
        manifest, "graph_run_manifest", f"{label} Graph manifest"
    )
    vector_manifest = _read_json(vector_manifest_path, f"{label} Vector manifest")
    graph_manifest = _read_json(graph_manifest_path, f"{label} Graph manifest")
    model = vector_manifest.get("model")
    encoder_id = model.get("id") if isinstance(model, dict) else None
    if encoder_id != expected_encoder:
        raise EncoderComparisonError(
            f"{label} encoder is {encoder_id!r}, expected {expected_encoder!r}"
        )
    graph_inputs = graph_manifest.get("inputs")
    graph_vector = (
        graph_inputs.get("vector_run_manifest")
        if isinstance(graph_inputs, dict)
        else None
    )
    if not isinstance(graph_vector, dict) or graph_vector.get("sha256") != sha256_file(
        vector_manifest_path
    ):
        raise EncoderComparisonError(
            f"{label} Graph did not consume the corresponding Vector manifest"
        )
    dense_seed_model = graph_manifest.get("dense_seed_model")
    if dense_seed_model is not None and (
        not isinstance(dense_seed_model, dict)
        or dense_seed_model.get("id") != expected_encoder
    ):
        raise EncoderComparisonError(f"{label} Graph seed encoder metadata mismatch")

    per_query = _read_csv(directory / PER_QUERY_FILENAME, f"{label} per-query metrics")
    diagnostics = _read_csv(
        directory / GRAPH_DIAGNOSTICS_FILENAME, f"{label} Graph diagnostics"
    )
    vector_rankings_path = _validated_input_path(
        manifest, "vector_core_rankings", f"{label} Vector Core rankings"
    )
    graph_rankings_path = _validated_input_path(
        manifest, "graph_core_rankings", f"{label} Graph Core rankings"
    )
    vector_core_rankings = _read_csv(
        vector_rankings_path, f"{label} Vector Core rankings"
    )
    graph_core_rankings = _read_csv(
        graph_rankings_path, f"{label} Graph Core rankings"
    )
    return _EvaluationRun(
        label,
        directory,
        manifest,
        manifest_path,
        str(encoder_id),
        per_query,
        diagnostics,
        vector_core_rankings,
        graph_core_rankings,
    )


def _query_ground_truth(run: _EvaluationRun) -> pd.DataFrame:
    columns = (
        "query_id",
        "user_id",
        "benchmark_tier",
        "query_type",
        "target_event_id",
    )
    _require_columns(run.per_query, columns + ("method",), f"{run.label} per-query metrics")
    values = run.per_query.loc[:, columns].drop_duplicates()
    if values["query_id"].duplicated().any():
        raise EncoderComparisonError(f"{run.label} has inconsistent query Ground Truth")
    return values.sort_values("query_id", kind="stable").reset_index(drop=True)


def _numeric_metrics(run: _EvaluationRun) -> pd.DataFrame:
    required = (
        "query_id",
        "benchmark_tier",
        "query_type",
        "method",
        "reciprocal_rank",
        "success_at_1",
        "success_at_5",
    )
    _require_columns(run.per_query, required, f"{run.label} per-query metrics")
    core = run.per_query.loc[run.per_query["benchmark_tier"].eq("core")].copy()
    if set(core["method"]) != {"flat", "vector", "graph"}:
        raise EncoderComparisonError(f"{run.label} has unexpected methods")
    if core.groupby("method")["query_id"].nunique().to_dict() != {
        "flat": 320,
        "graph": 320,
        "vector": 320,
    }:
        raise EncoderComparisonError(f"{run.label} does not contain 320 Core queries per method")
    try:
        core["reciprocal_rank"] = pd.to_numeric(
            core["reciprocal_rank"], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise EncoderComparisonError(
            f"{run.label} has invalid reciprocal-rank values"
        ) from exc
    for column in ("success_at_1", "success_at_5"):
        normalized = core[column].str.strip().str.casefold()
        if not normalized.isin({"true", "false", "1", "0"}).all():
            raise EncoderComparisonError(
                f"{run.label} has invalid {column} boolean values"
            )
        core[column] = normalized.isin({"true", "1"}).astype(int)
    return core


def _metric_row(run: _EvaluationRun, method: str, configuration: str) -> dict[str, object]:
    core = _numeric_metrics(run)
    selected = core.loc[core["method"].eq(method)]
    by_type = selected.groupby("query_type", sort=False)["reciprocal_rank"].mean()
    required_types = {"semantic", "temporal_spatial", "relational_after"}
    if set(by_type.index) != required_types:
        raise EncoderComparisonError(f"{run.label} has unexpected Core query types")
    return {
        "configuration": configuration,
        "mrr": selected["reciprocal_rank"].mean(),
        "success_at_1": selected["success_at_1"].mean(),
        "success_at_5": selected["success_at_5"].mean(),
        "semantic_mrr": by_type["semantic"],
        "temporal_spatial_mrr": by_type["temporal_spatial"],
        "relational_after_mrr": by_type["relational_after"],
    }


def _graph_row(run: _EvaluationRun, encoder: str) -> dict[str, object]:
    required = (
        "query_id",
        "benchmark_tier",
        "graph_outcome_vs_vector",
        "target_in_expansion",
        "target_first_discovery_relation",
    )
    _require_columns(run.diagnostics, required, f"{run.label} Graph diagnostics")
    core = run.diagnostics.loc[run.diagnostics["benchmark_tier"].eq("core")].copy()
    if len(core) != 320 or core["query_id"].nunique() != 320:
        raise EncoderComparisonError(f"{run.label} Graph diagnostics must have 320 Core rows")
    outcomes = core["graph_outcome_vs_vector"].value_counts().to_dict()
    if set(outcomes) - {"improved", "unchanged", "worsened"}:
        raise EncoderComparisonError(f"{run.label} has invalid Graph outcome values")
    relations = core["target_first_discovery_relation"].value_counts().to_dict()

    top1_frames = []
    for method, frame in (
        ("vector", run.vector_core_rankings),
        ("graph", run.graph_core_rankings),
    ):
        _require_columns(
            frame,
            ("query_id", "method", "rank", "retrieved_event_id"),
            f"{run.label} {method} rankings",
        )
        if set(frame["method"]) != {method}:
            raise EncoderComparisonError(f"{run.label} {method} rankings method mismatch")
        try:
            ranks = pd.to_numeric(frame["rank"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise EncoderComparisonError(f"{run.label} has invalid rank values") from exc
        top1 = frame.loc[ranks.eq(1), ["query_id", "retrieved_event_id"]].copy()
        if len(top1) != 320 or top1["query_id"].nunique() != 320:
            raise EncoderComparisonError(f"{run.label} {method} must have 320 Top-1 rows")
        top1 = top1.rename(columns={"retrieved_event_id": method})
        top1_frames.append(top1)
    merged = top1_frames[0].merge(top1_frames[1], on="query_id", validate="one_to_one")

    target_in_expansion = core["target_in_expansion"].str.casefold().isin({"true", "1"})
    previous_count = int(relations.get("PREVIOUS", 0))
    next_count = int(relations.get("NEXT", 0))
    structural_count = previous_count + next_count
    return {
        "encoder": encoder,
        "core_query_count": 320,
        "graph_improved_vs_dense": int(outcomes.get("improved", 0)),
        "graph_unchanged_vs_dense": int(outcomes.get("unchanged", 0)),
        "graph_worsened_vs_dense": int(outcomes.get("worsened", 0)),
        "target_first_reached_as_self": int(relations.get("SELF", 0)),
        "target_first_reached_as_previous": previous_count,
        "target_first_reached_as_next": next_count,
        "target_not_expanded": int((~target_in_expansion).sum()),
        "structural_first_reach_count": structural_count,
        "structural_first_reach_rate": structural_count / 320,
        "vector_graph_top1_identical_count": int(merged["vector"].eq(merged["graph"]).sum()),
    }


def _delta_row(minilm: dict[str, object], e5: dict[str, object]) -> dict[str, object]:
    return {
        column: (
            "E5_minus_MiniLM"
            if column == "encoder"
            else (
                float(e5[column]) - float(minilm[column])
                if column == "structural_first_reach_rate"
                else int(e5[column]) - int(minilm[column])
            )
        )
        for column in GRAPH_COMPARISON_COLUMNS
    }


def _add_metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = ((0, 1), (2, 3))
    metric_names = (
        "mrr",
        "success_at_1",
        "success_at_5",
        "semantic_mrr",
        "temporal_spatial_mrr",
        "relational_after_mrr",
    )
    for metric in metric_names:
        delta_column = f"{metric}_delta_vs_minilm"
        metrics[delta_column] = 0.0
        for minilm_index, e5_index in pairs:
            metrics.loc[e5_index, delta_column] = (
                float(metrics.loc[e5_index, metric])
                - float(metrics.loc[minilm_index, metric])
            )
    return metrics.loc[:, CORE_METRIC_COLUMNS]


def _format_metric(value: object) -> str:
    return f"{float(value):.4f}"


def _markdown_report(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    dense_old = metrics.loc[metrics["configuration"].eq("MiniLM Dense")].iloc[0]
    dense_new = metrics.loc[metrics["configuration"].eq("E5 Dense")].iloc[0]
    graph_old = metrics.loc[metrics["configuration"].eq("MiniLM Graph")].iloc[0]
    graph_new = metrics.loc[metrics["configuration"].eq("E5 Graph")].iloc[0]
    lines = [
        "# MiniLM vs E5 驗收報告",
        "",
        "本報告只比較 Main Core 320 題；E5 尚未取代正式 Stage 5。",
        "",
        "## Core metrics",
        "",
        "| Configuration | MRR | S@1 | S@5 | Semantic MRR | Temporal-Spatial MRR | Relational-After MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            "| {configuration} | {mrr} | {s1} | {s5} | {semantic} | {temporal} | {relational} |".format(
                configuration=row["configuration"],
                mrr=_format_metric(row["mrr"]),
                s1=_format_metric(row["success_at_1"]),
                s5=_format_metric(row["success_at_5"]),
                semantic=_format_metric(row["semantic_mrr"]),
                temporal=_format_metric(row["temporal_spatial_mrr"]),
                relational=_format_metric(row["relational_after_mrr"]),
            )
        )
    lines.extend(
        [
            "",
            "## Graph diagnosis",
            "",
            "| Encoder | Improved | Unchanged | Worsened | SELF | PREVIOUS | NEXT | Not expanded | Structural reach | Top-1 identical |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.to_dict("records"):
        lines.append(
            "| {encoder} | {graph_improved_vs_dense} | {graph_unchanged_vs_dense} | {graph_worsened_vs_dense} | {target_first_reached_as_self} | {target_first_reached_as_previous} | {target_first_reached_as_next} | {target_not_expanded} | {structural_first_reach_count} ({structural_first_reach_rate:.2%}) | {vector_graph_top1_identical_count} |".format(**row)
        )
    dense_delta = float(dense_new["mrr"]) - float(dense_old["mrr"])
    graph_delta = float(graph_new["mrr"]) - float(graph_old["mrr"])
    e5_graph_vs_dense = float(graph_new["mrr"]) - float(dense_new["mrr"])
    old_graph_diagnostic = diagnostics.loc[diagnostics["encoder"].eq("MiniLM")].iloc[0]
    e5_graph_diagnostic = diagnostics.loc[diagnostics["encoder"].eq("E5")].iloc[0]
    old_structural_reach = int(old_graph_diagnostic["target_first_reached_as_previous"]) + int(
        old_graph_diagnostic["target_first_reached_as_next"]
    )
    e5_structural_reach = int(e5_graph_diagnostic["target_first_reached_as_previous"]) + int(
        e5_graph_diagnostic["target_first_reached_as_next"]
    )
    lines.extend(
        [
            "",
            "## Validation answers",
            "",
            f"1. E5 Dense 相對 MiniLM Dense 的 Core MRR 絕對差為 {dense_delta:+.4f}；因此 E5 並未改善本 benchmark 的整體 Dense MRR。各 query type 與 Success 指標仍須分開閱讀。",
            f"2. E5 Graph 相對 MiniLM Graph 的 Core MRR 絕對差為 {graph_delta:+.4f}，相對同次 E5 Dense 則為 {e5_graph_vs_dense:+.4f}。PREVIOUS／NEXT 首次找到 target 的 Core 題數由 {old_structural_reach} 增至 {e5_structural_reach}，但 E5 Vector／Graph 的 Top-1 仍有 {int(e5_graph_diagnostic['vector_graph_top1_identical_count'])}/320 相同。這表示 E5 seeds 提高了結構可達性，但固定 Graph reranking 仍未轉化為整體 MRR 或 rank-1 優勢。",
            "",
            "這些差異是受控重跑的描述結果，不單獨構成因果證明。",
            "",
        ]
    )
    return "\n".join(lines)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_outputs_atomically(
    output_dir: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame, report: str
) -> None:
    destinations = {
        output_dir / CORE_METRICS_FILENAME: ("csv", metrics),
        output_dir / GRAPH_DIAGNOSTICS_COMPARISON_FILENAME: ("csv", diagnostics),
        output_dir / VALIDATION_REPORT_FILENAME: ("text", report),
    }
    temporary: dict[Path, Path] = {}
    try:
        for destination, (kind, value) in destinations.items():
            temp = _temporary_path(destination)
            temporary[destination] = temp
            if kind == "csv":
                value.to_csv(temp, index=False, encoding="utf-8", float_format="%.10f")
            else:
                temp.write_text(str(value), encoding="utf-8")
        for destination, temp in temporary.items():
            temp.replace(destination)
    except OSError as exc:
        raise EncoderComparisonError(f"could not write comparison outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def compare_encoder_runs(
    minilm_evaluation_dir: str | Path,
    e5_evaluation_dir: str | Path,
    output_dir: str | Path,
) -> EncoderComparisonSummary:
    """Validate and compare frozen MiniLM and candidate E5 Core evaluations."""

    minilm = _load_run(
        Path(minilm_evaluation_dir), "MiniLM", MINILM_SPEC.model_id
    )
    e5 = _load_run(Path(e5_evaluation_dir), "E5", E5_SMALL_SPEC.model_id)
    if not _query_ground_truth(minilm).equals(_query_ground_truth(e5)):
        raise EncoderComparisonError(
            "MiniLM and E5 evaluations do not contain identical queries and Ground Truth"
        )

    metrics = pd.DataFrame(
        [
            _metric_row(minilm, "vector", "MiniLM Dense"),
            _metric_row(e5, "vector", "E5 Dense"),
            _metric_row(minilm, "graph", "MiniLM Graph"),
            _metric_row(e5, "graph", "E5 Graph"),
        ],
    )
    metrics = _add_metric_deltas(metrics)
    minilm_graph = _graph_row(minilm, "MiniLM")
    e5_graph = _graph_row(e5, "E5")
    diagnostics = pd.DataFrame(
        [minilm_graph, e5_graph, _delta_row(minilm_graph, e5_graph)],
        columns=GRAPH_COMPARISON_COLUMNS,
    )
    report = _markdown_report(metrics, diagnostics)
    output_dir = Path(output_dir)
    _write_outputs_atomically(output_dir, metrics, diagnostics, report)
    return EncoderComparisonSummary(
        core_query_count=320,
        minilm_dense_mrr=float(metrics.iloc[0]["mrr"]),
        e5_dense_mrr=float(metrics.iloc[1]["mrr"]),
        minilm_graph_mrr=float(metrics.iloc[2]["mrr"]),
        e5_graph_mrr=float(metrics.iloc[3]["mrr"]),
        minilm_top1_identical_count=int(minilm_graph["vector_graph_top1_identical_count"]),
        e5_top1_identical_count=int(e5_graph["vector_graph_top1_identical_count"]),
        output_dir=output_dir,
    )
