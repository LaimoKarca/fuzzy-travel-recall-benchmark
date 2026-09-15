"""Evaluate the fixed post-hoc all-event PPR personalization sensitivity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import EvaluationError, _load_events, _load_queries, _load_ranking
from src.evaluate_graph_extensions import (
    GraphExtensionEvaluationError,
    _read_json_manifest,
    _validate_targets,
)
from src.model_download import sha256_file
from src.retrieve_graph_ppr import (
    EVENT_SCORES_FILENAME as TOP5_SCORES_FILENAME,
    METHOD as TOP5_METHOD,
    RANKINGS_FILENAME as TOP5_RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME as TOP5_MANIFEST_FILENAME,
)
from src.retrieve_graph_ppr_all_event import (
    EVENT_SCORES_FILENAME as ALL_SCORES_FILENAME,
    METHOD as ALL_METHOD,
    RANKINGS_FILENAME as ALL_RANKINGS_FILENAME,
    RUN_MANIFEST_FILENAME as ALL_MANIFEST_FILENAME,
)


PER_QUERY_FILENAME = "tokyo_ppr_personalization_per_query.csv"
OVERALL_FILENAME = "tokyo_ppr_personalization_overall_metrics.csv"
BY_TYPE_FILENAME = "tokyo_ppr_personalization_metrics_by_query_type.csv"
DIAGNOSTICS_FILENAME = "tokyo_ppr_personalization_rank_diagnostics.csv"
REPORT_FILENAME = "tokyo_ppr_personalization_validation_report.md"
FROZEN_AUDIT_FILENAME = "tokyo_frozen_artifact_hash_audit.csv"
RUN_MANIFEST_FILENAME = "tokyo_ppr_personalization_run_manifest.json"

CONFIGS = (
    "e5_dense", "top5_raw_ppr", "top5_ppr_rrf",
    "all_event_raw_ppr", "all_event_ppr_rrf",
)
LABELS = {
    "e5_dense": "E5 Dense",
    "top5_raw_ppr": "Top-5 Raw PPR",
    "top5_ppr_rrf": "Top-5 PPR-RRF",
    "all_event_raw_ppr": "All-event Raw PPR",
    "all_event_ppr_rrf": "All-event PPR-RRF",
}


class PPRPersonalizationEvaluationError(RuntimeError):
    """Raised when the personalization sensitivity cannot be evaluated safely."""


@dataclass(frozen=True)
class PPRPersonalizationEvaluationSummary:
    query_count: int
    all_event_mrr: float
    top5_mrr: float
    dense_mrr: float
    output_dir: Path


def _load_scores(path: Path, label: str, queries: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        raise PPRPersonalizationEvaluationError(f"could not read {label}: {exc}") from exc
    required = {"query_id", "user_id", "event_id", "dense_rank", "ppr_rank", "final_rank"}
    if not required.issubset(frame.columns):
        raise PPRPersonalizationEvaluationError(f"{label} has an invalid schema")
    if frame.duplicated(["query_id", "event_id"]).any():
        raise PPRPersonalizationEvaluationError(f"{label} contains duplicate query-event rows")
    for column in ("dense_rank", "ppr_rank", "final_rank"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        except ValueError as exc:
            raise PPRPersonalizationEvaluationError(f"{label} has invalid {column}") from exc
    expected_users = queries.set_index("query_id")["user_id"].to_dict()
    if set(frame["query_id"]) != set(queries["query_id"]):
        raise PPRPersonalizationEvaluationError(f"{label} query coverage differs")
    event_users = events.set_index("event_id")["user_id"].to_dict()
    for query_id, group in frame.groupby("query_id", sort=False):
        user_id = expected_users[query_id]
        if set(group["user_id"]) != {user_id}:
            raise PPRPersonalizationEvaluationError(f"{label} crosses users for {query_id}")
        expected = {event_id for event_id, owner in event_users.items() if owner == user_id}
        if set(group["event_id"]) != expected:
            raise PPRPersonalizationEvaluationError(f"{label} candidate set differs for {query_id}")
        for column in ("dense_rank", "ppr_rank", "final_rank"):
            if sorted(group[column].tolist()) != list(range(1, len(group) + 1)):
                raise PPRPersonalizationEvaluationError(f"{label} has discontinuous {column}")
    return frame


def _per_query(
    queries: pd.DataFrame,
    dense: pd.DataFrame,
    top5_scores: pd.DataFrame,
    top5_rrf: pd.DataFrame,
    all_scores: pd.DataFrame,
    all_rrf: pd.DataFrame,
) -> pd.DataFrame:
    ranking_groups = {
        "e5_dense": {k: g.sort_values("rank") for k, g in dense.groupby("query_id")},
        "top5_ppr_rrf": {k: g.sort_values("rank") for k, g in top5_rrf.groupby("query_id")},
        "all_event_ppr_rrf": {k: g.sort_values("rank") for k, g in all_rrf.groupby("query_id")},
    }
    score_groups = {
        "top5_raw_ppr": {k: g for k, g in top5_scores.groupby("query_id")},
        "all_event_raw_ppr": {k: g for k, g in all_scores.groupby("query_id")},
    }
    rows = []
    for query in queries.itertuples(index=False):
        target = query.target_event_id
        ranks: dict[str, int] = {}
        top1: dict[str, str] = {}
        for config, groups in ranking_groups.items():
            group = groups[query.query_id]
            match = group.loc[group["retrieved_event_id"].eq(target)]
            if len(match) != 1:
                raise PPRPersonalizationEvaluationError(f"missing target in {config}: {query.query_id}")
            ranks[config] = int(match.iloc[0]["rank"])
            top1[config] = str(group.iloc[0]["retrieved_event_id"])
        for config, groups in score_groups.items():
            group = groups[query.query_id]
            match = group.loc[group["event_id"].eq(target)]
            if len(match) != 1:
                raise PPRPersonalizationEvaluationError(f"missing target in {config}: {query.query_id}")
            ranks[config] = int(match.iloc[0]["ppr_rank"])
            top1[config] = str(group.sort_values(["ppr_rank", "event_id"]).iloc[0]["event_id"])
        rows.append({
            "query_id": query.query_id, "user_id": query.user_id,
            "query_type": query.query_type, "target_event_id": target,
            "target_is_dense_top5": ranks["e5_dense"] <= 5,
            **{f"{config}_target_rank": ranks[config] for config in CONFIGS},
            **{f"{config}_reciprocal_rank": 1.0 / ranks[config] for config in CONFIGS},
            **{f"{config}_top1_event_id": top1[config] for config in CONFIGS},
            "all_rrf_rank_gain_vs_dense": ranks["e5_dense"] - ranks["all_event_ppr_rrf"],
            "all_rrf_rank_gain_vs_top5_rrf": ranks["top5_ppr_rrf"] - ranks["all_event_ppr_rrf"],
            "all_raw_rank_gain_vs_top5_raw": ranks["top5_raw_ppr"] - ranks["all_event_raw_ppr"],
        })
    return pd.DataFrame(rows).sort_values(["user_id", "query_id"], kind="stable").reset_index(drop=True)


def _metrics(per_query: pd.DataFrame, by_type: bool) -> pd.DataFrame:
    rows = []
    groups = per_query.groupby("query_type", sort=True) if by_type else [("overall", per_query)]
    for query_type, group in groups:
        for config in CONFIGS:
            ranks = group[f"{config}_target_rank"]
            rows.append({
                "query_type": query_type, "configuration": config,
                "configuration_label": LABELS[config], "analysis_role": (
                    "post_hoc_sensitivity" if config.startswith("all_event") else "frozen_reference"
                ),
                "query_count": len(group), "user_count": group["user_id"].nunique(),
                "mrr": (1.0 / ranks).mean(),
                "success_at_1": ranks.le(1).mean(), "success_at_5": ranks.le(5).mean(),
            })
    return pd.DataFrame(rows)


def _outcome(gain: pd.Series) -> tuple[int, int, int]:
    return int(gain.gt(0).sum()), int(gain.eq(0).sum()), int(gain.lt(0).sum())


def _diagnostics(per_query: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("all_event_ppr_rrf_vs_e5_dense", "all_rrf_rank_gain_vs_dense", "e5_dense", "all_event_ppr_rrf"),
        ("all_event_ppr_rrf_vs_top5_ppr_rrf", "all_rrf_rank_gain_vs_top5_rrf", "top5_ppr_rrf", "all_event_ppr_rrf"),
        ("all_event_raw_ppr_vs_top5_raw_ppr", "all_raw_rank_gain_vs_top5_raw", "top5_raw_ppr", "all_event_raw_ppr"),
    )
    strata = (
        ("overall", per_query),
        ("target_in_dense_top5", per_query.loc[per_query["target_is_dense_top5"]]),
        ("target_outside_dense_top5", per_query.loc[~per_query["target_is_dense_top5"]]),
    )
    rows = []
    for stratum, group in strata:
        for comparison, gain_column, baseline, candidate in comparisons:
            improved, unchanged, worsened = _outcome(group[gain_column])
            rows.append({
                "stratum": stratum, "comparison": comparison, "query_count": len(group),
                "improved": improved, "unchanged": unchanged, "worsened": worsened,
                "mean_rank_gain": group[gain_column].mean(),
                "mean_baseline_target_rank": group[f"{baseline}_target_rank"].mean(),
                "mean_candidate_target_rank": group[f"{candidate}_target_rank"].mean(),
                "baseline_mrr": group[f"{baseline}_reciprocal_rank"].mean(),
                "candidate_mrr": group[f"{candidate}_reciprocal_rank"].mean(),
                "baseline_success_at_1": group[f"{baseline}_target_rank"].le(1).mean(),
                "candidate_success_at_1": group[f"{candidate}_target_rank"].le(1).mean(),
                "baseline_success_at_5": group[f"{baseline}_target_rank"].le(5).mean(),
                "candidate_success_at_5": group[f"{candidate}_target_rank"].le(5).mean(),
                "top1_identical_count": group[f"{baseline}_top1_event_id"].eq(
                    group[f"{candidate}_top1_event_id"]
                ).sum(),
                "top1_changed_count": group[f"{baseline}_top1_event_id"].ne(
                    group[f"{candidate}_top1_event_id"]
                ).sum(),
            })
    return pd.DataFrame(rows)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_files(project_root: Path, excluded: tuple[Path, ...]) -> list[Path]:
    workspace = project_root.parent
    roots = [
        project_root / "paper", workspace / "研究規劃.md", workspace / "研究實作.md",
        project_root / "data" / "raw", project_root / "data" / "outputs",
        *(project_root / "data" / "prepared" / name for name in (
            "stage_01_cohorts", "stage_02_canonical_events", "stage_03_queries", "stage_04_retrieval"
        )),
        project_root / "data" / "prepared" / "phase_02_graph_extensions" / "ppr",
        project_root / "data" / "prepared" / "phase_02_graph_extensions" / "structured_next",
        project_root / "data" / "prepared" / "phase_02_graph_extensions" / "evaluation",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted({p.resolve() for p in files if not any(x.resolve() in p.resolve().parents for x in excluded)})


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        try:
            relative = path.resolve().relative_to(root.resolve().parent)
            return f"../{relative.as_posix()}"
        except ValueError:
            return path.name


def _render_report(overall: pd.DataFrame, by_type: pd.DataFrame, diagnostics: pd.DataFrame) -> str:
    lookup = overall.set_index("configuration")
    dense = float(lookup.loc["e5_dense", "mrr"])
    top5 = float(lookup.loc["top5_ppr_rrf", "mrr"])
    all_event = float(lookup.loc["all_event_ppr_rrf", "mrr"])
    top5_raw = float(lookup.loc["top5_raw_ppr", "mrr"])
    all_raw = float(lookup.loc["all_event_raw_ppr", "mrr"])
    inside = diagnostics.loc[
        diagnostics["stratum"].eq("target_in_dense_top5")
        & diagnostics["comparison"].eq("all_event_ppr_rrf_vs_e5_dense")
    ].iloc[0]
    outside = diagnostics.loc[
        diagnostics["stratum"].eq("target_outside_dense_top5")
        & diagnostics["comparison"].eq("all_event_ppr_rrf_vs_e5_dense")
    ].iloc[0]
    outside_top5_hits = (
        0 if int(outside["query_count"]) == 0
        else int(round(outside["candidate_success_at_5"] * outside["query_count"]))
    )
    if all_event >= dense - 1e-10:
        conclusion = "All-event personalization reached or exceeded Dense; the PPR result is highly sensitive to personalization design."
    elif all_event > top5 + 1e-10:
        conclusion = "All-event personalization recovered part of the degradation, but static diffusion still did not exceed Dense."
    else:
        conclusion = "All-event personalization did not improve the Top-5-personalized PPR-RRF result; Top-5 truncation is not the main explanation."

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
                for c in columns
            ) + " |")
        return "\n".join(lines)

    return "\n".join([
        "# PPR Personalization Sensitivity Validation Report", "",
        "This is a fixed post-hoc sensitivity analysis. It is not a fifth primary retrieval configuration.", "",
        "## Overall metrics", "",
        table(overall, ["configuration_label", "query_count", "mrr", "success_at_1", "success_at_5"]), "",
        "## MRR by query type", "",
        table(by_type, ["query_type", "configuration_label", "mrr"]), "",
        "## Rank diagnostics", "",
        table(diagnostics, [
            "stratum", "comparison", "query_count", "improved", "unchanged",
            "worsened", "baseline_mrr", "candidate_mrr", "baseline_success_at_1",
            "candidate_success_at_1", "baseline_success_at_5",
            "candidate_success_at_5", "top1_changed_count",
        ]), "",
        "## Dense Top-5 strata", "",
        f"- Targets already in Dense Top-5: {int(inside['query_count'])}; All-event PPR-RRF produced {int(inside['improved'])} improvements, {int(inside['unchanged'])} unchanged ranks, and {int(inside['worsened'])} worsenings. Its S@5 in this stratum was {inside['candidate_success_at_5']:.4f}.",
        f"- Targets outside Dense Top-5: {int(outside['query_count'])}; All-event PPR-RRF produced {int(outside['improved'])} improvements, {int(outside['unchanged'])} unchanged ranks, and {int(outside['worsened'])} worsenings. It moved {outside_top5_hits} targets into the Top-5.",
        f"- Raw PPR MRR changed from {top5_raw:.4f} with Top-5 personalization to {all_raw:.4f} with all-event personalization; equal-weight RRF raised the all-event result to {all_event:.4f}.", "",
        "## Pre-specified interpretation", "", conclusion, "",
        "Raw PPR and RRF are reported separately. No new significance test was added, and no PPR, graph, or fusion parameter was tuned after observing the result.", "",
    ])


def _write(
    output_dir: Path, frames: dict[str, pd.DataFrame], report: str, manifest: dict[str, Any]
) -> None:
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
        raise PPRPersonalizationEvaluationError(f"could not write sensitivity outputs: {exc}") from exc
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)


def evaluate_ppr_personalization_sensitivity(
    events_path: str | Path,
    queries_path: str | Path,
    vector_dir: str | Path,
    top5_ppr_dir: str | Path,
    all_event_ppr_dir: str | Path,
    output_dir: str | Path,
) -> PPRPersonalizationEvaluationSummary:
    paths = tuple(map(Path, (
        events_path, queries_path, vector_dir, top5_ppr_dir, all_event_ppr_dir, output_dir
    )))
    events_path, queries_path, vector_dir, top5_ppr_dir, all_event_ppr_dir, output_dir = paths
    project_root = Path(__file__).resolve().parents[1]
    frozen = _frozen_files(project_root, (all_event_ppr_dir, output_dir))
    frozen_before = {path: _hash(path) for path in frozen}
    try:
        events = _load_events(events_path)
        queries = _load_queries(queries_path, "core", "Core queries")
        _validate_targets(events, queries)
        top5_manifest, top5_manifest_path = _read_json_manifest(
            top5_ppr_dir, TOP5_MANIFEST_FILENAME, "method", TOP5_METHOD
        )
        all_manifest, all_manifest_path = _read_json_manifest(
            all_event_ppr_dir, ALL_MANIFEST_FILENAME, "method", ALL_METHOD
        )
        if all_manifest.get("analysis_role") != "post_hoc_personalization_sensitivity":
            raise PPRPersonalizationEvaluationError("all-event manifest has an invalid analysis role")
        for manifest in (top5_manifest, all_manifest):
            if manifest.get("dense_seed_model", {}).get("id") != "intfloat/multilingual-e5-small":
                raise PPRPersonalizationEvaluationError("PPR manifest does not identify frozen E5")
        dense = _load_ranking(
            vector_dir / "tokyo_vector_core_rankings.csv", "vector", "core", queries, events
        )
        top5_rrf = _load_ranking(
            top5_ppr_dir / TOP5_RANKINGS_FILENAME, TOP5_METHOD, "core", queries, events
        )
        all_rrf = _load_ranking(
            all_event_ppr_dir / ALL_RANKINGS_FILENAME, ALL_METHOD, "core", queries, events
        )
        top5_scores = _load_scores(top5_ppr_dir / TOP5_SCORES_FILENAME, "Top-5 PPR scores", queries, events)
        all_scores = _load_scores(all_event_ppr_dir / ALL_SCORES_FILENAME, "all-event PPR scores", queries, events)
    except (
        EvaluationError, GraphExtensionEvaluationError,
        PPRPersonalizationEvaluationError, OSError, ValueError,
    ) as exc:
        raise PPRPersonalizationEvaluationError(str(exc)) from exc

    for query_id in queries["query_id"]:
        candidates = []
        for frame, id_column in (
            (dense, "retrieved_event_id"), (top5_rrf, "retrieved_event_id"),
            (all_rrf, "retrieved_event_id"), (top5_scores, "event_id"), (all_scores, "event_id"),
        ):
            candidates.append(frozenset(frame.loc[frame["query_id"].eq(query_id), id_column]))
        if any(value != candidates[0] for value in candidates[1:]):
            raise PPRPersonalizationEvaluationError(f"candidate sets differ for {query_id}")

    per_query = _per_query(queries, dense, top5_scores, top5_rrf, all_scores, all_rrf)
    overall = _metrics(per_query, False)
    by_type = _metrics(per_query, True)
    diagnostics = _diagnostics(per_query)
    report = _render_report(overall, by_type, diagnostics)

    frozen_after = {path: _hash(path) for path in frozen}
    if frozen_before != frozen_after:
        changed = [str(path) for path in frozen if frozen_before[path] != frozen_after[path]]
        raise PPRPersonalizationEvaluationError("frozen artifacts changed: " + ", ".join(changed[:5]))
    frozen_audit = pd.DataFrame([
        {
            "path": _portable(path, project_root), "sha256_before": frozen_before[path],
            "sha256_after": frozen_after[path], "status": "unchanged_during_evaluation",
        }
        for path in frozen
    ])
    inputs = {
        "canonical_events": events_path, "core_queries": queries_path,
        "dense_rankings": vector_dir / "tokyo_vector_core_rankings.csv",
        "top5_ppr_manifest": top5_manifest_path, "all_event_ppr_manifest": all_manifest_path,
    }
    manifest = {
        "evaluation": "ppr_personalization_sensitivity",
        "analysis_role": "post_hoc_sensitivity",
        "inputs": {
            key: {"path": _portable(path, project_root), "sha256": sha256_file(path)}
            for key, path in sorted(inputs.items())
        },
        "configurations": list(CONFIGS),
        "parameters": {
            "metrics": ["mrr", "success_at_1", "success_at_5"],
            "inferential_statistics": "none",
            "strata": ["overall", "target_in_dense_top5", "target_outside_dense_top5"],
        },
        "runtime_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "python": sys.version.split()[0],
        },
    }
    _write(output_dir, {
        PER_QUERY_FILENAME: per_query, OVERALL_FILENAME: overall,
        BY_TYPE_FILENAME: by_type, DIAGNOSTICS_FILENAME: diagnostics,
        FROZEN_AUDIT_FILENAME: frozen_audit,
    }, report, manifest)
    lookup = overall.set_index("configuration")
    return PPRPersonalizationEvaluationSummary(
        len(per_query), float(lookup.loc["all_event_ppr_rrf", "mrr"]),
        float(lookup.loc["top5_ppr_rrf", "mrr"]),
        float(lookup.loc["e5_dense", "mrr"]), output_dir,
    )
