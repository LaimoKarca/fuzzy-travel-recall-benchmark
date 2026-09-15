"""Generate a structured-cue oracle diagnostic for relational-after queries."""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.build_graph import GraphBuildError, build_travel_graph
from src.model_download import sha256_file


QUERY_TYPE = "relational_after"
CANDIDATES_FILENAME = "tokyo_structured_next_relational_after_candidates.csv"
SUMMARY_FILENAME = "tokyo_structured_next_relational_after_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_structured_next_run_manifest.json"

QUERY_COLUMNS = (
    "query_id", "user_id", "benchmark_tier", "query_type",
    "cue_previous_place", "cue_category", "is_high_memory_load",
)
CANDIDATE_COLUMNS = (
    "query_id", "user_id", "benchmark_tier", "query_type",
    "cue_previous_place", "cue_category", "anchor_event_count",
    "anchor_event_ids", "next_successor_count", "next_successor_event_ids",
    "category_matched_candidate_count", "category_matched_event_ids",
    "uniquely_selected_event_id", "candidate_status", "is_high_memory_load",
)
SUMMARY_COLUMNS = (
    "benchmark_tier", "query_type", "query_count", "users",
    "queries_with_previous_place_anchor", "total_previous_place_anchors",
    "queries_with_next_successor", "total_next_successors",
    "queries_with_category_match", "total_category_matched_candidates",
    "uniquely_selected_query_count", "ambiguous_query_count",
    "missing_anchor_query_count", "missing_next_query_count",
    "no_category_match_query_count",
)


class StructuredNextDiagnosticError(RuntimeError):
    """Raised when the structured NEXT diagnostic cannot be produced safely."""


@dataclass(frozen=True)
class StructuredNextDiagnosticSummary:
    query_count: int
    unique_candidate_count: int
    output_dir: Path


def normalize_cue(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _read_events(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise StructuredNextDiagnosticError(f"events CSV does not exist: {path}")
    try:
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        built = build_travel_graph(source)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, GraphBuildError) as exc:
        raise StructuredNextDiagnosticError(f"could not validate events: {exc}") from exc
    return built.events


def _read_queries(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise StructuredNextDiagnosticError(f"queries CSV does not exist: {path}")
    try:
        queries = pd.read_csv(
            path, dtype=str, keep_default_na=False, usecols=list(QUERY_COLUMNS)
        )
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
        raise StructuredNextDiagnosticError(f"could not parse allowed query cues: {exc}") from exc
    if queries.empty:
        raise StructuredNextDiagnosticError("queries CSV contains no rows")
    for column in QUERY_COLUMNS:
        if column not in queries:
            raise StructuredNextDiagnosticError(f"queries CSV is missing {column}")
    for column in ("query_id", "user_id", "benchmark_tier", "query_type"):
        if queries[column].str.strip().eq("").any():
            raise StructuredNextDiagnosticError(f"queries CSV contains blank {column}")
    if queries["query_id"].duplicated().any():
        raise StructuredNextDiagnosticError("queries CSV contains duplicate query_id values")
    queries = queries.loc[queries["query_type"].eq(QUERY_TYPE)].copy()
    if queries.empty or set(queries["benchmark_tier"]) != {"core"}:
        raise StructuredNextDiagnosticError("no valid Core relational-after queries found")
    for column in ("cue_previous_place", "cue_category"):
        if queries[column].map(normalize_cue).eq("").any():
            raise StructuredNextDiagnosticError(f"relational-after query contains blank {column}")
    normalized_bool = queries["is_high_memory_load"].str.strip().str.casefold()
    if not normalized_bool.isin({"true", "false"}).all():
        raise StructuredNextDiagnosticError("queries contain invalid is_high_memory_load values")
    queries["_is_high_memory_load"] = normalized_bool.eq("true")
    return queries.sort_values(["user_id", "query_id"], kind="stable").reset_index(drop=True)


def _generate(events: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    event_lookup = events.set_index("event_id")
    events_by_user = {
        user_id: group.copy() for user_id, group in events.groupby("user_id", sort=False)
    }
    rows = []
    for _, query in queries.iterrows():
        user_events = events_by_user.get(query["user_id"])
        if user_events is None:
            raise StructuredNextDiagnosticError(
                f"query user is absent from events: {query['user_id']}"
            )
        anchors = user_events.loc[
            user_events["name"].map(normalize_cue).eq(normalize_cue(query["cue_previous_place"]))
        ].sort_values("event_id", kind="stable")
        anchor_ids = anchors["event_id"].tolist()
        successors = sorted({value for value in anchors["next_event_id"] if value})
        for successor in successors:
            if successor not in event_lookup.index:
                raise StructuredNextDiagnosticError(f"NEXT successor is missing: {successor}")
            if event_lookup.loc[successor, "user_id"] != query["user_id"]:
                raise StructuredNextDiagnosticError("NEXT oracle crosses a user boundary")
        matches = sorted(
            successor for successor in successors
            if normalize_cue(event_lookup.loc[successor, "venue_category"])
            == normalize_cue(query["cue_category"])
        )
        if not anchor_ids:
            status = "missing_anchor"
        elif not successors:
            status = "missing_next"
        elif not matches:
            status = "no_category_match"
        elif len(matches) == 1:
            status = "unique_candidate"
        else:
            status = "ambiguous_candidates"
        rows.append({
            "query_id": query["query_id"],
            "user_id": query["user_id"],
            "benchmark_tier": query["benchmark_tier"],
            "query_type": query["query_type"],
            "cue_previous_place": query["cue_previous_place"],
            "cue_category": query["cue_category"],
            "anchor_event_count": len(anchor_ids),
            "anchor_event_ids": json.dumps(anchor_ids, ensure_ascii=False, separators=(",", ":")),
            "next_successor_count": len(successors),
            "next_successor_event_ids": json.dumps(successors, ensure_ascii=False, separators=(",", ":")),
            "category_matched_candidate_count": len(matches),
            "category_matched_event_ids": json.dumps(matches, ensure_ascii=False, separators=(",", ":")),
            "uniquely_selected_event_id": matches[0] if len(matches) == 1 else "",
            "candidate_status": status,
            "is_high_memory_load": bool(query["_is_high_memory_load"]),
        })
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _summarize(candidates: pd.DataFrame) -> pd.DataFrame:
    statuses = candidates["candidate_status"]
    return pd.DataFrame([{
        "benchmark_tier": "core",
        "query_type": QUERY_TYPE,
        "query_count": len(candidates),
        "users": candidates["user_id"].nunique(),
        "queries_with_previous_place_anchor": int(candidates["anchor_event_count"].gt(0).sum()),
        "total_previous_place_anchors": int(candidates["anchor_event_count"].sum()),
        "queries_with_next_successor": int(candidates["next_successor_count"].gt(0).sum()),
        "total_next_successors": int(candidates["next_successor_count"].sum()),
        "queries_with_category_match": int(candidates["category_matched_candidate_count"].gt(0).sum()),
        "total_category_matched_candidates": int(candidates["category_matched_candidate_count"].sum()),
        "uniquely_selected_query_count": int(statuses.eq("unique_candidate").sum()),
        "ambiguous_query_count": int(statuses.eq("ambiguous_candidates").sum()),
        "missing_anchor_query_count": int(statuses.eq("missing_anchor").sum()),
        "missing_next_query_count": int(statuses.eq("missing_next").sum()),
        "no_category_match_query_count": int(statuses.eq("no_category_match").sum()),
    }], columns=SUMMARY_COLUMNS)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write(output_dir: Path, candidates: pd.DataFrame, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temps: dict[str, Path] = {}
    try:
        for filename, frame in ((CANDIDATES_FILENAME, candidates), (SUMMARY_FILENAME, summary)):
            descriptor, name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=output_dir)
            os.close(descriptor)
            temp = Path(name)
            frame.to_csv(temp, index=False, encoding="utf-8")
            temps[filename] = temp
        manifest = dict(manifest)
        manifest["outputs"] = {name: sha256_file(path) for name, path in sorted(temps.items())}
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
        raise StructuredNextDiagnosticError(f"could not write Structured NEXT outputs: {exc}") from exc
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        if "manifest_temp" in locals():
            manifest_temp.unlink(missing_ok=True)


def diagnose_structured_next(
    events_path: str | Path,
    queries_path: str | Path,
    output_dir: str | Path,
) -> StructuredNextDiagnosticSummary:
    """Generate candidates without reading Ground Truth target columns."""

    events_path, queries_path, output_dir = Path(events_path), Path(queries_path), Path(output_dir)
    project_root = Path(__file__).resolve().parents[1]
    events = _read_events(events_path)
    queries = _read_queries(queries_path)
    candidates = _generate(events, queries)
    summary = _summarize(candidates)
    manifest = {
        "diagnostic": "structured_next_structured_cue_oracle",
        "inputs": {
            "canonical_events": {"path": _portable(events_path, project_root), "sha256": sha256_file(events_path)},
            "core_queries": {
                "path": _portable(queries_path, project_root),
                "sha256": sha256_file(queries_path),
                "allowed_columns": list(QUERY_COLUMNS),
                "forbidden_candidate_generation_columns": ["target_event_id", "cue_previous_event_id"],
            },
        },
        "parameters": {
            "query_type": QUERY_TYPE,
            "anchor": "normalized cue_previous_place equals Event.name",
            "traversal": "outgoing NEXT only",
            "filter": "normalized cue_category equals successor venue_category",
            "normalization": "Unicode NFKC, case folding, whitespace collapse",
        },
    }
    _write(output_dir, candidates, summary, manifest)
    return StructuredNextDiagnosticSummary(
        query_count=len(candidates),
        unique_candidate_count=int(candidates["candidate_status"].eq("unique_candidate").sum()),
        output_dir=output_dir,
    )
