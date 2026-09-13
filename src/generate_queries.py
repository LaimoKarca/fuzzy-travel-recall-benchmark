"""Generate deterministic incomplete-cue recall queries from canonical events."""

from __future__ import annotations

import calendar
import hashlib
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


CORE_MAIN_FILENAME = "tokyo_recall_queries_main.csv"
CORE_HIGH_LOAD_FILENAME = "tokyo_recall_queries_high_memory_load.csv"
BETWEEN_MAIN_FILENAME = "tokyo_relational_between_queries_main.csv"
BETWEEN_HIGH_LOAD_FILENAME = (
    "tokyo_relational_between_queries_high_memory_load.csv"
)
AUDIT_FILENAME = "tokyo_query_generation_audit.csv"
SUMMARY_FILENAME = "tokyo_query_generation_summary.csv"

QUERY_LANGUAGE = "dataset_native_mixed"
SELECTION_SEED = "icct-pacific-2027-os07-stage3-v1"

REQUIRED_COLUMNS = (
    "user_id",
    "trail_id",
    "event_id",
    "venue_id",
    "timestamp",
    "name",
    "venue_category",
    "venue_city",
    "previous_event_id",
    "previous_place",
    "next_event_id",
    "next_place",
    "is_high_memory_load",
    "is_target_eligible",
)

QUERY_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "query_language",
    "query_text",
    "target_event_id",
    "target_venue_id",
    "target_name",
    "target_trail_id",
    "target_timestamp",
    "target_category",
    "target_city",
    "cue_year",
    "cue_month",
    "cue_city",
    "cue_category",
    "cue_previous_event_id",
    "cue_previous_place",
    "cue_next_event_id",
    "cue_next_place",
    "is_high_memory_load",
)

AUDIT_COLUMNS = (
    "user_id",
    "benchmark_tier",
    "query_type",
    "eligible_target_count",
    "complete_cue_count",
    "unique_event_candidate_count",
    "leakage_safe_candidate_count",
    "status",
    "skip_reason",
    "selected_event_id",
    "selected_venue_id",
    "same_user_target_venue_query_count",
    "target_venue_reused_across_query_types",
    "is_high_memory_load",
)

SUMMARY_COLUMNS = (
    "benchmark_tier",
    "cohort",
    "query_type",
    "user_count",
    "generated_query_count",
    "skipped_user_count",
)


class QueryGenerationError(RuntimeError):
    """Raised when benchmark query generation cannot complete safely."""


@dataclass(frozen=True)
class QueryGenerationSummary:
    core_main: dict[str, int]
    core_high_load: dict[str, int]
    stress_main: dict[str, int]
    stress_high_load: dict[str, int]
    audit_rows: int


@dataclass(frozen=True)
class QuerySpec:
    benchmark_tier: str
    query_type: str
    cue_key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    render: Callable[[pd.Series], str]


def _render_semantic(row: pd.Series) -> str:
    return (
        f"Which {row['venue_category']} did I visit in "
        f"{row['venue_city']}?"
    )


def _render_temporal_spatial(row: pd.Series) -> str:
    timestamp = row["_parsed_timestamp"]
    month_name = calendar.month_name[int(timestamp.month)]
    return (
        f"Which {row['venue_category']} did I visit in {row['venue_city']} "
        f"in {month_name} {timestamp.year}?"
    )


def _render_relational_after(row: pd.Series) -> str:
    return (
        f"Which {row['venue_category']} did I visit after "
        f"{row['previous_place']}?"
    )


def _render_relational_between(row: pd.Series) -> str:
    return (
        f"Which {row['venue_category']} did I visit between "
        f"{row['previous_place']} and {row['next_place']}?"
    )


QUERY_SPECS = (
    QuerySpec(
        "core",
        "semantic",
        ("_category_key", "_city_key"),
        ("venue_category", "venue_city"),
        _render_semantic,
    ),
    QuerySpec(
        "core",
        "temporal_spatial",
        ("_year_key", "_month_key", "_city_key", "_category_key"),
        ("venue_category", "venue_city"),
        _render_temporal_spatial,
    ),
    QuerySpec(
        "core",
        "relational_after",
        ("_previous_place_key", "_category_key"),
        ("previous_event_id", "previous_place", "venue_category"),
        _render_relational_after,
    ),
    QuerySpec(
        "stress_test",
        "relational_between",
        ("_previous_place_key", "_next_place_key", "_category_key"),
        (
            "previous_event_id",
            "previous_place",
            "next_event_id",
            "next_place",
            "venue_category",
        ),
        _render_relational_between,
    ),
)


def _normalize(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def _parse_boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    normalized = frame[column].str.strip().str.casefold()
    invalid = ~normalized.isin(("true", "false"))
    if invalid.any():
        values = sorted(frame.loc[invalid, column].unique())
        raise QueryGenerationError(
            f"input CSV contains invalid {column} value(s): {values}"
        )
    return normalized.eq("true")


def _validate_event_links(frame: pd.DataFrame) -> None:
    lookup = frame.set_index("event_id", drop=False)

    for direction, relation_column, place_column in (
        ("previous", "previous_event_id", "previous_place"),
        ("next", "next_event_id", "next_place"),
    ):
        linked = frame[relation_column].str.strip().ne("")
        referenced_ids = frame.loc[linked, relation_column]
        missing = ~referenced_ids.isin(lookup.index)
        if missing.any():
            bad_id = referenced_ids.loc[missing].iloc[0]
            raise QueryGenerationError(
                f"input CSV contains unknown {relation_column}: {bad_id}"
            )

        if not linked.any():
            continue

        source = frame.loc[linked, ["user_id", "trail_id", place_column]].copy()
        referenced = lookup.loc[referenced_ids, ["user_id", "trail_id", "name"]]
        referenced.index = source.index

        wrong_scope = (
            source["user_id"].ne(referenced["user_id"])
            | source["trail_id"].ne(referenced["trail_id"])
        )
        if wrong_scope.any():
            raise QueryGenerationError(
                f"input CSV contains {direction} event link(s) across user or trail"
            )

        place_mismatch = source[place_column].map(_normalize).ne(
            referenced["name"].map(_normalize)
        )
        if place_mismatch.any():
            raise QueryGenerationError(
                f"input CSV contains {place_column} value(s) that do not match "
                f"the referenced event name"
            )

    previous_lookup = lookup["next_event_id"]
    has_previous = frame["previous_event_id"].str.strip().ne("")
    expected_targets = frame.loc[has_previous, "event_id"]
    actual_targets = previous_lookup.loc[
        frame.loc[has_previous, "previous_event_id"]
    ]
    actual_targets.index = expected_targets.index
    if actual_targets.ne(expected_targets).any():
        raise QueryGenerationError(
            "input CSV contains non-reciprocal previous/next event links"
        )

    next_lookup = lookup["previous_event_id"]
    has_next = frame["next_event_id"].str.strip().ne("")
    expected_sources = frame.loc[has_next, "event_id"]
    actual_sources = next_lookup.loc[frame.loc[has_next, "next_event_id"]]
    actual_sources.index = expected_sources.index
    if actual_sources.ne(expected_sources).any():
        raise QueryGenerationError(
            "input CSV contains non-reciprocal next/previous event links"
        )


def _read_source(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise QueryGenerationError(f"input CSV does not exist: {input_path}")

    try:
        frame = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except (EmptyDataError, ParserError, UnicodeDecodeError, OSError) as exc:
        raise QueryGenerationError(
            f"could not read input CSV {input_path}: {exc}"
        ) from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise QueryGenerationError(
            "input CSV is missing required column(s): " + ", ".join(missing)
        )

    blank_identifiers = [
        column
        for column in ("user_id", "trail_id", "event_id", "venue_id", "timestamp")
        if frame[column].str.strip().eq("").any()
    ]
    if blank_identifiers:
        raise QueryGenerationError(
            "input CSV contains blank value(s) in required identifier column(s): "
            + ", ".join(blank_identifiers)
        )

    if frame["event_id"].duplicated().any():
        raise QueryGenerationError("input CSV contains duplicate event_id values")

    try:
        parsed_timestamps = pd.to_datetime(
            frame["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise QueryGenerationError(
            f"input CSV contains invalid timestamp value(s): {exc}"
        ) from exc

    frame = frame.copy()
    frame["_parsed_timestamp"] = parsed_timestamps
    frame["_is_high_memory_load"] = _parse_boolean_column(
        frame, "is_high_memory_load"
    )
    frame["_is_target_eligible"] = _parse_boolean_column(
        frame, "is_target_eligible"
    )

    inconsistent_high_load = frame.groupby("user_id")[
        "_is_high_memory_load"
    ].nunique().gt(1)
    if inconsistent_high_load.any():
        raise QueryGenerationError(
            "input CSV contains inconsistent is_high_memory_load values within user"
        )

    expected_eligibility = (
        frame["name"].str.strip().ne("")
        & frame["venue_category"].str.strip().ne("")
        & frame["venue_city"].str.strip().ne("")
    )
    if frame["_is_target_eligible"].ne(expected_eligibility).any():
        raise QueryGenerationError(
            "input CSV contains is_target_eligible values inconsistent with "
            "name, venue_category, or venue_city"
        )

    _validate_event_links(frame)

    frame["_name_key"] = frame["name"].map(_normalize)
    frame["_category_key"] = frame["venue_category"].map(_normalize)
    frame["_city_key"] = frame["venue_city"].map(_normalize)
    frame["_previous_place_key"] = frame["previous_place"].map(_normalize)
    frame["_next_place_key"] = frame["next_place"].map(_normalize)
    frame["_year_key"] = frame["_parsed_timestamp"].dt.year.astype(str)
    frame["_month_key"] = frame["_parsed_timestamp"].dt.month.astype(str)
    return frame


def _has_complete_cue(frame: pd.DataFrame, spec: QuerySpec) -> pd.Series:
    return frame[list(spec.required_columns)].apply(
        lambda column: column.str.strip().ne("")
    ).all(axis=1)


def _leaks_target_name(row: pd.Series, query_text: str) -> bool:
    target_name = row["_name_key"]
    if not target_name:
        return True
    return target_name in _normalize(query_text)


def _selection_hash(user_id: str, query_type: str, event_id: str) -> str:
    value = f"{SELECTION_SEED}|{user_id}|{query_type}|{event_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_row(row: pd.Series, spec: QuerySpec, query_text: str) -> dict[str, object]:
    timestamp = row["_parsed_timestamp"]
    cue_year = str(timestamp.year) if spec.query_type == "temporal_spatial" else ""
    cue_month = (
        str(timestamp.month).zfill(2)
        if spec.query_type == "temporal_spatial"
        else ""
    )
    uses_previous = spec.query_type in (
        "relational_after",
        "relational_between",
    )
    uses_next = spec.query_type == "relational_between"
    return {
        "query_id": f"tokyo_{row['user_id']}_{spec.query_type}",
        "user_id": row["user_id"],
        "benchmark_tier": spec.benchmark_tier,
        "query_type": spec.query_type,
        "query_language": QUERY_LANGUAGE,
        "query_text": query_text,
        "target_event_id": row["event_id"],
        "target_venue_id": row["venue_id"],
        "target_name": row["name"],
        "target_trail_id": row["trail_id"],
        "target_timestamp": row["timestamp"],
        "target_category": row["venue_category"],
        "target_city": row["venue_city"],
        "cue_year": cue_year,
        "cue_month": cue_month,
        "cue_city": row["venue_city"]
        if spec.query_type in ("semantic", "temporal_spatial")
        else "",
        "cue_category": row["venue_category"],
        "cue_previous_event_id": row["previous_event_id"]
        if uses_previous
        else "",
        "cue_previous_place": row["previous_place"] if uses_previous else "",
        "cue_next_event_id": row["next_event_id"] if uses_next else "",
        "cue_next_place": row["next_place"] if uses_next else "",
        "is_high_memory_load": bool(row["_is_high_memory_load"]),
    }


def _generate_for_spec(
    frame: pd.DataFrame, spec: QuerySpec
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    complete = _has_complete_cue(frame, spec)
    complete_frame = frame.loc[complete].copy()
    if complete_frame.empty:
        complete_frame["_cue_match_count"] = pd.Series(dtype="int64")
    else:
        complete_frame["_cue_match_count"] = complete_frame.groupby(
            ["user_id", *spec.cue_key_columns]
        )["event_id"].transform("size")

    unique_candidates = complete_frame.loc[
        complete_frame["_cue_match_count"].eq(1)
        & complete_frame["_is_target_eligible"]
    ].copy()
    if unique_candidates.empty:
        unique_candidates["_query_text"] = pd.Series(dtype="object")
        safe_candidates = unique_candidates
    else:
        unique_candidates["_query_text"] = unique_candidates.apply(
            spec.render, axis=1
        )
        leakage = unique_candidates.apply(
            lambda row: _leaks_target_name(row, row["_query_text"]), axis=1
        )
        safe_candidates = unique_candidates.loc[~leakage].copy()
        safe_candidates["_selection_hash"] = safe_candidates.apply(
            lambda row: _selection_hash(
                row["user_id"], spec.query_type, row["event_id"]
            ),
            axis=1,
        )

    eligible_counts = frame.groupby("user_id")["_is_target_eligible"].sum()
    complete_counts = complete_frame.groupby("user_id").size()
    unique_counts = unique_candidates.groupby("user_id").size()
    safe_counts = safe_candidates.groupby("user_id").size()

    if safe_candidates.empty:
        selected = safe_candidates
    else:
        selected = (
            safe_candidates.sort_values(
                ["user_id", "_selection_hash", "event_id"], kind="stable"
            )
            .groupby("user_id", sort=False)
            .head(1)
        )
    selected_by_user = {
        row["user_id"]: row for _, row in selected.iterrows()
    }

    queries: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    user_high_load = frame.groupby("user_id")["_is_high_memory_load"].first()
    for user_id in sorted(frame["user_id"].unique()):
        eligible_count = int(eligible_counts.get(user_id, 0))
        complete_count = int(complete_counts.get(user_id, 0))
        unique_count = int(unique_counts.get(user_id, 0))
        safe_count = int(safe_counts.get(user_id, 0))
        selected_row = selected_by_user.get(user_id)

        if selected_row is not None:
            query_text = str(selected_row["_query_text"])
            query = _query_row(selected_row, spec, query_text)
            queries.append(query)
            status = "generated"
            skip_reason = ""
            selected_event_id = query["target_event_id"]
            selected_venue_id = query["target_venue_id"]
        else:
            status = "skipped"
            if complete_count == 0:
                skip_reason = "no_complete_cue"
            elif unique_count == 0:
                skip_reason = "no_unique_event_candidate"
            else:
                skip_reason = "target_name_leakage"
            selected_event_id = ""
            selected_venue_id = ""

        audit.append(
            {
                "user_id": user_id,
                "benchmark_tier": spec.benchmark_tier,
                "query_type": spec.query_type,
                "eligible_target_count": eligible_count,
                "complete_cue_count": complete_count,
                "unique_event_candidate_count": unique_count,
                "leakage_safe_candidate_count": safe_count,
                "status": status,
                "skip_reason": skip_reason,
                "selected_event_id": selected_event_id,
                "selected_venue_id": selected_venue_id,
                "same_user_target_venue_query_count": 0,
                "target_venue_reused_across_query_types": False,
                "is_high_memory_load": bool(user_high_load.loc[user_id]),
            }
        )

    return queries, audit


def _mark_reused_target_venues(audit: pd.DataFrame) -> pd.DataFrame:
    generated = audit["status"].eq("generated")
    counts = (
        audit.loc[generated]
        .groupby(["user_id", "selected_venue_id"])
        .size()
        .to_dict()
    )
    audit = audit.copy()
    for index in audit.index[generated]:
        key = (audit.at[index, "user_id"], audit.at[index, "selected_venue_id"])
        count = int(counts[key])
        audit.at[index, "same_user_target_venue_query_count"] = count
        audit.at[index, "target_venue_reused_across_query_types"] = count > 1
    return audit


def _sort_queries(frame: pd.DataFrame) -> pd.DataFrame:
    order = {spec.query_type: index for index, spec in enumerate(QUERY_SPECS)}
    frame = frame.copy()
    frame["_query_order"] = frame["query_type"].map(order)
    return (
        frame.sort_values(["user_id", "_query_order"], kind="stable")
        .drop(columns="_query_order")
        .reset_index(drop=True)
    )


def _summary_rows(
    core_main: pd.DataFrame,
    core_high: pd.DataFrame,
    stress_main: pd.DataFrame,
    stress_high: pd.DataFrame,
    user_counts: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets = (
        ("core", "main", core_main, user_counts["main"]),
        ("core", "high_memory_load", core_high, user_counts["high"]),
        ("stress_test", "main", stress_main, user_counts["main"]),
        (
            "stress_test",
            "high_memory_load",
            stress_high,
            user_counts["high"],
        ),
    )
    for tier, cohort, dataset, user_count in datasets:
        query_types = [
            spec.query_type
            for spec in QUERY_SPECS
            if spec.benchmark_tier == tier
        ]
        for query_type in query_types:
            generated = int(dataset["query_type"].eq(query_type).sum())
            rows.append(
                {
                    "benchmark_tier": tier,
                    "cohort": cohort,
                    "query_type": query_type,
                    "user_count": user_count,
                    "generated_query_count": generated,
                    "skipped_user_count": user_count - generated,
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _counts_by_type(frame: pd.DataFrame) -> dict[str, int]:
    return {
        query_type: int(count)
        for query_type, count in frame["query_type"].value_counts().items()
    }


def _write_csvs_atomically(
    outputs: dict[Path, pd.DataFrame], output_dir: Path
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise QueryGenerationError(
            f"could not create output directory {output_dir}: {exc}"
        ) from exc

    temporary_paths: list[Path] = []
    try:
        for destination, frame in outputs.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=output_dir
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            frame.to_csv(temporary_path, index=False, encoding="utf-8")

        for temporary_path, destination in zip(
            temporary_paths, outputs, strict=True
        ):
            temporary_path.replace(destination)
    except (OSError, ValueError) as exc:
        raise QueryGenerationError(
            f"could not write query CSV files: {exc}"
        ) from exc
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def generate_queries(
    input_path: str | Path, output_dir: str | Path
) -> QueryGenerationSummary:
    """Generate core and relational-between benchmark queries."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    source = _read_source(input_path)

    query_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for spec in QUERY_SPECS:
        spec_queries, spec_audit = _generate_for_spec(source, spec)
        query_rows.extend(spec_queries)
        audit_rows.extend(spec_audit)

    queries = _sort_queries(pd.DataFrame(query_rows, columns=QUERY_COLUMNS))
    audit = _mark_reused_target_venues(
        pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    )
    audit = _sort_queries(audit)

    core_main = queries.loc[queries["benchmark_tier"].eq("core")].copy()
    core_high = core_main.loc[core_main["is_high_memory_load"]].copy()
    stress_main = queries.loc[
        queries["benchmark_tier"].eq("stress_test")
    ].copy()
    stress_high = stress_main.loc[stress_main["is_high_memory_load"]].copy()

    user_counts = {
        "main": int(source["user_id"].nunique()),
        "high": int(
            source.loc[source["_is_high_memory_load"], "user_id"].nunique()
        ),
    }
    summary_frame = _summary_rows(
        core_main, core_high, stress_main, stress_high, user_counts
    )

    outputs = {
        output_dir / CORE_MAIN_FILENAME: core_main,
        output_dir / CORE_HIGH_LOAD_FILENAME: core_high,
        output_dir / BETWEEN_MAIN_FILENAME: stress_main,
        output_dir / BETWEEN_HIGH_LOAD_FILENAME: stress_high,
        output_dir / AUDIT_FILENAME: audit,
        output_dir / SUMMARY_FILENAME: summary_frame,
    }
    _write_csvs_atomically(outputs, output_dir)

    return QueryGenerationSummary(
        core_main=_counts_by_type(core_main),
        core_high_load=_counts_by_type(core_high),
        stress_main=_counts_by_type(stress_main),
        stress_high_load=_counts_by_type(stress_high),
        audit_rows=int(len(audit)),
    )
