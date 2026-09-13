"""Build shared event documents and run the Flat BM25 retriever."""

from __future__ import annotations

import calendar
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


DOCUMENT_FILENAME = "tokyo_event_documents.csv"
CORE_RANKINGS_FILENAME = "tokyo_flat_core_rankings.csv"
BETWEEN_RANKINGS_FILENAME = "tokyo_flat_between_rankings.csv"
SUMMARY_FILENAME = "tokyo_flat_retrieval_summary.csv"

BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25
METHOD = "flat"

EVENT_REQUIRED_COLUMNS = (
    "event_id",
    "user_id",
    "trail_id",
    "venue_id",
    "timestamp",
    "name",
    "venue_category",
    "venue_city",
    "address",
    "previous_event_id",
    "previous_place",
    "next_event_id",
    "next_place",
    "is_high_memory_load",
)

QUERY_REQUIRED_COLUMNS = (
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
    "query_text",
    "is_high_memory_load",
)

DOCUMENT_COLUMNS = EVENT_REQUIRED_COLUMNS + ("event_text",)

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

SUMMARY_COLUMNS = (
    "method",
    "benchmark_tier",
    "cohort",
    "query_type",
    "query_count",
    "ranked_candidate_count",
    "min_candidates_per_query",
    "max_candidates_per_query",
    "mean_candidates_per_query",
    "all_zero_score_query_count",
)

TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}


class FlatRetrievalError(RuntimeError):
    """Raised when Flat retrieval cannot complete safely."""


@dataclass(frozen=True)
class FlatRetrievalSummary:
    document_count: int
    core_query_count: int
    core_ranking_count: int
    stress_query_count: int
    stress_ranking_count: int
    high_load_core_query_count: int
    high_load_stress_query_count: int
    all_zero_score_query_count: int


def _is_japanese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or character in {"々", "〆", "ヶ"}
    )


def mixed_language_tokenize(value: object) -> list[str]:
    """Tokenize Latin words/numbers and Japanese character n-grams."""

    if value is None:
        return []
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = " ".join(normalized.split())

    tokens: list[str] = []
    segment: list[str] = []
    segment_kind: str | None = None

    def flush() -> None:
        nonlocal segment, segment_kind
        if not segment:
            return
        text = "".join(segment)
        if segment_kind == "japanese":
            tokens.append(text)
            for width in (2, 3):
                if len(text) > width:
                    tokens.extend(
                        text[index : index + width]
                        for index in range(len(text) - width + 1)
                    )
        else:
            tokens.append(text)
        segment = []
        segment_kind = None

    for character in normalized:
        if _is_japanese_character(character):
            kind = "japanese"
        else:
            category = unicodedata.category(character)
            kind = "word" if category[0] in {"L", "M", "N"} else None

        if kind is None:
            flush()
            continue
        if segment_kind is not None and kind != segment_kind:
            flush()
        segment_kind = kind
        segment.append(character)

    flush()
    return tokens


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FlatRetrievalError(f"{label} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise FlatRetrievalError(f"could not parse {label} CSV {path}: {exc}") from exc
    if frame.empty:
        raise FlatRetrievalError(f"{label} CSV contains no rows: {path}")
    return frame


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FlatRetrievalError(
            f"{label} CSV is missing required column(s): {', '.join(missing)}"
        )


def _require_nonblank(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    blank = [
        column
        for column in columns
        if frame[column].astype(str).str.strip().eq("").any()
    ]
    if blank:
        raise FlatRetrievalError(
            f"{label} CSV contains blank value(s) in: {', '.join(blank)}"
        )


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:3]
        raise FlatRetrievalError(
            f"{label} contains invalid boolean value(s): {', '.join(examples)}"
        )
    return normalized.isin(TRUE_VALUES)


def _prepare_events(path: Path) -> pd.DataFrame:
    events = _read_csv(path, "events")
    _require_columns(events, EVENT_REQUIRED_COLUMNS, "events")
    _require_nonblank(
        events,
        ("event_id", "user_id", "trail_id", "venue_id", "timestamp"),
        "events",
    )
    if events["event_id"].duplicated().any():
        raise FlatRetrievalError("events CSV contains duplicate event_id values")

    try:
        parsed = pd.to_datetime(
            events["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise FlatRetrievalError(
            f"events CSV contains invalid timestamp value(s): {exc}"
        ) from exc

    events = events.loc[:, EVENT_REQUIRED_COLUMNS].copy()
    events["_parsed_timestamp"] = parsed
    events["_is_high_memory_load"] = _parse_boolean(
        events["is_high_memory_load"], "events.is_high_memory_load"
    )

    inconsistent = events.groupby("user_id")["_is_high_memory_load"].nunique()
    if inconsistent.gt(1).any():
        raise FlatRetrievalError(
            "events CSV has inconsistent is_high_memory_load values within a user"
        )
    return events.sort_values(["user_id", "event_id"], kind="stable").reset_index(
        drop=True
    )


def _prepare_queries(path: Path, expected_tier: str, label: str) -> pd.DataFrame:
    queries = _read_csv(path, label)
    _require_columns(queries, QUERY_REQUIRED_COLUMNS, label)
    _require_nonblank(
        queries,
        ("query_id", "user_id", "benchmark_tier", "query_type", "query_text"),
        label,
    )
    if queries["query_id"].duplicated().any():
        raise FlatRetrievalError(f"{label} CSV contains duplicate query_id values")
    unexpected_tiers = sorted(set(queries["benchmark_tier"]) - {expected_tier})
    if unexpected_tiers:
        raise FlatRetrievalError(
            f"{label} CSV contains unexpected benchmark_tier value(s): "
            f"{', '.join(unexpected_tiers)}"
        )

    queries = queries.loc[:, QUERY_REQUIRED_COLUMNS].copy()
    queries["_is_high_memory_load"] = _parse_boolean(
        queries["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    return queries.sort_values(["user_id", "query_id"], kind="stable").reset_index(
        drop=True
    )


def _render_event_text(row: pd.Series) -> str:
    timestamp = row["_parsed_timestamp"]
    date = (
        f"{calendar.month_name[int(timestamp.month)]} {int(timestamp.day)}, "
        f"{int(timestamp.year)} at {timestamp.strftime('%H:%M')}"
    )

    name = row["name"].strip()
    category = row["venue_category"].strip()
    city = row["venue_city"].strip()
    if name and category and city:
        first = f"On {date}, the user visited {name}, a {category} in {city}."
    elif name:
        first = f"On {date}, the user visited {name}."
    else:
        first = f"On {date}, the user recorded a travel event."
        if category:
            first += f" Category: {category}."
        if city:
            first += f" City: {city}."

    parts = [first]
    for prefix, column in (
        ("Address", "address"),
        ("Previous place", "previous_place"),
        ("Next place", "next_place"),
    ):
        value = row[column].strip()
        if value:
            parts.append(f"{prefix}: {value}.")
    return " ".join(parts)


def build_event_documents(events: pd.DataFrame) -> pd.DataFrame:
    """Create the shared, deterministic one-document-per-event corpus."""

    documents = events.loc[:, EVENT_REQUIRED_COLUMNS].copy()
    documents["event_text"] = events.apply(_render_event_text, axis=1)
    return documents.loc[:, DOCUMENT_COLUMNS]


def _validate_query_users(
    events: pd.DataFrame, core: pd.DataFrame, stress: pd.DataFrame
) -> None:
    event_users = set(events["user_id"])
    query_users = set(core["user_id"]) | set(stress["user_id"])
    missing = sorted(query_users - event_users)
    if missing:
        raise FlatRetrievalError(
            "query CSV references user_id value(s) absent from events: "
            + ", ".join(missing[:5])
        )

    event_high = (
        events.groupby("user_id", sort=False)["_is_high_memory_load"].first().to_dict()
    )
    for label, queries in (("core queries", core), ("stress queries", stress)):
        mismatched = queries["_is_high_memory_load"].ne(
            queries["user_id"].map(event_high)
        )
        if mismatched.any():
            raise FlatRetrievalError(
                f"{label} has is_high_memory_load inconsistent with events"
            )


def _retrieve(
    events: pd.DataFrame, documents: pd.DataFrame, queries: pd.DataFrame
) -> tuple[pd.DataFrame, set[str]]:
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise FlatRetrievalError(
            "rank-bm25 is not installed; run "
            "'python -m pip install -r requirements.txt'"
        ) from exc

    event_lookup = events.set_index("event_id")
    document_groups = {
        user_id: group.sort_values("event_id", kind="stable").reset_index(drop=True)
        for user_id, group in documents.groupby("user_id", sort=False)
    }
    indexes = {
        user_id: BM25Okapi(
            [mixed_language_tokenize(text) for text in group["event_text"]],
            k1=BM25_K1,
            b=BM25_B,
            epsilon=BM25_EPSILON,
        )
        for user_id, group in document_groups.items()
    }

    rows: list[dict[str, object]] = []
    all_zero_queries: set[str] = set()
    for _, query in queries.iterrows():
        candidates = document_groups[query["user_id"]]
        scores = np.asarray(
            indexes[query["user_id"]].get_scores(
                mixed_language_tokenize(query["query_text"])
            ),
            dtype=float,
        )
        if not np.isfinite(scores).all():
            raise FlatRetrievalError(
                f"BM25 returned a non-finite score for query {query['query_id']}"
            )
        if np.all(scores == 0.0):
            all_zero_queries.add(query["query_id"])

        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates.at[index, "event_id"]),
        )
        for rank, index in enumerate(order, start=1):
            event_id = candidates.at[index, "event_id"]
            # Access through event_id only to make the user boundary auditable.
            if event_lookup.at[event_id, "user_id"] != query["user_id"]:
                raise FlatRetrievalError(
                    f"retrieval crossed a user boundary for query {query['query_id']}"
                )
            rows.append(
                {
                    "query_id": query["query_id"],
                    "user_id": query["user_id"],
                    "benchmark_tier": query["benchmark_tier"],
                    "query_type": query["query_type"],
                    "method": METHOD,
                    "rank": rank,
                    "retrieved_event_id": event_id,
                    "score": float(scores[index]),
                    "is_high_memory_load": bool(query["_is_high_memory_load"]),
                }
            )

    return pd.DataFrame(rows, columns=RANKING_COLUMNS), all_zero_queries


def _summary_for_dataset(
    rankings: pd.DataFrame,
    queries: pd.DataFrame,
    all_zero_queries: set[str],
    cohort: str,
) -> list[dict[str, object]]:
    selected = queries
    if cohort == "high_memory_load":
        selected = queries.loc[queries["_is_high_memory_load"]]

    rows: list[dict[str, object]] = []
    for query_type in sorted(selected["query_type"].unique()):
        query_ids = set(selected.loc[selected["query_type"].eq(query_type), "query_id"])
        subset = rankings.loc[rankings["query_id"].isin(query_ids)]
        candidates_per_query = subset.groupby("query_id").size()
        rows.append(
            {
                "method": METHOD,
                "benchmark_tier": selected["benchmark_tier"].iloc[0],
                "cohort": cohort,
                "query_type": query_type,
                "query_count": len(query_ids),
                "ranked_candidate_count": int(len(subset)),
                "min_candidates_per_query": int(candidates_per_query.min()),
                "max_candidates_per_query": int(candidates_per_query.max()),
                "mean_candidates_per_query": float(candidates_per_query.mean()),
                "all_zero_score_query_count": len(query_ids & all_zero_queries),
            }
        )
    return rows


def _write_csvs_atomically(outputs: dict[Path, pd.DataFrame]) -> None:
    temporary_paths: list[Path] = []
    try:
        for destination, frame in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
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
        raise FlatRetrievalError(f"could not write Flat retrieval CSV files: {exc}") from exc
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def retrieve_flat(
    events_path: str | Path,
    core_queries_path: str | Path,
    stress_queries_path: str | Path,
    output_dir: str | Path,
) -> FlatRetrievalSummary:
    """Build shared documents and retrieve complete per-user BM25 rankings."""

    events = _prepare_events(Path(events_path))
    core = _prepare_queries(Path(core_queries_path), "core", "core queries")
    stress = _prepare_queries(
        Path(stress_queries_path), "stress_test", "stress queries"
    )
    duplicate_query_ids = set(core["query_id"]) & set(stress["query_id"])
    if duplicate_query_ids:
        raise FlatRetrievalError(
            "core and stress query CSVs contain duplicate query_id values"
        )
    _validate_query_users(events, core, stress)

    documents = build_event_documents(events)
    core_rankings, core_zero = _retrieve(events, documents, core)
    stress_rankings, stress_zero = _retrieve(events, documents, stress)

    summary_rows: list[dict[str, object]] = []
    for cohort in ("main", "high_memory_load"):
        summary_rows.extend(
            _summary_for_dataset(core_rankings, core, core_zero, cohort)
        )
        summary_rows.extend(
            _summary_for_dataset(stress_rankings, stress, stress_zero, cohort)
        )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    output_dir = Path(output_dir)
    outputs = {
        output_dir / "common" / DOCUMENT_FILENAME: documents,
        output_dir / "flat" / CORE_RANKINGS_FILENAME: core_rankings,
        output_dir / "flat" / BETWEEN_RANKINGS_FILENAME: stress_rankings,
        output_dir / "flat" / SUMMARY_FILENAME: summary,
    }
    _write_csvs_atomically(outputs)

    return FlatRetrievalSummary(
        document_count=len(documents),
        core_query_count=len(core),
        core_ranking_count=len(core_rankings),
        stress_query_count=len(stress),
        stress_ranking_count=len(stress_rankings),
        high_load_core_query_count=int(core["_is_high_memory_load"].sum()),
        high_load_stress_query_count=int(stress["_is_high_memory_load"].sum()),
        all_zero_score_query_count=len(core_zero | stress_zero),
    )
