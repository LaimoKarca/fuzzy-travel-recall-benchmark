"""Clean trajectory anomalies and create canonical event datasets."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.prepare_cohorts import CohortCounts


MAX_ADJACENT_GAP = pd.Timedelta(hours=8)
MAIN_MIN_TRAILS = 10
HIGH_LOAD_MIN_TRAILS = 20

MAIN_EVENTS_FILENAME = (
    "tokyo_canonical_events_users_with_at_least_10_clean_trails.csv"
)
HIGH_LOAD_EVENTS_FILENAME = (
    "tokyo_canonical_events_users_with_at_least_20_clean_trails.csv"
)
REMOVED_TRAILS_FILENAME = "tokyo_removed_trails_exceeding_8_hour_gap.csv"
CLEAN_USER_STATS_FILENAME = "tokyo_clean_user_trail_and_checkin_counts.csv"

REQUIRED_COLUMNS = (
    "user_id",
    "trail_id",
    "timestamp",
    "name",
    "venue_category",
    "venue_city",
)


class EventPreparationError(RuntimeError):
    """Raised when canonical event preparation cannot complete safely."""


@dataclass(frozen=True)
class EventPreparationSummary:
    input_counts: CohortCounts
    cleaned_counts: CohortCounts
    main_counts: CohortCounts
    high_load_counts: CohortCounts
    removed_trails: int
    removed_checkins: int


def _read_source(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise EventPreparationError(f"input CSV does not exist: {input_path}")

    try:
        frame = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except (EmptyDataError, ParserError, UnicodeDecodeError, OSError) as exc:
        raise EventPreparationError(
            f"could not read input CSV {input_path}: {exc}"
        ) from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise EventPreparationError(
            "input CSV is missing required column(s): " + ", ".join(missing)
        )

    blank_identifiers = [
        column
        for column in ("user_id", "trail_id", "timestamp")
        if frame[column].str.strip().eq("").any()
    ]
    if blank_identifiers:
        raise EventPreparationError(
            "input CSV contains blank value(s) in required identifier column(s): "
            + ", ".join(blank_identifiers)
        )

    trail_owner_counts = frame.groupby("trail_id")["user_id"].nunique()
    if trail_owner_counts.gt(1).any():
        raise EventPreparationError(
            "input CSV contains trail_id values shared by multiple users"
        )

    try:
        parsed_timestamps = pd.to_datetime(
            frame["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise EventPreparationError(
            f"input CSV contains invalid timestamp value(s): {exc}"
        ) from exc

    frame = frame.copy()
    frame["_source_order"] = range(len(frame))
    frame["_parsed_timestamp"] = parsed_timestamps
    return frame.sort_values(
        ["user_id", "trail_id", "_parsed_timestamp", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)


def _counts(frame: pd.DataFrame) -> CohortCounts:
    return CohortCounts(
        users=int(frame["user_id"].nunique()),
        trails=int(len(frame[["user_id", "trail_id"]].drop_duplicates())),
        checkins=int(len(frame)),
    )


def _remove_invalid_trails(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["user_id", "trail_id"]
    frame = frame.copy()
    frame["_adjacent_gap"] = frame.groupby(group_columns)[
        "_parsed_timestamp"
    ].diff()

    removed_keys = frame.loc[
        frame["_adjacent_gap"] > MAX_ADJACENT_GAP, group_columns
    ].drop_duplicates()

    if removed_keys.empty:
        audit = pd.DataFrame(
            columns=[
                "user_id",
                "trail_id",
                "checkin_count",
                "start_timestamp",
                "end_timestamp",
                "max_adjacent_gap_hours",
                "removal_reason",
            ]
        )
        return frame.drop(columns="_adjacent_gap"), audit

    removed_index = pd.MultiIndex.from_frame(removed_keys)
    frame_index = pd.MultiIndex.from_frame(frame[group_columns])
    removed = frame.loc[frame_index.isin(removed_index)].copy()
    cleaned = frame.loc[~frame_index.isin(removed_index)].copy()

    audit = (
        removed.groupby(group_columns, sort=False)
        .agg(
            checkin_count=("trail_id", "size"),
            start_timestamp=("timestamp", "first"),
            end_timestamp=("timestamp", "last"),
            max_adjacent_gap=("_adjacent_gap", "max"),
        )
        .reset_index()
    )
    audit["max_adjacent_gap_hours"] = (
        audit.pop("max_adjacent_gap").dt.total_seconds() / 3600
    )
    audit["removal_reason"] = "adjacent_timestamp_gap_exceeds_8_hours"
    audit = audit.sort_values(group_columns, kind="stable").reset_index(drop=True)

    return cleaned.drop(columns="_adjacent_gap"), audit


def _build_user_stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = (
        frame.groupby("user_id", sort=False)
        .agg(
            clean_trail_count=("trail_id", "nunique"),
            clean_checkin_count=("trail_id", "size"),
        )
        .reset_index()
    )
    stats["in_main_cohort"] = stats["clean_trail_count"] >= MAIN_MIN_TRAILS
    stats["in_high_memory_load_cohort"] = (
        stats["clean_trail_count"] >= HIGH_LOAD_MIN_TRAILS
    )
    return stats.sort_values("user_id", kind="stable").reset_index(drop=True)


def _add_canonical_fields(
    frame: pd.DataFrame, clean_trail_counts: dict[str, int]
) -> pd.DataFrame:
    group_columns = ["user_id", "trail_id"]
    canonical = frame.copy()
    canonical["event_position"] = canonical.groupby(group_columns).cumcount() + 1
    position_text = canonical["event_position"].astype(str).str.zfill(3)
    canonical["event_id"] = (
        canonical["trail_id"] + "__event_" + position_text
    )
    canonical["previous_event_id"] = canonical.groupby(group_columns)[
        "event_id"
    ].shift(1, fill_value="")
    canonical["previous_place"] = canonical.groupby(group_columns)["name"].shift(
        1, fill_value=""
    )
    canonical["next_event_id"] = canonical.groupby(group_columns)["event_id"].shift(
        -1, fill_value=""
    )
    canonical["next_place"] = canonical.groupby(group_columns)["name"].shift(
        -1, fill_value=""
    )
    canonical["trail_checkin_count"] = canonical.groupby(group_columns)[
        "trail_id"
    ].transform("size")
    canonical["clean_trail_count"] = canonical["user_id"].map(clean_trail_counts)
    canonical["is_high_memory_load"] = (
        canonical["clean_trail_count"] >= HIGH_LOAD_MIN_TRAILS
    )
    canonical["is_target_eligible"] = (
        canonical["name"].str.strip().ne("")
        & canonical["venue_category"].str.strip().ne("")
        & canonical["venue_city"].str.strip().ne("")
    )

    internal_columns = ["_source_order", "_parsed_timestamp"]
    return canonical.drop(columns=internal_columns).reset_index(drop=True)


def _write_csvs_atomically(
    outputs: dict[Path, pd.DataFrame], output_dir: Path
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EventPreparationError(
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
        raise EventPreparationError(f"could not write prepared CSV files: {exc}") from exc
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def prepare_events(
    input_path: str | Path, output_dir: str | Path
) -> EventPreparationSummary:
    """Remove temporally invalid trails and create canonical event cohorts."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    source = _read_source(input_path)
    cleaned, removed_trails = _remove_invalid_trails(source)
    user_stats = _build_user_stats(cleaned)

    main_users = set(user_stats.loc[user_stats["in_main_cohort"], "user_id"])
    high_load_users = set(
        user_stats.loc[user_stats["in_high_memory_load_cohort"], "user_id"]
    )
    clean_trail_counts = dict(
        zip(user_stats["user_id"], user_stats["clean_trail_count"], strict=True)
    )

    canonical = _add_canonical_fields(cleaned, clean_trail_counts)
    main_events = canonical.loc[canonical["user_id"].isin(main_users)].copy()
    high_load_events = canonical.loc[
        canonical["user_id"].isin(high_load_users)
    ].copy()

    outputs = {
        output_dir / MAIN_EVENTS_FILENAME: main_events,
        output_dir / HIGH_LOAD_EVENTS_FILENAME: high_load_events,
        output_dir / REMOVED_TRAILS_FILENAME: removed_trails,
        output_dir / CLEAN_USER_STATS_FILENAME: user_stats,
    }
    _write_csvs_atomically(outputs, output_dir)

    removed_checkins = int(
        len(source) - len(cleaned)
    )
    return EventPreparationSummary(
        input_counts=_counts(source),
        cleaned_counts=_counts(cleaned),
        main_counts=_counts(main_events),
        high_load_counts=_counts(high_load_events),
        removed_trails=int(len(removed_trails)),
        removed_checkins=removed_checkins,
    )
