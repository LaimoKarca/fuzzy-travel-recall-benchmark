"""Create the fixed Tokyo check-in cohorts used by the benchmark."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


MAIN_MIN_TRAILS = 10
HIGH_LOAD_MIN_TRAILS = 20

MAIN_COHORT_FILENAME = "tokyo_checkins_users_with_at_least_10_trails.csv"
HIGH_LOAD_COHORT_FILENAME = "tokyo_checkins_users_with_at_least_20_trails.csv"
USER_STATS_FILENAME = "tokyo_user_trail_and_checkin_counts.csv"

REQUIRED_COLUMNS = ("user_id", "trail_id")


class CohortPreparationError(RuntimeError):
    """Raised when cohort preparation cannot complete safely."""


@dataclass(frozen=True)
class CohortCounts:
    users: int
    trails: int
    checkins: int


def _read_source(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise CohortPreparationError(f"input CSV does not exist: {input_path}")

    try:
        frame = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except (EmptyDataError, ParserError, UnicodeDecodeError, OSError) as exc:
        raise CohortPreparationError(
            f"could not read input CSV {input_path}: {exc}"
        ) from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise CohortPreparationError(
            "input CSV is missing required column(s): " + ", ".join(missing_columns)
        )

    blank_columns = [
        column
        for column in REQUIRED_COLUMNS
        if frame[column].str.strip().eq("").any()
    ]
    if blank_columns:
        raise CohortPreparationError(
            "input CSV contains blank value(s) in required column(s): "
            + ", ".join(blank_columns)
        )

    return frame


def _build_user_stats(frame: pd.DataFrame) -> pd.DataFrame:
    stats = (
        frame.groupby("user_id", sort=False)
        .agg(
            trail_count=("trail_id", "nunique"),
            checkin_count=("trail_id", "size"),
        )
        .reset_index()
    )
    stats["in_main_cohort"] = stats["trail_count"] >= MAIN_MIN_TRAILS
    stats["in_high_memory_load_cohort"] = (
        stats["trail_count"] >= HIGH_LOAD_MIN_TRAILS
    )
    return stats.sort_values("user_id", kind="stable").reset_index(drop=True)


def _count(frame: pd.DataFrame) -> CohortCounts:
    trail_pairs = frame[["user_id", "trail_id"]].drop_duplicates()
    return CohortCounts(
        users=int(frame["user_id"].nunique()),
        trails=int(len(trail_pairs)),
        checkins=int(len(frame)),
    )


def _write_csvs_atomically(
    outputs: dict[Path, pd.DataFrame], output_dir: Path
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CohortPreparationError(
            f"could not create output directory {output_dir}: {exc}"
        ) from exc

    temporary_paths: list[Path] = []
    try:
        for destination, frame in outputs.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=output_dir,
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
        raise CohortPreparationError(f"could not write prepared CSV files: {exc}") from exc
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def prepare_cohorts(
    input_path: str | Path, output_dir: str | Path
) -> dict[str, CohortCounts]:
    """Filter Tokyo check-ins into the fixed main and high-load cohorts.

    The cohort CSVs retain every source column and preserve source row order.
    Trail membership is based on each user's number of distinct ``trail_id`` values.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    frame = _read_source(input_path)
    stats = _build_user_stats(frame)

    main_users = set(stats.loc[stats["in_main_cohort"], "user_id"])
    high_load_users = set(
        stats.loc[stats["in_high_memory_load_cohort"], "user_id"]
    )

    main_cohort = frame.loc[frame["user_id"].isin(main_users)].copy()
    high_load_cohort = frame.loc[frame["user_id"].isin(high_load_users)].copy()

    outputs = {
        output_dir / MAIN_COHORT_FILENAME: main_cohort,
        output_dir / HIGH_LOAD_COHORT_FILENAME: high_load_cohort,
        output_dir / USER_STATS_FILENAME: stats,
    }
    _write_csvs_atomically(outputs, output_dir)

    return {
        "Original": _count(frame),
        f"Main (>={MAIN_MIN_TRAILS})": _count(main_cohort),
        f"High load (>={HIGH_LOAD_MIN_TRAILS})": _count(high_load_cohort),
    }
