"""Run deterministic per-user dense retrieval with multilingual MiniLM."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.model_download import (
    EXPECTED_DIMENSION,
    EXPECTED_MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    ModelDownloadError,
    manifest_path_for_model_dir,
    sha256_file,
    validate_model_installation,
)


METHOD = "vector"
DEVICE = "cpu"
BACKEND = "torch"
DTYPE = "float32"
BATCH_SIZE = 32

EVENT_EMBEDDINGS_FILENAME = "tokyo_vector_event_embeddings.npy"
EVENT_INDEX_FILENAME = "tokyo_vector_event_embedding_index.csv"
CORE_EMBEDDINGS_FILENAME = "tokyo_vector_core_query_embeddings.npy"
CORE_INDEX_FILENAME = "tokyo_vector_core_query_embedding_index.csv"
BETWEEN_EMBEDDINGS_FILENAME = "tokyo_vector_between_query_embeddings.npy"
BETWEEN_INDEX_FILENAME = "tokyo_vector_between_query_embedding_index.csv"
CORE_RANKINGS_FILENAME = "tokyo_vector_core_rankings.csv"
BETWEEN_RANKINGS_FILENAME = "tokyo_vector_between_rankings.csv"
SUMMARY_FILENAME = "tokyo_vector_retrieval_summary.csv"
RUN_MANIFEST_FILENAME = "tokyo_vector_run_manifest.json"

DOCUMENT_REQUIRED_COLUMNS = (
    "event_id",
    "user_id",
    "event_text",
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
EVENT_INDEX_COLUMNS = (
    "row_index",
    "event_id",
    "user_id",
    "is_high_memory_load",
)
QUERY_INDEX_COLUMNS = (
    "row_index",
    "query_id",
    "user_id",
    "benchmark_tier",
    "query_type",
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
SUMMARY_COLUMNS = (
    "method",
    "model_id",
    "model_revision",
    "device",
    "embedding_dimension",
    "batch_size",
    "max_seq_length",
    "benchmark_tier",
    "cohort",
    "query_type",
    "query_count",
    "ranked_candidate_count",
    "min_candidates_per_query",
    "max_candidates_per_query",
    "mean_candidates_per_query",
    "truncated_document_count",
    "truncated_query_count",
    "all_equal_score_query_count",
)
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}


class VectorRetrievalError(RuntimeError):
    """Raised when Vector retrieval cannot complete safely."""


@dataclass(frozen=True)
class VectorRetrievalSummary:
    document_count: int
    core_query_count: int
    stress_query_count: int
    core_ranking_count: int
    stress_ranking_count: int
    high_load_core_query_count: int
    high_load_stress_query_count: int
    truncated_document_count: int
    truncated_core_query_count: int
    truncated_stress_query_count: int
    all_equal_score_query_count: int


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise VectorRetrievalError(f"{label} CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, EmptyDataError, ParserError, UnicodeDecodeError) as exc:
        raise VectorRetrievalError(f"could not parse {label} CSV {path}: {exc}") from exc
    if frame.empty:
        raise VectorRetrievalError(f"{label} CSV contains no rows: {path}")
    return frame


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], label: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise VectorRetrievalError(
            f"{label} CSV is missing required column(s): {', '.join(missing)}"
        )


def _require_nonblank(
    frame: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    blank = [
        column for column in columns if frame[column].astype(str).str.strip().eq("").any()
    ]
    if blank:
        raise VectorRetrievalError(
            f"{label} CSV contains blank value(s) in: {', '.join(blank)}"
        )


def _parse_boolean(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:3]
        raise VectorRetrievalError(
            f"{label} contains invalid boolean value(s): {', '.join(examples)}"
        )
    return normalized.isin(TRUE_VALUES)


def _prepare_documents(path: Path) -> pd.DataFrame:
    documents = _read_csv(path, "documents")
    _require_columns(documents, DOCUMENT_REQUIRED_COLUMNS, "documents")
    _require_nonblank(documents, ("event_id", "user_id", "event_text"), "documents")
    if documents["event_id"].duplicated().any():
        raise VectorRetrievalError("documents CSV contains duplicate event_id values")
    documents = documents.loc[:, DOCUMENT_REQUIRED_COLUMNS].copy()
    documents["_is_high_memory_load"] = _parse_boolean(
        documents["is_high_memory_load"], "documents.is_high_memory_load"
    )
    inconsistent = documents.groupby("user_id")["_is_high_memory_load"].nunique()
    if inconsistent.gt(1).any():
        raise VectorRetrievalError(
            "documents CSV has inconsistent is_high_memory_load values within a user"
        )
    return documents.sort_values(["user_id", "event_id"], kind="stable").reset_index(
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
        raise VectorRetrievalError(f"{label} CSV contains duplicate query_id values")
    unexpected = sorted(set(queries["benchmark_tier"]) - {expected_tier})
    if unexpected:
        raise VectorRetrievalError(
            f"{label} CSV contains unexpected benchmark_tier value(s): "
            + ", ".join(unexpected)
        )
    queries = queries.loc[:, QUERY_REQUIRED_COLUMNS].copy()
    queries["_is_high_memory_load"] = _parse_boolean(
        queries["is_high_memory_load"], f"{label}.is_high_memory_load"
    )
    return queries.sort_values(["user_id", "query_id"], kind="stable").reset_index(
        drop=True
    )


def _validate_relationships(
    documents: pd.DataFrame, core: pd.DataFrame, stress: pd.DataFrame
) -> None:
    if set(core["query_id"]) & set(stress["query_id"]):
        raise VectorRetrievalError(
            "core and stress query CSVs contain duplicate query_id values"
        )
    event_users = set(documents["user_id"])
    missing = sorted((set(core["user_id"]) | set(stress["user_id"])) - event_users)
    if missing:
        raise VectorRetrievalError(
            "query CSV references user_id value(s) absent from documents: "
            + ", ".join(missing[:5])
        )
    event_high = (
        documents.groupby("user_id", sort=False)["_is_high_memory_load"]
        .first()
        .to_dict()
    )
    for label, queries in (("core queries", core), ("stress queries", stress)):
        if queries["_is_high_memory_load"].ne(queries["user_id"].map(event_high)).any():
            raise VectorRetrievalError(
                f"{label} has is_high_memory_load inconsistent with documents"
            )


def _count_truncations(model: Any, texts: list[str]) -> np.ndarray:
    truncated = np.zeros(len(texts), dtype=bool)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise VectorRetrievalError("model does not expose a tokenizer")
    for start in range(0, len(texts), 256):
        batch = texts[start : start + 256]
        try:
            tokenized = tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = tokenized["input_ids"]
        except Exception as exc:
            raise VectorRetrievalError(
                f"could not inspect tokenizer lengths: {exc}"
            ) from exc
        if len(input_ids) != len(batch):
            raise VectorRetrievalError("tokenizer returned an unexpected batch size")
        truncated[start : start + len(batch)] = [
            len(ids) > EXPECTED_MAX_SEQUENCE_LENGTH for ids in input_ids
        ]
    return truncated


def _encode(model: Any, texts: list[str], label: str) -> np.ndarray:
    try:
        values = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            precision="float32",
            device=DEVICE,
        )
    except Exception as exc:
        raise VectorRetrievalError(f"could not encode {label}: {exc}") from exc
    embeddings = np.asarray(values, dtype=np.float32)
    expected_shape = (len(texts), EXPECTED_DIMENSION)
    if embeddings.shape != expected_shape:
        raise VectorRetrievalError(
            f"{label} embeddings have shape {embeddings.shape}, expected {expected_shape}"
        )
    if embeddings.dtype != np.float32:
        raise VectorRetrievalError(f"{label} embeddings are not float32")
    if not np.isfinite(embeddings).all():
        raise VectorRetrievalError(f"{label} embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise VectorRetrievalError(f"{label} embeddings are not L2-normalized")
    return embeddings


def _event_index(documents: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": np.arange(len(documents), dtype=int),
            "event_id": documents["event_id"],
            "user_id": documents["user_id"],
            "is_high_memory_load": documents["_is_high_memory_load"].astype(bool),
        },
        columns=EVENT_INDEX_COLUMNS,
    )


def _query_index(queries: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": np.arange(len(queries), dtype=int),
            "query_id": queries["query_id"],
            "user_id": queries["user_id"],
            "benchmark_tier": queries["benchmark_tier"],
            "query_type": queries["query_type"],
            "is_high_memory_load": queries["_is_high_memory_load"].astype(bool),
        },
        columns=QUERY_INDEX_COLUMNS,
    )


def _rank(
    documents: pd.DataFrame,
    queries: pd.DataFrame,
    event_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, set[str]]:
    user_rows = {
        user_id: group.index.to_numpy(dtype=int)
        for user_id, group in documents.groupby("user_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    all_equal_queries: set[str] = set()
    for query_row, query in queries.iterrows():
        candidate_rows = user_rows[query["user_id"]]
        scores = event_embeddings[candidate_rows] @ query_embeddings[query_row]
        if not np.isfinite(scores).all():
            raise VectorRetrievalError(
                f"cosine scoring returned a non-finite value for {query['query_id']}"
            )
        if np.all(scores == scores[0]):
            all_equal_queries.add(query["query_id"])
        order = sorted(
            range(len(candidate_rows)),
            key=lambda index: (
                -float(scores[index]),
                documents.at[int(candidate_rows[index]), "event_id"],
            ),
        )
        for rank, local_index in enumerate(order, start=1):
            document_row = int(candidate_rows[local_index])
            rows.append(
                {
                    "query_id": query["query_id"],
                    "user_id": query["user_id"],
                    "benchmark_tier": query["benchmark_tier"],
                    "query_type": query["query_type"],
                    "method": METHOD,
                    "rank": rank,
                    "retrieved_event_id": documents.at[document_row, "event_id"],
                    "score": float(scores[local_index]),
                    "is_high_memory_load": bool(query["_is_high_memory_load"]),
                }
            )
    return pd.DataFrame(rows, columns=RANKING_COLUMNS), all_equal_queries


def _summary_rows(
    rankings: pd.DataFrame,
    queries: pd.DataFrame,
    query_truncated: np.ndarray,
    truncated_document_count: int,
    all_equal_queries: set[str],
    cohort: str,
) -> list[dict[str, object]]:
    selected_mask = np.ones(len(queries), dtype=bool)
    if cohort == "high_memory_load":
        selected_mask = queries["_is_high_memory_load"].to_numpy(dtype=bool)
    selected = queries.loc[selected_mask]
    selected_truncated = pd.Series(query_truncated[selected_mask], index=selected.index)

    rows: list[dict[str, object]] = []
    for query_type in sorted(selected["query_type"].unique()):
        type_mask = selected["query_type"].eq(query_type)
        query_ids = set(selected.loc[type_mask, "query_id"])
        subset = rankings.loc[rankings["query_id"].isin(query_ids)]
        candidates_per_query = subset.groupby("query_id").size()
        rows.append(
            {
                "method": METHOD,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "device": DEVICE,
                "embedding_dimension": EXPECTED_DIMENSION,
                "batch_size": BATCH_SIZE,
                "max_seq_length": EXPECTED_MAX_SEQUENCE_LENGTH,
                "benchmark_tier": selected["benchmark_tier"].iloc[0],
                "cohort": cohort,
                "query_type": query_type,
                "query_count": len(query_ids),
                "ranked_candidate_count": int(len(subset)),
                "min_candidates_per_query": int(candidates_per_query.min()),
                "max_candidates_per_query": int(candidates_per_query.max()),
                "mean_candidates_per_query": float(candidates_per_query.mean()),
                "truncated_document_count": truncated_document_count,
                "truncated_query_count": int(selected_truncated.loc[type_mask].sum()),
                "all_equal_score_query_count": len(query_ids & all_equal_queries),
            }
        )
    return rows


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _write_outputs_atomically(
    arrays: dict[Path, np.ndarray],
    frames: dict[Path, pd.DataFrame],
    manifest_destination: Path,
    manifest_base: dict[str, Any],
) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for destination, array in arrays.items():
            temp = _temporary_path(destination)
            temporary[destination] = temp
            with temp.open("wb") as target:
                np.save(target, array, allow_pickle=False)
        for destination, frame in frames.items():
            temp = _temporary_path(destination)
            temporary[destination] = temp
            frame.to_csv(temp, index=False, encoding="utf-8")

        manifest = dict(manifest_base)
        manifest["outputs"] = {
            destination.name: sha256_file(temp)
            for destination, temp in sorted(
                temporary.items(), key=lambda item: item[0].name
            )
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
        raise VectorRetrievalError(f"could not write Vector retrieval outputs: {exc}") from exc
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_versions() -> dict[str, str]:
    return {
        "huggingface_hub": _package_version("huggingface-hub"),
        "numpy": _package_version("numpy"),
        "os": platform.platform(),
        "pandas": _package_version("pandas"),
        "python": sys.version.split()[0],
        "sentence_transformers": _package_version("sentence-transformers"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }


def retrieve_vector(
    documents_path: str | Path,
    core_queries_path: str | Path,
    stress_queries_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    model: Any | None = None,
) -> VectorRetrievalSummary:
    """Encode shared texts and save complete per-user cosine rankings."""

    documents_path = Path(documents_path)
    core_queries_path = Path(core_queries_path)
    stress_queries_path = Path(stress_queries_path)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    documents = _prepare_documents(documents_path)
    core = _prepare_queries(core_queries_path, "core", "core queries")
    stress = _prepare_queries(stress_queries_path, "stress_test", "stress queries")
    _validate_relationships(documents, core, stress)

    try:
        model_manifest, loaded_model = validate_model_installation(
            model_dir, load_model=model is None
        )
    except ModelDownloadError as exc:
        raise VectorRetrievalError(str(exc)) from exc
    if model is None:
        model = loaded_model
    else:
        try:
            get_dimension = getattr(model, "get_embedding_dimension", None)
            if get_dimension is None:
                get_dimension = model.get_sentence_embedding_dimension
            if get_dimension() != EXPECTED_DIMENSION:
                raise VectorRetrievalError("mock/provided model has the wrong dimension")
            if int(model.max_seq_length) != EXPECTED_MAX_SEQUENCE_LENGTH:
                raise VectorRetrievalError("mock/provided model has the wrong max sequence length")
        except AttributeError as exc:
            raise VectorRetrievalError("provided model has an invalid interface") from exc

    document_texts = documents["event_text"].tolist()
    core_texts = core["query_text"].tolist()
    stress_texts = stress["query_text"].tolist()
    document_truncated = _count_truncations(model, document_texts)
    core_truncated = _count_truncations(model, core_texts)
    stress_truncated = _count_truncations(model, stress_texts)

    event_embeddings = _encode(model, document_texts, "event")
    core_embeddings = _encode(model, core_texts, "core query")
    stress_embeddings = _encode(model, stress_texts, "between query")

    event_index = _event_index(documents)
    core_index = _query_index(core)
    stress_index = _query_index(stress)
    core_rankings, core_equal = _rank(
        documents, core, event_embeddings, core_embeddings
    )
    stress_rankings, stress_equal = _rank(
        documents, stress, event_embeddings, stress_embeddings
    )

    summary_rows: list[dict[str, object]] = []
    for cohort in ("main", "high_memory_load"):
        summary_rows.extend(
            _summary_rows(
                core_rankings,
                core,
                core_truncated,
                int(document_truncated.sum()),
                core_equal,
                cohort,
            )
        )
        summary_rows.extend(
            _summary_rows(
                stress_rankings,
                stress,
                stress_truncated,
                int(document_truncated.sum()),
                stress_equal,
                cohort,
            )
        )
    summary_frame = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    arrays = {
        output_dir / EVENT_EMBEDDINGS_FILENAME: event_embeddings,
        output_dir / CORE_EMBEDDINGS_FILENAME: core_embeddings,
        output_dir / BETWEEN_EMBEDDINGS_FILENAME: stress_embeddings,
    }
    frames = {
        output_dir / EVENT_INDEX_FILENAME: event_index,
        output_dir / CORE_INDEX_FILENAME: core_index,
        output_dir / BETWEEN_INDEX_FILENAME: stress_index,
        output_dir / CORE_RANKINGS_FILENAME: core_rankings,
        output_dir / BETWEEN_RANKINGS_FILENAME: stress_rankings,
        output_dir / SUMMARY_FILENAME: summary_frame,
    }
    manifest_path = manifest_path_for_model_dir(model_dir)
    manifest_base = {
        "inputs": {
            "core_queries": {
                "path": str(core_queries_path.resolve()),
                "sha256": sha256_file(core_queries_path),
            },
            "documents": {
                "path": str(documents_path.resolve()),
                "sha256": sha256_file(documents_path),
            },
            "stress_queries": {
                "path": str(stress_queries_path.resolve()),
                "sha256": sha256_file(stress_queries_path),
            },
        },
        "method": METHOD,
        "model": {
            "directory": str(model_dir.resolve()),
            "id": model_manifest["model_id"],
            "manifest_sha256": sha256_file(manifest_path),
            "revision": model_manifest["revision"],
        },
        "parameters": {
            "backend": BACKEND,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "dtype": DTYPE,
            "embedding_dimension": EXPECTED_DIMENSION,
            "max_seq_length": EXPECTED_MAX_SEQUENCE_LENGTH,
            "normalize_embeddings": True,
            "ranking_depth": "all events belonging to the query user",
            "similarity": "cosine via normalized-vector dot product",
        },
        "runtime_versions": _runtime_versions(),
    }
    _write_outputs_atomically(
        arrays,
        frames,
        output_dir / RUN_MANIFEST_FILENAME,
        manifest_base,
    )

    return VectorRetrievalSummary(
        document_count=len(documents),
        core_query_count=len(core),
        stress_query_count=len(stress),
        core_ranking_count=len(core_rankings),
        stress_ranking_count=len(stress_rankings),
        high_load_core_query_count=int(core["_is_high_memory_load"].sum()),
        high_load_stress_query_count=int(stress["_is_high_memory_load"].sum()),
        truncated_document_count=int(document_truncated.sum()),
        truncated_core_query_count=int(core_truncated.sum()),
        truncated_stress_query_count=int(stress_truncated.sum()),
        all_equal_score_query_count=len(core_equal | stress_equal),
    )
