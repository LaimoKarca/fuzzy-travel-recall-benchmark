"""Download and validate pinned multilingual embedding research models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.model_specs import (
    DEFAULT_MODEL_KEY,
    MINILM_SPEC,
    ModelSpec,
    get_model_spec,
    model_spec_for_directory,
)

# Backward-compatible aliases for the original MiniLM experiment.
MODEL_ID = MINILM_SPEC.model_id
MODEL_REVISION = MINILM_SPEC.revision
MODEL_DIRECTORY_NAME = MINILM_SPEC.directory_name
EXPECTED_DIMENSION = MINILM_SPEC.embedding_dimension
EXPECTED_MAX_SEQUENCE_LENGTH = MINILM_SPEC.max_sequence_length
MODEL_MANIFEST_FILENAME = "model_manifest.json"
REFERENCE_RUNTIME_MANIFEST_FILENAME = "reference_runtime_manifest.json"
REQUIRED_MODEL_FILES = MINILM_SPEC.required_files


class ModelDownloadError(RuntimeError):
    """Raised when the fixed model cannot be prepared safely."""


@dataclass(frozen=True)
class ModelDownloadSummary:
    model_dir: Path
    manifest_path: Path
    runtime_manifest_path: Path
    downloaded: bool
    file_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for_model_dir(
    model_dir: Path, model_spec: ModelSpec | None = None
) -> Path:
    model_spec = model_spec or model_spec_for_directory(model_dir.name)
    if model_dir.parent.name == "sentence-transformers":
        return model_dir.parent.parent / model_spec.manifest_filename
    return model_dir.parent / model_spec.manifest_filename


def runtime_manifest_path_for_model_dir(
    model_dir: Path, model_spec: ModelSpec | None = None
) -> Path:
    model_spec = model_spec or model_spec_for_directory(model_dir.name)
    return manifest_path_for_model_dir(model_dir, model_spec).with_name(
        model_spec.runtime_manifest_filename
    )


def _stable_manifest(model_dir: Path, model_spec: ModelSpec) -> dict[str, Any]:
    return {
        "backend": "torch",
        "document_prefix": model_spec.document_prefix,
        "embedding_dimension": model_spec.embedding_dimension,
        "files": {
            relative: sha256_file(model_dir / relative)
            for relative in model_spec.required_files
        },
        "license": model_spec.license,
        "max_seq_length": model_spec.max_sequence_length,
        "model_id": model_spec.model_id,
        "query_prefix": model_spec.query_prefix,
        "revision": model_spec.revision,
    }


def _load_sentence_transformer(model_dir: Path) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, OSError) as exc:
        raise ModelDownloadError(
            "sentence-transformers is not available; run "
            "'python -m pip install -r requirements.txt'"
        ) from exc

    try:
        return SentenceTransformer(
            str(model_dir),
            device="cpu",
            backend="torch",
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:  # third-party loaders expose several exception classes
        raise ModelDownloadError(
            f"model cannot be loaded offline from {model_dir}: {exc}"
        ) from exc


def _validate_model_runtime(model: Any, model_spec: ModelSpec) -> None:
    get_dimension = getattr(model, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = model.get_sentence_embedding_dimension
    dimension = get_dimension()
    if dimension != model_spec.embedding_dimension:
        raise ModelDownloadError(
            "model embedding dimension is "
            f"{dimension}, expected {model_spec.embedding_dimension}"
        )
    max_sequence_length = int(model.max_seq_length)
    if max_sequence_length != model_spec.max_sequence_length:
        raise ModelDownloadError(
            "model max sequence length is "
            f"{max_sequence_length}, expected {model_spec.max_sequence_length}"
        )


def validate_model_installation(
    model_dir: str | Path,
    *,
    load_model: bool = True,
    model_loader: Callable[[Path], Any] | None = None,
    model_spec: ModelSpec | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Validate required files, pinned metadata, hashes, and offline loading."""

    model_dir = Path(model_dir)
    model_spec = model_spec or model_spec_for_directory(model_dir.name)
    manifest_path = manifest_path_for_model_dir(model_dir, model_spec)
    if not model_dir.is_dir():
        raise ModelDownloadError(f"model directory does not exist: {model_dir}")
    if not manifest_path.is_file():
        raise ModelDownloadError(f"model manifest does not exist: {manifest_path}")

    missing = [
        name for name in model_spec.required_files if not (model_dir / name).is_file()
    ]
    if missing:
        raise ModelDownloadError(
            "model installation is incomplete; missing: " + ", ".join(missing)
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelDownloadError(f"could not parse model manifest: {exc}") from exc

    expected_metadata = {
        "model_id": model_spec.model_id,
        "revision": model_spec.revision,
        "backend": "torch",
        "embedding_dimension": model_spec.embedding_dimension,
        "max_seq_length": model_spec.max_sequence_length,
        "query_prefix": model_spec.query_prefix,
        "document_prefix": model_spec.document_prefix,
    }
    mismatches = []
    for key, expected in expected_metadata.items():
        actual = manifest.get(key, "" if key.endswith("_prefix") else None)
        if actual != expected:
            mismatches.append(key)
    if mismatches:
        raise ModelDownloadError(
            "model manifest metadata mismatch in: " + ", ".join(mismatches)
        )

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise ModelDownloadError("model manifest has no valid files mapping")
    for relative in model_spec.required_files:
        expected_hash = manifest_files.get(relative)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ModelDownloadError(
                f"model manifest has no valid SHA-256 for {relative}"
            )
        actual_hash = sha256_file(model_dir / relative)
        if actual_hash != expected_hash:
            raise ModelDownloadError(f"model file hash mismatch: {relative}")

    model = None
    if load_model:
        model = (model_loader or _load_sentence_transformer)(model_dir)
        _validate_model_runtime(model, model_spec)
    return manifest, model


def _snapshot_download(model_dir: Path, model_spec: ModelSpec) -> None:
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as exc:
        raise ModelDownloadError(
            "huggingface-hub is not available; run "
            "'python -m pip install -r requirements.txt'"
        ) from exc
    try:
        snapshot_download(
            repo_id=model_spec.model_id,
            revision=model_spec.revision,
            local_dir=model_dir,
            allow_patterns=list(model_spec.required_files),
        )
    except Exception as exc:
        raise ModelDownloadError(
            "could not download "
            f"{model_spec.model_id} at revision {model_spec.revision}: {exc}"
        ) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _reference_runtime_manifest() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "packages": {
            "huggingface-hub": _package_version("huggingface-hub"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "python": sys.version.split()[0],
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        _write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_models(
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_KEY,
    snapshot_downloader: Callable[[Path], None] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> ModelDownloadSummary:
    """Download the pinned model to a temporary directory, then publish it."""

    model_dir = Path(output_dir)
    try:
        model_spec = get_model_spec(model_name)
    except ValueError as exc:
        raise ModelDownloadError(str(exc)) from exc
    manifest_path = manifest_path_for_model_dir(model_dir, model_spec)
    runtime_manifest_path = runtime_manifest_path_for_model_dir(model_dir, model_spec)
    if model_dir.exists() or manifest_path.exists():
        validate_model_installation(
            model_dir, model_loader=model_loader, model_spec=model_spec
        )
        _write_json_atomically(runtime_manifest_path, _reference_runtime_manifest())
        return ModelDownloadSummary(
            model_dir=model_dir,
            manifest_path=manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            downloaded=False,
            file_count=len(model_spec.required_files),
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".model-download-", dir=model_dir.parent)
    )
    temporary_model = temporary_root / model_dir.name
    temporary_model.mkdir()
    temporary_manifest = temporary_root / model_spec.manifest_filename
    temporary_runtime_manifest = temporary_root / model_spec.runtime_manifest_filename

    try:
        if snapshot_downloader is None:
            _snapshot_download(temporary_model, model_spec)
        else:
            snapshot_downloader(temporary_model)
        cache_dir = temporary_model / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        missing = [
            name
            for name in model_spec.required_files
            if not (temporary_model / name).is_file()
        ]
        if missing:
            raise ModelDownloadError(
                "downloaded model is incomplete; missing: " + ", ".join(missing)
            )

        model = (model_loader or _load_sentence_transformer)(temporary_model)
        _validate_model_runtime(model, model_spec)
        del model

        manifest = _stable_manifest(temporary_model, model_spec)
        _write_json(temporary_manifest, manifest)
        _write_json(temporary_runtime_manifest, _reference_runtime_manifest())

        temporary_model.replace(model_dir)
        temporary_manifest.replace(manifest_path)
        temporary_runtime_manifest.replace(runtime_manifest_path)
    except ModelDownloadError:
        raise
    except OSError as exc:
        raise ModelDownloadError(f"could not publish downloaded model: {exc}") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    validate_model_installation(
        model_dir, model_loader=model_loader, model_spec=model_spec
    )
    return ModelDownloadSummary(
        model_dir=model_dir,
        manifest_path=manifest_path,
        runtime_manifest_path=runtime_manifest_path,
        downloaded=True,
        file_count=len(model_spec.required_files),
    )
