"""Download and validate the fixed multilingual MiniLM research model."""

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


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_DIRECTORY_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EXPECTED_DIMENSION = 384
EXPECTED_MAX_SEQUENCE_LENGTH = 128
MODEL_MANIFEST_FILENAME = "model_manifest.json"
REFERENCE_RUNTIME_MANIFEST_FILENAME = "reference_runtime_manifest.json"
REQUIRED_MODEL_FILES = (
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "unigram.json",
)


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


def manifest_path_for_model_dir(model_dir: Path) -> Path:
    if model_dir.parent.name == "sentence-transformers":
        return model_dir.parent.parent / MODEL_MANIFEST_FILENAME
    return model_dir.parent / MODEL_MANIFEST_FILENAME


def runtime_manifest_path_for_model_dir(model_dir: Path) -> Path:
    return manifest_path_for_model_dir(model_dir).with_name(
        REFERENCE_RUNTIME_MANIFEST_FILENAME
    )


def _stable_manifest(model_dir: Path) -> dict[str, Any]:
    return {
        "backend": "torch",
        "embedding_dimension": EXPECTED_DIMENSION,
        "files": {
            relative: sha256_file(model_dir / relative)
            for relative in REQUIRED_MODEL_FILES
        },
        "license": "Apache-2.0",
        "max_seq_length": EXPECTED_MAX_SEQUENCE_LENGTH,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
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


def _validate_model_runtime(model: Any) -> None:
    get_dimension = getattr(model, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = model.get_sentence_embedding_dimension
    dimension = get_dimension()
    if dimension != EXPECTED_DIMENSION:
        raise ModelDownloadError(
            f"model embedding dimension is {dimension}, expected {EXPECTED_DIMENSION}"
        )
    max_sequence_length = int(model.max_seq_length)
    if max_sequence_length != EXPECTED_MAX_SEQUENCE_LENGTH:
        raise ModelDownloadError(
            "model max sequence length is "
            f"{max_sequence_length}, expected {EXPECTED_MAX_SEQUENCE_LENGTH}"
        )


def validate_model_installation(
    model_dir: str | Path,
    *,
    load_model: bool = True,
    model_loader: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Validate required files, pinned metadata, hashes, and offline loading."""

    model_dir = Path(model_dir)
    manifest_path = manifest_path_for_model_dir(model_dir)
    if not model_dir.is_dir():
        raise ModelDownloadError(f"model directory does not exist: {model_dir}")
    if not manifest_path.is_file():
        raise ModelDownloadError(f"model manifest does not exist: {manifest_path}")

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise ModelDownloadError(
            "model installation is incomplete; missing: " + ", ".join(missing)
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelDownloadError(f"could not parse model manifest: {exc}") from exc

    expected_metadata = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "torch",
        "embedding_dimension": EXPECTED_DIMENSION,
        "max_seq_length": EXPECTED_MAX_SEQUENCE_LENGTH,
    }
    mismatches = [
        key for key, expected in expected_metadata.items() if manifest.get(key) != expected
    ]
    if mismatches:
        raise ModelDownloadError(
            "model manifest metadata mismatch in: " + ", ".join(mismatches)
        )

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise ModelDownloadError("model manifest has no valid files mapping")
    for relative in REQUIRED_MODEL_FILES:
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
        _validate_model_runtime(model)
    return manifest, model


def _snapshot_download(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as exc:
        raise ModelDownloadError(
            "huggingface-hub is not available; run "
            "'python -m pip install -r requirements.txt'"
        ) from exc
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=model_dir,
            allow_patterns=list(REQUIRED_MODEL_FILES),
        )
    except Exception as exc:
        raise ModelDownloadError(
            f"could not download {MODEL_ID} at revision {MODEL_REVISION}: {exc}"
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
    snapshot_downloader: Callable[[Path], None] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> ModelDownloadSummary:
    """Download the pinned model to a temporary directory, then publish it."""

    model_dir = Path(output_dir)
    manifest_path = manifest_path_for_model_dir(model_dir)
    runtime_manifest_path = runtime_manifest_path_for_model_dir(model_dir)
    if model_dir.exists() or manifest_path.exists():
        validate_model_installation(model_dir, model_loader=model_loader)
        _write_json_atomically(runtime_manifest_path, _reference_runtime_manifest())
        return ModelDownloadSummary(
            model_dir=model_dir,
            manifest_path=manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            downloaded=False,
            file_count=len(REQUIRED_MODEL_FILES),
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".model-download-", dir=model_dir.parent)
    )
    temporary_model = temporary_root / model_dir.name
    temporary_model.mkdir()
    temporary_manifest = temporary_root / MODEL_MANIFEST_FILENAME
    temporary_runtime_manifest = temporary_root / REFERENCE_RUNTIME_MANIFEST_FILENAME

    try:
        (snapshot_downloader or _snapshot_download)(temporary_model)
        cache_dir = temporary_model / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        missing = [
            name for name in REQUIRED_MODEL_FILES if not (temporary_model / name).is_file()
        ]
        if missing:
            raise ModelDownloadError(
                "downloaded model is incomplete; missing: " + ", ".join(missing)
            )

        model = (model_loader or _load_sentence_transformer)(temporary_model)
        _validate_model_runtime(model)
        del model

        manifest = _stable_manifest(temporary_model)
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

    validate_model_installation(model_dir, model_loader=model_loader)
    return ModelDownloadSummary(
        model_dir=model_dir,
        manifest_path=manifest_path,
        runtime_manifest_path=runtime_manifest_path,
        downloaded=True,
        file_count=len(REQUIRED_MODEL_FILES),
    )
