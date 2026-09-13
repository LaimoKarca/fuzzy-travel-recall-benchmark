from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.model_download import (
    EXPECTED_DIMENSION,
    EXPECTED_MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_MODEL_FILES,
    ModelDownloadError,
    download_models,
)
from src.model_specs import E5_SMALL_SPEC, MINILM_SPEC


class _ValidModel:
    max_seq_length = EXPECTED_MAX_SEQUENCE_LENGTH

    def get_sentence_embedding_dimension(self) -> int:
        return EXPECTED_DIMENSION


class _ValidE5Model:
    max_seq_length = E5_SMALL_SPEC.max_sequence_length

    def get_sentence_embedding_dimension(self) -> int:
        return E5_SMALL_SPEC.embedding_dimension


class ModelDownloadTests(unittest.TestCase):
    def _downloader(self, destination: Path) -> None:
        for relative in REQUIRED_MODEL_FILES:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixed:{relative}".encode("utf-8"))
        cache = destination / ".cache" / "ignored"
        cache.parent.mkdir(parents=True)
        cache.write_text("cache", encoding="utf-8")

    def test_download_manifest_no_op_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = (
                Path(directory)
                / "models"
                / "sentence-transformers"
                / "paraphrase-multilingual-MiniLM-L12-v2"
            )
            result = download_models(
                model_dir,
                model_name=MINILM_SPEC.key,
                snapshot_downloader=self._downloader,
                model_loader=lambda _: _ValidModel(),
            )

            self.assertTrue(result.downloaded)
            self.assertFalse((model_dir / ".cache").exists())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_id"], MODEL_ID)
            self.assertEqual(manifest["revision"], MODEL_REVISION)
            self.assertEqual(set(manifest["files"]), set(REQUIRED_MODEL_FILES))
            self.assertNotIn("timestamp", manifest)
            runtime = json.loads(
                result.runtime_manifest_path.read_text(encoding="utf-8")
            )
            self.assertIn("python", runtime)
            self.assertIn("torch", runtime["packages"])
            self.assertNotIn("timestamp", runtime)

            second = download_models(
                model_dir,
                model_name=MINILM_SPEC.key,
                snapshot_downloader=lambda _: self.fail("unexpected download"),
                model_loader=lambda _: _ValidModel(),
            )
            self.assertFalse(second.downloaded)

            (model_dir / "config.json").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ModelDownloadError, "hash mismatch"):
                download_models(
                    model_dir,
                    model_name=MINILM_SPEC.key,
                    snapshot_downloader=lambda _: self.fail("unexpected overwrite"),
                    model_loader=lambda _: _ValidModel(),
                )

    def test_incomplete_download_leaves_no_final_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"

            def incomplete(destination: Path) -> None:
                (destination / "README.md").write_text("only one", encoding="utf-8")

            with self.assertRaisesRegex(ModelDownloadError, "incomplete"):
                download_models(
                    model_dir,
                    snapshot_downloader=incomplete,
                    model_loader=lambda _: _ValidModel(),
                )
            self.assertFalse(model_dir.exists())
            self.assertFalse((Path(directory) / "model_manifest.json").exists())

    def test_e5_uses_pinned_files_prefixes_and_separate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root
                / "models"
                / "sentence-transformers"
                / E5_SMALL_SPEC.directory_name
            )

            def downloader(destination: Path) -> None:
                for relative in E5_SMALL_SPEC.required_files:
                    path = destination / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"e5:{relative}", encoding="utf-8")

            result = download_models(
                model_dir,
                model_name=E5_SMALL_SPEC.key,
                snapshot_downloader=downloader,
                model_loader=lambda _: _ValidE5Model(),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_id"], E5_SMALL_SPEC.model_id)
            self.assertEqual(manifest["revision"], E5_SMALL_SPEC.revision)
            self.assertEqual(manifest["query_prefix"], "query: ")
            self.assertEqual(manifest["document_prefix"], "passage: ")
            self.assertEqual(manifest["max_seq_length"], 512)
            self.assertEqual(
                result.manifest_path.name, E5_SMALL_SPEC.manifest_filename
            )
            self.assertNotEqual(result.manifest_path.name, "model_manifest.json")


if __name__ == "__main__":
    unittest.main()
