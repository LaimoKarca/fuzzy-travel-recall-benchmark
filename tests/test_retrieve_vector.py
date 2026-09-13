from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_download import (
    EXPECTED_DIMENSION,
    EXPECTED_MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_MODEL_FILES,
)
from src.retrieve_vector import VectorRetrievalError, retrieve_vector
from src.model_specs import E5_SMALL_SPEC


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Tokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [
                list(range(129 if "LONG" in text else max(2, len(text.split()) + 2)))
                for text in texts
            ]
        }


class _MockModel:
    max_seq_length = EXPECTED_MAX_SEQUENCE_LENGTH
    tokenizer = _Tokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return EXPECTED_DIMENSION

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(EXPECTED_DIMENSION, dtype=np.float32)
            if "apple" in text:
                vector[0] = 1.0
            elif "banana" in text:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            rows.append(vector)
        return np.stack(rows)


class _RecordingE5Model(_MockModel):
    max_seq_length = E5_SMALL_SPEC.max_sequence_length

    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encoded_batches.append(list(texts))
        return super().encode(texts, **kwargs)


class VectorRetrievalTests(unittest.TestCase):
    def _model(self, root: Path) -> Path:
        model_dir = (
            root
            / "models"
            / "sentence-transformers"
            / "paraphrase-multilingual-MiniLM-L12-v2"
        )
        hashes = {}
        for relative in REQUIRED_MODEL_FILES:
            path = model_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
            hashes[relative] = _sha256(path)
        manifest = {
            "backend": "torch",
            "embedding_dimension": EXPECTED_DIMENSION,
            "files": hashes,
            "license": "Apache-2.0",
            "max_seq_length": EXPECTED_MAX_SEQUENCE_LENGTH,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
        }
        (root / "models" / "model_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return model_dir

    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        documents = pd.DataFrame(
            [
                {"event_id": "e2", "user_id": "u1", "event_text": "banana", "is_high_memory_load": "False"},
                {"event_id": "e1", "user_id": "u1", "event_text": "apple LONG", "is_high_memory_load": "False"},
                {"event_id": "e3", "user_id": "u2", "event_text": "apple", "is_high_memory_load": "True"},
            ]
        )
        core = pd.DataFrame(
            [
                {"query_id": "q1", "user_id": "u1", "benchmark_tier": "core", "query_type": "semantic", "query_text": "apple", "is_high_memory_load": "False", "target_event_id": "e2"},
                {"query_id": "q2", "user_id": "u2", "benchmark_tier": "core", "query_type": "semantic", "query_text": "apple LONG", "is_high_memory_load": "True", "target_event_id": "e1"},
            ]
        )
        stress = pd.DataFrame(
            [
                {"query_id": "q3", "user_id": "u1", "benchmark_tier": "stress_test", "query_type": "relational_between", "query_text": "unknown", "is_high_memory_load": "False", "target_event_id": "e1"},
            ]
        )
        paths = root / "documents.csv", root / "core.csv", root / "stress.csv"
        for frame, path in zip((documents, core, stress), paths, strict=True):
            frame.to_csv(path, index=False, encoding="utf-8")
        return paths

    def _e5_model(self, root: Path) -> Path:
        model_dir = (
            root
            / "models"
            / "sentence-transformers"
            / E5_SMALL_SPEC.directory_name
        )
        hashes = {}
        for relative in E5_SMALL_SPEC.required_files:
            path = model_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative, encoding="utf-8")
            hashes[relative] = _sha256(path)
        manifest = {
            "backend": "torch",
            "document_prefix": E5_SMALL_SPEC.document_prefix,
            "embedding_dimension": E5_SMALL_SPEC.embedding_dimension,
            "files": hashes,
            "license": E5_SMALL_SPEC.license,
            "max_seq_length": E5_SMALL_SPEC.max_sequence_length,
            "model_id": E5_SMALL_SPEC.model_id,
            "query_prefix": E5_SMALL_SPEC.query_prefix,
            "revision": E5_SMALL_SPEC.revision,
        }
        (root / "models" / E5_SMALL_SPEC.manifest_filename).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return model_dir

    def test_embeddings_indexes_rankings_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents, core, stress = self._inputs(root)
            model_dir = self._model(root)
            output = root / "vector"
            summary = retrieve_vector(
                documents, core, stress, model_dir, output, model=_MockModel()
            )

            self.assertEqual(summary.document_count, 3)
            self.assertEqual(summary.core_ranking_count, 3)
            self.assertEqual(summary.stress_ranking_count, 2)
            self.assertEqual(summary.truncated_document_count, 1)
            self.assertEqual(summary.truncated_core_query_count, 1)
            self.assertEqual(np.load(output / "tokyo_vector_event_embeddings.npy").shape, (3, 384))

            event_index = pd.read_csv(output / "tokyo_vector_event_embedding_index.csv")
            self.assertEqual(event_index["event_id"].tolist(), ["e1", "e2", "e3"])
            ranking = pd.read_csv(output / "tokyo_vector_core_rankings.csv")
            q1 = ranking.loc[ranking["query_id"].eq("q1")]
            self.assertEqual(q1["retrieved_event_id"].tolist(), ["e1", "e2"])
            self.assertEqual(q1["rank"].tolist(), [1, 2])
            self.assertEqual(set(ranking["method"]), {"vector"})
            self.assertTrue((q1["user_id"] == "u1").all())

            manifest = json.loads(
                (output / "tokyo_vector_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["parameters"]["device"], "cpu")
            self.assertIn("tokyo_vector_event_embeddings.npy", manifest["outputs"])

    def test_ground_truth_metadata_does_not_affect_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents, core, stress = self._inputs(root)
            model_dir = self._model(root)
            first = root / "first"
            second = root / "second"
            retrieve_vector(documents, core, stress, model_dir, first, model=_MockModel())
            core_frame = pd.read_csv(core, dtype=str)
            core_frame["target_event_id"] = "changed"
            core_frame["cue_city"] = "secret"
            core_frame.to_csv(core, index=False, encoding="utf-8")
            retrieve_vector(documents, core, stress, model_dir, second, model=_MockModel())

            for filename in (
                "tokyo_vector_core_query_embeddings.npy",
                "tokyo_vector_core_query_embedding_index.csv",
                "tokyo_vector_core_rankings.csv",
            ):
                self.assertEqual(_sha256(first / filename), _sha256(second / filename))

    def test_unknown_user_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents, core, stress = self._inputs(root)
            model_dir = self._model(root)
            frame = pd.read_csv(core, dtype=str)
            frame.loc[0, "user_id"] = "missing"
            frame.to_csv(core, index=False, encoding="utf-8")
            output = root / "vector"

            with self.assertRaisesRegex(VectorRetrievalError, "absent"):
                retrieve_vector(documents, core, stress, model_dir, output, model=_MockModel())
            self.assertFalse(output.exists())

    def test_e5_applies_query_and_passage_prefixes_without_rewriting_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents, core, stress = self._inputs(root)
            original_documents = documents.read_bytes()
            original_core = core.read_bytes()
            model_dir = self._e5_model(root)
            model = _RecordingE5Model()
            output = root / "e5-vector"

            retrieve_vector(
                documents, core, stress, model_dir, output, model=model
            )

            self.assertTrue(
                all(text.startswith("passage: ") for text in model.encoded_batches[0])
            )
            self.assertTrue(
                all(text.startswith("query: ") for text in model.encoded_batches[1])
            )
            self.assertTrue(
                all(text.startswith("query: ") for text in model.encoded_batches[2])
            )
            self.assertEqual(documents.read_bytes(), original_documents)
            self.assertEqual(core.read_bytes(), original_core)
            manifest = json.loads(
                (output / "tokyo_vector_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["model"]["id"], E5_SMALL_SPEC.model_id)
            self.assertEqual(manifest["parameters"]["query_prefix"], "query: ")
            self.assertEqual(manifest["parameters"]["document_prefix"], "passage: ")


if __name__ == "__main__":
    unittest.main()
