"""Pinned local embedding-model specifications used by retrieval stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    directory_name: str
    embedding_dimension: int
    max_sequence_length: int
    license: str
    required_files: tuple[str, ...]
    query_prefix: str = ""
    document_prefix: str = ""
    manifest_filename: str = "model_manifest.json"
    runtime_manifest_filename: str = "reference_runtime_manifest.json"


MINILM_SPEC = ModelSpec(
    key="multilingual-minilm",
    model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    directory_name="paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dimension=384,
    max_sequence_length=128,
    license="Apache-2.0",
    required_files=(
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
    ),
)

E5_SMALL_SPEC = ModelSpec(
    key="multilingual-e5-small",
    model_id="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    directory_name="multilingual-e5-small",
    embedding_dimension=384,
    max_sequence_length=512,
    license="MIT",
    required_files=(
        "1_Pooling/config.json",
        "README.md",
        "config.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    query_prefix="query: ",
    document_prefix="passage: ",
    manifest_filename="multilingual_e5_small_model_manifest.json",
    runtime_manifest_filename="multilingual_e5_small_runtime_manifest.json",
)

# E5 is the formal retrieval-oriented multilingual encoder.  MiniLM remains
# available for reproducing the archived encoder-sensitivity run.
DEFAULT_MODEL_KEY = E5_SMALL_SPEC.key
MODEL_SPECS = {
    MINILM_SPEC.key: MINILM_SPEC,
    E5_SMALL_SPEC.key: E5_SMALL_SPEC,
}
MODEL_SPECS_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS.values()}
MODEL_SPECS_BY_DIRECTORY = {
    spec.directory_name: spec for spec in MODEL_SPECS.values()
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported model key: {key}") from exc


def model_spec_for_directory(directory_name: str) -> ModelSpec:
    return MODEL_SPECS_BY_DIRECTORY.get(directory_name, MINILM_SPEC)


def model_spec_for_id(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"unsupported model ID: {model_id}") from exc
