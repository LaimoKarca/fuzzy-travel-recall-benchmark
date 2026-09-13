# Local model storage

Downloaded model weights belong under this directory, outside `src`:

```
models/
└── sentence-transformers/
    ├── paraphrase-multilingual-MiniLM-L12-v2/
    └── multilingual-e5-small/
```

Stage 4.1 Flat/BM25 does not use a model. Download the Stage 4.2 model with:

```bash
python main.py download-models
```

Download the candidate E5 encoder separately with:

```bash
python main.py download-models --model multilingual-e5-small
```

The command fixes Hugging Face revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42` and downloads only the 12 files
needed by the safetensors/Torch backend. It validates the model on CPU and
writes file hashes to `models/model_manifest.json` and reference environment
versions to `models/reference_runtime_manifest.json`. Vector and Graph
retrieval load only from this local path with network fallback disabled.

The E5 candidate is fixed to `intfloat/multilingual-e5-small` revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Its hashes are stored in
`models/multilingual_e5_small_model_manifest.json`, while reference runtime
versions are stored in `models/multilingual_e5_small_runtime_manifest.json`.
E5 retrieval uses `query: ` and `passage: ` prefixes as required by the model.

Model weights are intentionally ignored by Git. The download command, pinned
revision, dependency versions, Apache-2.0 license information, and generated
manifest are reproducible project metadata. If a local file no longer matches
the manifest, remove or repair the model installation explicitly before
downloading again; the command will not silently overwrite it.

MiniLM is licensed under Apache-2.0. Multilingual E5-small is licensed under
MIT. Neither model's weights are committed to this repository.
