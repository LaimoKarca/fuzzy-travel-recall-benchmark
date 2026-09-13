# Local model storage

Downloaded model weights belong under this directory, outside `src`:

```
models/
└── sentence-transformers/
    └── paraphrase-multilingual-MiniLM-L12-v2/
```

Stage 4.1 Flat/BM25 does not use a model. Download the Stage 4.2 model with:

```bash
python main.py download-models
```

The command fixes Hugging Face revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42` and downloads only the 12 files
needed by the safetensors/Torch backend. It validates the model on CPU and
writes file hashes to `models/model_manifest.json` and reference environment
versions to `models/reference_runtime_manifest.json`. Vector and Graph
retrieval load only from this local path with network fallback disabled.

Model weights are intentionally ignored by Git. The download command, pinned
revision, dependency versions, Apache-2.0 license information, and generated
manifest are reproducible project metadata. If a local file no longer matches
the manifest, remove or repair the model installation explicitly before
downloading again; the command will not silently overwrite it.
