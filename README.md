# Fuzzy Travel Recall Benchmark

[繁體中文](README.zh_TW.md) | English

A reproducible benchmark for retrieving a person's mobility event from
incomplete semantic, temporal-spatial, or relational cues. The repository
contains the complete command-line pipeline, the Tokyo research artifacts,
and the formal Flat/BM25, multilingual Vector, and Graph evaluation results.

## Highlights

- One `main.py` entry point for all five stages.
- Deterministic query generation and complete per-user rankings.
- Dataset-native English/Japanese mixed-language retrieval.
- Event-level Ground Truth with repeated visits retained as distractors.
- Reproducible local model download pinned to an immutable revision.
- Auditable outputs from raw check-ins through final metrics.

This project builds a reproducible incomplete-cue personal mobility recall
benchmark from the Massive-STEPS Tokyo check-in data.

## Quick start

```bash
git clone https://github.com/LaimoKarca/fuzzy-travel-recall-benchmark.git
cd fuzzy-travel-recall-benchmark
python -m venv .venv
```

Activate the environment and install dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
# Linux or macOS
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the full pipeline from the repository root:

```bash
python main.py prepare-cohorts
python main.py prepare-events
python main.py generate-queries
python main.py retrieve-flat
python main.py download-models
python main.py retrieve-vector
python main.py retrieve-graph
python main.py evaluate
```

Use `python main.py --help` or `python main.py <command> --help` to inspect
optional input and output paths. Research thresholds and retrieval parameters
are intentionally fixed by the implementation rather than exposed as CLI
options.

## Requirements

- Python 3.12 or later
- pandas 2.x
- `rank-bm25==0.2.2`
- `sentence-transformers==6.0.1`
- `torch==2.14.0`
- `transformers==5.17.0`
- `huggingface-hub==1.31.0`
- `networkx==3.6.1`
- `scipy==1.18.1`

Install the current dependency set:

```bash
python -m pip install -r requirements.txt
```

## Step 1: prepare user cohorts

Run from the project root:

```bash
python main.py prepare-cohorts
```

The command reads `data/raw/tokyo_checkins.csv` by default. Custom paths can
be supplied when needed:

```bash
python main.py prepare-cohorts --input path/to/checkins.csv --output-dir path/to/prepared
```

The research thresholds are fixed: the main cohort contains users with at
least 10 distinct trails, and the high-memory-load cohort contains users with
at least 20 distinct trails. This step only filters rows. It does not sort the
source data, remove check-ins, or add previous/next event columns.

Generated files in `data/prepared/stage_01_cohorts`:

| File | Description |
|---|---|
| `tokyo_checkins_users_with_at_least_10_trails.csv` | Complete check-in rows for the main cohort |
| `tokyo_checkins_users_with_at_least_20_trails.csv` | Complete check-in rows for the high-memory-load cohort |
| `tokyo_user_trail_and_checkin_counts.csv` | Per-user trail/check-in counts and cohort flags |

Expected counts for the current Tokyo source data:

| Dataset | Users | Trails | Check-ins |
|---|---:|---:|---:|
| Original | 764 | 5,482 | 13,839 |
| Main (at least 10 trails) | 132 | 2,565 | 6,669 |
| High load (at least 20 trails) | 44 | 1,412 | 3,735 |

These are preliminary cohorts based on the source `trail_id` values. They are
inputs to the temporal-integrity cleaning step, not the final experiment data.

## Step 2: prepare canonical events

Run:

```bash
python main.py prepare-events
```

This command reads the preliminary main cohort, orders events within each
source trail, and removes an entire trail when any adjacent timestamp gap is
greater than eight hours. It does not split or rewrite source trail IDs. The
10- and 20-clean-trail thresholds are then reapplied before previous/next event
links are created.

Custom paths can be supplied with `--input` and `--output-dir`.

Generated files in `data/prepared/stage_02_canonical_events`:

| File | Description |
|---|---|
| `tokyo_canonical_events_users_with_at_least_10_clean_trails.csv` | Final main-cohort canonical events |
| `tokyo_canonical_events_users_with_at_least_20_clean_trails.csv` | Final high-load canonical events |
| `tokyo_removed_trails_exceeding_8_hour_gap.csv` | Audit report for removed source trails |
| `tokyo_clean_user_trail_and_checkin_counts.csv` | Post-cleaning user counts and cohort flags |

Expected results:

| Stage | Users | Trails | Events |
|---|---:|---:|---:|
| Preliminary main input | 132 | 2,565 | 6,669 |
| After removing invalid trails | 132 | 2,223 | 5,585 |
| Final main (at least 10 clean trails) | 108 | 2,035 | 5,106 |
| Final high load (at least 20 clean trails) | 35 | 1,072 | 2,760 |

The current data-quality audit removes 342 source trails containing 1,084
check-ins. Raw data and Step 1 outputs remain unchanged.

## Step 3: generate benchmark queries

Run:

```bash
python main.py generate-queries
```

The command reads the final Stage 2 main-cohort canonical events. Custom paths
can be supplied with `--input` and `--output-dir`. It generates deterministic
dataset-native mixed-language queries: templates are English while category,
city, and adjacent-place cue values are copied from the CSV without
translation.

Ground truth is a unique event. Repeated visits remain in the canonical event
search space as distractors; a cue matching multiple events is skipped even
when those events share one venue. Candidate selection uses a fixed SHA-256
rule and does not depend on input row order.

Generated files in `data/prepared/stage_03_queries`:

| File | Description |
|---|---|
| `tokyo_recall_queries_main.csv` | Core Semantic, Temporal-spatial, and Relational-after queries |
| `tokyo_recall_queries_high_memory_load.csv` | High-load subset of the selected core queries |
| `tokyo_relational_between_queries_main.csv` | Optional dual-neighbor relational stress set |
| `tokyo_relational_between_queries_high_memory_load.csv` | High-load subset of the between stress set |
| `tokyo_query_generation_audit.csv` | One audit row per user and query type |
| `tokyo_query_generation_summary.csv` | Counts by benchmark tier, cohort, and query type |

Expected results:

| Set | Semantic | Temporal-spatial | Relational-after | Relational-between | Total |
|---|---:|---:|---:|---:|---:|
| Core main | 105 | 108 | 107 | — | 320 |
| Core high load | 34 | 35 | 35 | — | 104 |
| Between stress main | — | — | — | 98 | 98 |
| Between stress high load | — | — | — | 33 | 33 |

The between set is generated for reproducibility but remains optional for the
deadline. It must not be merged into the core overall score.

## Step 4.1: run Flat/BM25 retrieval

Install the updated dependencies, then run:

```bash
python -m pip install -r requirements.txt
python main.py retrieve-flat
```

The command reads the Stage 2 Main canonical events and the Stage 3 Main Core
and Between query files. Custom paths can be supplied with `--events`,
`--core-queries`, `--stress-queries`, and `--output-dir`.

Every canonical event is rendered as one shared English-framed event document.
Original Japanese names and addresses, English categories, Romanized cities,
timestamps, and previous/next places are preserved. Flat, Vector, and Graph
dense seeding must reuse this exact `event_text`.

BM25 is built separately for each user with fixed parameters `k1=1.5`,
`b=0.75`, and `epsilon=0.25`. Text is normalized with Unicode NFKC and case
folding. Latin text and numbers use word tokens; contiguous Japanese text keeps
the full span and adds character bigrams and trigrams. No translation,
Romanization, morphological dictionary, or model is used.

Generated files in `data/prepared/stage_04_retrieval`:

| File | Description |
|---|---|
| `common/tokyo_event_documents.csv` | One shared text document per canonical event |
| `flat/tokyo_flat_core_rankings.csv` | Complete per-user BM25 rankings for the 320 Core queries |
| `flat/tokyo_flat_between_rankings.csv` | Complete per-user BM25 rankings for the 98 Between stress queries |
| `flat/tokyo_flat_retrieval_summary.csv` | Counts and all-zero-score diagnostics by tier, cohort, and query type |

Expected results for the current data:

| Artifact | Records |
|---|---:|
| Event documents | 5,106 |
| Core queries | 320 |
| Core ranking rows | 15,188 |
| Core high-load queries | 104 |
| Between queries | 98 |
| Between ranking rows | 4,816 |
| Between high-load queries | 33 |
| All-zero-score queries | 0 |

Rankings use long format and include every event belonging to the query user.
They deliberately exclude Ground Truth; Stage 5 will join targets by
`query_id` and calculate MRR and Success metrics. Equal BM25 scores are ordered
by ascending `event_id`, so the output is reproducible even if input rows are
reordered.

## Step 4.2: run Vector retrieval

Download and validate the pinned model once:

```bash
python main.py download-models
```

The command downloads only the 12 safetensors, tokenizer, and Sentence
Transformers files required for CPU inference. It fixes model revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, validates a local offline load,
and writes per-file SHA-256 values to `models/model_manifest.json` plus the
reference Python, OS, and package versions to
`models/reference_runtime_manifest.json`. A valid existing installation is a
no-op for model files; an incomplete or modified installation is reported
without being overwritten.

The model is stored under:

```text
models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/
```

Model weights are ignored by Git and must not be placed in `src`. The small
model manifest remains project metadata.

Run Vector retrieval after Stage 4.1 has created the shared documents:

```bash
python main.py retrieve-vector
```

The retriever uses the local model only (`local_files_only=True`) on CPU with
float32, batch size 32, the native 128-token limit, and normalized 384-dimensional
embeddings. It does not use BM25 tokenization, prompts, fine-tuning, FAISS, a
reranker, target fields, or structured cue fields. Cosine similarity is computed
as a normalized-vector dot product against every event belonging to the query
user. Exact ties use ascending `event_id`.

Generated files in `data/prepared/stage_04_retrieval/vector`:

| File | Description |
|---|---|
| `tokyo_vector_event_embeddings.npy` | 5,106 event embeddings, shape `(5106, 384)` |
| `tokyo_vector_event_embedding_index.csv` | Event ID and user mapping for each 0-based embedding row |
| `tokyo_vector_core_query_embeddings.npy` | 320 Core query embeddings, shape `(320, 384)` |
| `tokyo_vector_core_query_embedding_index.csv` | Core query mapping for each embedding row |
| `tokyo_vector_between_query_embeddings.npy` | 98 Between query embeddings, shape `(98, 384)` |
| `tokyo_vector_between_query_embedding_index.csv` | Between query mapping for each embedding row |
| `tokyo_vector_core_rankings.csv` | 15,188 complete per-user Vector ranking rows |
| `tokyo_vector_between_rankings.csv` | 4,816 complete per-user Vector ranking rows |
| `tokyo_vector_retrieval_summary.csv` | Counts, model settings, truncations, and equal-score diagnostics |
| `tokyo_vector_run_manifest.json` | Input, model, output, parameter, and runtime hashes/versions |

The current run contains 104 High-load Core queries and 33 High-load Between
queries. No event document or query was truncated, and no query produced equal
scores for all candidates. Repeating the run in the same CPU environment
produced identical hashes for all ten outputs. The rankings themselves exclude
Ground Truth; the formal evaluator in Step 5 performs the controlled join.

## Step 4.3: run Graph retrieval

Run after Stage 4.2:

```bash
python main.py retrieve-graph
```

The command accepts `--events`, `--vector-dir`, and `--output-dir`. It validates
the Stage 4.2 run manifest and every referenced hash, then recomputes the dense
rankings from the saved embeddings. It does not load MiniLM, encode text, read
query Ground Truth, or use structured query cues.

`build_graph.py` creates a directed NetworkX graph with User, Event, POI,
Category, and Trail nodes. Its relations are `HAS`, `AT`, `CATEGORY`,
`IN_TRAIL`, and `NEXT`; `PREVIOUS` is obtained by traversing `NEXT` backwards.
The current graph has 9,030 nodes and 19,942 edges, including 3,071 `NEXT`
edges.

For every query type, the retriever takes the existing Top-5 dense seeds and
performs one symmetric hop over previous and next events. Traversal order is
fixed by seed rank, `SELF`, and neighbor `event_id`; the first discovery fixes
the expansion rank. Dense and expansion ranks are combined with equal-weight
RRF using `k=60`, while every event belonging to the query user remains in the
final ranking.

Generated files in `data/prepared/stage_04_retrieval/graph`:

| File | Description |
|---|---|
| `tokyo_graph_nodes.csv` | Portable typed-node table for rebuilding the graph |
| `tokyo_graph_edges.csv` | Typed directed-edge table |
| `tokyo_graph_core_expansion_audit.csv` | 3,629 Core seed/self/neighbor traversals |
| `tokyo_graph_between_expansion_audit.csv` | 1,183 Between traversals |
| `tokyo_graph_core_rankings.csv` | 15,188 complete Core Graph ranking rows |
| `tokyo_graph_between_rankings.csv` | 4,816 complete Between Graph ranking rows |
| `tokyo_graph_retrieval_summary.csv` | Graph, expansion, query, and cohort counts |
| `tokyo_graph_run_manifest.json` | Input/output hashes, parameters, and runtime versions |

The Core and Between audits contain 3,313 and 1,039 first-discovered expanded
candidates, respectively. High-load queries remain subsets of the Main runs
(104 Core and 33 Between); they are not executed separately.

## Step 5: run formal evaluation

Run after all three retrieval configurations are complete:

```bash
python main.py evaluate
```

The evaluator validates the Stage 3 query targets, complete per-user candidate
rankings, Vector and Graph manifests, artifact hashes, cohort flags, and user
boundaries before joining Ground Truth. Custom inputs can be supplied with
`--events`, `--core-queries`, `--between-queries`, `--flat-dir`,
`--vector-dir`, `--graph-dir`, and `--output-dir`.

Formal results are written to `data/outputs/stage_05_evaluation`:

| File | Description |
|---|---|
| `tokyo_per_query_metrics.csv` | 1,254 query-method target ranks and metrics |
| `tokyo_core_overall_metrics.csv` | Main Core MRR and Success results |
| `tokyo_core_metrics_by_query_type.csv` | Main Core results by cue type |
| `tokyo_high_memory_load_metrics.csv` | Main and High-load Core comparison |
| `tokyo_between_stress_metrics.csv` | Main and High-load Between results |
| `tokyo_pairwise_wilcoxon.csv` | User-level paired Wilcoxon tests and Holm correction |
| `tokyo_graph_query_diagnostics.csv` | All 418 Graph comparison cases |
| `tokyo_graph_diagnostic_summary.csv` | Graph outcomes, expansion, and repeated-visit counts |
| `tokyo_evaluation_run_manifest.json` | Input/output hashes and fixed evaluation settings |

Core Main results:

| Method | MRR | Success@1 | Success@5 |
|---|---:|---:|---:|
| Flat / BM25 | 0.9338 | 0.8875 | 0.9938 |
| Vector | 0.7908 | 0.6906 | 0.9344 |
| Graph | 0.7883 | 0.6906 | 0.9281 |

The fixed Graph configuration improves 7 of 418 target ranks relative to
Vector, leaves 378 unchanged, and worsens 33. Of 396 targets present in the
expansion list, 384 are existing dense seeds discovered as `SELF`; only 12 are
first discovered through `PREVIOUS` or `NEXT`. These diagnostics do not support
a general claim that Graph resolves repeated visits. Graph-minus-NEXT and
post-result parameter tuning are intentionally left for future work.

## Tests

```bash
python -m unittest discover -v
```

## Repository layout

```text
.
├── main.py                         # Unified CLI entry point
├── src/                            # Pipeline implementation
├── tests/                          # Unit and integration tests
├── data/
│   ├── raw/                        # Massive-STEPS Tokyo source CSV
│   ├── prepared/                   # Stages 1–4 reproducible artifacts
│   └── outputs/stage_05_evaluation # Formal evaluation results
├── models/                         # Model metadata; weights are Git-ignored
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Reproducibility and privacy

The three Stage 4/5 run manifests are intentionally excluded from Git because
they record machine-specific absolute paths. This does not remove the actual
rankings or formal result CSVs. Portable model metadata remains in
`models/model_manifest.json` and `models/reference_runtime_manifest.json`.

The repository contains anonymized identifiers supplied by the upstream
dataset. Do not attempt to re-identify individuals or combine the data with
external sources for that purpose.

## Data and model attribution

The Tokyo check-ins originate from
[Massive-STEPS](https://github.com/CRUISEResearchGroup/Massive-STEPS), by
Wilson Wongso, Hao Xue, and Flora D. Salim. The upstream repository is
distributed under Apache-2.0 and is based on Semantic Trails plus POI metadata.
Please cite the dataset paper when using these files:

```bibtex
@misc{wongso2025massivesteps,
  title         = {Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins -- Dataset and Benchmarks},
  author        = {Wilson Wongso and Hao Xue and Flora D. Salim},
  year          = {2025},
  eprint        = {2505.11239},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.11239}
}
```

Vector retrieval uses
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
licensed under Apache-2.0. Its weights are not stored in Git.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and
license boundaries.

## License

Project-authored source code and documentation are available under the
[MIT License](LICENSE). Third-party data, model artifacts, and dependencies
retain their respective licenses and terms.
