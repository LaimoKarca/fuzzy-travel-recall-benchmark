# 模糊旅遊記憶檢索基準

[English](README.md) | 繁體中文

本專案建立一套可重現的個人移動記憶檢索基準，目標是在使用者只記得不完整的語意、時間空間或前後地點線索時，找回唯一的造訪事件。倉庫包含完整 CLI 管線、Massive-STEPS Tokyo 研究資料，以及 Flat／BM25、多語 Vector 與 Graph 三種方法的正式評估結果。

## 專案特色

- 所有階段都由 `main.py` 作為統一 CLI 入口。
- 固定研究門檻、抽樣規則與檢索參數，避免執行時任意改動。
- 支援資料集原生的英文／日文混合文字。
- Ground Truth 為唯一 `target_event_id`，重複造訪保留為 distractors。
- MiniLM 固定 Hugging Face revision，下載後只從本機離線載入。
- 從原始 check-ins、清洗、queries、完整 rankings 到正式 metrics 都可稽核。

## 快速開始

```bash
git clone https://github.com/LaimoKarca/fuzzy-travel-recall-benchmark.git
cd fuzzy-travel-recall-benchmark
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux／macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依序執行完整管線：

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

可用以下方式查看所有指令，或查看個別指令的自訂輸入／輸出路徑：

```bash
python main.py --help
python main.py <command> --help
```

研究門檻與檢索參數刻意固定於程式碼中，不開放成 CLI 參數。

## 執行環境

- Python 3.12 或更新版本
- pandas 2.x
- `rank-bm25==0.2.2`
- `sentence-transformers==6.0.1`
- `torch==2.14.0`
- `transformers==5.17.0`
- `huggingface-hub==1.31.0`
- `networkx==3.6.1`
- `scipy==1.18.1`

完整版本以 [requirements.txt](requirements.txt) 為準。

## Stage 1：初步 Cohort 篩選

```bash
python main.py prepare-cohorts
```

預設輸入為 `data/raw/tokyo_checkins.csv`，輸出至 `data/prepared/stage_01_cohorts/`。此階段依每位使用者的 distinct `trail_id` 數量建立 ≥10 與 ≥20 trails 的初步 cohort，不排序、不清除 check-ins，也不建立 previous／next event。

| 資料集 | Users | Trails | Check-ins |
|---|---:|---:|---:|
| 原始 Tokyo | 764 | 5,482 | 13,839 |
| 初步 Main ≥10 | 132 | 2,565 | 6,669 |
| 初步 High-load ≥20 | 44 | 1,412 | 3,735 |

主要輸出：

- `tokyo_checkins_users_with_at_least_10_trails.csv`
- `tokyo_checkins_users_with_at_least_20_trails.csv`
- `tokyo_user_trail_and_checkin_counts.csv`

## Stage 2：Canonical Events 與八小時完整性檢查

```bash
python main.py prepare-events
```

此階段依 user、trail、timestamp 與原始列序穩定排序。若 source trail 內任一相鄰 timestamp gap 大於八小時，整條 trail 會被排除；`gap == 8 hours` 仍保留。清洗後重新套用 ≥10／≥20 clean trails 門檻，再建立 `event_id`、event position、previous／next references 與 target eligibility。

輸出位置：`data/prepared/stage_02_canonical_events/`

| 階段 | Users | Trails | Events |
|---|---:|---:|---:|
| 初步 Main 輸入 | 132 | 2,565 | 6,669 |
| 移除異常 trails 後 | 132 | 2,223 | 5,585 |
| 正式 Main ≥10 clean trails | 108 | 2,035 | 5,106 |
| 正式 High-load ≥20 clean trails | 35 | 1,072 | 2,760 |

主要輸出：

- `tokyo_canonical_events_users_with_at_least_10_clean_trails.csv`
- `tokyo_canonical_events_users_with_at_least_20_clean_trails.csv`
- `tokyo_removed_trails_exceeding_8_hour_gap.csv`
- `tokyo_clean_user_trail_and_checkin_counts.csv`

本次資料稽核共排除 342 條 source trails、1,084 筆 check-ins。Raw 與 Stage 1 輸出不會被修改。

## Stage 3：產生 Queries 與 Ground Truth

```bash
python main.py generate-queries
```

輸出位置：`data/prepared/stage_03_queries/`

核心 benchmark 包含三種 deterministic query：

| Query type | 唯一性線索 | Main | High-load |
|---|---|---:|---:|
| Semantic | category + city | 105 | 34 |
| Temporal-spatial | year + month + city + category | 108 | 35 |
| Relational-after | previous place + category | 107 | 35 |
| **Core total** | — | **320** | **104** |

另產生獨立的 `relational_between` stress test：Main 98 題、High-load 33 題，不併入 Core overall score。

模板使用英文框架，但 category、city 與相鄰 POI 等線索保留 CSV 原文。Cue 經 Unicode NFKC、case folding 與空白正規化後，必須在該使用者全部 canonical events 中恰好命中一個 event；target name 不得洩漏至 query。固定 SHA-256 抽樣不受輸入列順序影響。

主要輸出：

- `tokyo_recall_queries_main.csv`
- `tokyo_recall_queries_high_memory_load.csv`
- `tokyo_relational_between_queries_main.csv`
- `tokyo_relational_between_queries_high_memory_load.csv`
- `tokyo_query_generation_audit.csv`
- `tokyo_query_generation_summary.csv`

## Stage 4.1：Flat／BM25

```bash
python main.py retrieve-flat
```

Flat 不需要模型。每個 canonical event 會先渲染成共用 `event_text`；Flat、Vector 與 Graph dense seed 都使用相同文字。BM25 以每位使用者建立獨立 index，固定 `k1=1.5`、`b=0.75`、`epsilon=0.25`。Tokenizer 使用 NFKC、case folding、拉丁文字／數字 word tokens，以及日文完整片段與 character bigrams／trigrams。

主要輸出：

- `data/prepared/stage_04_retrieval/common/tokyo_event_documents.csv`：5,106 documents。
- `flat/tokyo_flat_core_rankings.csv`：15,188 rows。
- `flat/tokyo_flat_between_rankings.csv`：4,816 rows。
- `flat/tokyo_flat_retrieval_summary.csv`。

每題都保留該使用者所有事件的完整排名；Ground Truth 不會被寫入 ranking CSV。

## Stage 4.2：Vector／Multilingual MiniLM

先下載並驗證固定模型：

```bash
python main.py download-models
```

模型位置：

```text
models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/
```

固定 revision：`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`。模型權重不提交 Git；其他研究者可用同一指令下載，並透過 `models/model_manifest.json` 的 SHA-256 驗證。

接著執行：

```bash
python main.py retrieve-vector
```

正式設定為 CPU、float32、batch size 32、384 維 L2-normalized embeddings、模型原生 128-token 上限。Query 與 document 原文直接交由模型 tokenizer，不使用 BM25 n-grams、prompt、fine-tuning、FAISS 或 reranker。

主要輸出位於 `data/prepared/stage_04_retrieval/vector/`：三組 `.npy` embeddings、三份 row-index CSV、Core／Between 完整 rankings 與 summary。預期 shapes 為 events `(5106, 384)`、Core `(320, 384)`、Between `(98, 384)`。

## Stage 4.3：Graph／NetworkX

```bash
python main.py retrieve-graph
```

Graph 不重新載入 MiniLM 或 encode 文字，而是驗證並重用 Stage 4.2 embeddings 與 dense rankings。異質有向圖包含 User、Event、POI、Category、Trail 節點，以及 HAS、AT、CATEGORY、IN_TRAIL、NEXT edges；PREVIOUS 由反向遍歷 NEXT 取得。

每題固定取 Vector Top-5 seeds，對稱擴展 SELF／PREVIOUS／NEXT 一跳，再以等權 RRF、`k=60` 合併 dense rank 與 expansion rank。所有 user events 仍保留於最終完整排名。

圖規模：9,030 nodes、19,942 edges，其中 NEXT 3,071 條。主要輸出位於 `data/prepared/stage_04_retrieval/graph/`，包含 portable nodes／edges CSV、Core／Between expansion audit、完整 rankings 與 summary。

## Stage 5：正式評估

```bash
python main.py evaluate
```

正式結果寫入 `data/outputs/stage_05_evaluation/`。Evaluator 在 join Ground Truth 前會驗證 query、target、user boundary、rank continuity、三方法 candidate set，以及 Vector／Graph artifact hashes。

Core Main 結果：

| Method | MRR | Success@1 | Success@5 |
|---|---:|---:|---:|
| Flat／BM25 | 0.9338 | 0.8875 | 0.9938 |
| Vector | 0.7908 | 0.6906 | 0.9344 |
| Graph | 0.7883 | 0.6906 | 0.9281 |

固定 Graph 配置在全部 418 題中，相對 Vector 有 7 題改善、378 題不變、33 題退步。396 個出現在 expansion list 的 targets 中，有 384 個本來就是 SELF dense seeds，只有 12 個首次透過 PREVIOUS 或 NEXT 找到。因此目前結果不支持「Graph 普遍能解決 repeated visits」的結論，也未進行事後 Graph 調參。

主要輸出：

- `tokyo_per_query_metrics.csv`
- `tokyo_core_overall_metrics.csv`
- `tokyo_core_metrics_by_query_type.csv`
- `tokyo_high_memory_load_metrics.csv`
- `tokyo_between_stress_metrics.csv`
- `tokyo_pairwise_wilcoxon.csv`
- `tokyo_graph_query_diagnostics.csv`
- `tokyo_graph_diagnostic_summary.csv`

## 測試

```bash
python -m unittest discover -v
```

測試不會下載模型；模型下載與實際模型推論另由整合驗收處理。

## 專案結構

```text
.
├── main.py                         # 統一 CLI 入口
├── src/                            # 各階段實作
├── tests/                          # 單元與整合測試
├── data/
│   ├── raw/                        # Massive-STEPS Tokyo 原始 CSV
│   ├── prepared/                   # Stage 1–4 可重現產物
│   └── outputs/stage_05_evaluation # 正式評估結果
├── models/                         # 模型 metadata；權重由 Git 排除
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## 可重現性與隱私

三份 Stage 4／5 run manifests 含有本機 Windows 絕對路徑，因此由 `.gitignore` 精準排除：

- `tokyo_vector_run_manifest.json`
- `tokyo_graph_run_manifest.json`
- `tokyo_evaluation_run_manifest.json`

實際 rankings、embeddings、audit 與正式 result CSV 仍會提交。可攜的模型 revision、必要檔案 hash 與參考 runtime versions 則保留於 `models/model_manifest.json` 和 `models/reference_runtime_manifest.json`。

倉庫使用上游資料提供的匿名識別碼。請勿嘗試重新識別個人，或為此目的與外部資料進行串接。

## 資料與模型歸屬

Tokyo check-ins 來自 Wilson Wongso、Hao Xue 與 Flora D. Salim 建立的 [Massive-STEPS](https://github.com/CRUISEResearchGroup/Massive-STEPS)。上游 repository 採 Apache-2.0，並以 Semantic Trails 及 POI metadata 為基礎。使用本資料時請引用：

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

Vector 使用 Apache-2.0 的 [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)。模型權重不存放於 Git。

詳細來源與授權界線請見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 授權

本專案自行撰寫的程式碼與文件採 [MIT License](LICENSE)。第三方資料、模型與 dependencies 仍適用各自的授權與使用條款。

