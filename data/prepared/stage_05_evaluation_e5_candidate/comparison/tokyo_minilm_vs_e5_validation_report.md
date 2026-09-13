# MiniLM vs E5 驗收報告

本報告只比較 Main Core 320 題；E5 尚未取代正式 Stage 5。

## Core metrics

| Configuration | MRR | S@1 | S@5 | Semantic MRR | Temporal-Spatial MRR | Relational-After MRR |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM Dense | 0.7908 | 0.6906 | 0.9344 | 0.9310 | 0.8292 | 0.6144 |
| E5 Dense | 0.7554 | 0.6188 | 0.9406 | 0.8917 | 0.7488 | 0.6282 |
| MiniLM Graph | 0.7883 | 0.6906 | 0.9281 | 0.9319 | 0.8268 | 0.6084 |
| E5 Graph | 0.7539 | 0.6188 | 0.9344 | 0.8917 | 0.7423 | 0.6304 |

## Graph diagnosis

| Encoder | Improved | Unchanged | Worsened | SELF | PREVIOUS | NEXT | Not expanded | Structural reach | Top-1 identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | 5 | 289 | 26 | 295 | 1 | 6 | 18 | 7 (2.19%) | 320 |
| E5 | 12 | 280 | 28 | 250 | 2 | 57 | 11 | 59 (18.44%) | 320 |
| E5_minus_MiniLM | 7 | -9 | 2 | -45 | 1 | 51 | -7 | 52 (16.25%) | 0 |

## Validation answers

1. E5 Dense 相對 MiniLM Dense 的 Core MRR 絕對差為 -0.0354；因此 E5 並未改善本 benchmark 的整體 Dense MRR。各 query type 與 Success 指標仍須分開閱讀。
2. E5 Graph 相對 MiniLM Graph 的 Core MRR 絕對差為 -0.0343，相對同次 E5 Dense 則為 -0.0015。PREVIOUS／NEXT 首次找到 target 的 Core 題數由 7 增至 59，但 E5 Vector／Graph 的 Top-1 仍有 320/320 相同。這表示 E5 seeds 提高了結構可達性，但固定 Graph reranking 仍未轉化為整體 MRR 或 rank-1 優勢。

這些差異是受控重跑的描述結果，不單獨構成因果證明。
