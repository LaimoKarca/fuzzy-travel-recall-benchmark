# E5 Graph 方向性消融驗收報告

本報告只分析 107 題 relational-after；所有設定除展開方向外均固定。

## Metrics

| Configuration | MRR | S@1 | S@5 | ΔMRR vs Dense |
|---|---:|---:|---:|---:|
| E5 Dense | 0.6282 | 0.4112 | 0.9252 | +0.0000 |
| E5 Graph Symmetric | 0.6304 | 0.4112 | 0.9346 | +0.0022 |
| E5 Graph NEXT-only | 0.6362 | 0.4112 | 0.9252 | +0.0081 |

## Graph diagnosis

| Configuration | Improved | Unchanged | Worsened | SELF | PREVIOUS | NEXT | Not expanded | Structural reach | Top-1 changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E5 Graph Symmetric | 10 | 86 | 11 | 51 | 0 | 55 | 1 | 55 (51.40%) | 0 |
| E5 Graph NEXT-only | 11 | 89 | 7 | 51 | 0 | 55 | 1 | 55 (51.40%) | 0 |

## Validation answers

1. NEXT-only 相對 symmetric 的 MRR 差為 +0.0059。NEXT-only 的 MRR 較高，結果支持對稱鄰居升位是部分干擾來源。
2. NEXT-only 相對 symmetric：改善 6、不變 100、惡化 1 題。
3. Symmetric 的 structural first reach 為 55/107；NEXT-only 為 55/107。可達性是否轉成排序收益須與 MRR 與逐題 delta 一起解讀。
4. NEXT-only 改變 Dense Top-1 的題數為 0/107；本結果僅適用於 E5 Top-5、一跳、等權 RRF(k=60)。
