# Phase 2 Graph Extension Validation Report

## Core overall

| method_label | query_count | mrr | success_at_1 | success_at_5 |
| --- | --- | --- | --- | --- |
| Flat BM25 | 320 | 0.9338 | 0.8875 | 0.9938 |
| E5 Dense | 320 | 0.7554 | 0.6188 | 0.9406 |
| E5 Graph-1Hop-RRF | 320 | 0.7539 | 0.6188 | 0.9344 |
| E5 Graph-PPR-RRF | 320 | 0.6746 | 0.4750 | 0.9406 |

## MRR by query type

| query_type | method_label | mrr |
| --- | --- | --- |
| relational_after | Flat BM25 | 0.8445 |
| relational_after | E5 Dense | 0.6282 |
| relational_after | E5 Graph-1Hop-RRF | 0.6304 |
| relational_after | E5 Graph-PPR-RRF | 0.5583 |
| semantic | Flat BM25 | 0.9857 |
| semantic | E5 Dense | 0.8917 |
| semantic | E5 Graph-1Hop-RRF | 0.8917 |
| semantic | E5 Graph-PPR-RRF | 0.7724 |
| temporal_spatial | Flat BM25 | 0.9718 |
| temporal_spatial | E5 Dense | 0.7488 |
| temporal_spatial | E5 Graph-1Hop-RRF | 0.7423 |
| temporal_spatial | E5 Graph-PPR-RRF | 0.6947 |

## Graph-PPR-RRF diagnosis

- Versus E5 Dense: 28 improved, 203 unchanged, 89 worsened.
- Versus Graph-1Hop-RRF: 31 improved, 201 unchanged, 88 worsened.
- Top-1 identical to Dense: 234/320 (changed: 86).
- Any-relation seed-to-target distance: 0=301, 1=8, 2=4, 3+=6, unreachable=1.
- NEXT-only seed-to-target distance: 0=301, 1=8, 2=2, 3+=0, unreachable=9.

## Structured NEXT oracle

- Target reachable before category filtering: 107/107.
- Target uniquely resolved after category filtering: 107/107.
- This is an oracle structured-cue diagnostic, not general text-retrieval performance.

## User-level Wilcoxon comparisons

| comparison | observation_count | mean_paired_difference_b_minus_a | raw_p_value | holm_adjusted_p_value |
| --- | --- | --- | --- | --- |
| flat_vs_vector | 108 | -0.1790 | 0.0000 | 0.0000 |
| vector_vs_graph | 108 | -0.0013 | 0.4938 | 0.4938 |
| vector_vs_graph_ppr_rrf | 108 | -0.0797 | 0.0000 | 0.0000 |
| graph_vs_graph_ppr_rrf | 108 | -0.0784 | 0.0000 | 0.0000 |

## Interpretation boundary

Graph-PPR-RRF holds the encoder, Top-5 seeds, candidate set, and RRF parameters fixed, while replacing local sequential expansion with heterogeneous PPR evidence. Relation coverage and graph scoring also differ, so the comparison must not be described as changing hop depth alone.

## Acceptance questions

1. **Does PPR-RRF outperform Dense or Graph-1Hop-RRF?** No. Its Core MRR is 0.6746, compared with 0.7554 for E5 Dense and 0.7539 for Graph-1Hop-RRF. The user-level comparisons against both configurations remain different after Holm correction.
2. **Are rank changes concentrated in relational-after queries?** Not exclusively. Relational-after accounts for 15 of 28 improvements and 30 of 89 worsenings versus Dense; changes also occur in semantic and temporal-spatial queries.
3. **Does PPR-RRF change Top-1 decisions?** Yes. It changes 86/320 Top-1 events relative to Dense, but aggregate Success@1 falls from 0.6188 to 0.4750.
4. **How far are targets from the E5 Top-5 seeds?** Any-relation distances are 0=301, 1=8, 2=4, 3+=6, unreachable=1; NEXT-only distances are 0=301, 1=8, 2=2, 3+=0, unreachable=9. The difference identifies reachability introduced by POI, Category, and Trail links rather than sequential adjacency alone.
5. **Does the graph retain the benchmark's AFTER relation under oracle structured cues?** Yes for this dataset: 107/107 targets are reachable before category filtering and 107/107 are uniquely resolved after filtering. This is not text-query retrieval performance.
6. **What mechanism is most consistent with the combined evidence?** Broader heterogeneous PPR diffusion does not improve the local Graph-1Hop-RRF result, while the oracle confirms that the required NEXT structure is present. Together with the distance and rank-change diagnostics, the result is more consistent with limitations of generic static diffusion and structural-to-ranking conversion than with missing graph information alone. Query-conditioned relation semantics remain an untested alternative, so no causal claim is made.
