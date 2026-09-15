# PPR Personalization Sensitivity Validation Report

This is a fixed post-hoc sensitivity analysis. It is not a fifth primary retrieval configuration.

## Overall metrics

| configuration_label | query_count | mrr | success_at_1 | success_at_5 |
| --- | --- | --- | --- | --- |
| E5 Dense | 320 | 0.7554 | 0.6188 | 0.9406 |
| Top-5 Raw PPR | 320 | 0.6088 | 0.4125 | 0.9406 |
| Top-5 PPR-RRF | 320 | 0.6746 | 0.4750 | 0.9406 |
| All-event Raw PPR | 320 | 0.2945 | 0.1594 | 0.4188 |
| All-event PPR-RRF | 320 | 0.5386 | 0.3500 | 0.7969 |

## MRR by query type

| query_type | configuration_label | mrr |
| --- | --- | --- |
| relational_after | E5 Dense | 0.6282 |
| relational_after | Top-5 Raw PPR | 0.5708 |
| relational_after | Top-5 PPR-RRF | 0.5583 |
| relational_after | All-event Raw PPR | 0.3477 |
| relational_after | All-event PPR-RRF | 0.5297 |
| semantic | E5 Dense | 0.8917 |
| semantic | Top-5 Raw PPR | 0.6313 |
| semantic | Top-5 PPR-RRF | 0.7724 |
| semantic | All-event Raw PPR | 0.2781 |
| semantic | All-event PPR-RRF | 0.5353 |
| temporal_spatial | E5 Dense | 0.7488 |
| temporal_spatial | Top-5 Raw PPR | 0.6247 |
| temporal_spatial | Top-5 PPR-RRF | 0.6947 |
| temporal_spatial | All-event Raw PPR | 0.2578 |
| temporal_spatial | All-event PPR-RRF | 0.5506 |

## Rank diagnostics

| stratum | comparison | query_count | improved | unchanged | worsened | baseline_mrr | candidate_mrr | baseline_success_at_1 | candidate_success_at_1 | baseline_success_at_5 | candidate_success_at_5 | top1_changed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all_event_ppr_rrf_vs_e5_dense | 320 | 40 | 114 | 166 | 0.7554 | 0.5386 | 0.6188 | 0.3500 | 0.9406 | 0.7969 | 172 |
| overall | all_event_ppr_rrf_vs_top5_ppr_rrf | 320 | 57 | 108 | 155 | 0.6746 | 0.5386 | 0.4750 | 0.3500 | 0.9406 | 0.7969 | 173 |
| overall | all_event_raw_ppr_vs_top5_raw_ppr | 320 | 40 | 38 | 242 | 0.6088 | 0.2945 | 0.4125 | 0.1594 | 0.9406 | 0.4188 | 250 |
| target_in_dense_top5 | all_event_ppr_rrf_vs_e5_dense | 301 | 33 | 112 | 156 | 0.7976 | 0.5660 | 0.6578 | 0.3721 | 1.0000 | 0.8372 | 162 |
| target_in_dense_top5 | all_event_ppr_rrf_vs_top5_ppr_rrf | 301 | 49 | 108 | 144 | 0.7116 | 0.5660 | 0.5050 | 0.3721 | 1.0000 | 0.8372 | 163 |
| target_in_dense_top5 | all_event_raw_ppr_vs_top5_raw_ppr | 301 | 31 | 38 | 232 | 0.6421 | 0.3062 | 0.4385 | 0.1694 | 1.0000 | 0.4352 | 235 |
| target_outside_dense_top5 | all_event_ppr_rrf_vs_e5_dense | 19 | 7 | 2 | 10 | 0.0859 | 0.1057 | 0.0000 | 0.0000 | 0.0000 | 0.1579 | 10 |
| target_outside_dense_top5 | all_event_ppr_rrf_vs_top5_ppr_rrf | 19 | 8 | 0 | 11 | 0.0873 | 0.1057 | 0.0000 | 0.0000 | 0.0000 | 0.1579 | 10 |
| target_outside_dense_top5 | all_event_raw_ppr_vs_top5_raw_ppr | 19 | 9 | 0 | 10 | 0.0807 | 0.1099 | 0.0000 | 0.0000 | 0.0000 | 0.1579 | 15 |

## Dense Top-5 strata

- Targets already in Dense Top-5: 301; All-event PPR-RRF produced 33 improvements, 112 unchanged ranks, and 156 worsenings. Its S@5 in this stratum was 0.8372.
- Targets outside Dense Top-5: 19; All-event PPR-RRF produced 7 improvements, 2 unchanged ranks, and 10 worsenings. It moved 3 targets into the Top-5.
- Raw PPR MRR changed from 0.6088 with Top-5 personalization to 0.2945 with all-event personalization; equal-weight RRF raised the all-event result to 0.5386.

## Pre-specified interpretation

All-event personalization did not improve the Top-5-personalized PPR-RRF result; Top-5 truncation is not the main explanation.

Raw PPR and RRF are reported separately. No new significance test was added, and no PPR, graph, or fusion parameter was tuned after observing the result.
