# Tag Extraction A/B Report

## Experiment Setup

| Item | Value |
| --- | --- |
| total_queries | 24 |
| extraction_model | models/gemma-4-31b-it |
| embedding_model | models/gemini-embedding-2-preview |
| prediction_key | mapped_pred_tags |
| no_desc_prediction_key | mapped_pred_tags |
| with_desc_prediction_key | mapped_pred_tags |

## Metrics

| Metric | No Description | With Description | Delta |
| --- | --- | --- | --- |
| parse_success_rate | 1.0000 | 1.0000 | +0.0000 |
| required_exact_cover_rate | 0.9286 | 1.0000 | +0.0714 |
| blocked_clean_rate | 1.0000 | 1.0000 | +0.0000 |
| required_micro_f1 | 0.4270 | 0.4651 | +0.0381 |
| required_macro_f1 | 0.0573 | 0.0690 | +0.0117 |
| required_exact_match_rate | 0.0000 | 0.0714 | +0.0714 |
| raw_outside_taxonomy_rate | 0.1707 | 0.0179 | -0.1529 |
| avg_pred_tag_count | 5.0417 | 4.6667 | -0.3750 |
