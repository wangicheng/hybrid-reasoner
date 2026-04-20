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
| required_exact_cover_rate | 1.0000 | 1.0000 | +0.0000 |
| blocked_clean_rate | 1.0000 | 1.0000 | +0.0000 |
| required_micro_f1 | 0.4651 | 0.4819 | +0.0168 |
| required_macro_f1 | 0.0719 | 0.0694 | -0.0024 |
| required_exact_match_rate | 0.0714 | 0.0714 | +0.0000 |
| raw_outside_taxonomy_rate | 0.0273 | 0.0000 | -0.0273 |
| avg_pred_tag_count | 4.5833 | 4.5833 | +0.0000 |
