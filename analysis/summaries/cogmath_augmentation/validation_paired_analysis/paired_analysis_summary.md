# Paired Validation Robustness Analysis

`original_plus_all_full` answered +13 more validation records correctly than `original_only_large`.

## Where Did The Additional Correct Answers Come From?

Original: +3; Dim1 paraphrasing: +0; Dim2 word scrambling: +3; Dim4 irrelevant information: +7; Dim6 numerical variation: +0.

## Did Mixed Augmentation Improve Every Transformation?

No. The paired count differences are form-specific; see `paired_gain_loss.csv` for rescues and regressions.

## Statistical Support

Overall McNemar exact p-value: 0.2323. This does not support describing the overall paired difference as statistically significant at p < 0.05.

Dim4 Holm-adjusted p-value: 1. This does not support describing the Dim4 improvement as statistically significant after Holm correction.

## Cluster Bootstrap

Overall observed difference: 1.313 percentage points (95% CI -1.010, 3.737).
Mean variant observed difference: 1.263 percentage points (95% CI -1.136, 3.535).

## Consistent Performance

`original_only_large` average correct forms per base problem: 0.697; `original_plus_all_full`: 0.763.
`original_only_large` all-five-correct base problems: 3; `original_plus_all_full`: 4.

## Conditional Robustness

Among base problems where the original form was solved, average variant correctness was 0.477 for `original_only_large` and 0.521 for `original_plus_all_full`.

Do not call any result statistically significant unless the corresponding p-value/CI supports it.
