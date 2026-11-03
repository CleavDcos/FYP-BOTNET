# MLPClassifier: Binary

## Training and validation protocol

- **Training rows:** 2,567,965 (all eligible rows; no sampling or resampling)
- **Training class counts:** `{'0': 334, '1': 2567631}`
- **Rows processed per epoch:** 2,567,965
- **Optimization batch size:** 65,536; batches cover the full training set each epoch
- **Validation rows:** 550,278 (full validation task split)
- **Validation class counts:** `{'0': 72, '1': 550206}`
- **Class weighting:** Balanced per-row sample weights computed from the complete task training split.
- **Early stopping:** monitor validation macro-F1; best epoch 13 of 17 completed
- **Training time:** 69.16s
- **Model artifact:** `models\mlp_binary_full_data.joblib`
- **Test split used:** No

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.999967 |
| Balanced accuracy | 0.999984 |
| Macro precision | 0.900000 |
| Macro recall | 0.999984 |
| Macro F1 | 0.944436 |
| Weighted F1 | 0.999969 |
| False positive rate | 0.000000 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[72, 0], [18, 550188]]`
Counts: TN=72, FP=0, FN=18, TP=550188.

## Overfitting and class-balance notes

L2 regularization is used; the best validation macro-F1 checkpoint is retained and training stops after the configured patience or at the maximum epoch limit. Balanced sample weights change each row's loss contribution but do not change which rows are used. This experiment did not use test data.
The validation set contains only 72 Normal samples; Normal-class metrics are sensitive to a few records.
Validation metrics are for model comparison; the test set was not used for these full-data runs.
