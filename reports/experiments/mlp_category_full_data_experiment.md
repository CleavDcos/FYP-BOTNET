# MLPClassifier: Category

## Training and validation protocol

- **Training rows:** 2,567,631 (all eligible rows; no sampling or resampling)
- **Training class counts:** `{'DDoS': 1348005, 'DoS': 1155814, 'Reconnaissance': 63757, 'Theft': 55}`
- **Rows processed per epoch:** 2,567,631
- **Optimization batch size:** 65,536; batches cover the full training set each epoch
- **Validation rows:** 550,206 (full validation task split)
- **Validation class counts:** `{'DDoS': 289275, 'DoS': 247257, 'Reconnaissance': 13662, 'Theft': 12}`
- **Class weighting:** Balanced per-row sample weights computed from the complete task training split.
- **Early stopping:** monitor validation macro-F1; best epoch 19 of 20 completed
- **Training time:** 424.46s
- **Model artifact:** `models\mlp_category_full_data.joblib`
- **Test split used:** No

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.997388 |
| Balanced accuracy | 0.998478 |
| Macro precision | 0.855348 |
| Macro recall | 0.998478 |
| Macro F1 | 0.898340 |
| Weighted F1 | 0.997394 |

Confusion matrix label order: `['DDoS', 'DoS', 'Reconnaissance', 'Theft']`.

## Overfitting and class-balance notes

L2 regularization is used; the best validation macro-F1 checkpoint is retained and training stops after the configured patience or at the maximum epoch limit. Balanced sample weights change each row's loss contribution but do not change which rows are used. This experiment did not use test data.
The validation set contains only 12 Theft samples; Theft-class metrics are highly uncertain.
Validation metrics are for model comparison; the test set was not used for these full-data runs.
