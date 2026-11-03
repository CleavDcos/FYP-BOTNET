# LinearSVC: Binary

## Training and validation protocol

- **Training rows:** 2,567,965 (all eligible rows; no sampling or resampling)
- **Training class counts:** `{'0': 334, '1': 2567631}`
- **Validation rows:** 550,278 (full validation task split)
- **Validation class counts:** `{'0': 72, '1': 550206}`
- **Optimization:** LinearSVC fit on the complete sparse training matrix.
- **Class weighting:** LinearSVC class_weight='balanced'.
- **Regularization / solver:** L2 with C=0.1; solver stops by tolerance or max_iter. Iterations used: 5000; converged: False
- **Training time:** 475.07s
- **Model artifact:** `models\svm_binary_full_data.joblib`
- **Test split used:** No

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.999964 |
| Balanced accuracy | 0.993038 |
| Macro precision | 0.894444 |
| Macro recall | 0.993038 |
| Macro F1 | 0.938263 |
| Weighted F1 | 0.999966 |
| False positive rate | 0.013889 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[71, 1], [19, 550187]]`
Counts: TN=71, FP=1, FN=19, TP=550187.

## Overfitting and class-balance notes

L2 regularization (C=0.1) limits model complexity. The solver convergence flag above matters: the binary fit reached its iteration limit, so treat it as a provisional candidate. This experiment did not use test data.
The validation set contains only 72 Normal samples; Normal-class metrics are sensitive to a few records.
Validation metrics are for model comparison; the test set was not used for these full-data runs.
