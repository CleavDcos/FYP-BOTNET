# LinearSVC: Category

## Training and validation protocol

- **Training rows:** 2,567,631 (all eligible rows; no sampling or resampling)
- **Training class counts:** `{'DDoS': 1348005, 'DoS': 1155814, 'Reconnaissance': 63757, 'Theft': 55}`
- **Validation rows:** 550,206 (full validation task split)
- **Validation class counts:** `{'DDoS': 289275, 'DoS': 247257, 'Reconnaissance': 13662, 'Theft': 12}`
- **Optimization:** LinearSVC fit on the complete sparse training matrix.
- **Class weighting:** LinearSVC class_weight='balanced'.
- **Regularization / solver:** L2 with C=0.1; solver stops by tolerance or max_iter. Iterations used: 701; converged: True
- **Training time:** 327.27s
- **Model artifact:** `models\svm_category_full_data.joblib`
- **Test split used:** No

## Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.912229 |
| Balanced accuracy | 0.759578 |
| Macro precision | 0.747907 |
| Macro recall | 0.759578 |
| Macro F1 | 0.752831 |
| Weighted F1 | 0.912050 |

Confusion matrix label order: `['DDoS', 'DoS', 'Reconnaissance', 'Theft']`.

## Overfitting and class-balance notes

L2 regularization (C=0.1) limits model complexity. The solver convergence flag above matters: the binary fit reached its iteration limit, so treat it as a provisional candidate. This experiment did not use test data.
The validation set contains only 12 Theft samples; Theft-class metrics are highly uncertain.
Validation metrics are for model comparison; the test set was not used for these full-data runs.
